"""Training-only endpoints extracted from observed occupancy and true instances.

Boxes provide a conservative association proxy, not dense instance identity.
Unknown/unobserved voxels never produce targets or interpolated hidden states.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from mmdet.datasets.builder import PIPELINES
except ModuleNotFoundError:  # Pure CPU geometry tests do not need MMDetection.
    PIPELINES = None

SCHEMA_VERSION = 'endpoint-segments-v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stable_key(token):
    return hashlib.sha256(token.encode()).digest()


def observed_box_points(semantics, camera_mask, frame, pc_range, voxel_size):
    """Return visible class-matched voxel centers with unique compatible OBB owner."""
    semantics = np.asarray(semantics)
    camera_mask = np.asarray(camera_mask, dtype=bool)
    if semantics.shape != camera_mask.shape or semantics.ndim != 3:
        raise ValueError('Occupancy and visibility must have matching 3D shapes')
    shape = np.asarray(semantics.shape)
    lower = np.asarray(pc_range[:3],dtype=np.float64)
    spacing = np.asarray(voxel_size,dtype=np.float64)
    if not np.allclose(shape*spacing,np.asarray(pc_range[3:])-lower):
        raise ValueError('Occupancy shape does not match metric grid contract')
    owner = np.full(semantics.size,-1,dtype=np.int32)
    candidates = []
    signs = np.array([[i,j,k] for i in (-1,1) for j in (-1,1) for k in (-1,1)])
    for index,(center,R,size,label) in enumerate(zip(frame['centers'],frame['rotations'],frame['sizes'],frame['labels'])):
        center=np.asarray(center,dtype=np.float64);R=np.asarray(R,dtype=np.float64);half=np.asarray(size,dtype=np.float64)/2
        if not np.isfinite(center).all() or not np.isfinite(R).all() or not np.isfinite(half).all() or (half<=0).any():
            raise ValueError('Invalid annotation box')
        if not np.allclose(R.T@R,np.eye(3),atol=1e-4):
            raise ValueError('Annotation box rotation is not orthogonal')
        corners=(signs*half)@R.T+center
        lo=np.maximum(np.floor((corners.min(0)-lower)/spacing).astype(int),0)
        hi=np.minimum(np.ceil((corners.max(0)-lower)/spacing).astype(int),shape)
        if np.any(hi<=lo):
            candidates.append(np.empty(0,dtype=np.int64));continue
        indices=np.stack(np.meshgrid(*[np.arange(lo[d],hi[d]) for d in range(3)],indexing='ij'),-1).reshape(-1,3)
        xyz=(indices+.5)*spacing+lower
        inside=(np.abs((xyz-center)@R)<=half+1e-6).all(1)
        indices=indices[inside]
        valid=(semantics[tuple(indices.T)]==int(label)) & camera_mask[tuple(indices.T)]
        indices=indices[valid]
        flat=np.ravel_multi_index(indices.T,semantics.shape)
        old=owner[flat]
        owner[flat]=np.where(old==-1,index,-2)  # -2 stays ambiguous on any later overlap.
        candidates.append(flat)
    output=[]
    for index,flat in enumerate(candidates):
        flat=flat[owner[flat]==index]
        indices=np.stack(np.unravel_index(flat,semantics.shape),-1)
        output.append(((indices+.5)*spacing+lower).astype(np.float32))
    return output


class LoadEndpointSegments:
    def __init__(self,cache_root,split='train',max_instances=32,samples_per_endpoint=16,
                 pc_range=(-40.,-40.,-1.,40.,40.,5.4),voxel_size=(.4,.4,.4),cache_size=128):
        self.root=Path(cache_root);self.split=split
        self.max_instances=int(max_instances);self.samples_per_endpoint=int(samples_per_endpoint)
        if not 1<=self.max_instances<=32 or not 1<=self.samples_per_endpoint<=16:
            raise ValueError('Endpoint targets require 1..32 instances and 1..16 points')
        self.pc_range=tuple(pc_range);self.voxel_size=tuple(voxel_size);self.cache_size=int(cache_size)
        manifest=json.loads((self.root/'manifest.json').read_text())
        if manifest.get('schema_version')!=SCHEMA_VERSION or manifest.get('occupancy_labels_cached') is not False:
            raise ValueError('Unsupported endpoint metadata contract')
        if split not in manifest['splits']:
            raise ValueError('Endpoint split is missing')
        anchors_path=self.root/split/'anchors.json'
        if _sha(anchors_path)!=manifest['splits'][split]['anchors_sha256']:
            raise ValueError('Endpoint anchor index checksum mismatch')
        self.anchors=json.loads(anchors_path.read_text());self.horizons=manifest['horizons']
        if set(self.anchors)!=set(manifest['splits'][split]['anchor_tokens']):
            raise ValueError('Endpoint token manifest mismatch')
        self._frames=OrderedDict()

    def _frame(self,token):
        if token not in self._frames:
            with np.load(self.root/self.split/'frames'/(token+'.npz'),allow_pickle=False) as data:
                frame={k:data[k] for k in data.files}
            n=len(frame['instance_tokens'])
            if len(set(frame['instance_tokens'].tolist()))!=n:
                raise ValueError('Duplicated true instance token')
            for key,shape in [('centers',(n,3)),('rotations',(n,3,3)),('sizes',(n,3)),('labels',(n,))]:
                if frame[key].shape!=shape:
                    raise ValueError('Invalid endpoint frame array '+key)
            self._frames[token]=frame
            while len(self._frames)>self.cache_size:
                self._frames.popitem(last=False)
        self._frames.move_to_end(token)
        return self._frames[token]

    def __call__(self,results):
        if 'voxel_semantics' not in results or 'mask_camera' not in results:
            raise ValueError('Endpoint targets require already loaded training occupancy')
        horizons=[int(x) for x in results['fut_list']]
        if horizons!=self.horizons:
            raise ValueError('Endpoint horizons differ from frozen cache protocol')
        anchor=results['sample_idx']
        if anchor not in self.anchors:
            raise ValueError('Anchor is not in configured endpoint split')
        tokens=self.anchors[anchor];frames=[self._frame(t) for t in tokens];F=len(tokens)
        if len(results['voxel_semantics'])!=F or len(results['mask_camera'])!=F or len(results['fut2cur'])!=F:
            raise ValueError('Endpoint target horizon length mismatch')
        times=np.array([(int(f['timestamp_us'])-int(frames[0]['timestamp_us']))/1e6 for f in frames],dtype=np.float32)
        if times[0]!=0 or np.any(np.diff(times)<=0):
            raise ValueError('Endpoint physical timestamps are not increasing')
        R0=frames[0]['ego2global_rotation'];t0=frames[0]['ego2global_translation']
        per_instance={}
        for h,frame in enumerate(frames):
            T=np.asarray(results['fut2cur'][h],dtype=np.float64)
            expected=np.eye(4);expected[:3,:3]=R0.T@frame['ego2global_rotation'];expected[:3,3]=R0.T@(frame['ego2global_translation']-t0)
            if T.shape!=(4,4) or not np.allclose(T,expected,atol=5e-4,rtol=1e-5):
                raise ValueError('Endpoint pose differs from occupancy future-to-current pose')
            points=observed_box_points(results['voxel_semantics'][h],results['mask_camera'][h],frame,self.pc_range,self.voxel_size)
            for i,instance in enumerate(frame['instance_tokens'].tolist()):
                label=int(frame['labels'][i])
                if instance not in per_instance:per_instance[instance]={'label':label,'points':[None]*F}
                elif per_instance[instance]['label']!=label:raise ValueError('Instance semantic label changed')
                if len(points[i]):
                    xyz=points[i]@T[:3,:3].T+T[:3,3]
                    per_instance[instance]['points'][h]=xyz.astype(np.float32)
        keep=[]
        for token,item in per_instance.items():
            observed=np.array([p is not None for p in item['points']])
            if observed.sum()>=2 and observed[1:].any():keep.append(token)
        keep=sorted(keep,key=lambda token:(_stable_key(token),token))[:self.max_instances]
        S=self.samples_per_endpoint;out=np.zeros((len(keep),F,S,3),dtype=np.float32);valid=np.zeros((len(keep),F,S),dtype=bool);labels=[]
        for m,token in enumerate(keep):
            item=per_instance[token];labels.append(item['label'])
            for h,points in enumerate(item['points']):
                if points is None:continue
                # Stable hash-seeded selection, independent of worker RNG and box ordering.
                seed=int.from_bytes(_stable_key(token+':'+tokens[h])[:4],'little')
                chosen=np.random.RandomState(seed).permutation(len(points))[:S]
                out[m,h,:len(chosen)]=points[chosen];valid[m,h,:len(chosen)]=True
        results['endpoint_segments']={'points':out,'point_valid':valid,'observed':valid.any(-1),
            'labels':np.asarray(labels,dtype=np.int64),'times':times,'instance_tokens':keep}
        return results


if PIPELINES is not None:
    PIPELINES.register_module()(LoadEndpointSegments)
