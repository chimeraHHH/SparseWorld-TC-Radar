"""BS1 paired real-input cost; cached replay is gated by trained-output parity."""
import argparse,copy,hashlib,json,os,time,collections,gc
from pathlib import Path
from types import MethodType
import numpy as np
import torch
import mmcv
from mmcv.parallel import MMDataParallel,collate
from mmcv.runner import wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
import models,loaders
from tools.check_censored_path_contracts import compare_voxels
from tools.check_history_doppler_contracts import tensor_parity
from run_history_budget_campaign import ARMS


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class FrameCache:
    """Same per-view identity and transform, no cross-scene or future reuse."""
    def __init__(self,net,frames,max_groups=16):
        self.net,self.frames,self.maximum=net,frames,max_groups;self.entries=collections.OrderedDict();self.hits=self.misses=0
        self.original=net.extract_feat
    def clear(self):self.entries.clear()
    @property
    def bytes(self):return sum(f.numel()*f.element_size() for fs in self.entries.values() for f in fs)
    def extract(self,net,img,img_metas):
        assert img.shape[0]==1 and img.shape[1]==self.frames*6
        original=copy.deepcopy(img_metas[0]);levels=[]
        for slot in range(self.frames):
            ids=list(range(slot*6,(slot+1)*6));m={}
            for k,v in original.items():
                m[k]=[copy.deepcopy(v[i]) for i in ids] if isinstance(v,(list,tuple)) and len(v)==self.frames*6 else copy.deepcopy(v)
            key=(tuple(m['filename']),tuple(img.shape[-2:]),str(img.dtype))
            # Image identity/resolution fix deterministic test augmentation; projections remain anchor-specific.
            if key in self.entries:self.hits+=1;features=self.entries.pop(key)
            else:
                self.misses+=1;budget=net.visual_history_frames;net.visual_history_frames=None
                try:features=self.original(img[:,ids], [m])
                finally:net.visual_history_frames=budget
            self.entries[key]=features
            while len(self.entries)>self.maximum:self.entries.popitem(last=False)
            levels.append(features)
        result=[torch.cat([slot[level] for slot in levels],dim=1) for level in range(len(levels[0]))]
        if self.frames==2:
            result=[torch.cat([f[:,:6],f[:,6:12].repeat(1,7,1,1,1)],dim=1) for f in result]
            indices=list(range(6))+list(range(6,12))*7
            from models.sparse_world import _VISUAL_VIEW_META_KEYS
            for key in _VISUAL_VIEW_META_KEYS:
                value=img_metas[0].get(key)
                if isinstance(value,(list,tuple,np.ndarray)) and len(value)==12:
                    img_metas[0][key]=[copy.deepcopy(value[i]) for i in indices]
        # Original augmentation also writes shapes; test resolution is identical for each group.
        shape=(img.shape[-2],img.shape[-1],img.shape[-3])
        for k in ('img_shape','ori_shape','pad_shape'):img_metas[0][k]=[shape]*48
        img_metas[0]['input_shape']=tuple(img.shape[-2:])
        return result


def summary(rows):
    times=np.array([r['end_to_end_seconds'] for r in rows]);return dict(
        predictions=len(rows),p50_ms=float(np.percentile(times,50)*1000),p95_ms=float(np.percentile(times,95)*1000),
        serial_predictions_per_second=float(len(times)/times.sum()),
        p50_pipeline_ms=float(np.percentile([r['pipeline_seconds'] for r in rows],50)*1000),
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),
        maximum_feature_cache_bytes=max(r['feature_cache_bytes'] for r in rows),
        maximum_cpu_rss_bytes=max(r['cpu_rss_bytes'] for r in rows))


def rss():
    return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    root=Path(args.campaign);torch.set_num_threads(4);torch.manual_seed(0)
    cfg0=mmcv.Config.fromfile('configs/sw-budget-h8-velocity.py');base=build_dataset(cfg0.data.val)
    # Fixed, chronological scene sample with16 consecutive eligible anchors in16 scenes.
    scenes=collections.defaultdict(list)
    for i,x in enumerate(base.data_infos):scenes[x['scene_name']].append(i)
    indices=[i for name in sorted(scenes)[:16] for i in sorted(scenes[name],key=lambda i:base.data_infos[i]['timestamp'])[:16]]
    assert len(indices)==256
    assert next(x for x in cfg0.data.val.pipeline if x['type']=='RandomTransformImage')['training'] is False
    output=dict(protocol=dict(batch_size=1,precision='original wrap_fp16_model',warmup=30,replays=3,
        anchors=256,scenes=16,indices=indices,tokens=[base.data_infos[i]['token'] for i in indices],
        complete_horizons_seconds=[0,1,2,3],filesystem_cache='uncontrolled OS cache; never claimed cold disk',
        cache_scope='causal frame-feature reuse; loader still decodes full inputs; no historical I/O saving claimed',
        scope='matched corrected version; offline measured costs, cache replay diagnostic after16 exact-voxel/raw parity',
        energy_scope='GPU cumulative energy only; unavailable if unsupported; no whole-system estimate'),arms={})
    try:
        import pynvml
        pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByUUID(os.environ['CUDA_VISIBLE_DEVICES'])
        def energy():
            try:return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)/1000
            except Exception:return None
    except Exception:
        def energy():return None
    schedule=[list(ARMS),list(reversed(ARMS)),list(ARMS)]
    for replay,arms in enumerate(schedule):
        for arm in arms:
            cfg=mmcv.Config.fromfile(f'configs/sw-budget-{arm}.py');ds=build_dataset(copy.deepcopy(cfg.data.val));checkpoint=Path(json.loads((root/(arm+'_result.json')).read_text())['final_audit']['path'])
            started=time.monotonic();net=build_model(copy.deepcopy(cfg.model));net.init_weights();state=torch.load(checkpoint,map_location='cpu')['state_dict'];net.load_state_dict(state,strict=True);del state
            net.cuda().eval();wrap_fp16_model(net);net.simple_test=net.simple_test_offline;wrapper=MMDataParallel(net,[0]);cache=FrameCache(net,cfg.model.visual_history_frames)
            torch.cuda.synchronize();load_seconds=time.monotonic()-started
            row=output['arms'].setdefault(arm,dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),replays=[],cache_parity=[]))
            def data(i):return collate([ds[i]],samples_per_gpu=1)
            with torch.no_grad():
                warm=data(indices[0]);warm_started=time.monotonic()
                for _ in range(30):wrapper(return_loss=False,rescale=True,**copy.deepcopy(warm))
                torch.cuda.synchronize();row['model_loading_seconds']=load_seconds;row['warmup_seconds']=time.monotonic()-warm_started
                captures=[]
                def capture(_m,_inputs,value):captures.append([v.detach().cpu().clone() for v in [value['init_points']]+value['all_cls_scores']+value['all_refine_pts']])
                hook=net.pts_bbox_head.register_forward_hook(capture);accepted=True
                try:
                    for index in indices[:16]:
                        batch=data(index);captures.clear();left=wrapper(return_loss=False,rescale=True,**copy.deepcopy(batch));raw=captures[-1]
                        cache.clear();net.extract_feat=MethodType(cache.extract,net)
                        try:right=wrapper(return_loss=False,rescale=True,**copy.deepcopy(batch))
                        finally:net.extract_feat=cache.original
                        try:
                            proof=[tensor_parity(a,b,f'{arm}:{index}:{j}') for j,(a,b) in enumerate(zip(raw,captures[-1]))];compare_voxels(left,right)
                            row['cache_parity'].append(dict(index=index,replay=replay,raw_max_abs=max(x['max_abs_difference'] for x in proof),voxels_exact=True))
                        except (AssertionError,ValueError,FloatingPointError) as error:
                            accepted=False;row['cache_rejected_reason']=repr(error);break
                finally:hook.remove()
                for mode in ('independent_anchor','chronological_no_feature_cache','chronological_feature_cache'):
                    if mode=='chronological_feature_cache' and not accepted:continue
                    cache.clear();cache.hits=cache.misses=0;previous_scene=None;measure=[];initial_energy=energy()
                    if mode=='chronological_feature_cache':net.extract_feat=MethodType(cache.extract,net)
                    try:
                        for index in indices:
                            scene=ds.data_infos[index]['scene_name'];scene_switch=scene!=previous_scene
                            if scene_switch or mode=='independent_anchor':cache.clear()
                            prepared=ds.get_data_info(index)
                            # Read source identity without decoding; planned input byte counts are logical, not physical disk I/O.
                            from loaders.pipelines.history_budget import plan_history,CAMERAS
                            prepared['filename']=prepared['img_filename'];plans=plan_history(prepared,cfg.model.visual_history_frames,True)
                            files=list(prepared['filename']);stamps=list(prepared['img_timestamp_us'])
                            for choice in plans:
                                for j,c in enumerate(CAMERAS):
                                    sensor=prepared['cam_sweeps']['prev'][choice][c] if choice is not None else None
                                    files.append(sensor['data_path'] if sensor else files[j]);stamps.append(int(sensor['timestamp']) if sensor else stamps[j])
                            logical=sum(Path(f).stat().st_size for f in files)
                            previous_scene=scene;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                            batch=data(index);pipeline=time.perf_counter()-begin
                            prediction=wrapper(return_loss=False,rescale=True,**batch);torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                            measure.append(dict(index=index,scene=scene,scene_switch=scene_switch,end_to_end_seconds=elapsed,pipeline_seconds=pipeline,
                                physical_input_views=6*cfg.model.visual_history_frames,unique_input_files=len(set(files)),logical_file_bytes=logical,
                                actual_history_span_seconds=(max(stamps)-min(stamps))/1e6,
                                h2d_image_bytes=6*cfg.model.visual_history_frames*3*256*704,
                                feature_cache_bytes=cache.bytes,feature_cache_hits=cache.hits,feature_cache_misses=cache.misses,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),cpu_rss_bytes=rss()))
                            del prediction,batch
                    finally:net.extract_feat=cache.original
                    end_energy=energy();block=dict(replay=replay,mode=mode,rows=measure,summary=summary(measure),
                        gpu_j_per_prediction=(end_energy-initial_energy)/len(measure) if initial_energy is not None and end_energy is not None else None,
                        startup_scope='empty model feature cache with dataset-provided causal historical inputs; no missing-history accuracy claim')
                    row['replays'].append(block);Path(args.out).write_text(json.dumps(output,indent=2,allow_nan=False))
            cache.clear();del wrapper,net,cache,ds,warm;gc.collect();torch.cuda.empty_cache()
    output['status']='complete';Path(args.out).write_text(json.dumps(output,indent=2,allow_nan=False));print(json.dumps({a:[x['summary'] for x in r['replays']] for a,r in output['arms'].items()},indent=2))


if __name__=='__main__':main()
