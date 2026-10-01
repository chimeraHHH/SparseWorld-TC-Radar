"""Explicit real-history selection, source identity and integer camera timestamps."""
import copy
import os
from pathlib import Path
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from .loading import compose_lidar2img

CAMERAS = ('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT')


def candidate_choices(count, test_mode, rng=np.random, interval=None):
    if count == 0: return [None] * 7
    if test_mode: choices = [(k+1)*6-1 for k in range(7)]
    elif count <= 7: choices = list(range(count)) + [count-1]*(7-count)
    else:
        maximum = min(count//7, 8);minimum = min(maximum, 4)
        step = int(rng.randint(minimum, maximum+1)) if interval is None else interval
        if not minimum <= step <= maximum: raise ValueError('Invalid training interval')
        choices = [(k+1)*step-1 for k in range(7)]
    return [min(i,count-1) for i in choices]


def plan_history(results, frames, test_mode, rng=np.random, interval=None):
    if frames not in (2,8): raise ValueError('Only the authorized H2/H8 budgets')
    times = results['img_timestamp_us'];files=results['filename']
    if len(times)!=6 or len(files)!=6: raise ValueError('Expected six current camera identities')
    prev=results['cam_sweeps']['prev'];plans=[]
    for choice in candidate_choices(len(prev),test_mode,rng,interval):
        if choice is None: plans.append(None);continue
        # Match the original complete-group fallback, without negative-index wraparound.
        order=list(range(choice,-1,-1))+list(range(choice+1,len(prev)))
        valid=None
        for i in order:
            sweep=prev[i]
            if not all(c in sweep for c in CAMERAS): continue
            if all(int(sweep[c]['timestamp']) < int(times[j]) and
                   os.path.normpath(sweep[c]['data_path']) != os.path.normpath(files[j])
                   for j,c in enumerate(CAMERAS)):
                valid=i;break
        plans.append(valid)
    if frames==2:
        eligible=[i for i in plans if i is not None]
        # Choose the nearest complete causal candidate, preserving the H8 candidate pool.
        nearest=max(eligible,key=lambda i: min(int(prev[i][c]['timestamp']) for c in CAMERAS)) if eligible else None
        plans=[nearest]
    return plans


@PIPELINES.register_module()
class LoadBudgetedVisualHistory:
    def __init__(self, frames, test_mode=False, color_type='color'):
        self.frames,self.test_mode,self.color_type=frames,test_mode,color_type
        if frames not in (2,8): raise ValueError('Only H2/H8 are authorized')
        try: mmcv.use_backend('turbojpeg')
        except ImportError: mmcv.use_backend('cv2')

    def __call__(self, results):
        plans=plan_history(results,self.frames,self.test_mode)
        reference=int(results['reference_timestamp_us'])
        results['visual_source_kind']=['current']*6
        results['visual_source_id']=[os.path.normpath(f) for f in results['filename']]
        results['visual_time_delta_s']=[np.float32((int(t)-reference)/1e6) for t in results['img_timestamp_us']]
        seen=set(results['visual_source_id'])
        for choice in plans:
            for j,camera in enumerate(CAMERAS):
                if choice is None:
                    for key in ('img','filename','img_timestamp','img_timestamp_us','lidar2img','intrinsics','extrinsics','lidar2cam'):
                        if key in results: results[key].append(copy.deepcopy(results[key][j]))
                    source=results['visual_source_id'][j];kind='missing_fill';stamp=results['img_timestamp_us'][j]
                else:
                    sensor=results['cam_sweeps']['prev'][choice][camera]
                    source=os.path.normpath(sensor['data_path']);stamp=int(sensor['timestamp'])
                    kind='duplicate_history' if source in seen else 'real_history'
                    results['img'].append(mmcv.imread(sensor['data_path'],self.color_type))
                    results['filename'].append(os.path.relpath(sensor['data_path']))
                    results['img_timestamp'].append(stamp/1e6);results['img_timestamp_us'].append(stamp)
                    projection=compose_lidar2img(results['ego2global_translation'],results['ego2global_rotation'],
                        results['lidar2ego_translation'],results['lidar2ego_rotation'],
                        sensor['sensor2global_translation'],sensor['sensor2global_rotation'],sensor['cam_intrinsic'])
                    results['lidar2img'].append(projection)
                    intrinsic=np.eye(4);shape=sensor['cam_intrinsic'].shape;intrinsic[:shape[0],:shape[1]]=sensor['cam_intrinsic']
                    results['intrinsics'].append(intrinsic)
                    extrinsic=np.eye(4);extrinsic[:3,:3]=sensor['sensor2ego_rotation'];extrinsic[:3,3]=sensor['sensor2ego_translation']
                    results['extrinsics'].append(extrinsic)
                    # Original pipelines left this list at6; now every selected view has its own matrix.
                    results['lidar2cam'].append(np.linalg.inv(intrinsic)@projection)
                results['visual_source_id'].append(source);results['visual_source_kind'].append(kind)
                results['visual_time_delta_s'].append(np.float32((int(stamp)-reference)/1e6));seen.add(source)
        results['visual_history_frames']=self.frames
        results['visual_history_choices']=plans
        results['visual_history_encoded_images']=6*self.frames
        return results
