"""One velocity-only cost block; scientific model and measurement body unchanged."""
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
ARMS = ('h8-velocity', 'h2-velocity')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


from velocity_feature_cache import FrameCache


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


def image_storage(batch):
    value=batch['img']
    while not torch.is_tensor(value):
        if isinstance(value,(list,tuple)):
            assert len(value)==1;value=value[0]
        else:value=value.data
    return dict(bytes=value.numel()*value.element_size(),shape=list(value.shape),dtype=str(value.dtype))


def save_output(path,value):
    target=Path(path);temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(target)


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--out',required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--replay',type=int,choices=(-1,0,1,2),required=True)
    p.add_argument('--cache-blocked-evidence',help='Preserved exact-native gate rejection; never time or claim this cached path');args=p.parse_args()
    assert not Path(args.out).exists(), 'Existing block is evidence; never overwrite or automatically retry'
    root=Path(args.campaign);torch.set_num_threads(4);torch.manual_seed(0)
    cfg0=mmcv.Config.fromfile('configs/sw-budget-h8-velocity.py');base=build_dataset(cfg0.data.val)
    # Fixed, chronological scene sample with16 consecutive eligible anchors in16 scenes.
    scenes=collections.defaultdict(list)
    for i,x in enumerate(base.data_infos):scenes[x['scene_name']].append(i)
    indices=[i for name in sorted(scenes)[:16] for i in sorted(scenes[name],key=lambda i:base.data_infos[i]['timestamp'])[:16]]
    assert len(indices)==256
    assert next(x for x in cfg0.data.val.pipeline if x['type']=='RandomTransformImage')['training'] is False
    output=dict(protocol=dict(batch_size=1,precision='original wrap_fp16_model',warmup=30,replays=1,planned_paired_replays=3,
        anchors=256,scenes=16,indices=indices,tokens=[base.data_infos[i]['token'] for i in indices],
        complete_horizons_seconds=[0,1,2,3],filesystem_cache='uncontrolled OS cache; never claimed cold disk',
        cache_scope='identical whole-image-batch and slot hit reuse; any miss recomputes original whole-anchor batch; loader still decodes full inputs',
        cache_implementation='whole_anchor_miss_slot_bound_reuse_v2',
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
    output['measurement_role']='preliminary_single_arm' if args.replay==-1 else 'paired_primary'
    output['replay']=args.replay
    schedule=[(args.replay,[args.arm])]
    for replay,arms in schedule:
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
                hook=net.pts_bbox_head.register_forward_hook(capture);accepted=not bool(args.cache_blocked_evidence)
                if not accepted:
                    row['cache_rejected_reason']='Cached comparison excluded by preserved H8 exact-native gate failure; paired H2 cache is unmeasured, not a failure claim'
                    row['cache_blocked_evidence']=dict(path=args.cache_blocked_evidence,sha256=sha(args.cache_blocked_evidence))
                try:
                    for index in indices[:16] if accepted else ():
                        batch=data(index);captures.clear();left=wrapper(return_loss=False,rescale=True,**copy.deepcopy(batch));raw=captures[-1]
                        cache.clear();net.extract_feat=MethodType(cache.extract,net)
                        try:right=wrapper(return_loss=False,rescale=True,**copy.deepcopy(batch))
                        finally:net.extract_feat=cache.original
                        gate_phase='cold_full_batch_miss'
                        try:
                            assert len(raw)==len(captures[-1])==13, 'Require every raw output'
                            proof=[tensor_parity(a,b,f'{arm}:{index}:{j}') for j,(a,b) in enumerate(zip(raw,captures[-1]))];compare_voxels(left,right)
                            gate_phase='warm_all_slot_hit'
                            recomputations=cache.full_batch_recomputations
                            net.extract_feat=MethodType(cache.extract,net)
                            try:right=wrapper(return_loss=False,rescale=True,**copy.deepcopy(batch))
                            finally:net.extract_feat=cache.original
                            assert cache.full_batch_recomputations==recomputations and cache.reuse_only_calls>0
                            assert len(captures[-1])==13
                            warm_proof=[tensor_parity(a,b,f'{arm}:{index}:warm:{j}') for j,(a,b) in enumerate(zip(raw,captures[-1]))];compare_voxels(left,right)
                            row['cache_parity'].append(dict(index=index,replay=replay,raw_max_abs=max(x['max_abs_difference'] for x in proof),voxels_exact=True,
                                warm_raw_max_abs=max(x['max_abs_difference'] for x in warm_proof),warm_voxels_exact=True,warm_reuse_without_extraction=True))
                        except (AssertionError,ValueError,FloatingPointError) as error:
                            accepted=False;row['cache_rejected_reason']=repr(error)
                            diagnostic=Path(args.out).with_suffix('.cache_rejected.pt')
                            torch.save(dict(index=index,gate_phase=gate_phase,reference_raw=raw,cached_raw=captures[-1],reference_voxels=left,cached_voxels=right,error=repr(error)),diagnostic)
                            row['cache_rejection_evidence']=dict(path=str(diagnostic),sha256=sha(diagnostic))
                            break
                finally:hook.remove()
                for mode in ('independent_anchor','chronological_no_feature_cache','chronological_feature_cache'):
                    if mode=='chronological_feature_cache' and not accepted:continue
                    cache.clear();cache.hits=cache.misses=0;cache.full_batch_recomputations=cache.reuse_only_calls=0;previous_scene=None;measure=[];initial_energy=energy()
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
                                h2d_image_bytes=image_storage(batch)['bytes'],image_tensor=image_storage(batch),
                                feature_cache_bytes=cache.bytes,feature_cache_hits=cache.hits,feature_cache_misses=cache.misses,
                                feature_cache_full_batch_recomputations=cache.full_batch_recomputations,feature_cache_reuse_only_calls=cache.reuse_only_calls,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),cpu_rss_bytes=rss()))
                            del prediction,batch
                    finally:net.extract_feat=cache.original
                    end_energy=energy();block=dict(replay=replay,mode=mode,rows=measure,summary=summary(measure),
                        gpu_j_per_prediction=(end_energy-initial_energy)/len(measure) if initial_energy is not None and end_energy is not None else None,
                        startup_scope='empty model feature cache with dataset-provided causal historical inputs; no missing-history accuracy claim')
                    row['replays'].append(block);save_output(args.out,output)
            cache.clear();del wrapper,net,cache,ds,warm;gc.collect();torch.cuda.empty_cache()
    output['status']='complete';save_output(args.out,output);print(json.dumps({a:[x['summary'] for x in r['replays']] for a,r in output['arms'].items()},indent=2))


if __name__=='__main__':main()
