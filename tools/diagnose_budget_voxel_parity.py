"""Zero-optimization diagnosis of native voxelization repeatability.

The frozen scientific model and native get_occ are unchanged. Extra get_occ
calls and deterministic voxelizer calls are diagnostic only, never admissions,
training, evaluation results or a substitute for the failed exact-voxel gate.
"""
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SCIENCE = Path('/home/huayiming/Workspace/SparseWorld-TC-forecast-0f33492d1b63')
ROOT = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
UUID = 'GPU-000b6236-3632-a001-9667-1f02cbb61c8b'


def main():
    output = Path(sys.argv[1]);output.mkdir(exist_ok=False)
    os.chdir(SCIENCE);sys.path.insert(0,str(SCIENCE));sys.path.insert(0,str(SCIENCE/'tools'))
    from gpu_capacity import CapacityWindow, memory_snapshot
    from run_history_doppler_experiment import verify_snapshot, write_json
    manifest=json.loads((SCIENCE/'code_manifest.json').read_text());assert manifest['git_revision']=='0f33492d1b639da716897d8faa4a1df293354c49';verify_snapshot(SCIENCE,manifest)
    report=dict(status='diagnostic_running',optimization_steps=0,science_revision=manifest['git_revision'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='fixed official initialization; repeated postprocessing of identical prediction tensors only; no training, no gate relaxation',raw_tensors=[],postprocessing=[])
    def save():write_json(output/'diagnostic.json',report)
    save()
    with (ROOT/'forecast_gpu1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);window=CapacityWindow()
        while True:
            snap=memory_snapshot(UUID);compute=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
            available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
            tick=time.monotonic();ok=window.observe(snap['free_mib'] if not compute and available>=64*1024**3 else 0,tick)
            report['admission']=dict(**snap,other_compute=compute,host_available_bytes=available,stable_seconds=tick-window.sufficient_since if window.sufficient_since else 0);save()
            if ok:break
            time.sleep(15)
        os.environ['CUDA_VISIBLE_DEVICES']=UUID
        import torch,numpy as np
        from mmcv import Config
        from mmdet3d.models import build_model
        import models,loaders
        import tools.check_history_budget_contracts as checker
        from official_init import initialize_official
        from models.sparse_world_head import SparseWorldHead
        from tools.check_censored_path_contracts import compare_voxels
        torch.set_num_threads(2);torch.manual_seed(0)
        cfg=Config.fromfile('configs/sw-budget-h8-geometry.py');baseline=Config.fromfile('configs/sw-radar-forecast-transport.py')
        contract=checker.configuration_contract(cfg,baseline,'configs/sw-budget-h8-geometry.py')
        new=build_model(copy.deepcopy(cfg.model));new.init_weights();initialize_official(new,cfg.load_from)
        ref_cfg=copy.deepcopy(baseline.model);ref_cfg['visual_history_frames']=None;ref_cfg['pts_bbox_head']['transformer']['radar_cfg']=None
        reference=build_model(ref_cfg);reference.init_weights();initialize_official(reference,cfg.load_from)
        for key,value in reference.state_dict().items():assert torch.equal(value,new.state_dict()[key]),key
        original_occ=SparseWorldHead.get_occ;original_parity=checker.tensor_parity
        heads={id(new.pts_bbox_head):'new',id(reference.pts_bbox_head):'reference'}
        def equal(a,b):
            try:compare_voxels(a,b);return True
            except AssertionError:return False
        def stats(a,b):
            result=[]
            for horizon,(x,y) in enumerate(zip(a,b)):
                left={tuple(c):int(v) for c,v in zip(x['occ_loc'],x['sem_pred'])};right={tuple(c):int(v) for c,v in zip(y['occ_loc'],y['sem_pred'])}
                result.append(dict(horizon=horizon,left_voxels=len(left),right_voxels=len(right),coordinate_symmetric_difference=len(set(left)^set(right)),label_disagreements_on_common=sum(left[c]!=right[c] for c in set(left)&set(right))))
            return result
        def parity(a,b,label):
            row=original_parity(a,b,label);report['raw_tensors'].append(dict(label=label,**row));save();return row
        def occurrence(self,predictions,metadata,rescale=False):
            call=len(report['postprocessing']);tag=heads[id(self)]
            assert self.voxel_generator.deterministic is False
            # No mutation of frozen tensors. All native repetitions use the exact
            # same object, dtype, values and native non-deterministic setting.
            native=[original_occ(self,predictions,metadata,rescale) for _ in range(3)]
            try:
                self.voxel_generator.deterministic=True
                deterministic=[original_occ(self,predictions,metadata,rescale) for _ in range(2)]
            finally:self.voxel_generator.deterministic=False
            mismatch=not all(equal(native[0],x) for x in native[1:])
            row=dict(call=call,model=tag,native_deterministic_flag=False,same_prediction_object=True,
                native_repeat_exact=[equal(native[0],x) for x in native[1:]],
                native_repeat_differences=[stats(native[0],x) for x in native[1:]],
                diagnostic_deterministic_repeat_exact=equal(*deterministic),
                native_vs_diagnostic_deterministic=stats(native[0],deterministic[0]))
            report['postprocessing'].append(row)
            if mismatch:
                raw=[predictions['init_points']]+predictions['all_cls_scores']+predictions['all_refine_pts']
                torch.save([x.detach().cpu().clone() for x in raw],output/f'raw_{call}_{tag}.pt')
                for mode,values in [('native',native),('deterministic_diagnostic',deterministic)]:
                    for repeat,value in enumerate(values):
                        np.savez_compressed(output/f'voxels_{call}_{tag}_{mode}_{repeat}.npz',**{f'{h}_{k}':v[k] for h,v in enumerate(value) for k in ('occ_loc','sem_pred')})
            save();return native[0]
        SparseWorldHead.get_occ=occurrence;checker.tensor_parity=parity
        try:
            report['original_contract_result']=checker.gpu_contract(new,reference,cfg,contract)
            report['original_contract_completed']=True
        except BaseException as error:
            report['original_contract_completed']=False;report['original_contract_error']=repr(error);report['traceback']=traceback.format_exc()
        finally:
            SparseWorldHead.get_occ=original_occ;checker.tensor_parity=original_parity
        report.update(status='diagnostic_complete',native_postprocess_repeatability_failed=any(not all(x['native_repeat_exact']) for x in report['postprocessing']),
                      raw_parity_tensors_observed=len(report['raw_tensors']),raw_parity_all_exact=bool(report['raw_tensors']) and all(x['exact'] for x in report['raw_tensors']),finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        report['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name!='diagnostic.json'};save();print(json.dumps({k:v for k,v in report.items() if k not in ['raw_tensors','original_contract_result']},indent=2))


if __name__=='__main__':main()
