"""One-shot GPU1 continuation with immutable old scientific source and new receipts."""
import datetime
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from gpu_capacity import CapacityWindow, memory_snapshot

ROOT = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
CAMPAIGN = ROOT/'analysis/transport_reliable_extension_v3_20261001'
WORK = ROOT/'work_dirs/transport_reliable_extend20_v3_20261001'
OLD = ROOT/'work_dirs/radar_forecast_transport-reliable_seed0'
UUID = 'GPU-000b6236-3632-a001-9667-1f02cbb61c8b'
CONFIG = 'configs/sw-radar-transport-reliable-extend20.py'


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, data):
    temp = path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def audit(path, epoch=None, optimizer_steps=None):
    import torch
    torch.set_num_threads(2)
    c = torch.load(path,map_location='cpu')
    assert len(c['state_dict']) == 801 and all(torch.isfinite(x).all() for x in c['state_dict'].values())
    assert all(torch.isfinite(v).all() for s in c['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
    steps = {int(s['step']) for s in c['optimizer']['state'].values() if 'step' in s}
    assert len(steps)==1
    if optimizer_steps is not None: assert steps == {optimizer_steps},steps
    if epoch is not None:
        assert c['meta']['epoch']==epoch and c['meta']['iter']==epoch*2992,c['meta']
    result=dict(epoch=c['meta']['epoch'],iterations=c['meta']['iter'],optimizer_steps=min(steps),
        model_tensors=801,model_finite=True,optimizer_finite=True,source=str(path),
        continuation=c['meta'].get('transport_extension'))
    del c;gc.collect();return result


def main():
    code=Path(__file__).resolve().parents[1];os.chdir(code)
    manifest=json.loads((code/'code_manifest.json').read_text())
    CAMPAIGN.mkdir(parents=True,exist_ok=True)
    status=dict(state='preflight',controller_pid=os.getpid(),code=str(code),git_revision=manifest['git_revision'],
        scientific_base_revision=manifest['scientific_base_revision'],gpu=1,gpu_uuid=UUID,at_utc=utc(),completed_stages=[])
    with (CAMPAIGN/'submission.json').open('x') as f: json.dump(status,f,indent=2)
    def record(**values):
        status.update(values,at_utc=utc());write(CAMPAIGN/'status.json',status)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(code),PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),TMPDIR='/tmp')
    scratch=Path('/home/huayiming/Workspace/SparseWorld-cache/eval_spool_extension_20261001')
    scratch.mkdir(parents=True,exist_ok=True)
    env.update(SPARSEWORLD_EVAL_TMPDIR=str(scratch),
        SPARSEWORLD_FULL_EVAL_LOCK='/home/huayiming/Workspace/SparseWorld-cache/eval_spool_history_recovery_20260928/full-evaluation.lock')
    python=str(ROOT/'envs/hym_sparseworld/bin/python')
    timings=[]
    def stage(name,arguments,gpu=False):
        if gpu:
            window=CapacityWindow()
            while True:
                snapshot=memory_snapshot(UUID)
                compute=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
                available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
                now=time.monotonic()
                admitted=window.observe(snapshot['free_mib'] if not compute and available>=64*1024**3 else 0,now)
                record(state='waiting_for_resources',stage=name,child_pid=None,
                    admission=dict(**snapshot,host_available_bytes=available,other_compute=compute,
                        stable_seconds=now-window.sufficient_since if window.sufficient_since else 0))
                if admitted: break
                time.sleep(15)
        stage_env=dict(env,CUDA_VISIBLE_DEVICES=UUID if gpu else '')
        log=CAMPAIGN/(name+'.log');started=time.monotonic()
        with log.open('x') as f:
            p=subprocess.Popen([python]+arguments,cwd=code,env=stage_env,stdout=f,stderr=subprocess.STDOUT)
            record(state='running',stage=name,child_pid=p.pid,command=[python]+arguments,log=str(log))
            while p.poll() is None:
                try: p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    record(child_pid=p.pid)
                    if gpu:
                        snap=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
                        with (CAMPAIGN/'gpu.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(at_utc=utc(),stage=name,gpu=snap))+'\n')
            timing=dict(stage=name,elapsed_seconds=time.monotonic()-started,gpu=gpu,returncode=p.returncode)
            timings.append(timing);write(CAMPAIGN/'timings.json',timings)
            record(child_pid=None,child_returncode=p.returncode)
            if p.returncode: raise RuntimeError(f'{name} exited {p.returncode}; evidence retained at {log}')
            status['completed_stages'].append(name);record()
    try:
        for name,expected in manifest['sha256'].items():
            assert hashlib.sha256((code/name).read_bytes()).hexdigest()==expected,name
        # All original scientific files are byte-identical; only additive wrappers differ.
        original=json.loads(Path('/home/huayiming/Workspace/SparseWorld-TC-forecast-2fc2feaf5edb/code_manifest.json').read_text())
        for name,expected in original['sha256'].items():
            assert hashlib.sha256((code/name).read_bytes()).hexdigest()==expected,name
        failed=ROOT/'analysis/transport_reliable_extension_20261001'
        failure=json.loads((failed/'status.json').read_text())
        assert failure['state']=='failed_evidence_preserved' and not Path('/proc/'+str(failure['controller_pid'])).exists()
        for x in Path('/proc').glob('[0-9]*/cmdline'):
            try: command=x.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            except (FileNotFoundError,PermissionError,ProcessLookupError): continue
            assert 'tools/train_transport_extension.py' not in command, (str(x),command)
        prior=json.loads((ROOT/'analysis/transport_reliable_campaign_20260922/transport-reliable_status.json').read_text())
        assert prior['state']=='complete' and not Path('/proc/'+str(prior['controller_pid'])).exists()
        previous=ROOT/'analysis/transport_reliable_extension_v2_20261001'
        previous_status=json.loads((previous/'status.json').read_text())
        assert previous_status['state']=='failed_evidence_preserved' and previous_status['child_returncode']==0
        assert 'checkpoints=list' in previous_status['traceback']
        assert not Path('/proc/'+str(previous_status['controller_pid'])).exists()
        assert previous_status['completed_stages']==['cpu_tests','cpu_resume_cache','cuda_contracts','frozen_storage','resume_smoke']
        previous_code=Path(previous_status['code']);previous_manifest=json.loads((previous_code/'code_manifest.json').read_text())
        for name in ('tools/transport_extension_hooks.py','tools/transport_extension_spool.py',
                     'tools/train_transport_extension.py','tools/evaluate_transport_extension.py','tools/check_transport_extension.py'):
            assert manifest['sha256'][name]==previous_manifest['sha256'][name],name
        assert not WORK.exists() or not any(WORK.iterdir())
        record(state='waiting_for_gpu_lock')
        with (ROOT/'forecast_gpu1.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            stage('cpu_tests',['-m','pytest','-q','tests/test_transport_reliability.py','tests/test_radar_belief.py',
                'tests/test_forecast_improvements.py','tests/test_forecast_comparison.py','tests/test_radar_fusion.py',
                'tests/test_official_init.py','tests/test_radar_cache.py','tests/test_m0_contracts.py','tests/test_gpu_capacity.py',
                'tests/test_transport_extension_resume.py','tests/test_transport_extension_spool.py'])
            stage('cpu_resume_cache',['tools/check_transport_extension.py','--device','cpu','--out',str(CAMPAIGN/'cpu_resume_cache.json')])
            # Only controller discovery/output paths changed. Reuse already completed
            # CUDA/storage/smoke evidence after source-hash equality, never smoke weights.
            for name in ('frozen_storage.json',):
                receipt=json.loads((previous/name).read_text());assert receipt['status']=='passed' and receipt['all_metrics_exact'] and receipt['scene_confusions_exact']
            smoke=ROOT/'work_dirs/transport_reliable_extend20_smoke_v2_20261001'
            checkpoint=smoke/'iter_29944.pth';assert checkpoint.is_file()
            resume=json.loads((smoke/'resume_integrity.json').read_text())
            assert all(resume[k] for k in ('model_exact','optimizer_exact','amp_scaler_exact')) and resume['iterations']==29920
            proof=audit(checkpoint,optimizer_steps=29915+24)
            proof.update(admission_evidence_campaign=str(previous),admission_git_revision=previous_manifest['git_revision'],
                evidence_reused_after_identical_scientific_source_hashes=True,smoke_weights_reused_for_formal=False)
            components=[];joint=[]
            for line in (smoke/'train.log').read_text().splitlines():
                if 'RADAR_COMPONENTS' in line:
                    d=json.loads(line[line.index('{'):]);assert set(d)=={'velocity_scale','reliability_gate','readout'}
                    assert all(math.isfinite(v[k]) and v[k]>0 for v in d.values() for k in ('gradient_sq','delta_sq'))
                    components.append(d)
                if 'JOINT_LEARNING' in line:
                    d=json.loads(line[line.index('{'):]);assert all(math.isfinite(v[k]) and v[k]>0 for v in d.values() for k in ('gradient_norm','parameter_delta'))
                    joint.append(d)
            assert len(components)==len(joint)==6
            proof.update(component_windows=components,joint_windows=joint);write(CAMPAIGN/'smoke_audit.json',proof)
            # Independent formal process reloads the original epoch10, never the smoke.
            stage('train',['tools/train_transport_extension.py','--config',CONFIG],gpu=True)
            final=audit(WORK/'epoch_20.pth',epoch=20);write(CAMPAIGN/'final_audit.json',final)
            for epoch in (15,20):
                target=WORK/f'validation_epoch_{epoch:02d}_full.json'
                assert json.loads(target.read_text())['samples']==5119
                stage(f'compare_e{epoch}_vs_e10',['tools/compare_forecast_results.py','--reference',str(OLD/'confusions_epoch_10_full'),
                    '--candidate',str(WORK/f'confusions_epoch_{epoch:02d}_full'),'--out',str(CAMPAIGN/f'e{epoch}_vs_e10.json')])
            selection=json.loads((WORK/'best_future.json').read_text())
            best_path=Path(selection.get('checkpoint',WORK/'best_future.pth'))
            if selection['epoch']<=10:
                best_dir=ROOT/'analysis/transport_reliable_campaign_20260922/transport-reliable_best_full'
                best_metrics=json.loads((best_dir/'normal.json').read_text())
            else:
                best_dir=CAMPAIGN/'best_full'
                stage('best_full',['tools/evaluate_transport_extension.py','--config',CONFIG,'--checkpoint',str(best_path),
                    '--samples','0','--output-dir',str(best_dir)],gpu=True)
                best_metrics=json.loads((best_dir/'normal.json').read_text())
                stage('best_interventions',['tools/evaluate_transport_extension.py','--config',CONFIG,'--checkpoint',str(best_path),
                    '--samples','256','--modes','normal','drop','zero_velocity','shuffle_velocity',
                    '--output-dir',str(CAMPAIGN/'best_interventions')],gpu=True)
                stage('compare_best',['tools/compare_forecast_results.py','--reference',str(ROOT/'analysis/transport_reliable_campaign_20260922/transport-reliable_best_full/confusions_normal'),
                    '--candidate',str(best_dir/'confusions_normal'),'--out',str(CAMPAIGN/'best_vs_original_best.json')])
            final_metrics=json.loads((WORK/'validation_epoch_20_full.json').read_text())
            result=dict(state='complete',git_revision=manifest['git_revision'],scientific_base_revision=manifest['scientific_base_revision'],
                final_audit=final,best_epoch=selection['epoch'],best_checkpoint=str(best_path),
                final_future_mean_miou=final_metrics['future_mean_miou'],best_future_mean_miou=best_metrics['future_mean_miou'],
                final_comparison=json.loads((CAMPAIGN/'e20_vs_e10.json').read_text()),timings=timings,
                limits=['single seed','fixed256 reused inside5119','epoch-boundary RNG not saved originally',
                    'old T/G layout preserved','extra budget does not isolate module contribution'])
            write(CAMPAIGN/'result.json',result)
            delta=result['final_comparison']['future_mean_delta_pp'];ci=result['final_comparison']['future_delta_scene_bootstrap_ci95']
            (CAMPAIGN/'REPORT_ZH.md').write_text(f'# 旧组合低学习率续训至20轮\n\n第20轮未来mIoU {result["final_future_mean_miou"]:.8f}%，相对原第10轮 {delta:+.8f}pp，场景配对95%区间 {ci}。\n\n最佳轮为 {selection["epoch"]}，未来mIoU {result["best_future_mean_miou"]:.8f}%。保留全部15/20轮时域和类别矩阵。新增训练预算不能当新增模块贡献；单种子与验证复用限制仍在。\n')
            record(state='complete',stage='all_training_and_evaluation_complete',result=str(CAMPAIGN/'result.json'))
    except BaseException as error:
        record(state='failed_evidence_preserved',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__': main()
