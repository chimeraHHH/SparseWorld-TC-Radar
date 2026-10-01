"""Authorized GPU0 serial matched H8/H2 campaign; exact admissions before each arm."""
import datetime,fcntl,json,os,subprocess,time,traceback,hashlib
from pathlib import Path
from gpu_capacity import CapacityWindow,memory_snapshot
from run_history_doppler_experiment import verify_snapshot,audit_smoke_log,write_json
from run_censored_path_experiment import audit_checkpoint,require_fresh_initialization,require_full_result
ROOT=Path('/storage/data/metaiot_data/huayiming/SparseWorld')
CAMPAIGN=ROOT/'analysis/history_budget_mainline_20261001'
ARMS=('h8-velocity','h8-geometry','h2-velocity','h2-geometry')
UUID='GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe'
CACHE='/home/huayiming/Workspace/SparseWorld-cache/radar_single_sweep_v1_20260927'
TESTS=['tests/test_visual_history_budget.py','tests/test_history_budget_mainline.py',
 'tests/test_single_sweep_radar.py','tests/test_single_sweep_cache.py','tests/test_history_radar_membership.py',
 'tests/test_history_doppler_contracts.py','tests/test_forecast_improvements.py','tests/test_forecast_comparison.py',
 'tests/test_radar_fusion.py','tests/test_official_init.py','tests/test_m0_contracts.py','tests/test_gpu_capacity.py',
 'tests/test_finetune_prediction_spool.py']


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    code=Path(__file__).resolve().parents[1];os.chdir(code)
    manifest=json.loads((code/'code_manifest.json').read_text());verify_snapshot(code,manifest)
    CAMPAIGN.mkdir(parents=True,exist_ok=True)
    status=dict(state='cpu_preflight',controller_pid=os.getpid(),gpu=0,gpu_uuid=UUID,code=str(code),
        git_revision=manifest['git_revision'],at_utc=now(),completed_arms=[],completed_stages=[],stage_seconds={})
    with (CAMPAIGN/'submission.json').open('x') as f:json.dump(status,f,indent=2)
    def record(**kw):status.update(kw,at_utc=now());write_json(CAMPAIGN/'status.json',status)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(code),PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TMPDIR='/tmp',
        SPARSEWORLD_EVAL_TMPDIR='/home/huayiming/Workspace/SparseWorld-cache/eval_spool_budget_20261001',
        SPARSEWORLD_FULL_EVAL_LOCK='/home/huayiming/Workspace/SparseWorld-cache/eval_spool_history_recovery_20260928/full-evaluation.lock',
        TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'))
    Path(env['SPARSEWORLD_EVAL_TMPDIR']).mkdir(parents=True,exist_ok=True)
    python=str(ROOT/'envs/hym_sparseworld/bin/python')
    def admit():
        window=CapacityWindow()
        while True:
            snapshot=memory_snapshot(UUID);compute=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
            available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
            t=time.monotonic();ok=window.observe(snapshot['free_mib'] if not compute and available>=64*1024**3 else 0,t)
            record(state='waiting_for_resources',child_pid=None,admission=dict(**snapshot,other_compute=compute,host_available_bytes=available,stable_seconds=t-window.sufficient_since if window.sufficient_since else 0))
            if ok:return
            time.sleep(15)
    def run(name,args,gpu=False):
        record(stage=name)
        if gpu:admit()
        started=time.monotonic();log=CAMPAIGN/(name+'.log')
        with log.open('x') as f:
            child=subprocess.Popen([python]+args,cwd=code,env=dict(env,CUDA_VISIBLE_DEVICES=UUID if gpu else ''),stdout=f,stderr=subprocess.STDOUT)
            record(state='running',child_pid=child.pid,command=[python]+args,log=str(log))
            while child.poll() is None:
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    record(child_pid=child.pid)
                    if gpu:
                        try:
                            snap=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
                            with (CAMPAIGN/'gpu.jsonl').open('a') as st:st.write(json.dumps(dict(at_utc=now(),stage=name,gpu=snap))+'\n')
                        except (OSError,subprocess.SubprocessError) as error:print('TELEMETRY_WARNING',repr(error),flush=True)
            status['stage_seconds'][name]=time.monotonic()-started;record(child_pid=None,child_returncode=child.returncode)
            if child.returncode:raise RuntimeError(f'{name} exited{child.returncode}; original evidence retained')
            status['completed_stages'].append(name);record()
    try:
        for arm in ARMS:
            for suffix in ('seed0','smoke_seed0'):
                assert not (ROOT/f'work_dirs/history_budget_{arm}_20261001_{suffix}').exists()
        with (ROOT/'forecast_gpu0.lock').open('a') as lock:
            record(state='waiting_for_original_gpu0_lock');fcntl.flock(lock,fcntl.LOCK_EX)
            run('cpu_tests',['-m','pytest','-q',*TESTS])
            run('single_sweep_cache',['tools/precompute_single_sweep_radar.py','--config','configs/sw-budget-h8-velocity.py','--output',CACHE,'--workers','8','--verify-existing'])
            run('all_visual_sources',['tools/audit_history_budget_sources.py','--out',str(CAMPAIGN/'all_visual_sources.json')])
            for arm in ARMS:
                run('cpu_'+arm,['tools/check_history_budget_contracts.py','--config',f'configs/sw-budget-{arm}.py','--device','cpu','--out',str(CAMPAIGN/f'cpu_{arm}.json')])
            for arm in ARMS:
                record(arm=arm)
                with (CAMPAIGN/(arm+'_claim.json')).open('x') as f:json.dump(dict(arm=arm,gpu=0,controller_pid=os.getpid(),git_revision=manifest['git_revision'],at_utc=now()),f,indent=2)
                cpu=json.loads((CAMPAIGN/f'cpu_{arm}.json').read_text());assert cpu['status']=='passed' and cpu['git_revision']==manifest['git_revision']
                schema=cpu['initialization']['state_schema'];cfg=f'configs/sw-budget-{arm}.py'
                work=ROOT/f'work_dirs/history_budget_{arm}_20261001_seed0';smoke=ROOT/f'work_dirs/history_budget_{arm}_20261001_smoke_seed0'
                run(arm+'_cuda',['-m','pytest','-q','tests/test_m0_contracts.py','tests/test_history_budget_mainline.py','-k','cuda'],True)
                run(arm+'_parity',['tools/check_history_budget_contracts.py','--config',cfg,'--device','cuda','--out',str(CAMPAIGN/f'gpu_{arm}.json')],True)
                run(arm+'_smoke',['train.py','--config',f'configs/sw-budget-{arm}-smoke.py'],True)
                smoke_audit=audit_checkpoint(smoke/'iter_24.pth',schema,exact_steps=24);smoke_audit.update(audit_smoke_log((smoke/'train.log').read_text()));write_json(CAMPAIGN/f'smoke_{arm}.json',smoke_audit)
                run(arm+'_train',['train.py','--config',cfg],True)
                init=require_fresh_initialization(work/'official_initialization.json',cpu['initialization'])
                final=audit_checkpoint(work/'epoch_10.pth',schema,expected_epoch=10);assert final['iterations']==29920
                meta=json.loads((work/'best_future.json').read_text());assert meta['epoch'] in range(1,11) and meta['samples']==256
                best=audit_checkpoint(work/'best_future.pth',schema,expected_epoch=meta['epoch'])
                final_json=work/'validation_epoch_10_full.json';final_conf=work/'confusions_epoch_10_full';require_full_result(final_json,final_conf)
                if meta['epoch']==10:best_json,best_conf=final_json,final_conf
                else:
                    dst=CAMPAIGN/f'{arm}_best_full';run(arm+'_best_full',['tools/evaluate_radar_experiment.py','--config',cfg,'--checkpoint',str(work/'best_future.pth'),'--samples','0','--output-dir',str(dst)],True)
                    best_json,best_conf=dst/'normal.json',dst/'confusions_normal'
                require_full_result(best_json,best_conf)
                modes=['normal','drop'] if arm.endswith('geometry') else ['normal','drop','zero_velocity','shuffle_velocity']
                dst=CAMPAIGN/f'{arm}_interventions';run(arm+'_interventions',['tools/evaluate_radar_experiment.py','--config',cfg,'--checkpoint',str(work/'best_future.pth'),'--samples','256','--modes',*modes,'--output-dir',str(dst)],True)
                for mode in modes:
                    r=json.loads((dst/(mode+'.json')).read_text());assert r['samples']==256 and r['indices']==json.loads((work/f'validation_epoch_{meta["epoch"]:02d}_subset.json').read_text())['indices']
                result=dict(arm=arm,gpu=0,git_revision=manifest['git_revision'],official_initialization=init,final_audit=final,best_audit=best,best_epoch=meta['epoch'],final_json=str(final_json),best_json=str(best_json),final_confusions=str(final_conf),best_confusions=str(best_conf),interventions=str(dst),stage_seconds={k:v for k,v in status['stage_seconds'].items() if k.startswith(arm+'_')})
                write_json(CAMPAIGN/(arm+'_result.json'),result);status['completed_arms'].append(arm);record(state='arm_complete')
            run('four_way_bootstrap',['tools/summarize_history_budget.py','--campaign',str(CAMPAIGN)])
            run('cost_benchmark',['tools/benchmark_history_budget.py','--campaign',str(CAMPAIGN),'--out',str(CAMPAIGN/'cost_benchmark.json')],True)
            record(state='complete',stage='four_arms_accuracy_and_cost_complete',result=str(CAMPAIGN/'comparison.json'),cost_result=str(CAMPAIGN/'cost_benchmark.json'))
    except BaseException as error:
        record(state='failed_evidence_preserved',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
