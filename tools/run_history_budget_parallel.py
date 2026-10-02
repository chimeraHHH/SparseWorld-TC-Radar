"""Scheduler-only handoff of the frozen four-arm mainline to two single H200s.

Science, admissions and evaluation commands run from the verified 0f snapshot.
GPU0 adopts its existing H8 trainer without restarting; GPU1 takes geometry.
There is no automatic retry, work-directory reuse, or extra experimental arm.
"""
import argparse
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

REVISION = '0f33492d1b639da716897d8faa4a1df293354c49'
SCIENCE = Path('/home/huayiming/Workspace/SparseWorld-TC-forecast-0f33492d1b63')
ROOT = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
CAMPAIGN = ROOT / 'analysis/history_budget_mainline_20261001'
GPUS = {0: 'GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe',
        1: 'GPU-000b6236-3632-a001-9667-1f02cbb61c8b'}
ASSIGNMENTS = {0: ('h8-velocity', 'h2-velocity'), 1: ('h8-geometry', 'h2-geometry')}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_stat(contents):
    # Linux comm may contain spaces and ')' (e.g. a data-loader queue thread).
    f = contents.rsplit(')', 1)[1].split()
    return dict(state=f[0], ppid=int(f[1]), pgid=int(f[2]), sid=int(f[3]),
                starttime=int(f[19]))


def process(pid):
    p = Path('/proc') / str(pid)
    try:
        result = dict(pid=pid, **parse_stat((p/'stat').read_text()))
        if result['state'] == 'Z':
            return result
        return dict(result, uid=p.stat().st_uid, cwd=os.readlink(p/'cwd'),
                    command=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (FileNotFoundError, ProcessLookupError):
        return dict(pid=pid, alive=False)


def same_process(current, expected):
    return (current.get('alive') is not False and current.get('state') != 'Z'
            and all(current.get(k) == expected.get(k)
                    for k in ('pid', 'starttime', 'uid', 'cwd', 'command')))


def exclusive_json(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def claim(campaign, arm, gpu, scheduler_revision):
    if arm not in ASSIGNMENTS[gpu]:
        raise ValueError('Arm is not assigned to this GPU')
    with (campaign/'claims.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exclusive_json(campaign/(arm+'_claim.json'),
                       dict(arm=arm, gpu=gpu, controller_pid=os.getpid(),
                            git_revision=REVISION, scheduler_revision=scheduler_revision,
                            at_utc=now()))


def adopted_completion(campaign, work, log):
    # Reparented trainer has no waitpid return code. Never invent one: require
    # complete artifacts, then the unchanged checkpoint/schema/finite audits.
    text = log.read_text()
    if any(x in text for x in ('Traceback (most recent call last)', 'CUDA out of memory',
                              'Segmentation fault', 'Killed', 'RuntimeError:')):
        raise RuntimeError('Adopted trainer ended with an error; evidence retained')
    if not (work/'epoch_10.pth').is_file() or not (work/'validation_epoch_10_full.json').is_file():
        raise RuntimeError('Adopted trainer exited without final artifacts; no retry')
    return dict(returncode=None, success_basis='final artifacts plus checkpoint and metric audits',
                exit_observation='non-parent process disappearance or zombie')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    scheduler = Path(__file__).resolve().parents[1]
    scheduler_manifest = json.loads((scheduler/'code_manifest.json').read_text())
    sys.path.insert(0, str(SCIENCE/'tools'))
    from gpu_capacity import CapacityWindow, memory_snapshot
    from run_history_doppler_experiment import verify_snapshot, audit_smoke_log, write_json
    from run_censored_path_experiment import audit_checkpoint, require_fresh_initialization, require_full_result
    verify_snapshot(scheduler, scheduler_manifest)
    manifest = json.loads((SCIENCE/'code_manifest.json').read_text())
    assert manifest['git_revision'] == REVISION
    verify_snapshot(SCIENCE, manifest)
    # Every old tracked file, including the old scheduler, must remain exact.
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((scheduler/name).read_bytes()).hexdigest() == digest, name
    handoff = json.loads((CAMPAIGN/'parallel_handoff_20261002.json').read_text())
    assert handoff['state'] == 'old_controller_exited_trainer_preserved'
    assert handoff['scheduler_revision'] == scheduler_manifest['git_revision']
    old = handoff['original_status']
    required = ['cpu_tests', 'single_sweep_cache', 'all_visual_sources'] + [
        'cpu_'+arm for arms in ASSIGNMENTS.values() for arm in arms]
    assert all(x in old['completed_stages'] for x in required)
    for arm in sum(ASSIGNMENTS.values(), ()):
        cpu = json.loads((CAMPAIGN/f'cpu_{arm}.json').read_text())
        assert cpu['status'] == 'passed' and cpu['git_revision'] == REVISION
    gpu = args.gpu
    uuid = GPUS[gpu]
    status = dict(state='prepared', controller_pid=os.getpid(), gpu=gpu, gpu_uuid=uuid,
                  git_revision=REVISION, code=str(SCIENCE), scheduler_code=str(scheduler),
                  scheduler_revision=scheduler_manifest['git_revision'], assigned_arms=list(ASSIGNMENTS[gpu]),
                  completed_arms=[], completed_stages=[], stage_seconds={}, stage_timing_basis={}, at_utc=now())
    if gpu == 0:
        status['completed_stages'] = list(old['completed_stages'])
        status['stage_seconds'] = dict(old['stage_seconds'])
    exclusive_json(CAMPAIGN/f'parallel_gpu{gpu}_submission.json', status)
    def record(**values):
        status.update(values, at_utc=now())
        write_json(CAMPAIGN/f'parallel_gpu{gpu}_status.json', status)
        if gpu == 0:
            write_json(CAMPAIGN/'status.json', status)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(SCIENCE), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TMPDIR='/tmp',
               SPARSEWORLD_EVAL_TMPDIR='/home/huayiming/Workspace/SparseWorld-cache/eval_spool_budget_20261001',
               SPARSEWORLD_FULL_EVAL_LOCK='/home/huayiming/Workspace/SparseWorld-cache/eval_spool_history_recovery_20260928/full-evaluation.lock',
               TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'))
    python = str(ROOT/'envs/hym_sparseworld/bin/python')
    def telemetry(name):
        try:
            snap = subprocess.check_output(['nvidia-smi', '-i', uuid,
                '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'],
                text=True, timeout=15).strip()
            with (CAMPAIGN/f'parallel_gpu{gpu}_gpu.jsonl').open('a') as f:
                f.write(json.dumps(dict(at_utc=now(), stage=name, gpu=snap))+'\n')
        except (OSError, subprocess.SubprocessError) as error:
            print('TELEMETRY_WARNING', repr(error), flush=True)
    def admit():
        window = CapacityWindow()
        while True:
            snap = memory_snapshot(uuid)
            compute = subprocess.check_output(['nvidia-smi', '-i', uuid,
                '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
            available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                                 if x.startswith('MemAvailable:')))*1024
            tick = time.monotonic()
            ok = window.observe(snap['free_mib'] if not compute and available >= 64*1024**3 else 0, tick)
            record(state='waiting_for_resources', child_pid=None, admission=dict(**snap,
                other_compute=compute, host_available_bytes=available,
                stable_seconds=tick-window.sufficient_since if window.sufficient_since else 0))
            if ok:
                return
            time.sleep(15)
    def run(name, commands, gpu_stage=False):
        record(stage=name)
        if gpu_stage:
            admit()
        started = time.monotonic()
        log = CAMPAIGN/(name+'.log')
        with log.open('x') as stream:
            child = subprocess.Popen([python]+commands, cwd=SCIENCE,
                env=dict(env, CUDA_VISIBLE_DEVICES=uuid if gpu_stage else ''), stdout=stream, stderr=subprocess.STDOUT)
            record(state='running', child_pid=child.pid, child_identity=process(child.pid),
                   child_returncode=None, command=[python]+commands, log=str(log))
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    record()
                    if gpu_stage:
                        telemetry(name)
        status['stage_seconds'][name] = time.monotonic()-started
        status['stage_timing_basis'][name] = 'parent-monitored stage elapsed including subprocess internal waits'
        record(child_pid=None, child_returncode=child.returncode)
        if child.returncode:
            raise RuntimeError(f'{name} exited {child.returncode}; evidence retained; no automatic retry')
        status['completed_stages'].append(name)
        record()
    def adopt():
        trainer = handoff['trainer']
        current = process(trainer['pid'])
        assert same_process(current, trainer), current
        original_claim = json.loads((CAMPAIGN/'h8-velocity_claim.json').read_text())
        assert original_claim['controller_pid'] == old['controller_pid'] and original_claim['git_revision'] == REVISION
        name = 'h8-velocity_train'
        exclusive_json(CAMPAIGN/'h8-velocity_adoption.json', dict(trainer=trainer,
            old_controller_pid=old['controller_pid'], new_controller_pid=os.getpid(),
            science_revision=REVISION, scheduler_revision=scheduler_manifest['git_revision'], at_utc=now()))
        record(state='running_adopted', stage=name, child_pid=trainer['pid'], child_identity=trainer,
               child_returncode=None, command=old['command'], log=old['log'])
        last_alive_tick = time.monotonic()
        while same_process(current, trainer):
            last_alive_tick = time.monotonic()
            time.sleep(15)
            current = process(trainer['pid'])
            exit_observed_tick = time.monotonic()
            if same_process(current, trainer):
                telemetry(name)
                record()
        assert current.get('alive') is False or current.get('state') == 'Z', current
        proof = adopted_completion(CAMPAIGN, ROOT/'work_dirs/history_budget_h8-velocity_20261001_seed0', Path(old['log']))
        elapsed = exit_observed_tick - handoff['trainer_start_boot_seconds']
        lower = last_alive_tick - handoff['trainer_start_boot_seconds']
        status['stage_seconds'][name] = elapsed
        status['stage_timing_basis'][name] = 'Linux process lifetime to first confirmed exit; actual exit-observation interval separately recorded; excludes pre-process stage setup; internal waits retained'
        exclusive_json(CAMPAIGN/'h8-velocity_adopted_exit.json', dict(**proof, at_utc=now(),
            lifetime_upper_bound_seconds=elapsed, lifetime_lower_bound_seconds=lower,
            exit_observation_interval_seconds=elapsed-lower, clock_resolution_seconds=1/os.sysconf('SC_CLK_TCK')))
        status['completed_stages'].append(name)
        record(child_pid=None)
    def finish_arm(arm):
        cpu = json.loads((CAMPAIGN/f'cpu_{arm}.json').read_text())
        schema = cpu['initialization']['state_schema']
        cfg = f'configs/sw-budget-{arm}.py'
        work = ROOT/f'work_dirs/history_budget_{arm}_20261001_seed0'
        init = require_fresh_initialization(work/'official_initialization.json', cpu['initialization'])
        final = audit_checkpoint(work/'epoch_10.pth', schema, expected_epoch=10)
        assert final['iterations'] == 29920
        meta = json.loads((work/'best_future.json').read_text())
        assert meta['epoch'] in range(1, 11) and meta['samples'] == 256
        best = audit_checkpoint(work/'best_future.pth', schema, expected_epoch=meta['epoch'])
        final_json = work/'validation_epoch_10_full.json'
        final_conf = work/'confusions_epoch_10_full'
        require_full_result(final_json, final_conf)
        if meta['epoch'] == 10:
            best_json, best_conf = final_json, final_conf
        else:
            dst = CAMPAIGN/f'{arm}_best_full'
            run(arm+'_best_full', ['tools/evaluate_radar_experiment.py', '--config', cfg,
                '--checkpoint', str(work/'best_future.pth'), '--samples', '0', '--output-dir', str(dst)], True)
            best_json, best_conf = dst/'normal.json', dst/'confusions_normal'
        require_full_result(best_json, best_conf)
        modes = ['normal', 'drop'] if arm.endswith('geometry') else ['normal', 'drop', 'zero_velocity', 'shuffle_velocity']
        dst = CAMPAIGN/f'{arm}_interventions'
        run(arm+'_interventions', ['tools/evaluate_radar_experiment.py', '--config', cfg,
            '--checkpoint', str(work/'best_future.pth'), '--samples', '256', '--modes', *modes,
            '--output-dir', str(dst)], True)
        for mode in modes:
            r = json.loads((dst/(mode+'.json')).read_text())
            assert r['samples'] == 256 and r['indices'] == json.loads(
                (work/f'validation_epoch_{meta["epoch"]:02d}_subset.json').read_text())['indices']
        result = dict(arm=arm, gpu=gpu, git_revision=REVISION, scheduler_revision=scheduler_manifest['git_revision'],
            official_initialization=init, final_audit=final, best_audit=best, best_epoch=meta['epoch'],
            final_json=str(final_json), best_json=str(best_json), final_confusions=str(final_conf),
            best_confusions=str(best_conf), interventions=str(dst),
            stage_seconds={k:v for k,v in status['stage_seconds'].items() if k.startswith(arm+'_')},
            stage_timing_basis={k:v for k,v in status['stage_timing_basis'].items() if k.startswith(arm+'_')})
        exclusive_json(CAMPAIGN/(arm+'_result.json'), result)
        status['completed_arms'].append(arm)
        record(state='arm_complete')
    try:
        with (ROOT/f'forecast_gpu{gpu}.lock').open('a') as lock:
            record(state='waiting_for_original_gpu_lock')
            fcntl.flock(lock, fcntl.LOCK_EX)
            for arm in ASSIGNMENTS[gpu]:
                record(arm=arm)
                if gpu == 0 and arm == 'h8-velocity':
                    adopt()
                else:
                    for suffix in ('seed0', 'smoke_seed0'):
                        assert not (ROOT/f'work_dirs/history_budget_{arm}_20261001_{suffix}').exists()
                    claim(CAMPAIGN, arm, gpu, scheduler_manifest['git_revision'])
                    cpu = json.loads((CAMPAIGN/f'cpu_{arm}.json').read_text())
                    schema = cpu['initialization']['state_schema']
                    cfg = f'configs/sw-budget-{arm}.py'
                    run(arm+'_cuda', ['-m', 'pytest', '-q', 'tests/test_m0_contracts.py',
                        'tests/test_history_budget_mainline.py', '-k', 'cuda'], True)
                    run(arm+'_parity', ['tools/check_history_budget_contracts.py', '--config', cfg,
                        '--device', 'cuda', '--out', str(CAMPAIGN/f'gpu_{arm}.json')], True)
                    run(arm+'_smoke', ['train.py', '--config', f'configs/sw-budget-{arm}-smoke.py'], True)
                    smoke = ROOT/f'work_dirs/history_budget_{arm}_20261001_smoke_seed0'
                    audit = audit_checkpoint(smoke/'iter_24.pth', schema, exact_steps=24)
                    audit.update(audit_smoke_log((smoke/'train.log').read_text()))
                    write_json(CAMPAIGN/f'smoke_{arm}.json', audit)
                    run(arm+'_train', ['train.py', '--config', cfg], True)
                finish_arm(arm)
            if gpu == 1:
                record(state='worker_queue_finished', stage='geometry_arms_complete')
                return
            record(state='waiting_for_peer', stage='waiting_for_four_results', child_pid=None)
            while not all((CAMPAIGN/(a+'_result.json')).exists() for a in sum(ASSIGNMENTS.values(), ())):
                peer = CAMPAIGN/'parallel_gpu1_status.json'
                if peer.exists() and json.loads(peer.read_text())['state'] == 'failed_evidence_preserved':
                    raise RuntimeError('Peer worker failed; no automatic retries or duplicate claims')
                record()
                time.sleep(30)
            run('four_way_bootstrap', ['tools/summarize_history_budget.py', '--campaign', str(CAMPAIGN)])
            run('cost_benchmark', ['tools/benchmark_history_budget.py', '--campaign', str(CAMPAIGN),
                '--out', str(CAMPAIGN/'cost_benchmark.json')], True)
            record(state='complete', stage='four_arms_accuracy_and_cost_complete',
                result=str(CAMPAIGN/'comparison.json'), cost_result=str(CAMPAIGN/'cost_benchmark.json'))
    except BaseException as error:
        record(state='failed_evidence_preserved', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
