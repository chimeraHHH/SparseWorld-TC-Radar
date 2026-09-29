"""Reserve and supervise one read-only H1 diagnostic under existing GPU gates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpu_capacity import CapacityWindow, memory_snapshot
from run_history_doppler_experiment import GPUS, ROOT, other_compute, utcnow, verify_snapshot, write_json
from recover_history_doppler_campaign import (
    EVAL_SPOOL, HOST_MINIMUM_BYTES, host_available_bytes, require_old_processes_absent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', required=True, type=Path)
    parser.add_argument('--recovery-campaign', type=Path,
                        help='After exact frozen-output proof, admit this NEW recovery campaign')
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    os.chdir(code)
    manifest = json.loads((code / 'code_manifest.json').read_text())
    verify_snapshot(code, manifest)
    campaign = args.campaign_dir.resolve()
    campaign.mkdir(parents=True, exist_ok=False)
    status = dict(controller_pid=os.getpid(), code=str(code), git_revision=manifest['git_revision'],
                  gpu=1, gpu_uuid=GPUS[1], state='precheck', child_pid=None, stages={})

    def record(**values):
        status.update(values, at_utc=utcnow())
        write_json(campaign / 'status.json', status)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(code), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TMPDIR='/tmp',
               SPARSEWORLD_EVAL_TMPDIR=EVAL_SPOOL,
               SPARSEWORLD_FULL_EVAL_LOCK=str(ROOT / 'history_doppler_full_eval.lock'),
               TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/torch_extensions'))

    def run(stage, command, child_env):
        started = time.monotonic()
        with (campaign / (stage + '.log')).open('xb') as stream:
            child = subprocess.Popen(command, cwd=code, env=child_env, stdout=stream,
                                     stderr=subprocess.STDOUT)
            record(state='running', stage=stage, child_pid=child.pid, command=command)
            returncode = child.wait()
        status['stages'][stage] = dict(returncode=returncode, seconds=time.monotonic() - started)
        record(child_pid=None)
        if returncode:
            raise RuntimeError('%s failed with returncode %d' % (stage, returncode))

    record()
    try:
        process_checks = {}
        for name in ('history_doppler_campaign_20260927', 'history_doppler_recovery_20260928'):
            old_campaign = ROOT / 'analysis' / name
            old_submission = json.loads((old_campaign / 'submission.json').read_text())
            process_checks[name] = require_old_processes_absent(Path(old_submission['code']), old_campaign)
        record(process_checks=process_checks)
        run('cpu_tests', [sys.executable, '-m', 'pytest', '-q',
                         'tests/test_history_eval_diagnostic.py',
                         'tests/test_finetune_prediction_spool.py'], env)
        with (ROOT / 'forecast_gpu1.lock').open('a') as lock:
            record(state='waiting_for_gpu_lock')
            fcntl.flock(lock, fcntl.LOCK_EX)
            window = CapacityWindow(minimum_free_mib=132000, quiet_seconds=60)
            host_since = None
            while True:
                now = time.monotonic()
                memory = memory_snapshot(GPUS[1])
                processes = other_compute(GPUS[1])
                gpu_ready = window.observe(0 if processes else memory['free_mib'], now)
                host_available = host_available_bytes()
                host_since = (now if host_since is None else host_since) if host_available >= HOST_MINIMUM_BYTES else None
                admission = dict(**memory, other_compute=processes, host_available_bytes=host_available,
                    minimum_free_mib=132000, host_minimum_bytes=HOST_MINIMUM_BYTES, required_seconds=60,
                    gpu_stable_seconds=now-window.sufficient_since if window.sufficient_since is not None else 0,
                    host_stable_seconds=now-host_since if host_since is not None else 0)
                record(state='waiting_for_resources', admission=admission)
                if gpu_ready and host_since is not None and now - host_since >= 60:
                    break
                time.sleep(15)
            env['CUDA_VISIBLE_DEVICES'] = GPUS[1]
            run('diagnostic', [sys.executable, 'tools/diagnose_history_eval_parity.py',
                '--config', 'configs/sw-history-h1-geometry.py', '--checkpoint',
                str(ROOT / 'work_dirs/history_h1-geometry_seed0/epoch_10.pth'),
                '--original-code', '/home/huayiming/Workspace/SparseWorld-TC-forecast-1e95cf1d53c8',
                '--output-dir', str(campaign / 'observations'), '--samples', '16', '--repeats', '5',
                '--workers', '0'], env)
        result = json.loads((campaign / 'observations/diagnostic.json').read_text())
        if result['status'] != 'complete_observations_only':
            raise ValueError('Diagnostic did not finish')
        if result.get('optimizer_steps_executed') != 0 or result.get('checkpoint_unchanged') is not True:
            raise ValueError('Read-only checkpoint diagnostic contract failed')
        for policy in result['policies'].values():
            comparison = policy['fixed_output_storage_comparison']
            if not (policy['spool_arrays_exact'] and comparison['metrics_exact'] and comparison['all_scene_arrays_exact']):
                raise ValueError('Frozen-prediction storage parity failed; no threshold relaxation applies')
        record(state='complete', stage='observations_complete', result=str(campaign / 'observations/diagnostic.json'))
        if args.recovery_campaign:
            env['CUDA_VISIBLE_DEVICES'] = ''
            run('recovery_admission', [sys.executable, 'tools/recover_history_doppler_campaign.py',
                '--campaign-dir', str(args.recovery_campaign.resolve()), '--recovery-of',
                str(ROOT / 'analysis/history_doppler_campaign_20260927'), '--previous-recovery',
                str(ROOT / 'analysis/history_doppler_recovery_20260928'), '--diagnostic-result',
                str(campaign / 'observations/diagnostic.json')], env)
            record(state='complete', stage='recovery_admission_complete',
                   recovery_campaign=str(args.recovery_campaign.resolve()))
    except BaseException as error:
        record(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
