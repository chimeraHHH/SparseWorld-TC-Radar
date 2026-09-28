"""Recover two completed H1 checkpoints and untouched H8 arms in a new receipt tree.

This entry point never resumes training H1, edits the failed campaign, or
reuses smoke weights. Only evaluation/storage/scheduling code may differ from
the original immutable experiment. Each submission path is reserved once.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from run_history_doppler_experiment import (
    ARMS, CACHE, ROOT, CAMPAIGN, audit_smoke_log, utcnow, write_json, verify_snapshot)
from run_censored_path_experiment import validate_state_schema

ORIGINAL_REVISION = '1e95cf1d53c8607efea04341e3d864099a2e48cc'
EVAL_SPOOL = '/home/huayiming/Workspace/SparseWorld-cache/eval_spool_history_recovery_20260928'
HOST_MINIMUM_BYTES = 64 * 1024 ** 3
ALLOWED_SOURCE_CHANGES = {'finetune_hooks.py', 'tools/run_history_doppler_experiment.py'}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def host_available_bytes(meminfo=Path('/proc/meminfo')):
    for line in meminfo.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise ValueError('Host MemAvailable is unavailable')


def require_h1_parity(previous, current):
    old, new = (json.loads(Path(p).read_text()) for p in (previous, current))
    for key in ('samples', 'indices', 'metrics', 'future_mean_miou'):
        if old[key] != new[key]:
            raise ValueError('Recovered evaluation differs from saved H1 subset: ' + key)
    if old['samples'] != 256 or new['mode'] != 'normal':
        raise ValueError('Recovery parity requires the original fixed 256 normal anchors')
    return dict(status='passed', samples=256, exact_metrics=True, exact_indices=True,
                previous=str(previous), current=str(current), previous_sha256=sha256_file(previous),
                current_sha256=sha256_file(current))


def require_old_processes_absent(original_code, original, proc=Path('/proc')):
    """Fail closed for recorded live PIDs or own residuals in the old checkout."""
    recorded = set()
    for name in ('gpu0_worker.json', 'gpu1_worker.json'):
        payload = json.loads((original / name).read_text())
        recorded.update(int(payload[k]) for k in ('controller_pid', 'child_pid') if payload.get(k))
    alive = [pid for pid in sorted(recorded) if (proc / str(pid)).exists()]
    residuals = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            cwd = (directory / 'cwd').resolve(strict=True)
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if cwd == original_code.resolve() or str(original_code) in command:
                residuals.append(int(directory.name))
        except FileNotFoundError:
            continue  # The process exited during the read.
    if alive or residuals:
        raise ValueError('Original processes still present: ' + str(dict(recorded=alive, residuals=residuals)))
    return dict(recorded_pids=sorted(recorded), live_recorded_pids=[], own_original_snapshot_processes=[])


def validate_recovery_source(original, code):
    original, code = Path(original), Path(code)
    submission = json.loads((original / 'submission.json').read_text())
    preflight = json.loads((original / 'preflight.json').read_text())
    if preflight['state'] != 'passed' or preflight['git_revision'] != ORIGINAL_REVISION:
        raise ValueError('Original CPU preflight did not pass')
    old_code = Path(submission['code'])
    manifest = json.loads((old_code / 'code_manifest.json').read_text())
    if manifest['git_revision'] != ORIGINAL_REVISION or code.resolve() == old_code.resolve():
        raise ValueError('Recovery requires the known original source and a separate immutable checkout')
    verify_snapshot(old_code, manifest)
    process_audit = require_old_processes_absent(old_code, original)
    protected = {}
    for name, expected in manifest['sha256'].items():
        if name.endswith('.py') and not name.startswith('tests/') and name not in ALLOWED_SOURCE_CHANGES:
            if sha256_file(code / name) != expected:
                raise ValueError('Scientific or unapproved source changed: ' + name)
            protected[name] = expected
    for gpu in (0, 1):
        status = json.loads((original / ('gpu%d_worker.json' % gpu)).read_text())
        if status['state'] != 'failed' or status['stage'] != 'train' or status.get('child_returncode') != -9:
            raise ValueError('Expected recorded SIGKILL during original train/final evaluation')
    evidence = {}
    for arm in ARMS:
        work = ROOT / ('work_dirs/history_' + arm + '_seed0')
        smoke = ROOT / ('work_dirs/history_' + arm + '_smoke_seed0')
        if arm.startswith('h8-'):
            if work.exists() or smoke.exists() or (original / (arm + '_claim.json')).exists():
                raise FileExistsError('H8 has already been claimed or started: ' + arm)
            continue
        required = ('cpu_', 'gpu_', 'smoke_')
        receipts = {prefix: json.loads((original / (prefix + arm + '.json')).read_text()) for prefix in required}
        for prefix in ('cpu_', 'gpu_'):
            if receipts[prefix]['status'] != 'passed' or receipts[prefix]['git_revision'] != ORIGINAL_REVISION:
                raise ValueError('Original admission did not pass: ' + prefix + arm)
        smoke_audit = receipts['smoke_']
        if not (smoke_audit['iterations'] in (23, 24) and smoke_audit['optimizer_steps'] == 24
                and smoke_audit['optimizer_finite'] and smoke_audit['state_schema_matches_cpu']):
            raise ValueError('Original smoke checkpoint audit incomplete')
        audit_smoke_log((smoke / 'train.log').read_text())
        best = json.loads((work / 'best_future.json').read_text())
        if best['epoch'] != 10 or best['samples'] != 256:
            raise ValueError('Only the observed H1 epoch10 selections may be recovered')
        for filename in ('epoch_10.pth', 'best_future.pth', 'official_initialization.json', 'validation_epoch_10_subset.json'):
            if not (work / filename).is_file():
                raise FileNotFoundError(work / filename)
        if (original / (arm + '_result.json')).exists() or (work / 'validation_epoch_10_full.json').exists():
            raise FileExistsError('H1 full evaluation already finished; do not duplicate it')
        evidence[arm] = dict(receipt_sha256={prefix: sha256_file(original / (prefix + arm + '.json')) for prefix in required},
                             smoke_iterations_metadata=smoke_audit['iterations'], smoke_optimizer_steps=24)
    return dict(recovery_of=str(original), original_git_revision=ORIGINAL_REVISION,
                original_code=str(old_code), process_audit=process_audit,
                protected_source_sha256=protected, original_admission_sha256=evidence)


def audit_h1_checkpoints(work, schema):
    """Read each model/optimizer once; compare model content, not zip filenames."""
    import torch
    torch.set_num_threads(2)
    reports, file_hashes, model_hashes = {}, {}, {}
    for filename in ('epoch_10.pth', 'best_future.pth'):
        path = work / filename
        checkpoint = torch.load(path, map_location='cpu')
        state, optimizer = checkpoint['state_dict'], checkpoint['optimizer']['state']
        validate_state_schema(state, schema)
        digest = hashlib.sha256()
        for key in sorted(state):
            value = state[key]
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite H1 model state')
            digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        steps = [int(item['step']) for item in optimizer.values() if 'step' in item]
        if not steps or min(steps) != max(steps):
            raise ValueError('Missing or inconsistent optimizer steps')
        if not all(torch.isfinite(value).all() for item in optimizer.values()
                   for value in item.values() if torch.is_tensor(value)):
            raise FloatingPointError('Nonfinite H1 optimizer state')
        meta = checkpoint['meta']
        if meta['epoch'] != 10 or meta['iter'] != 29920:
            raise ValueError('H1 has not completed the exact formal iteration budget')
        reports[filename] = dict(path=str(path), epoch=10, iterations=29920, optimizer_steps=min(steps),
            finite_model_tensors=len(state), state_schema_matches_cpu=True, optimizer_finite=True)
        model_hashes[filename] = digest.hexdigest()
        del checkpoint, state, optimizer
        gc.collect()
        file_hashes[filename] = sha256_file(path)
    if len(set(model_hashes.values())) != 1:
        raise ValueError('Selected H1 model differs from the final epoch10 model')
    return reports['epoch_10.pth'], reports['best_future.pth'], dict(files=file_hashes, model_state=model_hashes)


def require_disk_spool(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    fs = subprocess.check_output(['findmnt', '-n', '-T', str(path), '-o', 'FSTYPE'], text=True).strip()
    if not fs or fs in ('tmpfs', 'ramfs'):
        raise ValueError('Evaluation spool must be on a real disk, not memory-backed storage')
    free = shutil.disk_usage(path).free
    if free < 32 * 1024 ** 3:
        raise ValueError('Evaluation spool requires at least 32 GiB free disk space')
    return dict(path=str(path), filesystem=fs, free_bytes=free)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--recovery-of', type=Path, default=ROOT / 'analysis' / CAMPAIGN)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Recovery CPU preflight must hide all GPUs')
    code = Path(__file__).resolve().parents[1]
    os.chdir(code)
    manifest = json.loads((code / 'code_manifest.json').read_text())
    verify_snapshot(code, manifest)
    if manifest['git_revision'] == ORIGINAL_REVISION:
        raise ValueError('Recovery requires its own new committed source snapshot')
    recovery = validate_recovery_source(args.recovery_of, code)
    campaign = args.campaign_dir
    if campaign.resolve() == args.recovery_of.resolve():
        raise ValueError('Do not overwrite the failed campaign')
    campaign.mkdir(parents=True, exist_ok=False)
    report = dict(state='preflight', launcher_pid=os.getpid(), code=str(code),
                  git_revision=manifest['git_revision'], at_utc=utcnow(), **recovery)
    write_json(campaign / 'submission.json', report)
    write_json(campaign / 'recovery_source.json', recovery)
    try:
        report['spool'] = require_disk_spool(EVAL_SPOOL)
        def run(stage, arguments):
            report.update(stage=stage, at_utc=utcnow())
            write_json(campaign / 'preflight.json', report)
            with (campaign / (stage + '.log')).open('xb') as log:
                subprocess.run([sys.executable] + arguments, cwd=code, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        from launch_history_doppler_campaign import TESTS
        run('cpu_tests', ['-m', 'pytest', '-q', *TESTS,
            'tests/test_history_doppler_recovery.py', 'tests/test_finetune_prediction_spool.py',
            '--junitxml=' + str(campaign / 'cpu_tests.xml')])
        run('single_sweep_cache', ['tools/precompute_single_sweep_radar.py',
            '--config', 'configs/sw-history-h1-velocity.py', '--output', CACHE, '--workers', '8', '--verify-existing'])
        for arm in ARMS:
            run('cpu_' + arm, ['tools/check_history_doppler_contracts.py', '--config',
                'configs/sw-history-' + arm + '.py', '--device', 'cpu', '--out', str(campaign / ('cpu_' + arm + '.json'))])
            item = json.loads((campaign / ('cpu_' + arm + '.json')).read_text())
            old = json.loads((args.recovery_of / ('cpu_' + arm + '.json')).read_text())
            if item['status'] != 'passed' or item['git_revision'] != manifest['git_revision']:
                raise ValueError('Exact-recovery CPU contract failed: ' + arm)
            for key in ('state_schema', 'sha256'):
                if item['initialization'][key] != old['initialization'][key]:
                    raise ValueError('Recovery initialization differs from original: ' + arm)
            if item['input_contract'] != old['input_contract'] or item['budget'] != old['budget']:
                raise ValueError('Scientific input/budget contract changed: ' + arm)
        report.update(state='passed', at_utc=utcnow())
        write_json(campaign / 'preflight.json', report)
        # Recheck live originals after lengthy CPU preflight, before any worker.
        require_old_processes_absent(Path(recovery['original_code']), args.recovery_of)
        controllers = {}
        for gpu in (1, 0):
            log_path = campaign / ('gpu%d_controller.log' % gpu)
            with log_path.open('xb') as log:
                child = subprocess.Popen([sys.executable, 'tools/run_history_doppler_experiment.py',
                    '--gpu', str(gpu), '--campaign-dir', str(campaign), '--recovery-of', str(args.recovery_of)],
                    cwd=code, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            controllers[str(gpu)] = dict(pid=child.pid, log=str(log_path))
            report.update(state='submitting', controllers=dict(controllers), at_utc=utcnow())
            write_json(campaign / 'submission.json', report)
        report.update(state='submitted', controllers=controllers, at_utc=utcnow())
        write_json(campaign / 'submission.json', report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        report.update(state='failed', error=repr(error), traceback=traceback.format_exc(), at_utc=utcnow())
        write_json(campaign / 'preflight.json', report)
        write_json(campaign / 'submission.json', report)
        raise


if __name__ == '__main__':
    main()
