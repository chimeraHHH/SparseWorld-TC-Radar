"""Recover two completed H1 checkpoints and untouched H8 arms in a new receipt tree.

This entry point never resumes training H1, edits the failed campaign, or
reuses smoke weights. Only evaluation/storage/scheduling code may differ from
the original immutable experiment. Each submission path is reserved once.
"""
import argparse
from decimal import Decimal
import gc
import hashlib
import json
import math
import os
import pwd
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
PREVIOUS_RECOVERY_REVISION = '3aaf8693d1e270a69fad940145fb28b47859631a'
HORIZONS = ('0.0s', '1.0s', '2.0s', '3.0s')
CLASS_METRICS = tuple(name + '_IoU' for name in (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'))
AGGREGATE_METRICS = ('Semantic mIoU', 'Binary IoU')
PARITY_POLICY = dict(
    authorization='User explicitly permitted relaxing evaluation parity tolerance on 2026-09-29',
    units='percentage points', aggregate_atol=0.001, per_class_atol=0.01, rtol=0,
    scope='Independent saved-checkpoint fixed256 evaluation only; frozen-output storage must remain exact')


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
    """Audit independent inference within explicitly authorized absolute bounds.

    The original inline and standalone evaluators have different metadata
    schemas. The metric schema and anchor identity remain exact. This tolerance
    never applies to the same-prediction list/disk comparison.
    """
    old, new = (json.loads(Path(p).read_text()) for p in (previous, current))
    if (old.get('scope') != 'subset' or old.get('epoch') != 10
            or old.get('dataset_samples') != 5119 or old.get('mode', 'normal') != 'normal'
            or new.get('mode') != 'normal'
            or new.get('checkpoint_meta') != {'epoch': 10, 'iter': 29920}):
        raise ValueError('Recovery parity requires original epoch10 inline subset and normal standalone schemas')
    if any(type(item.get('samples')) is not int or item['samples'] != 256 for item in (old, new)):
        raise ValueError('Recovery parity requires the original fixed 256 normal anchors')
    indices = old.get('indices')
    if (not isinstance(indices, list) or len(indices) != 256 or len(set(indices)) != 256
            or any(type(index) is not int or not 0 <= index < 5119 for index in indices)
            or indices != sorted(indices) or new.get('indices') != indices
            or any(type(index) is not int for index in new['indices'])):
        raise ValueError('Recovered evaluation differs from saved H1 subset: ordered indices')
    for item in (old, new):
        if not isinstance(item.get('metrics'), dict) or set(item['metrics']) != set(HORIZONS):
            raise ValueError('Recovery parity horizon schema differs')
        for horizon in HORIZONS:
            metrics = item['metrics'][horizon]
            expected = set(AGGREGATE_METRICS + CLASS_METRICS + ('evaluated_samples',))
            if not isinstance(metrics, dict) or set(metrics) != expected:
                raise ValueError('Recovery parity metric schema differs: ' + horizon)
            if type(metrics['evaluated_samples']) is not int or metrics['evaluated_samples'] != 256:
                raise ValueError('Recovery parity evaluated_samples must remain exactly 256: ' + horizon)

    differences, violations = {}, []
    def compare(name, reference, observed, tolerance, allow_undefined=False):
        both_null = reference is None and observed is None
        both_nan = (isinstance(reference, float) and isinstance(observed, float)
                    and math.isnan(reference) and math.isnan(observed))
        # Undefined per-class IoU is produced for unsupported classes; inline
        # json_safe encodes it as null. Never tolerate a newly undefined class.
        if allow_undefined and (both_null or both_nan):
            differences[name] = dict(previous=None, current=None, signed_delta_pp=None,
                absolute_delta_pp=None, atol_pp=tolerance, within_tolerance=True,
                matching_undefined='null' if both_null else 'NaN')
            return
        if (type(reference) not in (int, float) or type(observed) not in (int, float)
                or not math.isfinite(reference) or not math.isfinite(observed)):
            raise ValueError('Recovery parity requires finite metrics or matching undefined classes: ' + name)
        if not 0 <= reference <= 100 or not 0 <= observed <= 100:
            raise ValueError('Recovery parity metric outside percentage range: ' + name)
        # Compare the serialized decimal values to make the inclusive boundary
        # unambiguous (23.001 - 23.0 must satisfy a 0.001 pp allowance).
        delta = Decimal(str(observed)) - Decimal(str(reference))
        passed = abs(delta) <= Decimal(str(tolerance))
        differences[name] = dict(previous=reference, current=observed,
            signed_delta_pp=float(delta), absolute_delta_pp=float(abs(delta)),
            atol_pp=tolerance, within_tolerance=passed)
        if not passed:
            violations.append(name)

    for horizon in HORIZONS:
        for key in AGGREGATE_METRICS + CLASS_METRICS:
            compare(horizon + '/' + key, old['metrics'][horizon][key], new['metrics'][horizon][key],
                    PARITY_POLICY['aggregate_atol'] if key in AGGREGATE_METRICS else PARITY_POLICY['per_class_atol'],
                    allow_undefined=key in CLASS_METRICS)
    compare('future_mean_miou', old.get('future_mean_miou'), new.get('future_mean_miou'),
            PARITY_POLICY['aggregate_atol'])
    audit = dict(status='failed' if violations else 'passed', samples=256,
                comparison='absolute_tolerance', policy=dict(PARITY_POLICY),
                exact_indices=True, exact_metric_schema=True, exact_evaluated_samples=True,
                schemas=dict(previous='inline_epoch10_subset', current='standalone_normal_epoch10'),
                differences=differences, violations=violations,
                previous=str(previous), current=str(current), previous_sha256=sha256_file(previous),
                current_sha256=sha256_file(current))
    if violations:
        error = ValueError('Recovered evaluation exceeds authorized parity bounds: ' + ', '.join(violations))
        error.parity_audit = audit
        raise error
    return audit


def require_old_processes_absent(original_code, original, proc=Path('/proc')):
    """Exclude live originals; only verified system helpers may hide their cwd."""
    recorded = set()
    for name in ('gpu0_worker.json', 'gpu1_worker.json'):
        payload = json.loads((original / name).read_text())
        recorded.update(int(payload[k]) for k in ('controller_pid', 'child_pid') if payload.get(k))
    submission_path = original / 'submission.json'
    if submission_path.exists():
        submission = json.loads(submission_path.read_text())
        if submission.get('launcher_pid'):
            recorded.add(int(submission['launcher_pid']))
        recorded.update(int(item['pid']) for item in submission.get('controllers', {}).values() if item.get('pid'))
    alive = [pid for pid in sorted(recorded) if (proc / str(pid)).exists()]
    # Even a reused or helper-shaped PID is not silently accepted as exited.
    if alive:
        raise ValueError('Original processes still present: ' + str(dict(recorded=alive)))
    residuals, protected_helpers = [], []
    uid = os.getuid()
    username = pwd.getpwuid(uid).pw_name
    original_path = original_code.resolve()
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != uid:
                continue
            fields = dict(line.split(':', 1) for line in (directory / 'status').read_text().splitlines()
                          if ':' in line)
            uids = [int(value) for value in fields['Uid'].split()]
            name = fields['Name'].strip()
            if len(uids) != 4:
                raise ValueError('Malformed UID evidence for process ' + directory.name)
            if uids[1] != uid:
                continue
            command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
            if str(original_code) in command:
                residuals.append(int(directory.name))
                continue
            try:
                cwd = (directory / 'cwd').resolve(strict=True)
            except PermissionError as error:
                # Observed Ubuntu login-session helpers are nondumpable despite
                # owning UID 1031. Their exact status + command identity rules
                # out a Python trainer; unfamiliar protected processes block.
                known_helper = (uids == [uid] * 4 and (
                    (name == '(sd-pam)' and command == '(sd-pam)') or
                    (name == 'sshd' and command == 'sshd: ' + username + '@notty')))
                if not known_helper:
                    raise PermissionError('Cannot exclude old-snapshot process with unreadable cwd: '
                                          + directory.name + ' name=' + name) from error
                protected_helpers.append(dict(pid=int(directory.name), name=name, command=command,
                    uids=uids, reason='verified nondumpable login-session helper; cwd unreadable'))
                continue
            if cwd == original_path:
                residuals.append(int(directory.name))
        except FileNotFoundError:
            continue  # The process exited during the read.
    if residuals:
        raise ValueError('Original processes still present: ' + str(dict(residuals=residuals)))
    return dict(recorded_pids=sorted(recorded), live_recorded_pids=[], own_original_snapshot_processes=[],
                protected_session_helpers=protected_helpers)


def require_previous_recovery_absent(previous, original, code):
    """Preserve and exclude the observed failed strict-parity recovery."""
    previous, original, code = Path(previous), Path(original), Path(code)
    if previous.resolve() == original.resolve():
        raise ValueError('Previous recovery must be distinct from the original campaign')
    submission = json.loads((previous / 'submission.json').read_text())
    preflight = json.loads((previous / 'preflight.json').read_text())
    old_code = Path(submission['code'])
    if (submission['git_revision'] != PREVIOUS_RECOVERY_REVISION
            or preflight['git_revision'] != PREVIOUS_RECOVERY_REVISION
            or preflight['state'] != 'passed'
            or Path(submission['recovery_of']).resolve() != original.resolve()
            or code.resolve() == old_code.resolve()):
        raise ValueError('Unexpected previous recovery identity or source reuse')
    manifest = json.loads((old_code / 'code_manifest.json').read_text())
    if manifest['git_revision'] != PREVIOUS_RECOVERY_REVISION:
        raise ValueError('Unexpected previous recovery manifest')
    verify_snapshot(old_code, manifest)
    states = {}
    for gpu in (0, 1):
        path = previous / ('gpu%d_worker.json' % gpu)
        state = json.loads(path.read_text())
        if state.get('state') != 'failed' or state.get('stage') != 'recovery_subset_parity':
            raise ValueError('Previous recovery worker is not terminal at the known parity failure')
        states[str(gpu)] = dict(state=state['state'], stage=state['stage'], sha256=sha256_file(path))
    for arm in ARMS:
        if (previous / (arm + '_result.json')).exists():
            raise FileExistsError('Previous recovery already completed an arm: ' + arm)
        if arm.startswith('h8-') and (previous / (arm + '_claim.json')).exists():
            raise FileExistsError('Previous recovery already claimed H8: ' + arm)
    process_audit = require_old_processes_absent(old_code, previous)
    return dict(campaign=str(previous), code=str(old_code), git_revision=PREVIOUS_RECOVERY_REVISION,
                submission_sha256=sha256_file(previous / 'submission.json'),
                preflight_sha256=sha256_file(previous / 'preflight.json'),
                worker_states=states, process_audit=process_audit, preserved=True)


def require_exact_storage_diagnostic(path, code, revision):
    """Tolerance concerns separate inference; the storage implementation is exact."""
    path, code = Path(path), Path(code)
    result = json.loads(path.read_text())
    if (result.get('status') != 'complete_observations_only'
            or result.get('checkpoint_unchanged') is not True
            or type(result.get('optimizer_steps_executed')) is not int
            or result['optimizer_steps_executed'] != 0):
        raise ValueError('Read-only completed storage diagnostic is required')
    if (result.get('source_revision') != revision
            or result.get('original_revision') != ORIGINAL_REVISION
            or result.get('current_evaluator_sha256') != sha256_file(code / 'finetune_hooks.py')):
        raise ValueError('Storage diagnostic must use this exact evaluation source')
    if result.get('checkpoint_meta') != {'epoch': 10, 'iter': 29920}:
        raise ValueError('Storage diagnostic must use a completed H1 checkpoint')
    policies = result.get('policies', {})
    required = ('offline_manual_seed', 'training_deterministic_seed')
    if set(policies) != set(required):
        raise ValueError('Storage diagnostic must cover both inference policies')
    for name in required:
        policy = policies[name]
        comparison = policy.get('fixed_output_storage_comparison', {})
        if (policy.get('spool_arrays_exact') is not True
                or comparison.get('metrics_exact') is not True
                or comparison.get('all_scene_arrays_exact') is not True):
            raise ValueError('Frozen-output storage equality failed: ' + name)
    return dict(path=str(path), sha256=sha256_file(path), source_revision=revision,
                frozen_output_arrays_exact=True, metrics_exact=True, all_scene_arrays_exact=True,
                checkpoint_unchanged=True, optimizer_steps_executed=0)


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
    parser.add_argument('--previous-recovery', type=Path)
    parser.add_argument('--diagnostic-result', type=Path, required=True,
                        help='Exact-version completed frozen-output list/disk diagnostic')
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
    recovery['parity_policy'] = dict(PARITY_POLICY)
    recovery['exact_storage_diagnostic'] = require_exact_storage_diagnostic(
        args.diagnostic_result, code, manifest['git_revision'])
    if args.previous_recovery:
        recovery['previous_recovery'] = require_previous_recovery_absent(
            args.previous_recovery, args.recovery_of, code)
    campaign = args.campaign_dir
    if (campaign.resolve() == args.recovery_of.resolve()
            or args.previous_recovery and campaign.resolve() == args.previous_recovery.resolve()):
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
        require_exact_storage_diagnostic(args.diagnostic_result, code, manifest['git_revision'])
        if args.previous_recovery:
            require_previous_recovery_absent(args.previous_recovery, args.recovery_of, code)
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
