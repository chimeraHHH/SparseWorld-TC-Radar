"""Isolated censored-path queue; never resumes or mutates another experiment.

The controller owns the existing per-card lock throughout the queue, and admits
each GPU subprocess only after 132000 MiB have remained free for 60 seconds.
"""
import argparse
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

if __package__:
    from .gpu_capacity import CapacityWindow, memory_snapshot
else:
    from gpu_capacity import CapacityWindow, memory_snapshot


ARM = 'censored-path'
CAMPAIGN = 'censored_path_campaign_20260926'
ROOT = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
GPU_UUID = 'GPU-000b6236-3632-a001-9667-1f02cbb61c8b'
CONFIG = 'configs/sw-radar-forecast-censored-path.py'
SMOKE_CONFIG = 'configs/sw-radar-forecast-censored-path-smoke.py'
HORIZONS = ('0.0s', '1.0s', '2.0s', '3.0s')
INTERVENTIONS = ('normal', 'drop', 'zero_velocity', 'shuffle_velocity')


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_state_schema(state, schema):
    """Compare every tensor key and shape to the exact CPU preflight model."""
    if not isinstance(schema, dict) or not schema:
        raise ValueError('CPU preflight has no exact state_schema')
    if set(state) != set(schema):
        raise ValueError(dict(missing=sorted(set(schema) - set(state)),
                              unexpected=sorted(set(state) - set(schema))))
    mismatched = {name: (list(state[name].shape), list(shape))
                  for name, shape in schema.items()
                  if list(state[name].shape) != list(shape)}
    if mismatched:
        raise ValueError('Checkpoint shapes differ from CPU preflight: ' + str(mismatched))


def audit_checkpoint(path, schema, expected_epoch=None, exact_steps=None):
    import torch
    torch.set_num_threads(2)
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint['state_dict']
    validate_state_schema(state, schema)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError('Nonfinite model state: ' + str(path))
    optimizer = checkpoint['optimizer']['state']
    steps = [int(value['step']) for value in optimizer.values() if 'step' in value]
    if not steps or min(steps) != max(steps):
        raise ValueError('Missing or inconsistent optimizer step counters')
    if not all(torch.isfinite(value).all() for entry in optimizer.values()
               for value in entry.values() if torch.is_tensor(value)):
        raise FloatingPointError('Nonfinite optimizer state: ' + str(path))
    if exact_steps is not None and min(steps) != exact_steps:
        raise ValueError('Smoke requires %d optimizer updates, found %d' % (exact_steps, min(steps)))
    if expected_epoch is not None and checkpoint['meta']['epoch'] != expected_epoch:
        raise ValueError('Unexpected checkpoint epoch')
    result = dict(path=str(path), epoch=checkpoint['meta']['epoch'],
                  iterations=checkpoint['meta']['iter'], optimizer_steps=min(steps),
                  finite_model_tensors=len(state), state_schema_matches_cpu=True,
                  optimizer_finite=True)
    del checkpoint, state, optimizer
    gc.collect()
    return result


def _finite_nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid learning audit value for ' + name)
    return value


def audit_learning_log(contents, expected_components):
    """Check actual gradients and parameter changes, allowing zero-init warmup."""
    if (not expected_components or len(set(expected_components)) != len(expected_components)
            or not all(isinstance(name, str) and name for name in expected_components)):
        raise ValueError('CPU preflight must declare distinct worldline_components')
    radar, joint, path_windows = [], [], []
    for line in contents.splitlines():
        if 'RADAR_LEARNING' in line:
            match = re.search(r'microstep_grad_norm=([^ ]+) parameter_delta=([^ ]+)', line)
            if match is None:
                raise ValueError('Malformed RADAR_LEARNING audit')
            values = [_finite_nonnegative(v, 'radar') for v in match.groups()]
            if not all(value > 0 for value in values):
                raise ValueError('Radar parameters did not learn')
            radar.append(values)
        if 'JOINT_LEARNING' in line:
            payload = json.loads(line[line.index('{'):])
            if set(payload) != {'img_backbone', 'img_neck', 'pts_bbox_head'}:
                raise ValueError('Missing pretrained component audit')
            for name, values in payload.items():
                for field in ('gradient_norm', 'parameter_delta'):
                    if _finite_nonnegative(values[field], name) <= 0:
                        raise ValueError('Pretrained component did not learn: ' + name)
            joint.append(payload)
        if 'CENSORED_PATH_LEARNING' in line:
            match = re.search(r'CENSORED_PATH_LEARNING iter=(\d+)\s+(\{.*\})', line)
            if match is None:
                raise ValueError('Malformed CENSORED_PATH_LEARNING audit')
            iteration, payload = int(match.group(1)), json.loads(match.group(2))
            if set(payload) != set(expected_components):
                raise ValueError('Worldline component audit differs from CPU preflight')
            for name, values in payload.items():
                for field in ('gradient_norm', 'parameter_delta'):
                    _finite_nonnegative(values[field], name)
            if path_windows and iteration <= path_windows[-1]['iter']:
                raise ValueError('Worldline audit iterations must increase')
            path_windows.append(dict(iter=iteration, components=payload))
    if len(radar) != 6 or len(joint) != 6:
        raise ValueError('24-step smoke must contain six radar and pretrained audit windows')
    if len(path_windows) < 2:
        raise ValueError('Need at least two worldline learning audit windows')
    if not all(0 < window['iter'] <= 24 for window in path_windows):
        raise ValueError('Worldline smoke audit lies outside the 24-step run')
    totals = {name: {field: sum(window['components'][name][field]
                               for window in path_windows[-2:])
                     for field in ('gradient_norm', 'parameter_delta')}
              for name in expected_components}
    if not all(value > 0 for values in totals.values() for value in values.values()):
        raise ValueError('A worldline component has no real update in the last two windows')
    return dict(radar_audit_windows=len(radar), pretrained_audit_windows=len(joint),
                worldline_audit_windows=path_windows,
                last_two_worldline_windows_totals=totals,
                all_audited_components_updated=True)


def require_full_result(path, confusion_dir):
    payload = json.loads(path.read_text())
    if payload['samples'] != 5119:
        raise ValueError('Full evaluation requires exactly 5119 anchors')
    for horizon in HORIZONS:
        if not (confusion_dir / (horizon + '.npz')).is_file():
            raise FileNotFoundError(confusion_dir / (horizon + '.npz'))
    return payload


def require_fresh_initialization(path, cpu_initialization):
    initialization = json.loads(path.read_text())
    for key, expected in (('loaded_tensors', 669), ('optimizer_restored', False),
                          ('epoch_reset_to', 0), ('all_camera_tensors_exact', True)):
        if initialization.get(key) != expected:
            raise ValueError('Formal initialization failed contract: ' + key)
    if initialization['sha256'] != cpu_initialization['sha256']:
        raise ValueError('Formal initialization differs from audited official checkpoint')
    return initialization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[1], default=1)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    os.chdir(code)
    campaign = ROOT / 'analysis' / CAMPAIGN
    if not campaign.is_dir():
        raise FileNotFoundError('Launch only through launch_censored_path_campaign.py')
    status_path = campaign / (ARM + '_status.json')
    manifest = json.loads((code / 'code_manifest.json').read_text())
    status = dict(arm=ARM, gpu=args.gpu, gpu_uuid=GPU_UUID, controller_pid=os.getpid(),
                  git_revision=manifest['git_revision'], code=str(code),
                  completed_stages=[], state='preparing', at_utc=utcnow())
    # Atomic reservation also protects against bypassing the launcher twice.
    with status_path.open('x') as stream:
        json.dump(status, stream, indent=2, allow_nan=False)

    def record(**values):
        status.update(values, at_utc=utcnow())
        write_json(status_path, status)

    try:
        for name, expected in manifest['sha256'].items():
            if hashlib.sha256((code / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Immutable snapshot hash differs: ' + name)
        cpu = json.loads((campaign / ('cpu_' + ARM + '.json')).read_text())
        preflight = json.loads((campaign / 'preflight.json').read_text())
        if (cpu['status'] != 'passed' or preflight['state'] != 'passed'
                or cpu['git_revision'] != manifest['git_revision']
                or preflight['git_revision'] != manifest['git_revision']):
            raise ValueError('Exact-revision CPU preflight has not passed')
        schema = cpu['initialization']['state_schema']
        components = cpu['worldline_components']
        if not schema or not components:
            raise ValueError('Incomplete CPU contract')
        baseline = json.loads((ROOT / 'analysis/forecast_campaign_20260920/transport_result.json').read_text())
        if baseline['arm'] != 'transport':
            raise ValueError('Expected completed A reference')
        for scope in ('best', 'final'):
            require_full_result(Path(baseline[scope + '_json']), Path(baseline[scope + '_confusions']))
        work = ROOT / 'work_dirs/radar_forecast_censored-path_seed0'
        smoke = ROOT / 'work_dirs/radar_forecast_censored-path_smoke_seed0'
        for directory in (work, smoke):
            if directory.exists():
                raise FileExistsError('New-arm output path already exists: ' + str(directory))
        record(state='waiting_for_gpu_lock')
        lock = (ROOT / ('forecast_gpu%d.lock' % args.gpu)).open('a')
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', PYTHONPATH=str(code),
                   PYTHONUNBUFFERED='1', TMPDIR='/tmp',
                   TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/torch_extensions'))
        python = str(ROOT / 'envs/hym_sparseworld/bin/python')
        telemetry = campaign / (ARM + '_gpu.jsonl')

        def run(stage, arguments):
            window = CapacityWindow(minimum_free_mib=132000, quiet_seconds=60)
            while True:
                snapshot, now = memory_snapshot(GPU_UUID), time.monotonic()
                admitted = window.observe(snapshot['free_mib'], now)
                admission = dict(**snapshot, minimum_free_mib=window.minimum_free_mib,
                    required_stable_seconds=window.quiet_seconds,
                    observed_stable_seconds=(now - window.sufficient_since
                        if window.sufficient_since is not None else 0.))
                if admitted:
                    record(memory_admission=admission)
                    break
                record(state='waiting_for_gpu_memory', stage=stage, child_pid=None,
                       memory_admission=admission)
                time.sleep(15)
            command = [python] + arguments
            log_path = campaign / (ARM + '_' + stage + '.log')
            with log_path.open('xb') as log:
                child = subprocess.Popen(command, cwd=code, env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
                record(state='running', stage=stage, child_pid=child.pid,
                       command=command, log=str(log_path))
                while True:
                    try:
                        returncode = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        snapshot = subprocess.check_output(['nvidia-smi', '-i', GPU_UUID,
                            '--query-gpu=utilization.gpu,memory.used,power.draw',
                            '--format=csv,noheader,nounits'], text=True).strip()
                        with telemetry.open('a') as stream:
                            stream.write(json.dumps(dict(at_utc=utcnow(), stage=stage, gpu=snapshot)) + '\n')
                        record(gpu_snapshot=snapshot)
                record(child_pid=None, child_returncode=returncode)
                if returncode:
                    raise RuntimeError('%s exited %d; see %s' % (stage, returncode, log_path))
            status['completed_stages'].append(stage)
            record(stage=stage)

        run('gpu_contracts', ['-m', 'pytest', '-q', 'tests/test_m0_contracts.py',
                             'tests/test_censored_path.py', '-k', 'cuda'])
        run('gpu_parity', ['tools/check_censored_path_contracts.py', '--config', CONFIG,
                          '--device', 'cuda', '--out', str(campaign / ('gpu_' + ARM + '.json'))])
        parity = json.loads((campaign / ('gpu_' + ARM + '.json')).read_text())
        if parity['status'] != 'passed' or parity['git_revision'] != manifest['git_revision']:
            raise ValueError('GPU parity did not pass for this snapshot')
        run('smoke', ['train.py', '--config', SMOKE_CONFIG])
        smoke_audit = audit_checkpoint(smoke / 'iter_24.pth', schema, exact_steps=24)
        smoke_audit.update(audit_learning_log((smoke / 'train.log').read_text(), components))
        write_json(campaign / ('smoke_' + ARM + '.json'), smoke_audit)
        # A new train.py process reloads only the official model; never smoke state.
        run('train', ['train.py', '--config', CONFIG])
        initialization = require_fresh_initialization(work / 'official_initialization.json', cpu['initialization'])
        final_audit = audit_checkpoint(work / 'epoch_10.pth', schema, expected_epoch=10)
        if final_audit['iterations'] != 29920:
            raise ValueError('Formal budget must be exactly 10 epochs / 29920 iterations')
        final_json, final_confusions = work / 'validation_epoch_10_full.json', work / 'confusions_epoch_10_full'
        require_full_result(final_json, final_confusions)
        best_meta = json.loads((work / 'best_future.json').read_text())
        if best_meta['epoch'] not in range(1, 11) or best_meta['samples'] != 256:
            raise ValueError('Selection must use a trained epoch and the fixed 256-anchor subset')
        best_audit = audit_checkpoint(work / 'best_future.pth', schema, expected_epoch=best_meta['epoch'])
        if best_meta['epoch'] == 10:
            best_json, best_confusions = final_json, final_confusions
        else:
            best_dir = campaign / (ARM + '_best_full')
            run('best_full', ['tools/evaluate_radar_experiment.py', '--config', CONFIG,
                '--checkpoint', str(work / 'best_future.pth'), '--samples', '0',
                '--output-dir', str(best_dir)])
            best_json, best_confusions = best_dir / 'normal.json', best_dir / 'confusions_normal'
        require_full_result(best_json, best_confusions)
        intervention_dir = campaign / (ARM + '_counterfactual')
        run('counterfactual', ['tools/evaluate_radar_experiment.py', '--config', CONFIG,
            '--checkpoint', str(work / 'best_future.pth'), '--samples', '256',
            '--modes', *INTERVENTIONS, '--output-dir', str(intervention_dir)])
        indices = None
        for mode in INTERVENTIONS:
            intervention = json.loads((intervention_dir / (mode + '.json')).read_text())
            if intervention['samples'] != 256 or intervention['mode'] != mode:
                raise ValueError('Invalid fixed-subset radar intervention: ' + mode)
            if indices is not None and indices != intervention['indices']:
                raise ValueError('Radar interventions used different anchors')
            indices = intervention['indices']
        result = dict(arm=ARM, gpu=args.gpu, git_revision=manifest['git_revision'],
                      best_epoch=best_meta['epoch'], final_audit=final_audit, best_audit=best_audit,
                      official_initialization=initialization, final_json=str(final_json),
                      best_json=str(best_json), best_confusions=str(best_confusions),
                      final_confusions=str(final_confusions), interventions=str(intervention_dir))
        comparisons = {}
        for scope in ('best', 'final'):
            destination = campaign / ('A_vs_censored_path_' + scope + '.json')
            command = [python, 'tools/compare_forecast_results.py',
                       '--reference', baseline[scope + '_confusions'],
                       '--candidate', result[scope + '_confusions'], '--out', str(destination)]
            with (campaign / ('comparison_' + scope + '.log')).open('xb') as log:
                subprocess.run(command, cwd=code, env=dict(env, CUDA_VISIBLE_DEVICES=''),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            comparisons[scope] = str(destination)
        result['paired_A_comparisons'] = comparisons
        write_json(campaign / (ARM + '_result.json'), result)
        record(state='complete', stage='all_training_and_evaluation_complete', result=result)
    except BaseException as error:
        record(state='failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
