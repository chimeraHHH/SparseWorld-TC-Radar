"""Four matched arms, claimed once by available single-H200 workers.

No canceled campaign is resumed. Training always starts from official weights
after its own audited smoke. An occupied GPU is a queue condition, not failure.
"""
import argparse
import datetime
import fcntl
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
    from .run_censored_path_experiment import audit_checkpoint, require_full_result, require_fresh_initialization
else:
    from gpu_capacity import CapacityWindow, memory_snapshot
    from run_censored_path_experiment import audit_checkpoint, require_full_result, require_fresh_initialization

ROOT = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
CAMPAIGN = 'history_doppler_campaign_20260927'
ARMS = ('h1-velocity', 'h1-geometry', 'h8-velocity', 'h8-geometry')
GPUS = {0: 'GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe',
        1: 'GPU-000b6236-3632-a001-9667-1f02cbb61c8b'}
CACHE = '/home/huayiming/Workspace/SparseWorld-cache/radar_single_sweep_v1_20260927'


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, payload):
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def unclaimed(campaign):
    return [arm for arm in ARMS if not (campaign / (arm + '_claim.json')).exists()]


def claim_arm(campaign, gpu, revision):
    with (campaign / 'claims.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending = unclaimed(campaign)
        if not pending:
            return None
        arm = pending[0]
        with (campaign / (arm + '_claim.json')).open('x') as stream:
            json.dump(dict(arm=arm, gpu=gpu, controller_pid=os.getpid(),
                           git_revision=revision, at_utc=utcnow()), stream, indent=2)
        return arm


def other_compute(uuid):
    lines = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    return [line for line in lines.splitlines() if line.strip()]


def audit_smoke_log(contents):
    radar, joint = [], []
    for line in contents.splitlines():
        if 'RADAR_LEARNING' in line:
            match = re.search(r'microstep_grad_norm=([^ ]+) parameter_delta=([^ ]+)', line)
            if match is None:
                raise ValueError('Malformed radar learning audit')
            values = [float(v) for v in match.groups()]
            if not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError('No finite positive radar update')
            radar.append(values)
        if 'JOINT_LEARNING' in line:
            payload = json.loads(line[line.index('{'):])
            if set(payload) != {'img_backbone', 'img_neck', 'pts_bbox_head'}:
                raise ValueError('Incomplete pretrained-component audit')
            for item in payload.values():
                if not all(math.isfinite(item[k]) and item[k] > 0 for k in ('gradient_norm', 'parameter_delta')):
                    raise ValueError('Pretrained component did not update')
            joint.append(payload)
    if len(radar) != 6 or len(joint) != 6:
        raise ValueError('Require six real-update windows during the 24-step smoke')
    return dict(radar_update_windows=radar, pretrained_update_windows=joint)


def verify_snapshot(code, manifest):
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((code / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Immutable source changed: ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True, type=int, choices=[0, 1])
    parser.add_argument('--campaign-dir', type=Path)
    parser.add_argument('--recovery-of', type=Path)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    os.chdir(code)
    campaign = args.campaign_dir or ROOT / 'analysis' / CAMPAIGN
    if bool(args.campaign_dir) != bool(args.recovery_of):
        raise ValueError('Custom campaign requires its explicit recovery source')
    recovery = None
    if args.recovery_of:
        from recover_history_doppler_campaign import validate_recovery_source
        recovery = validate_recovery_source(args.recovery_of, code)
        if campaign.resolve() == args.recovery_of.resolve():
            raise ValueError('Never overwrite the failed campaign')
    manifest = json.loads((code / 'code_manifest.json').read_text())
    verify_snapshot(code, manifest)
    preflight = json.loads((campaign / 'preflight.json').read_text())
    if preflight['state'] != 'passed' or preflight['git_revision'] != manifest['git_revision']:
        raise ValueError('This exact version has not passed CPU preflight')
    worker_path = campaign / ('gpu%d_worker.json' % args.gpu)
    status = dict(controller_pid=os.getpid(), gpu=args.gpu, gpu_uuid=GPUS[args.gpu],
                  code=str(code), git_revision=manifest['git_revision'], at_utc=utcnow(),
                  state='waiting_for_gpu_lock', completed_arms=[], recovery_of=str(args.recovery_of) if recovery else None,
                  original_git_revision=recovery['original_git_revision'] if recovery else None)
    with worker_path.open('x') as stream:
        json.dump(status, stream, indent=2)
    arm_status = None
    active_child = None

    def record(**values):
        status.update(values, at_utc=utcnow())
        write_json(worker_path, status)
        if arm_status is not None:
            write_json(arm_status, status)

    python = str(ROOT / 'envs/hym_sparseworld/bin/python')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[args.gpu], OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(code),
               PYTHONUNBUFFERED='1', TMPDIR='/tmp',
               TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/torch_extensions'))
    if recovery:
        from recover_history_doppler_campaign import EVAL_SPOOL, HOST_MINIMUM_BYTES, host_available_bytes
        env['SPARSEWORLD_EVAL_TMPDIR'] = EVAL_SPOOL
        env['SPARSEWORLD_FULL_EVAL_LOCK'] = str(ROOT / 'history_doppler_full_eval.lock')

    def admit(abandon_if_claimed=False):
        window = CapacityWindow(minimum_free_mib=132000, quiet_seconds=60)
        host_since = None
        while True:
            if abandon_if_claimed and not unclaimed(campaign):
                return False
            snapshot, now = memory_snapshot(GPUS[args.gpu]), time.monotonic()
            processes = other_compute(GPUS[args.gpu])
            allowed = window.observe(0 if processes else snapshot['free_mib'], now)
            available = host_available_bytes() if recovery else None
            if recovery:
                host_since = ((now if host_since is None else host_since)
                              if available >= HOST_MINIMUM_BYTES else None)
                allowed = allowed and host_since is not None and now - host_since >= 60
            admission = dict(**snapshot, other_compute=processes,
                minimum_free_mib=132000, required_stable_seconds=60,
                observed_stable_seconds=now-window.sufficient_since if window.sufficient_since is not None else 0,
                host_available_bytes=available, host_minimum_bytes=HOST_MINIMUM_BYTES if recovery else None,
                host_observed_stable_seconds=now-host_since if host_since is not None else 0)
            record(state='waiting_for_gpu_memory', child_pid=None, memory_admission=admission)
            if allowed:
                return True
            time.sleep(15)

    def run(arm, stage, arguments):
        nonlocal active_child
        record(stage=stage)
        admit()
        started = time.monotonic()
        command = [python] + arguments
        log_path = campaign / (arm + '_' + stage + '.log')
        with log_path.open('xb') as log:
            child = subprocess.Popen(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
            active_child = child
            record(state='running', stage=stage, child_pid=child.pid, command=command, log=str(log_path))
            while True:
                try:
                    returncode = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        snapshot = subprocess.check_output(['nvidia-smi', '-i', GPUS[args.gpu],
                            '--query-gpu=utilization.gpu,memory.used,power.draw',
                            '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
                        with (campaign / (arm + '_gpu.jsonl')).open('a') as stream:
                            stream.write(json.dumps(dict(at_utc=utcnow(), stage=stage, gpu=snapshot))+'\n')
                        record(gpu_snapshot=snapshot)
                    except (OSError, subprocess.SubprocessError) as warning:
                        # Telemetry is not a training failure. Keep supervising
                        # the real child and holding the card lock.
                        print('TELEMETRY_WARNING', repr(warning), flush=True)
            record(child_pid=None, child_returncode=returncode)
            if returncode:
                raise RuntimeError('%s exited %s; see %s' % (stage, returncode, log_path))
            active_child = None
        status['stage_seconds'][stage] = time.monotonic()-started
        status['completed_stages'].append(stage)
        record()

    try:
        with (ROOT / ('forecast_gpu%d.lock' % args.gpu)).open('a') as lock:
            while True:
                if not unclaimed(campaign):
                    record(state='complete', stage='worker_queue_finished', arm=None, child_pid=None)
                    return
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    record(state='waiting_for_gpu_lock')
                    time.sleep(15)
            while unclaimed(campaign):
                # Claim only when ready; an occupied card must not strand a job.
                record(stage='awaiting_unclaimed_arm', arm=None)
                if not admit(abandon_if_claimed=True):
                    break
                arm = claim_arm(campaign, args.gpu, manifest['git_revision'])
                if arm is None:
                    break
                arm_status = campaign / (arm + '_status.json')
                record(arm=arm, completed_stages=[], stage_seconds={})
                cfg = 'configs/sw-history-' + arm + '.py'
                work = ROOT / ('work_dirs/history_' + arm + '_seed0')
                smoke = ROOT / ('work_dirs/history_' + arm + '_smoke_seed0')
                recovering_h1 = bool(recovery and arm.startswith('h1-'))
                if not recovering_h1 and (work.exists() or smoke.exists()):
                    raise FileExistsError('New-arm output already exists')
                cpu = json.loads((campaign / ('cpu_' + arm + '.json')).read_text())
                if cpu['status'] != 'passed' or cpu['git_revision'] != manifest['git_revision']:
                    raise ValueError('Arm CPU preflight mismatch')
                schema = cpu['initialization']['state_schema']
                if not recovering_h1:
                    run(arm, 'cuda_contracts', ['-m', 'pytest', '-q', 'tests/test_m0_contracts.py', '-k', 'cuda'])
                    run(arm, 'parity', ['tools/check_history_doppler_contracts.py', '--config', cfg,
                        '--device', 'cuda', '--out', str(campaign / ('gpu_' + arm + '.json'))])
                    parity = json.loads((campaign / ('gpu_' + arm + '.json')).read_text())
                    if parity['status'] != 'passed' or parity['git_revision'] != manifest['git_revision']:
                        raise ValueError('GPU parity receipt mismatch')
                    run(arm, 'smoke', ['train.py', '--config', 'configs/sw-history-' + arm + '-smoke.py'])
                    smoke_audit = audit_checkpoint(smoke / 'iter_24.pth', schema, exact_steps=24)
                    smoke_audit.update(audit_smoke_log((smoke / 'train.log').read_text()))
                    write_json(campaign / ('smoke_' + arm + '.json'), smoke_audit)
                    # Separate process: no smoke weights, optimizer or scaler reused.
                    run(arm, 'train', ['train.py', '--config', cfg])
                initialization = require_fresh_initialization(work / 'official_initialization.json', cpu['initialization'])
                checkpoint_hashes = None
                if recovering_h1:
                    from recover_history_doppler_campaign import audit_h1_checkpoints
                    final, best, checkpoint_hashes = audit_h1_checkpoints(work, schema)
                else:
                    final = audit_checkpoint(work / 'epoch_10.pth', schema, expected_epoch=10)
                if final['iterations'] != 29920:
                    raise ValueError('Expected exactly 29920 formal iterations')
                final_json = work / 'validation_epoch_10_full.json'
                final_confusions = work / 'confusions_epoch_10_full'
                best_meta = json.loads((work / 'best_future.json').read_text())
                if best_meta['epoch'] not in range(1, 11) or best_meta['samples'] != 256:
                    raise ValueError('Invalid fixed-subset checkpoint selection')
                if recovering_h1:
                    from recover_history_doppler_campaign import require_h1_parity
                    if best_meta['epoch'] != 10:
                        raise ValueError('This recovery is restricted to the observed epoch10 selections')
                    write_json(campaign / (arm + '_recovery_checkpoint_audit.json'),
                        dict(original_git_revision=recovery['original_git_revision'],
                             training_source=str(args.recovery_of), evaluation_git_revision=manifest['git_revision'],
                             initialization=initialization, final_audit=final, checkpoint_sha256=checkpoint_hashes))
                    parity_dir = campaign / (arm + '_recovery_subset_parity')
                    run(arm, 'recovery_subset_parity', ['tools/evaluate_radar_experiment.py', '--config', cfg,
                        '--checkpoint', str(work / 'epoch_10.pth'), '--samples', '256', '--output-dir', str(parity_dir)])
                    parity_audit = require_h1_parity(work / 'validation_epoch_10_subset.json', parity_dir / 'normal.json')
                    write_json(campaign / (arm + '_recovery_subset_parity.json'), parity_audit)
                    destination = campaign / (arm + '_final_full')
                    run(arm, 'final_full', ['tools/evaluate_radar_experiment.py', '--config', cfg,
                        '--checkpoint', str(work / 'epoch_10.pth'), '--samples', '0', '--output-dir', str(destination)])
                    final_json, final_confusions = destination / 'normal.json', destination / 'confusions_normal'
                else:
                    best = audit_checkpoint(work / 'best_future.pth', schema, expected_epoch=best_meta['epoch'])
                require_full_result(final_json, final_confusions)
                if best_meta['epoch'] == 10:
                    best_json, best_confusions = final_json, final_confusions
                else:
                    destination = campaign / (arm + '_best_full')
                    run(arm, 'best_full', ['tools/evaluate_radar_experiment.py', '--config', cfg,
                        '--checkpoint', str(work / 'best_future.pth'), '--samples', '0',
                        '--output-dir', str(destination)])
                    best_json, best_confusions = destination / 'normal.json', destination / 'confusions_normal'
                require_full_result(best_json, best_confusions)
                interventions = campaign / (arm + '_interventions')
                modes = ['normal', 'drop'] if arm.endswith('geometry') else ['normal', 'drop', 'zero_velocity', 'shuffle_velocity']
                run(arm, 'interventions', ['tools/evaluate_radar_experiment.py', '--config', cfg,
                    '--checkpoint', str(work / 'best_future.pth'), '--samples', '256',
                    '--modes', *modes, '--output-dir', str(interventions)])
                reference_indices = None
                for mode in modes:
                    item = json.loads((interventions / (mode + '.json')).read_text())
                    if item['samples'] != 256 or item['mode'] != mode:
                        raise ValueError('Invalid intervention receipt')
                    if reference_indices is not None and reference_indices != item['indices']:
                        raise ValueError('Intervention anchors differ')
                    reference_indices = item['indices']
                result = dict(arm=arm, gpu=args.gpu, git_revision=manifest['git_revision'],
                    best_epoch=best_meta['epoch'], official_initialization=initialization,
                    final_audit=final, best_audit=best, final_json=str(final_json), best_json=str(best_json),
                    final_confusions=str(final_confusions), best_confusions=str(best_confusions),
                    interventions=str(interventions), stage_seconds=dict(status['stage_seconds']),
                    recovery_of=str(args.recovery_of) if recovery else None,
                    training_git_revision=recovery['original_git_revision'] if recovering_h1 else manifest['git_revision'],
                    evaluation_git_revision=manifest['git_revision'], checkpoint_sha256=checkpoint_hashes,
                    stage_seconds_scope='recovery evaluation only' if recovering_h1 else 'new arm',
                    original_training_log=str(args.recovery_of / (arm + '_train.log')) if recovering_h1 else None,
                    original_gpu_telemetry=str(args.recovery_of / (arm + '_gpu.jsonl')) if recovering_h1 else None)
                write_json(campaign / (arm + '_result.json'), result)
                status['completed_arms'].append(arm)
                record(state='complete', stage='arm_complete', result=result)
                arm_status = None
                status.pop('result', None)
                # Last worker finishes the CPU-only four-way comparison once.
                if all((campaign / (a + '_result.json')).exists() for a in ARMS):
                    with (campaign / 'summary.lock').open('a') as summary_lock:
                        fcntl.flock(summary_lock, fcntl.LOCK_EX)
                        if not (campaign / 'comparison.json').exists():
                            with (campaign / 'comparison.log').open('xb') as log:
                                subprocess.run([python, 'tools/summarize_history_doppler.py', '--campaign', str(campaign)],
                                    env=dict(env, CUDA_VISIBLE_DEVICES=''), cwd=code, stdout=log,
                                    stderr=subprocess.STDOUT, check=True)
            record(state='complete', stage='worker_queue_finished', arm=None, child_pid=None)
    except BaseException as error:
        record(state='failed', error=repr(error), traceback=traceback.format_exc(),
               child_alive=bool(active_child is not None and active_child.poll() is None),
               child_pid=(active_child.pid if active_child is not None else None))
        raise


if __name__ == '__main__':
    main()
