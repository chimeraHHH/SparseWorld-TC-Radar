"""One GPU, one arm: parity -> smoke -> training -> selected-model evaluation.

The parent holds a per-GPU lock for the entire queue. Every failure stops the
queue and is recorded; there is no automatic restart or hyperparameter fallback.
"""
import argparse
import datetime
import fcntl
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=['transport', 'balanced'])
    parser.add_argument('--gpu', required=True, type=int, choices=[0, 1])
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    os.chdir(code)
    root = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
    campaign = root/'analysis/forecast_campaign_20260920'
    campaign.mkdir(parents=True, exist_ok=True)
    status_path = campaign/(args.arm + '_status.json')
    if status_path.exists():
        raise FileExistsError('A controller receipt already exists: ' + str(status_path))
    manifest = json.loads((code/'code_manifest.json').read_text())
    uuid = ['GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe',
            'GPU-000b6236-3632-a001-9667-1f02cbb61c8b'][args.gpu]
    status = dict(arm=args.arm, gpu=args.gpu, gpu_uuid=uuid, controller_pid=os.getpid(),
                  git_revision=manifest['git_revision'], code=str(code), completed_stages=[])
    def record(**values):
        status.update(values, at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False))
        temporary.replace(status_path)
    record(state='preparing')
    try:
        for name, expected in manifest['sha256'].items():
            assert hashlib.sha256((code/name).read_bytes()).hexdigest() == expected, name
        audit = json.loads((campaign/('cpu_' + args.arm + '.json')).read_text())
        assert audit['status'] == 'passed' and audit['git_revision'] == manifest['git_revision']
        common = json.loads((campaign/'preflight.json').read_text())
        assert common['state'] == 'passed' and common['git_revision'] == manifest['git_revision']
        record(state='waiting_for_gpu_lock')
        lock = open(root/f'forecast_gpu{args.gpu}.lock', 'w')
        fcntl.flock(lock, fcntl.LOCK_EX)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                        '--format=csv,noheader'], text=True)
        assert uuid not in apps, 'Selected GPU has another process; stopping without interference'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', PYTHONPATH=str(code),
                   PYTHONUNBUFFERED='1', TMPDIR='/tmp',
                   TORCH_EXTENSIONS_DIR=str(root/'cache/torch_extensions'))
        python = str(root/'envs/hym_sparseworld/bin/python')
        config = 'configs/sw-radar-forecast-' + args.arm + '.py'
        smoke_config = 'configs/sw-radar-forecast-' + args.arm + '-smoke.py'
        work = root/('work_dirs/radar_forecast_' + args.arm + '_seed0')
        smoke = root/('work_dirs/radar_forecast_' + args.arm + '_smoke_seed0')
        for directory in (work, smoke):
            assert not directory.exists() or not any(directory.iterdir()), directory
        telemetry = campaign/(args.arm + '_gpu.jsonl')
        def run(stage, arguments):
            log_path = campaign/(args.arm + '_' + stage + '.log')
            command = [python] + arguments
            with log_path.open('wb') as log:
                process = subprocess.Popen(command, cwd=code, env=env, stdout=log,
                                           stderr=subprocess.STDOUT)
                record(state='running', stage=stage, child_pid=process.pid,
                       command=command, log=str(log_path))
                while True:
                    try:
                        returncode = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        snapshot = subprocess.check_output([
                            'nvidia-smi', '-i', uuid,
                            '--query-gpu=utilization.gpu,memory.used,power.draw',
                            '--format=csv,noheader,nounits'], text=True).strip()
                        with telemetry.open('a') as stream:
                            stream.write(json.dumps(dict(at_utc=datetime.datetime.now(
                                datetime.timezone.utc).isoformat(), stage=stage, gpu=snapshot))+'\n')
                        record(gpu_snapshot=snapshot)
                if returncode:
                    raise RuntimeError('%s exited %d; see %s' % (stage, returncode, log_path))
                status['completed_stages'].append(stage)
                record(child_pid=None, stage=stage)

        run('gpu_contracts', ['-m', 'pytest', '-q', 'tests/test_m0_contracts.py', '-k', 'cuda'])
        run('gpu_parity', ['tools/check_forecast_contracts.py', '--config', config,
                          '--device', 'cuda', '--out', str(campaign/('gpu_' + args.arm + '.json'))])
        run('smoke', ['train.py', '--config', smoke_config])

        def audit_checkpoint(path, expected_epoch=None, exact_steps=None):
            import torch
            torch.set_num_threads(2)
            checkpoint = torch.load(path, map_location='cpu')
            state = checkpoint['state_dict']
            assert len(state) == 771
            assert all(torch.isfinite(v).all() for v in state.values())
            optimizer = checkpoint['optimizer']['state']
            steps = [int(s['step']) for s in optimizer.values() if 'step' in s]
            assert steps and min(steps) == max(steps)
            assert all(torch.isfinite(v).all() for s in optimizer.values()
                       for v in s.values() if torch.is_tensor(v))
            if exact_steps is not None:
                assert min(steps) == exact_steps, steps[:4]
            if expected_epoch is not None:
                assert checkpoint['meta']['epoch'] == expected_epoch
            result = dict(path=str(path), epoch=checkpoint['meta']['epoch'],
                          iterations=checkpoint['meta']['iter'], optimizer_steps=min(steps),
                          finite_model_tensors=len(state), optimizer_finite=True)
            del checkpoint, state, optimizer
            gc.collect()
            return result

        smoke_audit = audit_checkpoint(smoke/'iter_24.pth', exact_steps=24)
        radar_updates, joint_updates = [], []
        for line in (smoke/'train.log').read_text().splitlines():
            if 'JOINT_LEARNING' in line or 'RADAR_LEARNING' in line:
                assert 'nan' not in line.lower() and 'inf' not in line.lower().replace('[info]', '')
            if 'RADAR_LEARNING' in line:
                values = re.search(r'microstep_grad_norm=([\d.e+-]+) parameter_delta=([\d.e+-]+)', line)
                assert values and all(float(v) > 0 for v in values.groups())
                radar_updates.append(line)
            if 'JOINT_LEARNING' in line:
                payload = json.loads(line[line.index('{'):])
                assert all(v['gradient_norm'] > 0 and v['parameter_delta'] > 0 for v in payload.values())
                joint_updates.append(payload)
        assert len(radar_updates) == len(joint_updates) == 6
        smoke_audit.update(radar_audit_windows=6, pretrained_audit_windows=6,
                           all_audited_gradient_and_update_norms_positive=True)
        (campaign/('smoke_' + args.arm + '.json')).write_text(json.dumps(smoke_audit, indent=2))
        # train.py resets to the official checkpoint; smoke weights are never used.
        run('train', ['train.py', '--config', config])
        final_audit = audit_checkpoint(work/'epoch_10.pth', expected_epoch=10)
        assert final_audit['iterations'] == 29920
        final_json = work/'validation_epoch_10_full.json'
        final = json.loads(final_json.read_text())
        assert final['samples'] == 5119
        best_meta = json.loads((work/'best_future.json').read_text())
        best_audit = audit_checkpoint(work/'best_future.pth', expected_epoch=best_meta['epoch'])
        best_full_dir = campaign/(args.arm + '_best_full')
        if best_meta['epoch'] == 10:
            best_json = final_json
            best_confusions = work/'confusions_epoch_10_full'
        else:
            run('best_full', ['tools/evaluate_radar_experiment.py', '--config', config,
                '--checkpoint', str(work/'best_future.pth'), '--samples', '0',
                '--output-dir', str(best_full_dir)])
            best_json = best_full_dir/'normal.json'
            best_confusions = best_full_dir/'confusions_normal'
        run('counterfactual', ['tools/evaluate_radar_experiment.py', '--config', config,
            '--checkpoint', str(work/'best_future.pth'), '--samples', '256',
            '--modes', 'normal', 'drop', 'zero_velocity',
            '--output-dir', str(campaign/(args.arm + '_counterfactual'))])
        # Re-evaluate fixed references with scene histograms for paired intervals.
        baseline_dir = campaign/('reference_official' if args.arm == 'transport' else 'reference_m0_final')
        reference_args = ['tools/evaluate_radar_experiment.py', '--config',
                          'configs/sw-radar-m0-official-ft.py', '--samples', '0',
                          '--output-dir', str(baseline_dir)]
        if args.arm == 'transport':
            reference_args += ['--official', '--checkpoint', str(root/'checkpoints/sw-tc-small.pth')]
        else:
            reference_args += ['--checkpoint', str(root/'work_dirs/m0_official_ft_seed0/epoch_10.pth')]
        run('reference_full', reference_args)
        result = dict(arm=args.arm, gpu=args.gpu, git_revision=manifest['git_revision'],
                      best_epoch=best_meta['epoch'], final_audit=final_audit, best_audit=best_audit,
                      final_json=str(final_json), best_json=str(best_json),
                      best_confusions=str(best_confusions),
                      final_confusions=str(work/'confusions_epoch_10_full'))
        (campaign/(args.arm + '_result.json')).write_text(json.dumps(result, indent=2))
        record(state='complete', stage='all_training_and_evaluation_complete', result=result)
        # Whichever arm finishes last writes the shared scientific comparison.
        subprocess.run([python, 'tools/finish_forecast_campaign.py', '--campaign', str(campaign)],
                       cwd=code, env=dict(env, CUDA_VISIBLE_DEVICES=''), check=True)
    except BaseException as error:
        record(state='failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
