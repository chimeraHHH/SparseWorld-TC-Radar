"""Recovery must not repeat H1 training or bypass the original evidence gates."""
import importlib.util
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import recover_history_doppler_campaign as recovery
from run_history_doppler_experiment import claim_arm, ARMS


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def learning_log():
    joint = {key: dict(gradient_norm=1, parameter_delta=1)
             for key in ('img_backbone', 'img_neck', 'pts_bbox_head')}
    return '\n'.join(['RADAR_LEARNING microstep_grad_norm=1 parameter_delta=1\n'
                      + 'JOINT_LEARNING ' + json.dumps(joint)] * 6)


class RecoveryGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.new = self.root / 'old_code', self.root / 'new_code'
        self.original = self.root / 'analysis/original'
        self.old.mkdir(); self.new.mkdir(); self.original.mkdir(parents=True)
        self.old.joinpath('model.py').write_text('science = 1\n')
        self.new.joinpath('model.py').write_text('science = 1\n')
        put(self.old / 'code_manifest.json', dict(git_revision=recovery.ORIGINAL_REVISION,
            sha256={'model.py': recovery.sha256_file(self.old / 'model.py')}))
        put(self.original / 'submission.json', dict(code=str(self.old)))
        put(self.original / 'preflight.json', dict(state='passed', git_revision=recovery.ORIGINAL_REVISION))
        for gpu in (0, 1):
            put(self.original / ('gpu%d_worker.json' % gpu), dict(state='failed', stage='train',
                child_returncode=-9, controller_pid=90000000 + gpu, child_pid=90000002 + gpu))
        for arm in ARMS[:2]:
            for prefix in ('cpu_', 'gpu_'):
                put(self.original / (prefix + arm + '.json'), dict(status='passed', git_revision=recovery.ORIGINAL_REVISION))
            put(self.original / ('smoke_' + arm + '.json'), dict(iterations=23, optimizer_steps=24,
                optimizer_finite=True, state_schema_matches_cpu=True))
            work = self.root / ('work_dirs/history_' + arm + '_seed0')
            put(work / 'best_future.json', dict(epoch=10, samples=256))
            for file in ('epoch_10.pth', 'best_future.pth', 'official_initialization.json', 'validation_epoch_10_subset.json'):
                (work / file).write_text('evidence')
            smoke = self.root / ('work_dirs/history_' + arm + '_smoke_seed0')
            smoke.mkdir(); (smoke / 'train.log').write_text(learning_log())
        self.root_patch = patch.object(recovery, 'ROOT', self.root)
        self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        self.dead_patch = patch.object(recovery, 'require_old_processes_absent', return_value={'live_recorded_pids': []})
        self.dead_patch.start(); self.addCleanup(self.dead_patch.stop)

    def test_preserves_original_claims_and_claims_recovery_arms_exactly_once(self):
        (self.original / 'h1-velocity_claim.json').write_text('immutable evidence')
        result = recovery.validate_recovery_source(self.original, self.new)
        self.assertEqual(result['original_git_revision'], recovery.ORIGINAL_REVISION)
        fresh = self.root / 'new_campaign'; fresh.mkdir()
        self.assertEqual([claim_arm(fresh, n % 2, 'new') for n in range(5)], list(ARMS) + [None])
        self.assertEqual((self.original / 'h1-velocity_claim.json').read_text(), 'immutable evidence')

    def test_rejects_scientific_source_change(self):
        (self.new / 'model.py').write_text('science = 2\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            recovery.validate_recovery_source(self.original, self.new)

    def test_rejects_started_h8_and_finished_h1(self):
        path = self.original / 'h8-velocity_claim.json'
        path.write_text('{}')
        with self.assertRaisesRegex(FileExistsError, 'H8'):
            recovery.validate_recovery_source(self.original, self.new)
        path.unlink()
        path = self.root / 'work_dirs/history_h1-velocity_seed0/validation_epoch_10_full.json'
        path.write_text('{}')
        with self.assertRaisesRegex(FileExistsError, 'already finished'):
            recovery.validate_recovery_source(self.original, self.new)

    def test_rejects_missing_original_admission(self):
        put(self.original / 'gpu_h1-geometry.json', dict(status='failed', git_revision=recovery.ORIGINAL_REVISION))
        with self.assertRaisesRegex(ValueError, 'admission'):
            recovery.validate_recovery_source(self.original, self.new)

    def test_rejects_recorded_live_pid(self):
        self.dead_patch.stop()
        fake_proc = self.root / 'proc'; fake_proc.mkdir()
        (fake_proc / '90000000').mkdir()
        with self.assertRaisesRegex(ValueError, 'Original processes still present'):
            recovery.require_old_processes_absent(self.old, self.original, proc=fake_proc)
        self.dead_patch.start()

    def fake_protected_process(self, proc, pid, name, command):
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        uid = os.getuid()
        (directory / 'status').write_text('Name:\t' + name + '\nUid:\t' + '\t'.join([str(uid)] * 4) + '\n')
        (directory / 'cmdline').write_bytes(command.encode() + b'\0')
        return directory / 'cwd'

    def test_verified_protected_login_helpers_are_recorded(self):
        self.dead_patch.stop()
        proc = self.root / 'proc'
        username = recovery.pwd.getpwuid(os.getuid()).pw_name
        protected = [self.fake_protected_process(proc, 101, '(sd-pam)', '(sd-pam)'),
                     self.fake_protected_process(proc, 102, 'sshd', 'sshd: ' + username + '@notty')]
        real_resolve = Path.resolve
        def resolve(path, *args, **kwargs):
            if path in protected:
                raise PermissionError('nondumpable helper')
            return real_resolve(path, *args, **kwargs)
        with patch.object(Path, 'resolve', new=resolve):
            audit = recovery.require_old_processes_absent(self.old, self.original, proc=proc)
        self.assertEqual(sorted(p['pid'] for p in audit['protected_session_helpers']), [101, 102])
        self.dead_patch.start()

    def test_unrecognized_protected_python_process_fails_closed(self):
        self.dead_patch.stop()
        proc = self.root / 'proc'
        protected = self.fake_protected_process(proc, 101, 'python', 'python train.py')
        real_resolve = Path.resolve
        def resolve(path, *args, **kwargs):
            if path == protected:
                raise PermissionError('unknown process')
            return real_resolve(path, *args, **kwargs)
        with patch.object(Path, 'resolve', new=resolve):
            with self.assertRaisesRegex(PermissionError, 'Cannot exclude old-snapshot'):
                recovery.require_old_processes_absent(self.old, self.original, proc=proc)
        self.dead_patch.start()

    def test_helper_shaped_recorded_pid_is_still_rejected(self):
        self.dead_patch.stop()
        proc = self.root / 'proc'
        self.fake_protected_process(proc, 90000000, '(sd-pam)', '(sd-pam)')
        with self.assertRaisesRegex(ValueError, 'Original processes still present'):
            recovery.require_old_processes_absent(self.old, self.original, proc=proc)
        self.dead_patch.start()

    def parity_payloads(self):
        metrics = {h: dict({'Semantic mIoU': 23.0, 'Binary IoU': 40.0, 'evaluated_samples': 256},
                          **{key: 23.0 for key in recovery.CLASS_METRICS}) for h in recovery.HORIZONS}
        old = dict(samples=256, indices=list(range(256)), metrics=metrics, future_mean_miou=23.0,
                   epoch=10, scope='subset', dataset_samples=5119)
        new = dict(samples=256, indices=list(range(256)), metrics=copy.deepcopy(metrics),
                   future_mean_miou=23.0, mode='normal', checkpoint_meta={'epoch': 10, 'iter': 29920})
        return old, new

    def check_parity(self, old, new):
        previous, current = self.root / 'old.json', self.root / 'new.json'
        put(previous, old); put(current, new)
        return recovery.require_h1_parity(previous, current)

    def test_authorized_absolute_parity_boundaries_and_complete_audit(self):
        old, new = self.parity_payloads()
        new['metrics']['1.0s']['Semantic mIoU'] = 23.001
        new['metrics']['2.0s']['Binary IoU'] = 39.999
        new['metrics']['3.0s']['car_IoU'] = 23.01
        new['future_mean_miou'] = 22.999
        result = self.check_parity(old, new)
        self.assertEqual(result['status'], 'passed')
        self.assertNotIn('exact_metrics', result)
        self.assertEqual(result['policy']['rtol'], 0)
        self.assertEqual(len(result['differences']), 77)
        self.assertEqual(result['differences']['1.0s/Semantic mIoU']['signed_delta_pp'], .001)
        self.assertEqual(result['differences']['2.0s/Binary IoU']['signed_delta_pp'], -.001)
        self.assertTrue(result['exact_evaluated_samples'])

    def test_outside_each_tolerance_fails_with_full_delta_audit(self):
        for key, value in (('Semantic mIoU', 23.0010001), ('Binary IoU', 40.0010001),
                           ('car_IoU', 23.0100001), ('future_mean_miou', 23.0010001)):
            with self.subTest(key=key):
                old, new = self.parity_payloads()
                if key == 'future_mean_miou':
                    new[key] = value
                else:
                    new['metrics']['1.0s'][key] = value
                with self.assertRaisesRegex(ValueError, 'authorized parity bounds') as caught:
                    self.check_parity(old, new)
                self.assertEqual(caught.exception.parity_audit['status'], 'failed')
                self.assertEqual(len(caught.exception.parity_audit['differences']), 77)

    def test_missing_extra_or_renamed_metric_and_horizon_keys_rejected(self):
        for kind in ('missing', 'extra', 'renamed', 'horizon'):
            old, new = self.parity_payloads()
            if kind == 'horizon':
                new['metrics']['0s'] = new['metrics'].pop('0.0s')
            elif kind == 'extra':
                new['metrics']['1.0s']['unexpected_IoU'] = 23
            else:
                new['metrics']['1.0s'].pop('car_IoU')
                if kind == 'renamed':
                    new['metrics']['1.0s']['vehicle_IoU'] = 23
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'schema'):
                self.check_parity(old, new)

    def test_anchor_mode_and_counts_remain_exact(self):
        changes = (lambda new: new.update(indices=list(reversed(new['indices']))),
                   lambda new: new['indices'].__setitem__(0, 0.0),
                   lambda new: new.update(mode='no_radar'),
                   lambda new: new.update(samples=255),
                   lambda new: new['metrics']['1.0s'].update(evaluated_samples=255),
                   lambda new: new['metrics']['1.0s'].update(evaluated_samples=256.0),
                   lambda new: new.update(checkpoint_meta={'epoch': 9, 'iter': 26928}))
        for change in changes:
            old, new = self.parity_payloads()
            change(new)
            with self.assertRaises(ValueError):
                self.check_parity(old, new)

    def test_only_matching_undefined_per_class_iou_is_permitted(self):
        for missing in (None, float('nan')):
            old, new = self.parity_payloads()
            old['metrics']['1.0s']['car_IoU'] = missing
            new['metrics']['1.0s']['car_IoU'] = missing
            audit = self.check_parity(old, new)
            self.assertIsNone(audit['differences']['1.0s/car_IoU']['absolute_delta_pp'])
        for key, old_value, new_value in (('car_IoU', 23.0, float('nan')),
                ('car_IoU', float('inf'), float('inf')), ('Semantic mIoU', float('nan'), float('nan')),
                ('Binary IoU', None, None), ('car_IoU', True, True)):
            old, new = self.parity_payloads()
            old['metrics']['1.0s'][key] = old_value
            new['metrics']['1.0s'][key] = new_value
            with self.subTest(key=key, old=old_value), self.assertRaisesRegex(ValueError, 'finite metrics'):
                self.check_parity(old, new)

    def previous_recovery(self):
        previous, previous_code = self.root / 'previous', self.root / 'previous_code'
        previous.mkdir(); previous_code.mkdir()
        put(previous_code / 'code_manifest.json', dict(git_revision=recovery.PREVIOUS_RECOVERY_REVISION, sha256={}))
        put(previous / 'submission.json', dict(code=str(previous_code), recovery_of=str(self.original),
            git_revision=recovery.PREVIOUS_RECOVERY_REVISION, launcher_pid=91000000))
        put(previous / 'preflight.json', dict(state='passed', git_revision=recovery.PREVIOUS_RECOVERY_REVISION))
        for gpu in (0, 1):
            put(previous / ('gpu%d_worker.json' % gpu), dict(state='failed', stage='recovery_subset_parity',
                controller_pid=91000001 + gpu, child_pid=91000003 + gpu))
        return previous, previous_code

    def test_previous_recovery_must_be_failed_and_untouched(self):
        previous, old_code = self.previous_recovery()
        before = {p.name: p.read_bytes() for p in previous.iterdir()}
        result = recovery.require_previous_recovery_absent(previous, self.original, self.new)
        self.assertEqual(result['git_revision'], recovery.PREVIOUS_RECOVERY_REVISION)
        self.assertEqual(before, {p.name: p.read_bytes() for p in previous.iterdir()})
        with self.assertRaisesRegex(ValueError, 'source reuse'):
            recovery.require_previous_recovery_absent(previous, self.original, old_code)
        put(previous / 'gpu0_worker.json', dict(state='running', stage='recovery_subset_parity'))
        with self.assertRaisesRegex(ValueError, 'not terminal'):
            recovery.require_previous_recovery_absent(previous, self.original, self.new)

    def test_previous_recovery_started_arm_blocks_relaunch(self):
        previous, _ = self.previous_recovery()
        path = previous / 'h8-velocity_claim.json'; path.write_text('{}')
        with self.assertRaisesRegex(FileExistsError, 'claimed H8'):
            recovery.require_previous_recovery_absent(previous, self.original, self.new)
        path.unlink()
        (previous / 'h1-velocity_result.json').write_text('{}')
        with self.assertRaisesRegex(FileExistsError, 'completed an arm'):
            recovery.require_previous_recovery_absent(previous, self.original, self.new)

    def test_previous_recovery_launcher_pid_also_must_be_absent(self):
        previous, old_code = self.previous_recovery()
        self.dead_patch.stop()
        proc = self.root / 'proc'; proc.mkdir(); (proc / '91000000').mkdir()
        with self.assertRaisesRegex(ValueError, 'Original processes still present'):
            recovery.require_old_processes_absent(old_code, previous, proc=proc)
        self.dead_patch.start()

    def diagnostic(self):
        (self.new / 'finetune_hooks.py').write_text('exact evaluator\n')
        report = dict(status='complete_observations_only', checkpoint_unchanged=True,
            optimizer_steps_executed=0, source_revision='new', original_revision=recovery.ORIGINAL_REVISION,
            current_evaluator_sha256=recovery.sha256_file(self.new / 'finetune_hooks.py'),
            checkpoint_meta={'epoch': 10, 'iter': 29920}, policies={})
        for name in ('offline_manual_seed', 'training_deterministic_seed'):
            report['policies'][name] = dict(spool_arrays_exact=True,
                fixed_output_storage_comparison=dict(metrics_exact=True, all_scene_arrays_exact=True))
        return report

    def test_diagnostic_is_exact_version_read_only_and_storage_exact(self):
        report = self.diagnostic(); path = self.root / 'diagnostic.json'; put(path, report)
        result = recovery.require_exact_storage_diagnostic(path, self.new, 'new')
        self.assertTrue(result['all_scene_arrays_exact'])
        mutations = (lambda r: r.update(status='running'), lambda r: r.update(optimizer_steps_executed=1),
            lambda r: r.update(checkpoint_unchanged=False), lambda r: r.update(source_revision='previous'),
            lambda r: r.update(current_evaluator_sha256='different'),
            lambda r: r['policies']['offline_manual_seed'].update(spool_arrays_exact=False),
            lambda r: r['policies']['training_deterministic_seed']['fixed_output_storage_comparison'].update(metrics_exact=False),
            lambda r: r['policies']['offline_manual_seed']['fixed_output_storage_comparison'].update(all_scene_arrays_exact=False))
        for mutate in mutations:
            changed = copy.deepcopy(report); mutate(changed); put(path, changed)
            with self.assertRaises(ValueError):
                recovery.require_exact_storage_diagnostic(path, self.new, 'new')

    def test_host_available_memory_is_parsed_in_bytes(self):
        path = self.root / 'meminfo'
        path.write_text('MemTotal: 123 kB\nMemAvailable: 67108864 kB\n')
        self.assertEqual(recovery.host_available_bytes(path), recovery.HOST_MINIMUM_BYTES)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is audited on the H200 CPU runtime')
class CheckpointRecovery(unittest.TestCase):
    def test_saved_zip_names_can_differ_but_models_must_equal(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            schema = {'weight': [2]}
            checkpoint = dict(state_dict={'weight': torch.tensor([1., 2.])},
                optimizer={'state': {0: {'step': 29918, 'exp_avg': torch.tensor([0., 1.])}}},
                meta={'epoch': 10, 'iter': 29920})
            for name in ('epoch_10.pth', 'best_future.pth'):
                torch.save(checkpoint, work / name)
            final, best, hashes = recovery.audit_h1_checkpoints(work, schema)
            self.assertEqual(final['iterations'], 29920)
            self.assertEqual(final['optimizer_steps'], 29918)  # Recovered AMP skips are allowed.
            self.assertEqual(len(set(hashes['model_state'].values())), 1)
            checkpoint['state_dict']['weight'][0] = 2
            torch.save(checkpoint, work / 'best_future.pth')
            with self.assertRaisesRegex(ValueError, 'differs from'):
                recovery.audit_h1_checkpoints(work, schema)

    def test_rejects_partial_budget_or_nonfinite_optimizer(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            checkpoint = dict(state_dict={'weight': torch.ones(2)},
                optimizer={'state': {0: {'step': 29920, 'exp_avg': torch.ones(2)}}},
                meta={'epoch': 10, 'iter': 29919})
            for name in ('epoch_10.pth', 'best_future.pth'):
                torch.save(checkpoint, work / name)
            with self.assertRaisesRegex(ValueError, 'iteration budget'):
                recovery.audit_h1_checkpoints(work, {'weight': [2]})
            checkpoint['meta']['iter'] = 29920
            checkpoint['optimizer']['state'][0]['exp_avg'][0] = float('nan')
            torch.save(checkpoint, work / 'epoch_10.pth')
            with self.assertRaises(FloatingPointError):
                recovery.audit_h1_checkpoints(work, {'weight': [2]})


if __name__ == '__main__':
    unittest.main()
