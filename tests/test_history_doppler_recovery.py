"""Recovery must not repeat H1 training or bypass the original evidence gates."""
import importlib.util
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

    def test_exact_subset_parity_and_rejection_of_anchor_or_metric_changes(self):
        previous, current = self.root / 'old.json', self.root / 'new.json'
        payload = dict(samples=256, indices=list(range(256)), metrics={'1.0s': {'Semantic mIoU': 23.0}},
                       future_mean_miou=23.0, mode='normal')
        put(previous, payload); put(current, payload)
        self.assertTrue(recovery.require_h1_parity(previous, current)['exact_metrics'])
        payload['metrics']['1.0s']['Semantic mIoU'] = 23.000001
        put(current, payload)
        with self.assertRaisesRegex(ValueError, 'metrics'):
            recovery.require_h1_parity(previous, current)

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
