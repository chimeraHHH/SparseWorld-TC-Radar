"""Resume integrity and validation hooks; original model/scientific code stays frozen."""
import gc
import json
import math
from pathlib import Path
import torch
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks.lr_updater import LrUpdaterHook
import finetune_hooks
from transport_extension_spool import evaluate_subset

# Both the original validation hook and evaluator CLI resolve this function.
finetune_hooks.evaluate_subset = evaluate_subset


@HOOKS.register_module()
class CheckpointConstantLrUpdaterHook(LrUpdaterHook):
    """Capture actual resumed LRs, ignoring the original cosine initial_lr."""
    def before_run(self, runner):
        if isinstance(runner.optimizer, dict):
            raise TypeError('Only the original single AdamW optimizer is supported')
        self.base_lr = [g['lr'] for g in runner.optimizer.param_groups]
        if not all(math.isfinite(lr) and lr > 0 for lr in self.base_lr):
            raise ValueError('Invalid checkpoint learning rates')

    def get_lr(self, runner, base_lr):
        return base_lr


def assert_equal_state(actual, expected):
    """Optimizer loading can move tensors; values must remain exactly equal."""
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu().to(actual.dtype))
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_equal_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_equal_state(a, e)
    else:
        assert actual == expected, (actual, expected)


@HOOKS.register_module()
class ExtensionResumeAuditHook(Hook):
    def __init__(self, checkpoint, smoke=False):
        self.checkpoint, self.smoke = checkpoint, smoke

    def before_run(self, runner):
        c = torch.load(self.checkpoint, map_location='cpu')
        assert runner.epoch == 10 and runner.iter == 29920
        actual = runner.model.module.state_dict()
        expected = {k.removeprefix('module.'): v for k, v in c['state_dict'].items()}
        assert len(actual) == len(expected) == 801
        assert_equal_state(actual, expected)
        assert_equal_state(runner.optimizer.state_dict(), c['optimizer'])
        scalers = [h.loss_scaler for h in runner._hooks if hasattr(h, 'loss_scaler')]
        assert len(scalers) == 1
        # IterBasedRunner.resume omits checkpoint meta; explicitly restore saved AMP state.
        scalers[0].load_state_dict(c['meta']['fp16']['loss_scaler'])
        runner.meta['fp16'] = c['meta']['fp16']
        assert_equal_state(scalers[0].state_dict(), c['meta']['fp16']['loss_scaler'])
        steps = {int(s['step']) for s in runner.optimizer.state.values() if 'step' in s}
        assert steps == {29915}
        rates = [g['lr'] for g in runner.optimizer.param_groups]
        self.rates = rates
        runner.meta['transport_extension'] = dict(source=self.checkpoint, initial_epoch=10,
            initial_iterations=29920, initial_optimizer_steps=29915,
            learning_rate_policy='constant_actual_checkpoint_group_lrs', smoke=self.smoke)
        receipt = dict(epoch=runner.epoch, iterations=runner.iter, optimizer_steps=29915,
            model_tensors=801, model_exact=True, optimizer_exact=True, amp_scaler_exact=True,
            group_lrs=rates, source=self.checkpoint, smoke=self.smoke)
        Path(runner.work_dir, 'resume_integrity.json').write_text(json.dumps(receipt, indent=2))
        runner.logger.info('EXTENSION_RESUME_INTEGRITY_PASSED %s', json.dumps({
            k:v for k,v in receipt.items() if k != 'group_lrs'}))
        del c, actual, expected
        gc.collect()

    def before_train_iter(self, runner):
        assert [g['lr'] for g in runner.optimizer.param_groups] == self.rates


@HOOKS.register_module()
class ExtensionValidationHook(finetune_hooks.FinetuneValidationHook):
    def before_run(self, runner):
        super().before_run(runner)
        old = Path('/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/radar_forecast_transport-reliable_seed0')
        selection = json.loads((old/'best_future.json').read_text())
        prior = json.loads((old/'validation_epoch_10_subset.json').read_text())
        assert self.indices == prior['indices'] and len(self.dataset) == 5119
        assert [self.dataset.data_infos[i]['token'] for i in self.indices] == prior['tokens']
        self.best = selection['future_mean_miou']
        selection.update(checkpoint=str(old/'best_future.pth'), origin='original_ten_epoch_selection')
        Path(runner.work_dir, 'best_future.json').write_text(json.dumps(selection, indent=2))
        runner.logger.info('EXTENSION_SELECTION_BASELINE epoch=%s score=%s', selection['epoch'], self.best)

    def after_train_epoch(self, runner):
        super().after_train_epoch(runner)
        if runner.epoch + 1 == 15:
            self._evaluate(runner, 15, list(range(len(self.dataset))), 'full')
