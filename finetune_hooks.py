"""Joint-update audits and repeatable validation for official-weight fine-tuning."""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.datasets import build_dataset
from torch.utils.data import Subset

from loaders.builder import build_dataloader


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_subset(model, dataset, indices, workers=4, logger=None, confusion_dir=None):
    """Keep full dataset history while selecting anchors; restore training RNG."""
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all())
    module = model.module
    training = module.training
    original_test = module.simple_test
    fp16 = [(m, m.fp16_enabled) for m in module.modules() if hasattr(m, 'fp16_enabled')]
    try:
        model.eval()
        # Use all 48 input images directly. Online inference changes FP16 flags
        # and caches features, which must not leak across optimizer updates.
        module.simple_test = module.simple_test_offline
        loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1,
                                  workers_per_gpu=workers, dist=False,
                                  shuffle=False, seed=0, pin_memory=True)
        results = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                prediction = model(return_loss=False, rescale=True, **data)
                if len(prediction) != len(dataset.future_frames):
                    raise ValueError('Expected one result per forecast horizon')
                results.extend(prediction)
                if logger and ((i + 1) % 64 == 0 or i + 1 == len(indices)):
                    logger.info('FINETUNE_VALIDATION_PROGRESS %d/%d', i + 1, len(indices))
        if confusion_dir is not None:
            Path(confusion_dir).mkdir(parents=True, exist_ok=True)
        metrics = {str(h * .5) + 's': dataset.evaluate(
            results[j::len(dataset.future_frames)], h, sample_indices=indices,
            confusion_path=(str(Path(confusion_dir) / ('%.1fs.npz' % (h * .5)))
                            if confusion_dir is not None else None))
            for j, h in enumerate(dataset.future_frames)}
        return metrics
    finally:
        module.simple_test = original_test
        for m, enabled in fp16:
            m.fp16_enabled = enabled
        model.train(training)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])


@HOOKS.register_module()
class FinetuneValidationHook(Hook):
    def __init__(self, config, samples=256, full_at_end=True, keep_best_trained=False,
                 save_scene_confusions=False):
        self.config = config
        self.samples = samples
        self.full_at_end = full_at_end
        self.keep_best_trained = keep_best_trained
        self.save_scene_confusions = save_scene_confusions

    def before_run(self, runner):
        cfg = Config.fromfile(self.config)
        self.dataset = build_dataset(cfg.data.val)
        self.indices = sorted(np.random.RandomState(20260918).choice(
            len(self.dataset), min(self.samples, len(self.dataset)), replace=False).tolist())
        self.best = -float('inf')
        if runner.epoch == 0:
            self._evaluate(runner, 0, self.indices, 'subset')
            if self.keep_best_trained:
                # Keep the best trained checkpoint even for a negative experiment.
                self.best = -float('inf')

    def _evaluate(self, runner, epoch, indices, scope):
        started = time.monotonic()
        confusion_dir = (Path(runner.work_dir) / ('confusions_epoch_%02d_%s' % (epoch, scope))
                         if self.save_scene_confusions else None)
        metrics = evaluate_subset(runner.model, self.dataset, indices, logger=runner.logger,
                                  confusion_dir=confusion_dir)
        score = float(np.mean([metrics[h]['Semantic mIoU'] for h in ('1.0s', '2.0s', '3.0s')]))
        if not math.isfinite(score):
            raise FloatingPointError('Nonfinite validation future mIoU')
        report = dict(epoch=epoch, scope=scope, samples=len(indices),
                      dataset_samples=len(self.dataset), indices=indices,
                      tokens=[self.dataset.data_infos[i]['token'] for i in indices],
                      future_mean_miou=score, metrics=metrics,
                      elapsed_seconds=time.monotonic() - started)
        path = Path(runner.work_dir) / ('validation_epoch_%02d_%s.json' % (epoch, scope))
        path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
        runner.logger.info('FINETUNE_VALIDATION epoch=%d scope=%s samples=%d future_mean_miou=%.6f seconds=%.1f',
                           epoch, scope, len(indices), score, report['elapsed_seconds'])
        if scope == 'subset' and score > self.best:
            self.best = score
            if epoch > 0:
                runner.save_checkpoint(runner.work_dir, filename_tmpl='best_future.pth', create_symlink=False)
                Path(runner.work_dir, 'best_future.json').write_text(json.dumps(
                    dict(epoch=epoch, future_mean_miou=score, samples=len(indices)), indent=2))
        return report

    def after_train_epoch(self, runner):
        self._evaluate(runner, runner.epoch + 1, self.indices, 'subset')
        if self.full_at_end and runner.epoch + 1 == runner.max_epochs:
            self._evaluate(runner, runner.epoch + 1, list(range(len(self.dataset))), 'full')


@HOOKS.register_module()
class JointLearningAuditHook(Hook):
    """Track representative pretrained tensors in addition to the radar audit."""
    def __init__(self, interval=10):
        self.interval = interval

    def before_run(self, runner):
        named = list(runner.model.named_parameters())
        self.params = {}
        for group in ('img_backbone', 'img_neck', 'pts_bbox_head'):
            name, param = next((n, p) for n, p in named if group in n and
                               'radar_fusion' not in n and p.requires_grad and p.ndim > 1)
            self.params[group] = (name, param)
        self.previous = {g: p.detach().clone() for g, (_, p) in self.params.items()}
        self.grad = {}
        self.scale = 1.
        self.handles = []
        for group, (_, param) in self.params.items():
            def record(gradient, group=group):
                self.grad[group] = (gradient.detach().float() / self.scale).square().sum()
            self.handles.append(param.register_hook(record))

    def before_train_iter(self, runner):
        self.grad.clear()
        for hook in runner.hooks:
            scaler = getattr(hook, 'loss_scaler', None)
            if scaler is not None and hasattr(scaler, 'get_scale'):
                self.scale = scaler.get_scale()
                break

    def after_train_iter(self, runner):
        if (runner.iter + 1) % self.interval:
            return
        report = {}
        for group, (name, param) in self.params.items():
            delta = float((param.detach().float() - self.previous[group].float()).norm())
            grad = math.sqrt(float(self.grad.get(group, 0.)))
            if not math.isfinite(delta) or not math.isfinite(grad):
                raise FloatingPointError('Nonfinite joint update: ' + group)
            report[group] = dict(parameter=name, gradient_norm=grad, parameter_delta=delta)
            self.previous[group].copy_(param.detach())
        runner.logger.info('JOINT_LEARNING iter=%d %s', runner.iter + 1, json.dumps(report))

    def after_run(self, runner):
        for handle in self.handles:
            handle.remove()
