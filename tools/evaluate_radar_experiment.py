"""Offline, strict checkpoint evaluation and causal radar counterfactuals."""
import argparse
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import models
import loaders
from finetune_hooks import evaluate_subset
from official_init import initialize_official


def perturb_metadata(_module, inputs, mode):
    inputs = list(inputs)
    metadata = []
    for original in inputs[3]:
        item = dict(original)
        points = np.array(item['radar_points'], copy=True)
        if mode == 'drop':
            points = points[:0]
        elif mode == 'zero_velocity':
            points[:, 3:5] = 0
            points[:, 7] = 0
        elif mode == 'shuffle_velocity' and len(points):
            seed = int(hashlib.sha256(str(item['sample_idx']).encode()).hexdigest()[:8], 16)
            order = np.random.RandomState(seed).permutation(len(points))
            points[:, 3:5] = points[order, 3:5]
            points[:, 7] = (points[:, 3:5] * points[:, 8:10]).sum(-1)
        item['radar_points'] = points
        metadata.append(item)
    inputs[3] = metadata
    return tuple(inputs)


def require_finite_decoder(_module, _inputs, outputs):
    for group in outputs:
        for tensor in group:
            if not torch.isfinite(tensor).all():
                raise FloatingPointError('Nonfinite raw decoder output during evaluation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--official', action='store_true')
    parser.add_argument('--samples', type=int, default=256, help='0 means full validation')
    parser.add_argument('--modes', nargs='+', default=['normal'],
                        choices=['normal', 'drop', 'zero_velocity', 'shuffle_velocity'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)
    indices = (list(range(len(dataset))) if args.samples == 0 else
               sorted(np.random.RandomState(20260918).choice(
                   len(dataset), min(args.samples, len(dataset)), replace=False).tolist()))
    module = build_model(cfg.model)
    module.init_weights()
    if args.official:
        checkpoint_meta = initialize_official(module, args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state = {k.removeprefix('module.'): v for k, v in checkpoint['state_dict'].items()}
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise FloatingPointError('Nonfinite checkpoint')
        module.load_state_dict(state, strict=True)
        checkpoint_meta = {k: checkpoint.get('meta', {}).get(k) for k in ('epoch', 'iter')}
        del checkpoint, state
    module.cuda().eval()
    wrap_fp16_model(module)
    model = MMDataParallel(module, [0])
    decoder = module.pts_bbox_head.transformer.decoder
    finite_hook = decoder.register_forward_hook(require_finite_decoder)
    for mode in args.modes:
        destination = output / (mode + '.json')
        if destination.exists():
            raise FileExistsError(destination)
        hook = (decoder.register_forward_pre_hook(
            lambda m, x, mode=mode: perturb_metadata(m, x, mode)) if mode != 'normal' else None)
        started = time.monotonic()
        try:
            metrics = evaluate_subset(model, dataset, indices, workers=args.workers,
                                      logger=logging.getLogger(),
                                      confusion_dir=output / ('confusions_' + mode))
        finally:
            if hook is not None:
                hook.remove()
        future_mean = float(np.mean([metrics[h]['Semantic mIoU']
                                     for h in ('1.0s', '2.0s', '3.0s')]))
        manifest = Path('code_manifest.json')
        revision = (json.loads(manifest.read_text())['git_revision'] if manifest.exists() else
                    subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        report = dict(config=args.config, checkpoint=str(Path(args.checkpoint).resolve()),
                      checkpoint_meta=checkpoint_meta, mode=mode, samples=len(indices),
                      indices=indices, future_mean_miou=future_mean, metrics=metrics,
                      elapsed_seconds=time.monotonic() - started,
                      git_revision=revision)
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(destination)
        logging.info('EVALUATION_COMPLETE mode=%s samples=%d future_mean_miou=%.6f seconds=%.1f',
                     mode, len(indices), future_mean, report['elapsed_seconds'])
    finite_hook.remove()


if __name__ == '__main__':
    main()
