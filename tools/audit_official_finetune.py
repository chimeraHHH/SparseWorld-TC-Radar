"""CPU-only coverage, optimizer, cache and configuration audit."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from mmcv import Config
from mmcv.runner import build_optimizer
from mmdet3d.models import build_model
import models
import loaders
from official_init import initialize_official
from loaders.pipelines.radar import LoadCausalRadar

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
args = parser.parse_args()
torch.set_num_threads(2)
assert not torch.cuda.is_available()
cfg = Config.fromfile('configs/sw-radar-m0-official-ft.py')
assert cfg.resume_from is None and cfg.total_epochs == 10
assert cfg.batch_size == 8 and cfg.optimizer_config.cumulative_iters == 1
torch.manual_seed(cfg.seed)
model = build_model(cfg.model)
model.init_weights()
report = initialize_official(model, cfg.load_from)
assert report['loaded_tensors'] == 669 and report['source_epoch'] == 70
assert len(report['zero_residual_outputs']) == 6
optimizer = build_optimizer(model, cfg.optimizer)
groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
counts = Counter()
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if 'radar_fusion' in name:
        category, expected = 'radar', 2e-4
    elif 'img_backbone' in name or 'sampling_offset' in name:
        category, expected = 'backbone_or_sampling_offset', 2e-6
    else:
        category, expected = 'pretrained_world_model', 2e-5
    assert abs(groups[id(param)]['lr'] - expected) < 1e-12, name
    counts[category] += param.numel()
assert all(counts[g] > 0 for g in ('radar', 'backbone_or_sampling_offset', 'pretrained_world_model'))
stage = dict(next(s for s in cfg.data.train.pipeline if s.type == 'LoadCausalRadar'))
stage.pop('type')
cache = LoadCausalRadar(**stage)
assert len(cache.cache.entries) == 23930
report.update(status='passed', trainable_numel_by_group=dict(counts),
              total_trainable_numel=sum(counts.values()),
              optimizer_state_initially_empty=not bool(optimizer.state),
              batch_size=cfg.batch_size, total_epochs=cfg.total_epochs,
              cache_samples=len(cache.cache.entries))
Path(args.out).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
