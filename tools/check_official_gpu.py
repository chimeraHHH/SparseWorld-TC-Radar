"""Real-input zero-residual parity and validation round-trip on GPU0."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import Subset
import models
import loaders
from loaders.builder import build_dataloader
from official_init import initialize_official
from finetune_hooks import evaluate_subset, json_safe

torch.set_num_threads(2)
torch.manual_seed(0)
cfg = Config.fromfile('configs/sw-radar-m0-official-ft.py')
dataset = build_dataset(cfg.data.val)
module = build_model(cfg.model)
module.init_weights()
initialization = initialize_official(module, cfg.load_from)
module.cuda().eval()
wrap_fp16_model(module)
module.simple_test = module.simple_test_offline
model = MMDataParallel(module, [0])
indices = [0, 1024, 2048, 4096]
loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1,
                          workers_per_gpu=2, dist=False, shuffle=False, seed=0)
captured = []
def capture(_module, _inputs, outputs):
    tensors = {}
    def walk(value, prefix=''):
        if torch.is_tensor(value):
            tensors[prefix] = value.detach().cpu().clone()
        elif isinstance(value, dict):
            for key, item in value.items(): walk(item, prefix + '/' + key)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value): walk(item, prefix + '/' + str(i))
    walk(outputs)
    captured.append(tensors)
handle = module.pts_bbox_head.register_forward_hook(capture)
decoder = module.pts_bbox_head.transformer.decoder
layers = list(decoder.decoder_layers)
radar_modules = [layer.radar_fusion for layer in layers]
parity = []
with torch.no_grad():
    for index, data in zip(indices, loader):
        captured.clear()
        model(return_loss=False, rescale=True, **copy.deepcopy(data))
        try:
            decoder.radar_enabled = False
            for layer in layers: layer.radar_fusion = None
            model(return_loss=False, rescale=True, **copy.deepcopy(data))
        finally:
            decoder.radar_enabled = True
            for layer, radar in zip(layers, radar_modules): layer.radar_fusion = radar
        left, right = captured
        assert left.keys() == right.keys()
        assert all(torch.isfinite(v).all() for v in left.values())
        maximum = max(float((left[k] - right[k]).abs().max()) for k in left)
        assert maximum == 0., maximum
        parity.append(dict(index=index, raw_head_tensors=len(left), max_abs_difference=maximum))
        print('OFFICIAL_CAMERA_PARITY', parity[-1], flush=True)
handle.remove()
module.train()
enabled = [(m, m.fp16_enabled) for m in module.modules() if hasattr(m, 'fp16_enabled')]
rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
metrics = evaluate_subset(model, dataset, indices, workers=2)
assert module.training and all(m.fp16_enabled == old for m, old in enabled)
assert random.getstate() == rng[0]
assert np.array_equal(np.random.get_state()[1], rng[1][1])
assert torch.equal(torch.get_rng_state(), rng[2])
assert torch.equal(torch.cuda.get_rng_state(), rng[3])
assert all(v['evaluated_samples'] == len(indices) for v in metrics.values())
report = dict(status='passed', initialization=initialization, parity=parity,
              rng_and_training_state_restored=True, diagnostic_only_metrics=metrics)
Path('official_gpu_audit.json').write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
print('OFFICIAL_GPU_AUDIT_PASSED', flush=True)
