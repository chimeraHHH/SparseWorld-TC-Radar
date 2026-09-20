"""Check exact official initialization, optimizer groups and real-input parity."""
import argparse
import copy
import json
from pathlib import Path

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import Subset

import models
import loaders
from loaders.builder import build_dataloader
from loaders.pipelines.radar import LoadCausalRadar
from official_init import initialize_official


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = Config.fromfile(args.config)
    baseline = Config.fromfile('configs/sw-radar-m0-official-ft.py')
    for key in ('batch_size', 'total_epochs', 'optimizer', 'optimizer_config',
                'lr_config', 'data', 'dataloader_options', 'load_from', 'seed'):
        assert cfg[key] == baseline[key], key
    assert cfg.batch_size == 8 and cfg.total_epochs == 10 and cfg.resume_from is None
    assert cfg.model.samplewise_loss
    module = build_model(cfg.model)
    module.init_weights()
    initialization = initialize_official(module, cfg.load_from)
    assert initialization['loaded_tensors'] == 669
    camera = cfg.model.pts_bbox_head.transformer.radar_cfg is None
    assert initialization['new_radar_tensors'] == (0 if camera else 38)
    assert len(initialization['zero_residual_outputs']) == (0 if camera else 6)
    report = dict(config=args.config, device=args.device, initialization=initialization,
                  git_revision=json.loads(Path('code_manifest.json').read_text())['git_revision'])
    if args.device == 'cpu':
        assert not torch.cuda.is_available(), 'CPU audit must hide CUDA devices'
        optimizer = build_optimizer(module, cfg.optimizer)
        assert not optimizer.state
        groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
        counts = {}
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if 'radar_fusion' in name:
                kind, expected = 'radar', 2e-4
            elif 'img_backbone' in name or 'sampling_offset' in name:
                kind, expected = 'backbone_or_sampling_offset', 2e-6
            else:
                kind, expected = 'pretrained_world', 2e-5
            assert abs(groups[id(parameter)]['lr'] - expected) < 1e-12, name
            counts[kind] = counts.get(kind, 0) + parameter.numel()
        stage = dict(next(s for s in cfg.data.train.pipeline if s.type == 'LoadCausalRadar'))
        stage.pop('type')
        cache = LoadCausalRadar(**stage)
        assert len(cache.cache.entries) == 23930
        report.update(trainable_parameters=counts, fresh_optimizer=True,
                      radar_cache_samples=len(cache.cache.entries))
    else:
        dataset = build_dataset(cfg.data.val)
        module.cuda().eval()
        wrap_fp16_model(module)
        module.simple_test = module.simple_test_offline
        model = MMDataParallel(module, [0])
        decoder = module.pts_bbox_head.transformer.decoder
        layers = list(decoder.decoder_layers)
        radars = [layer.radar_fusion for layer in layers]
        captured = []
        def capture(_module, _inputs, output):
            values = [output['init_points']] + output['all_cls_scores'] + output['all_refine_pts']
            captured.append([value.detach().cpu().clone() for value in values])
        handle = module.pts_bbox_head.register_forward_hook(capture)
        indices = [0, 1024, 2048, 4096]
        dataloader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1,
                                     workers_per_gpu=2, dist=False, shuffle=False, seed=0)
        parity = []
        with torch.no_grad():
            for index, data in zip(indices, dataloader):
                captured.clear()
                model(return_loss=False, rescale=True, **copy.deepcopy(data))
                try:
                    decoder.radar_enabled = False
                    for layer in layers:
                        layer.radar_fusion = None
                    model(return_loss=False, rescale=True, **copy.deepcopy(data))
                finally:
                    decoder.radar_enabled = not camera
                    for layer, radar in zip(layers, radars):
                        layer.radar_fusion = radar
                left, right = captured
                assert all(torch.isfinite(x).all() for x in left + right)
                maximum = max(float((a-b).abs().max()) for a, b in zip(left, right))
                assert maximum == 0., (index, maximum)
                parity.append(dict(index=index, tensors=len(left), max_abs_difference=maximum))
                print('FORECAST_INITIALIZATION_PARITY', parity[-1], flush=True)
        handle.remove()
        report['real_input_parity'] = parity
    report['status'] = 'passed'
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
