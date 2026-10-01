"""CPU resume contracts and real frozen-output storage parity for continuation."""
import argparse
import copy
import gc
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
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
from transport_extension_spool import _DiskPredictionStore
from transport_extension_hooks import assert_equal_state

ANCHOR_HASH = '663a7b4bbb6b06721af04324277dddf52889c39fc87ecbe7331349d999cfd8bd'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = Config.fromfile('configs/sw-radar-transport-reliable-extend20.py')
    original = Config.fromfile('configs/sw-radar-forecast-transport-reliable.py')
    for key in ('model','data','batch_size','optimizer','optimizer_config','dataloader_options','seed'):
        assert cfg[key] == original[key], key
    assert cfg.total_epochs == 20 and cfg.batch_size == 8
    assert cfg.optimizer_config.cumulative_iters == 1 and cfg.resume_epoch_boundary is False
    assert sha(cfg.resume_from) == ANCHOR_HASH
    c = torch.load(cfg.resume_from, map_location='cpu')
    assert c['meta']['epoch'] == 10 and c['meta']['iter'] == 29920
    assert len(c['state_dict']) == 801
    assert all(torch.isfinite(v).all() for v in c['state_dict'].values())
    assert {int(s['step']) for s in c['optimizer']['state'].values() if 'step' in s} == {29915}
    assert all(torch.isfinite(v).all() for s in c['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
    module = build_model(copy.deepcopy(cfg.model)); module.init_weights()
    module.load_state_dict({k.removeprefix('module.'):v for k,v in c['state_dict'].items()}, strict=True)
    report = dict(status='passed', device=args.device, checkpoint_sha256=ANCHOR_HASH,
        epoch=10, iterations=29920, optimizer_steps=29915, model_tensors=801)
    if args.device == 'cpu':
        assert not torch.cuda.is_available()
        optimizer = build_optimizer(module, cfg.optimizer)
        optimizer.load_state_dict(c['optimizer'])
        assert_equal_state(optimizer.state_dict(), c['optimizer'])
        report.update(optimizer_exact=True, learning_rates=sorted({g['lr'] for g in optimizer.param_groups}),
            fp16_scaler=c['meta']['fp16']['loss_scaler'])
        stage = dict(next(s for s in cfg.data.train.pipeline if s.type == 'LoadCausalRadar'))
        stage.pop('type'); loader = LoadCausalRadar(**stage)
        assert len(loader.cache.entries) == 23930
        for token, entry in loader.cache.entries.items():
            loader.cache.load(dict(sample_idx=token, timestamp=entry['reference_timestamp_us']/1e6))
        nusc = loaders.pipelines.loading.get_nusc(cfg.dataset_root)
        tokens = sorted(loader.cache.entries)
        for index in np.random.RandomState(20261001).choice(len(tokens), 128, replace=False):
            token = tokens[index]; sample = nusc.get('sample', token)
            sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            meta = dict(sample_idx=token, timestamp=sd['timestamp']/1e6)
            cached = loader(copy.deepcopy(meta))['radar_points']
            online = loader._load_online(copy.deepcopy(meta))['radar_points']
            assert np.array_equal(cached, online), token
        train = build_dataset(cfg.data.train); val = build_dataset(cfg.data.val)
        assert len(train) == 23930 and len(val) == 5119
        report.update(cache_validated=23930, online_rechecks=128,
            train_samples=len(train), val_samples=len(val))
    else:
        del c; gc.collect()
        dataset = build_dataset(cfg.data.val)
        prior = json.loads(Path(cfg.resume_from).with_name('validation_epoch_10_subset.json').read_text())
        indices = prior['indices'][:16]
        assert [dataset.data_infos[i]['token'] for i in indices] == prior['tokens'][:16]
        module.cuda().eval(); wrap_fp16_model(module)
        module.simple_test = module.simple_test_offline
        model = MMDataParallel(module, [0])
        predictions = []
        with tempfile.TemporaryDirectory(prefix='extension-storage-', dir=__import__('os').environ['SPARSEWORLD_EVAL_TMPDIR']) as tmp:
            root = Path(tmp); store = _DiskPredictionStore(root)
            batches = build_dataloader(Subset(dataset, indices), samples_per_gpu=1, workers_per_gpu=2,
                dist=False, shuffle=False, seed=0)
            with torch.no_grad():
                for data in batches:
                    result = model(return_loss=False, rescale=True, **data)
                    predictions.append(result); store.append(result)
            for j, horizon in enumerate(dataset.future_frames):
                list_path, disk_path = root/f'list{j}.npz', root/f'disk{j}.npz'
                left = dataset.evaluate([x[j] for x in predictions], horizon,
                    sample_indices=indices, confusion_path=str(list_path))
                right = dataset.evaluate(store.horizon(j), horizon,
                    sample_indices=indices, confusion_path=str(disk_path))
                assert set(left) == set(right)
                for key in left: np.testing.assert_equal(left[key], right[key])
                with np.load(list_path) as a, np.load(disk_path) as b:
                    assert a.files == b.files
                    for key in a.files: np.testing.assert_array_equal(a[key], b[key])
            report.update(real_anchors=16, frozen_storage_arrays_exact=True,
                all_metrics_exact=True, scene_confusions_exact=True, optimizer_steps_executed=0)
        assert sha(cfg.resume_from) == ANCHOR_HASH
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__': main()
