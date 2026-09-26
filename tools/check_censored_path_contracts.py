"""Exact CPU contracts and independent official-model real-input GPU parity.

CUDA execution belongs to the campaign controller, after its capacity/lock
gate. This script never allocates a GPU when --device=cpu. The CUDA check uses
a separately constructed camera-only official reference, not a switched-off
branch of the new model. Training endpoints are absent from inference inputs.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer, MMDataParallel
from mmcv.runner import build_optimizer, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import Subset

import models
import loaders
from loaders.builder import build_dataloader
from loaders.pipelines.endpoint_segments import LoadEndpointSegments
from loaders.pipelines.radar import LoadCausalRadar
from official_init import initialize_official


COMPONENTS = ['path_encoder', 'path_head', 'mixture_head']
INDICES = [0, 1024, 2048, 4096]
ENDPOINT_CACHE = '/home/huayiming/Workspace/SparseWorld-cache/endpoint_segments_v1_20260926'


def normalized(value):
    """Config list/tuple containers are equivalent; their values are not."""
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


def require_equal(left, right, name):
    if normalized(left) != normalized(right):
        raise AssertionError('Resolved configuration differs from A: ' + name)


def configuration_contract(cfg, baseline):
    for name in ('batch_size', 'total_epochs', 'optimizer_config', 'lr_config',
                 'dataloader_options', 'load_from', 'seed', 'future_frames',
                 'fp16', 'runner', 'checkpoint_config'):
        require_equal(cfg.get(name), baseline.get(name), name)
    assert cfg.batch_size == 8 and cfg.total_epochs == 10
    assert cfg.optimizer_config.cumulative_iters == 1
    assert cfg.resume_from is None and cfg.resume_epoch_boundary is False
    assert cfg.official_finetune is True and cfg.revise_keys is None
    assert cfg.model.samplewise_loss is True
    assert list(cfg.future_frames) == [0, 2, 4, 6]
    validations = [hook for hook in cfg.custom_hooks if hook.type == 'FinetuneValidationHook']
    assert len(validations) == 1
    validation = validations[0]
    assert validation.samples == 256 and validation.full_at_end is True
    assert validation.keep_best_trained is True and validation.save_scene_confusions is True
    assert sum(hook.type == 'CensoredPathLearningAuditHook' for hook in cfg.custom_hooks) == 1
    architecture = copy.deepcopy(cfg.model)
    path = architecture.pts_bbox_head.pop('censored_path')
    require_equal(path, dict(num_modes=2, max_residual=6., temperature=.25, track_weight=.2), 'path design')
    require_equal(architecture, baseline.model, 'model except censored_path')
    optimizer = copy.deepcopy(cfg.optimizer)
    group = optimizer.paramwise_cfg.custom_keys.pop('censored_path')
    require_equal(group, dict(lr_mult=10.), 'path optimizer group')
    require_equal(optimizer, baseline.optimizer, 'optimizer except path group')

    data = copy.deepcopy(cfg.data)
    pipeline = data.train.pipeline
    stages = [stage for stage in pipeline if stage['type'] == 'LoadEndpointSegments']
    assert len(stages) == 1, 'Exactly one training endpoint stage is required'
    endpoint_stage = copy.deepcopy(stages[0])
    assert endpoint_stage.get('split', 'train') == 'train'
    assert str(Path(endpoint_stage['cache_root'])) == ENDPOINT_CACHE
    assert endpoint_stage.get('max_instances', 32) == 32
    assert endpoint_stage.get('samples_per_endpoint', 16) == 16
    assert next(i for i, stage in enumerate(pipeline) if stage['type'] == 'LoadOccFromFile') < pipeline.index(stages[0])
    data.train.pipeline = [stage for stage in pipeline if stage['type'] != 'LoadEndpointSegments']
    collectors = [stage for stage in data.train.pipeline if stage['type'] == 'Collect3D']
    assert len(collectors) == 1
    meta = list(collectors[0]['meta_keys'])
    assert meta.count('endpoint_segments') == 1, 'Endpoints must be collected only as training metadata'
    collectors[0]['meta_keys'] = [name for name in meta if name != 'endpoint_segments']
    require_equal(data, baseline.data, 'data except training endpoints')
    # Exact validation/test pipeline equality also excludes hidden labels and
    # endpoint loading from deployment. Check this explicitly for the receipt.
    for split in ('val', 'test'):
        require_equal(cfg.data[split], baseline.data[split], split + ' split')
        text = repr(normalized(cfg.data[split].pipeline))
        assert 'LoadEndpointSegments' not in text and 'endpoint_segments' not in text
        occ = [s for s in cfg.data[split].pipeline if s['type'] == 'LoadOccFromFile']
        assert len(occ) == 1 and occ[0].get('load_labels') is False
    return endpoint_stage


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def endpoint_contract(cfg, stage):
    arguments = dict(stage)
    arguments.pop('type')
    loader = LoadEndpointSegments(**arguments)
    root = Path(arguments['cache_root'])
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['schema_version'] == 'endpoint-segments-v1'
    assert manifest['occupancy_labels_cached'] is False
    assert manifest['horizons'] == [0, 2, 4, 6]
    builder_sources = [source for name, source in manifest['sources'].items()
                       if Path(name).name == 'build_endpoint_cache.py']
    assert len(builder_sources) == 1
    builder_hash = sha256(Path(__file__).with_name('build_endpoint_cache.py'))
    assert builder_sources[0]['sha256'] == builder_hash, 'Cache builder differs from this exact snapshot'
    assert Path(manifest['data_root']).resolve() == Path(cfg.data.train.data_root).resolve()
    assert Path(manifest['occ_root']).resolve() == Path(cfg.data.train.occ_root).resolve()
    anchor_sets, target_sets, sizes = {}, {}, {}
    for split, expected in (('train', 23930), ('val', 5119)):
        path = Path(cfg.data[split].ann_file)
        with path.open('rb') as stream:
            infos = pickle.load(stream)['infos']
        tokens = {info['token'] for info in infos}
        assert len(infos) == len(tokens) == expected, split
        source = manifest['sources'][str(path.resolve())]
        assert source['sha256'] == sha256(path), 'Endpoint source annotation changed'
        anchors_path = root / split / 'anchors.json'
        assert sha256(anchors_path) == manifest['splits'][split]['anchors_sha256']
        anchors = json.loads(anchors_path.read_text())
        assert set(anchors) == tokens == set(manifest['splits'][split]['anchor_tokens'])
        assert manifest['splits'][split]['anchors'] == expected
        assert all(len(values) == 4 and values[0] == key for key, values in anchors.items())
        anchor_sets[split] = tokens
        target_sets[split] = {token for values in anchors.values() for token in values}
        sizes[split] = expected
    assert not anchor_sets['train'] & anchor_sets['val']
    assert not target_sets['train'] & target_sets['val']
    assert set(loader.anchors) == anchor_sets['train']
    return dict(manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
                samples=sizes, train_val_anchor_tokens_disjoint=True,
                train_val_target_tokens_disjoint=True, occupancy_labels_cached=False,
                builder_sha256=builder_hash, builder_matches_snapshot=True,
                frame_payloads_rehashed=False, training_only=True)


def optimizer_contract(module, config):
    optimizer = build_optimizer(module, config)
    assert not optimizer.state, 'The optimizer must be fresh'
    groups = {}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            assert id(parameter) not in groups, 'Duplicate optimizer parameter'
            groups[id(parameter)] = group
    counts = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        assert id(parameter) in groups, 'Missing trainable parameter: ' + name
        if name.startswith('pts_bbox_head.censored_path.'):
            component = name[len('pts_bbox_head.censored_path.'):].split('.')[0]
            assert component in COMPONENTS, 'Unaudited path component: ' + name
            kind, expected = component, 2e-4
        elif 'radar_fusion' in name:
            kind, expected = 'radar', 2e-4
        elif 'img_backbone' in name or 'sampling_offset' in name:
            kind, expected = 'backbone_or_sampling_offset', 2e-6
        else:
            kind, expected = 'pretrained_world', 2e-5
        assert abs(groups[id(parameter)]['lr'] - expected) < 1e-12, name
        counts[kind] = counts.get(kind, 0) + parameter.numel()
    assert all(counts.get(name, 0) > 0 for name in COMPONENTS)
    return counts


def reject_target_inputs(value):
    """Reject labels/endpoint caches at the actual evaluation forward boundary."""
    if isinstance(value, DataContainer):
        reject_target_inputs(value.data)
    elif isinstance(value, dict):
        for name, child in value.items():
            assert name not in ('endpoint_segments', 'voxel_semantics', 'mask_camera',
                                'point_valid', 'observed', 'instance_tokens'), name
            assert not name.startswith('gt_'), name
            reject_target_inputs(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_target_inputs(child)


def compare_tensor(left, right, label):
    assert left.shape == right.shape and left.dtype == right.dtype, label
    assert torch.isfinite(left).all() and torch.isfinite(right).all(), label
    assert torch.equal(left, right), (label, float((left.float() - right.float()).abs().max()))


def compare_voxels(left, right):
    """Coordinates define identity; CUDA voxel enumeration order is irrelevant."""
    assert len(left) == len(right) == 4
    sizes = []
    for horizon, (a, b) in enumerate(zip(left, right)):
        assert set(a) == set(b) == {'sem_pred', 'occ_loc'}
        ordered = []
        for result in (a, b):
            locations = np.asarray(result['occ_loc'])
            labels = np.asarray(result['sem_pred'])
            assert locations.ndim == 2 and locations.shape[1] == 3
            assert labels.shape == (len(locations),)
            order = np.lexsort((locations[:, 2], locations[:, 1], locations[:, 0]))
            ordered.append((locations[order], labels[order]))
        assert np.array_equal(ordered[0][0], ordered[1][0]), ('voxel coordinates', horizon)
        assert np.array_equal(ordered[0][1], ordered[1][1]), ('voxel labels', horizon)
        sizes.append(len(ordered[0][0]))
    return sizes


def gpu_contract(module, reference, cfg):
    assert torch.cuda.is_available(), 'CUDA parity was requested without a visible GPU'
    assert torch.cuda.device_count() == 1, 'Controller must expose exactly its locked GPU'
    dataset = build_dataset(copy.deepcopy(cfg.data.val))
    assert len(dataset) == 5119
    modules = {'new': module, 'official': reference}
    wrappers, captures, handles = {}, {}, []

    def hook(name):
        def capture(_module, _inputs, output):
            raw = [output['init_points']] + output['all_cls_scores'] + output['all_refine_pts']
            assert len(raw) == 13
            record = dict(raw=[value.detach().cpu().clone() for value in raw])
            if name == 'new':
                paths = output['censored_paths']
                record['paths'] = {key: paths[key].detach().cpu().clone()
                                   for key in ('mode_points', 'mode_scores', 'mode_logits', 'mode_probability')}
            else:
                assert 'censored_paths' not in output
            captures.setdefault(name, []).append(record)
        return capture

    path_calls = []
    def check_path_arguments(_module, inputs):
        assert len(inputs) == 5, 'Path inference must receive features/base outputs/poses/times only'
        features, points, scores, poses, times = inputs
        assert features.ndim == 3 and points.ndim == scores.ndim == 4
        assert len(poses) == len(times) == 4
        path_calls.append(True)

    expected = ['query_features', 'base_points', 'base_scores', 'fut2cur', 'fut_list']
    assert list(inspect.signature(module.pts_bbox_head.censored_path.forward).parameters) == expected
    for name, instance in modules.items():
        instance.cuda().eval()
        wrap_fp16_model(instance)
        instance.simple_test = instance.simple_test_offline
        wrappers[name] = MMDataParallel(instance, [0])
        handles.append(instance.pts_bbox_head.register_forward_hook(hook(name)))
    handles.append(module.pts_bbox_head.censored_path.register_forward_pre_hook(check_path_arguments))
    dataloader = build_dataloader(Subset(dataset, INDICES), samples_per_gpu=1,
                                 workers_per_gpu=2, dist=False, shuffle=False, seed=0)
    parity = []
    try:
        with torch.no_grad():
            for index, data in zip(INDICES, dataloader):
                assert set(data) == {'img', 'img_metas', 'fut2cur', 'fut_list'}
                reject_target_inputs(data)
                captures.clear()
                path_calls.clear()
                new_voxels = wrappers['new'](return_loss=False, rescale=True, **copy.deepcopy(data))
                old_voxels = wrappers['official'](return_loss=False, rescale=True, **copy.deepcopy(data))
                assert len(captures['new']) == len(captures['official']) == len(path_calls) == 1
                new, old = captures['new'][0], captures['official'][0]
                for tensor_index, (a, b) in enumerate(zip(new['raw'], old['raw'])):
                    compare_tensor(a, b, 'raw tensor %d sample %d' % (tensor_index, index))
                paths = new['paths']
                assert paths['mode_points'].shape[0] == paths['mode_scores'].shape[0] == 2
                for mode in range(2):
                    compare_tensor(paths['mode_points'][mode], old['raw'][-1], 'mode points %d' % mode)
                    compare_tensor(paths['mode_scores'][mode], old['raw'][6], 'mode scores %d' % mode)
                assert torch.equal(paths['mode_logits'], torch.zeros(1, 2))
                assert torch.equal(paths['mode_probability'], torch.full((1, 2), .5))
                voxel_sizes = compare_voxels(new_voxels, old_voxels)
                parity.append(dict(index=index, tensors=13, modes=2, max_abs_difference=0.,
                                   selected_voxels_equal=True, voxels_per_horizon=voxel_sizes,
                                   selected_mode=0, no_target_input=True, path_forward_calls=1))
                print('CENSORED_PATH_INITIALIZATION_PARITY', parity[-1], flush=True)
    finally:
        for handle in handles:
            handle.remove()
    assert [row['index'] for row in parity] == INDICES
    return parity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.device == 'cpu':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU contract must hide CUDA devices'
        assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = Config.fromfile(args.config)
    baseline = Config.fromfile('configs/sw-radar-forecast-transport.py')
    stage = configuration_contract(cfg, baseline)
    manifest = json.loads(Path('code_manifest.json').read_text())
    module = build_model(copy.deepcopy(cfg.model))
    module.init_weights()
    initialization = initialize_official(module, cfg.load_from)
    assert initialization['loaded_tensors'] == 669
    assert initialization['new_radar_tensors'] == 102
    assert initialization['new_path_tensors'] > 0
    assert len(initialization['zero_residual_outputs']) == 6
    assert initialization['all_camera_tensors_exact'] is True
    assert initialization['optimizer_restored'] is False and initialization['epoch_reset_to'] == 0
    state = module.state_dict()
    assert all(torch.isfinite(tensor).all() for tensor in state.values())
    initialization['state_schema'] = {name: list(value.shape) for name, value in state.items()}
    new_path_names = [name for name in state if name.startswith('pts_bbox_head.censored_path.')]
    assert len(new_path_names) == initialization['new_path_tensors']
    assert {name.split('.')[2] for name in new_path_names} == set(COMPONENTS)

    # A separately built reference removes all new branches before construction.
    reference_cfg = copy.deepcopy(baseline.model)
    reference_cfg.pts_bbox_head.transformer.radar_cfg = None
    reference_cfg.pts_bbox_head.censored_path = None
    reference = build_model(reference_cfg)
    reference.init_weights()
    reference_initialization = initialize_official(reference, cfg.load_from)
    assert reference_initialization['new_radar_tensors'] == reference_initialization['new_path_tensors'] == 0
    assert len(reference.state_dict()) == 669
    for name, tensor in reference.state_dict().items():
        assert torch.equal(state[name], tensor), 'Original tensor differs: ' + name
    initialization['independent_reference_tensors_exact'] = 669
    report = dict(config=args.config, device=args.device, initialization=initialization,
                  reference_initialization=reference_initialization,
                  git_revision=manifest['git_revision'], worldline_components=COMPONENTS,
                  official_reference_independent=True, new_parameters_finite=True,
                  budget=dict(batch_size=8, epochs=10, seed=0, fresh_optimizer=True),
                  validation_has_no_endpoints=True)
    if args.device == 'cpu':
        report['trainable_parameters'] = optimizer_contract(module, cfg.optimizer)
        report['fresh_optimizer'] = True
        radar_stage = dict(next(s for s in cfg.data.train.pipeline if s.type == 'LoadCausalRadar'))
        radar_stage.pop('type')
        cache = LoadCausalRadar(**radar_stage)
        assert len(cache.cache.entries) == 23930
        report['radar_cache_samples'] = len(cache.cache.entries)
        report['endpoint_cache'] = endpoint_contract(cfg, stage)
    else:
        report['real_input_parity'] = gpu_contract(module, reference, cfg)
    report['status'] = 'passed'
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    # The saved receipt contains the full schema; terminal output stays bounded.
    display = copy.deepcopy(report)
    display['initialization']['state_schema'] = {'tensor_count': len(initialization['state_schema'])}
    print(json.dumps(display, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
