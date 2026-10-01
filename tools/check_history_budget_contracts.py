"""Admission contracts for the four history x processed-radar-velocity arms.

H1 preserves the released eight-slot predictor but supplies only current camera
evidence. Its reference independently repeats current features and calibrations;
this never asserts equivalence to the official eight-history input prediction.
Heavy training imports are confined to execution so the contracts are testable
without an MMCV installation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from types import MethodType


INDICES = [0, 1024, 2048, 4096]
CACHE_ROOT = '/home/huayiming/Workspace/SparseWorld-cache/radar_single_sweep_v1_20260927'
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-5
VIEW_KEYS = ('filename', 'img_timestamp', 'lidar2img', 'lidar2cam',
             'intrinsics', 'extrinsics', 'img_shape', 'ori_shape', 'pad_shape',
             'img_timestamp_us','visual_source_kind','visual_source_id','visual_time_delta_s')


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(child) for child in value]
    return value


def require_equal(left, right, label):
    if normalized(left) != normalized(right):
        raise ValueError('Uncontrolled configuration difference: ' + label)


EXTRA_META = {'reference_timestamp_us','img_timestamp_us','visual_source_kind','visual_source_id','visual_time_delta_s','visual_history_choices'}


def configuration_contract(cfg, baseline, config_path=None):
    from tools.check_history_doppler_contracts import configuration_contract as original_contract
    adapted=copy.deepcopy(cfg);history=adapted['model']['visual_history_frames']
    if history not in (2,8): raise ValueError('Only the four authorized H2/H8 arms')
    adapted['model']['visual_history_frames']=8
    for split in ('train','val','test'):
        stages=adapted['data'][split]['pipeline']
        for i,stage in enumerate(stages):
            if stage['type']=='LoadBudgetedVisualHistory':
                assert stage['frames']==history
                assert set(stage)==({'type','frames'} if split=='train' else {'type','frames','test_mode'})
                if split!='train': assert stage['test_mode'] is True
                stages[i]=dict(next(x for x in baseline['data'][split]['pipeline'] if x['type']=='LoadMultiViewImageFromMultiSweeps'))
        def restore_meta(stage):
            if 'meta_keys' in stage:
                assert EXTRA_META <= set(stage['meta_keys'])
                stage['meta_keys']=tuple(k for k in stage['meta_keys'] if k not in EXTRA_META)
            for child in stage.get('transforms',[]):restore_meta(child)
        for stage in stages:restore_meta(stage)
    contract=original_contract(adapted,baseline,config_path)
    contract.update(visual_history_frames=history,physical_images_per_anchor=6*history,
        layout='B,T,G shared by features coordinates and scale weights',
        short_history_mapping='current slot0; nearest retained real history in slots1..7',
        timestamp_precision='raw int64 microseconds subtracted before float32')
    return contract


def repeat_reference_metadata(metadata):
    """Independent audit implementation; never import the production helper."""
    for item in metadata:
        for key in VIEW_KEYS:
            value = item.get(key)
            if value is not None and hasattr(value, '__len__') and len(value) == 6:
                item[key] = [copy.deepcopy(value[camera])
                             for _ in range(8) for camera in range(6)]
        for key in ('filename', 'img_timestamp', 'lidar2img'):
            if key not in item or len(item[key]) != 48:
                raise ValueError('Independent H1 reference lacks repeated ' + key)


def tensor_parity(left, right, label):
    import torch
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('Parity shape/dtype mismatch: ' + label)
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise FloatingPointError('Nonfinite parity tensor: ' + label)
    error = float((left.float() - right.float()).abs().max()) if left.numel() else 0.
    torch.testing.assert_close(left, right, atol=PARITY_ATOL, rtol=PARITY_RTOL,
                               msg='Initialization parity: ' + label)
    return dict(exact=bool(torch.equal(left, right)), max_abs_difference=error,
                dtype=str(left.dtype), shape=list(left.shape))


def paired_radar_views(processed, zero):
    """Zeroing velocity must not change geometry, order, age, or provenance."""
    import numpy as np
    left, right = processed['radar_points'], zero['radar_points']
    if (left.dtype != np.float32 or right.dtype != np.float32
            or left.ndim != 2 or left.shape[1] != 10 or left.shape != right.shape
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError('Radar treatment shape/dtype/finite contract failed')
    geometry = [0, 1, 2, 5, 6, 8, 9]
    if not np.array_equal(left[:, geometry], right[:, geometry]):
        raise ValueError('Velocity treatment changed radar geometry or membership')
    if np.any(right[:, [3, 4, 7]] != 0):
        raise ValueError('Geometry treatment retained velocity information')
    if (processed['radar_reference_timestamp_us'] != zero['radar_reference_timestamp_us']
            or processed['radar_sweep_provenance'] != zero['radar_sweep_provenance']
            or not np.array_equal(processed['radar_point_sensor_indices'],
                                  zero['radar_point_sensor_indices'])):
        raise ValueError('Radar treatment changed sensor/time provenance')
    return dict(points=len(left), dtype=str(left.dtype),
                identical_geometry_membership_and_provenance=True,
                zero_velocity_columns=[3, 4, 7],
                processed_nonzero_velocity_points=int(np.any(left[:, [3, 4, 7]] != 0, axis=1).sum()),
                per_sensor_provenance=processed['radar_sweep_provenance'])


def optimizer_contract(module, config):
    from mmcv.runner import build_optimizer
    optimizer = build_optimizer(module, config)
    if optimizer.state:
        raise ValueError('Optimizer must start without state')
    groups, counts = {}, {}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if id(parameter) in groups:
                raise ValueError('Duplicate optimizer parameter')
            groups[id(parameter)] = group
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        kind, expected = ('radar', 2e-4) if 'radar_fusion' in name else (
            ('backbone_or_sampling_offset', 2e-6)
            if 'img_backbone' in name or 'sampling_offset' in name
            else ('pretrained_world', 2e-5))
        if id(parameter) not in groups or abs(groups[id(parameter)]['lr'] - expected) > 1e-12:
            raise ValueError('Wrong/missing optimizer parameter group: ' + name)
        counts[kind] = counts.get(kind, 0) + parameter.numel()
    if set(counts) != {'radar', 'backbone_or_sampling_offset', 'pretrained_world'}:
        raise ValueError('Missing trainable component')
    return counts


def gpu_contract(module, reference, cfg, contract):
    import numpy as np
    import torch
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import wrap_fp16_model
    from mmdet3d.datasets import build_dataset
    from torch.utils.data import Subset
    from loaders.builder import build_dataloader
    from tools.check_censored_path_contracts import compare_voxels, reject_target_inputs
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('GPU contract requires exactly the controller-locked GPU')
    history = contract['visual_history_frames']
    expected_images = history * 6
    dataset = build_dataset(copy.deepcopy(cfg.data.val))
    if len(dataset) != 5119:
        raise ValueError('Validation anchors must remain 5119')
    captures, image_counts, input_counts, metadata_reports = {}, {}, {}, {}
    handles, wrappers = [], {}
    # Capture the unmodified None-history method, then independently construct
    # the current-evidence eight-slot reference after actual six-image encoding.
    reference_extract = reference.extract_feat
    if history == 2:
        def independent_reference(self, img, img_metas):
            if img.ndim!=5 or img.shape[1]!=12: raise ValueError('H2 physical input must be12 views')
            features=reference_extract(img,img_metas)
            indices=list(range(6))+[camera+6 for _ in range(7) for camera in range(6)]
            for item in img_metas:
                for key in VIEW_KEYS:
                    value=item.get(key)
                    if value is not None and hasattr(value,'__len__') and len(value)==12:
                        item[key]=[copy.deepcopy(value[i]) for i in indices]
            return [torch.cat([f[:,:6]]+[f[:,6:12] for _ in range(7)],dim=1) for f in features]
        reference.extract_feat=MethodType(independent_reference,reference)

    for name, instance in (('new', module), ('camera_reference', reference)):
        instance.cuda().eval()
        wrap_fp16_model(instance)
        instance.simple_test = instance.simple_test_offline
        original_extract = instance.extract_feat
        def observed_extract(self, img, img_metas, name=name, original=original_extract):
            input_counts.setdefault(name, []).append(int(img.shape[1]))
            if img.shape[1] != expected_images:
                raise ValueError('Pipeline supplied undeclared physical image count')
            return original(img, img_metas)
        instance.extract_feat = MethodType(observed_extract, instance)
        def backbone_input(_module, inputs, name=name):
            image_counts.setdefault(name, []).append(int(inputs[0].shape[0]))
        def head_input(_module, inputs, name=name):
            features, metadata = inputs[:2]
            if any(feature.shape[1] != 48 for feature in features):
                raise ValueError('Predictor must retain 48 feature slots')
            item = metadata[0]
            for key in ('filename', 'img_timestamp', 'lidar2img'):
                if len(item[key]) != 48:
                    raise ValueError('Projection metadata does not match eight slots')
                if history == 2:
                    for slot in range(2, 8):
                        if not np.array_equal(np.asarray(item[key][6:12]),
                                              np.asarray(item[key][slot * 6:(slot + 1) * 6])):
                            raise ValueError('H2 replication differs: ' + key)
            points = np.asarray(item['radar_points'])
            if (points.dtype != np.float32 or points.ndim != 2
                    or points.shape[1] != 10 or not np.isfinite(points).all()):
                raise ValueError('Invalid radar features at model boundary')
            if contract['velocity_mode'] == 'zero' and np.any(points[:, [3, 4, 7]] != 0):
                raise ValueError('Geometry arm leaked a velocity channel')
            metadata_reports[name] = dict(radar_points=len(points),
                unique_image_files=len(set(item['filename'])), feature_slots=48,
                radar_dtype=str(points.dtype),
                radar_age_min=float(points[:, 6].min()) if len(points) else None,
                radar_age_max=float(points[:, 6].max()) if len(points) else None)
        def capture(_module, _inputs, output, name=name):
            values = [output['init_points']] + output['all_cls_scores'] + output['all_refine_pts']
            if len(values) != 13:
                raise ValueError('Unexpected decoder structure')
            captures.setdefault(name, []).append([value.detach().cpu().clone() for value in values])
        handles.extend([instance.img_backbone.register_forward_pre_hook(backbone_input),
                        instance.pts_bbox_head.register_forward_pre_hook(head_input),
                        instance.pts_bbox_head.register_forward_hook(capture)])
        wrappers[name] = MMDataParallel(instance, [0])
    dataloader = build_dataloader(Subset(dataset, INDICES), samples_per_gpu=1,
        workers_per_gpu=2, dist=False, shuffle=False, seed=0)
    results = []
    try:
        with torch.no_grad():
            for index, data in zip(INDICES, dataloader):
                if set(data) != {'img', 'img_metas', 'fut2cur', 'fut_list'}:
                    raise ValueError('Unexpected inference inputs')
                reject_target_inputs(data)
                captures.clear(); image_counts.clear(); input_counts.clear(); metadata_reports.clear()
                left = wrappers['new'](return_loss=False, rescale=True, **copy.deepcopy(data))
                right = wrappers['camera_reference'](return_loss=False, rescale=True, **copy.deepcopy(data))
                for name in wrappers:
                    if (image_counts.get(name) != [expected_images]
                            or input_counts.get(name) != [expected_images]
                            or len(captures.get(name, [])) != 1):
                        raise ValueError('Actual image encoding count differs from treatment')
                tensors = [tensor_parity(a, b, 'sample %d tensor %d' % (index, i))
                           for i, (a, b) in enumerate(zip(captures['new'][0], captures['camera_reference'][0]))]
                sizes = compare_voxels(left, right)
                row = dict(index=index, tensors=tensors,
                    max_abs_difference=max(value['max_abs_difference'] for value in tensors),
                    all_raw_tensors_exact=all(value['exact'] for value in tensors),
                    voxels_equal=True, voxels_per_horizon=sizes,
                    actual_backbone_images=image_counts.copy(), metadata=metadata_reports.copy(),
                    no_target_input=True)
                results.append(row)
                print('HISTORY_DOPPLER_INITIALIZATION_PARITY', index,
                      row['max_abs_difference'], row['actual_backbone_images'], flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if [result['index'] for result in results] != INDICES:
        raise ValueError('Incomplete four-real-anchor parity audit')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import torch
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet.datasets.builder import PIPELINES
    import models  # noqa: F401
    import loaders  # noqa: F401
    from official_init import initialize_official
    if args.device == 'cpu' and (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available()):
        raise ValueError('CPU contract must hide all GPUs')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = Config.fromfile(args.config)
    baseline = Config.fromfile('configs/sw-radar-forecast-transport.py')
    contract = configuration_contract(cfg, baseline, args.config)
    module = build_model(copy.deepcopy(cfg.model))
    module.init_weights()
    initialization = initialize_official(module, cfg.load_from)
    if not (initialization['loaded_tensors'] == 669 and initialization['new_radar_tensors'] == 102
            and initialization['new_path_tensors'] == 0 and len(initialization['zero_residual_outputs']) == 6
            and initialization['all_camera_tensors_exact'] and not initialization['optimizer_restored']):
        raise ValueError('Official tensor/schema/residual contract failed')
    state = module.state_dict()
    if not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise FloatingPointError('Nonfinite initialized model tensor')
    initialization['state_schema'] = {key: list(value.shape) for key, value in state.items()}
    reference_cfg = copy.deepcopy(baseline.model)
    reference_cfg['visual_history_frames'] = None
    reference_cfg['pts_bbox_head']['transformer']['radar_cfg'] = None
    reference = build_model(reference_cfg)
    reference.init_weights()
    reference_initialization = initialize_official(reference, cfg.load_from)
    if len(reference.state_dict()) != 669:
        raise ValueError('Independent camera reference includes unexpected parameters')
    for key, value in reference.state_dict().items():
        if not torch.equal(state[key], value):
            raise ValueError('Released tensor differs from independent reference: ' + key)
    manifest = json.loads(Path('code_manifest.json').read_text())
    report = dict(config=args.config, device=args.device, git_revision=manifest['git_revision'],
        initialization=initialization, reference_initialization=reference_initialization,
        input_contract=contract, independent_camera_reference=True,
        reference_input=('current plus one actual historical group encoded; independent feature/calibration replication'
                         if contract['visual_history_frames'] == 2 else 'matched eight real/categorized timesteps'),
        parity_atol=PARITY_ATOL, parity_rtol=PARITY_RTOL,
        equivalence_claim='same corrected sampler and transformed evidence only; no old-layout baseline comparison',
        budget=dict(batch_size=8, epochs=10, seed=0, cumulative_iters=1, fresh_optimizer=True))
    if args.device == 'cpu':
        report['trainable_parameters'] = optimizer_contract(module, cfg.optimizer)
        report['fresh_optimizer'] = True
        train = build_dataset(copy.deepcopy(cfg.data.train))
        val = build_dataset(copy.deepcopy(cfg.data.val))
        if (len(train), len(val)) != (23930, 5119):
            raise ValueError('Anchor budget changed')
        train_tokens = {item['token'] for item in train.data_infos}
        val_tokens = {item['token'] for item in val.data_infos}
        if len(train_tokens) != 23930 or len(val_tokens) != 5119 or train_tokens & val_tokens:
            raise ValueError('Duplicate or overlapping train/validation anchors')
        loader = PIPELINES.build(contract['radar_stages']['train'])
        cache = loader.cache
        if cache is None or set(cache.entries) != train_tokens | val_tokens:
            raise ValueError('Single-sweep cache must cover exactly the fixed train/validation anchors')
        report['cache'] = dict(root=CACHE_ROOT, entries=len(cache.entries),
            protocol_sha256=cache.protocol_sha256, train_samples=23930, val_samples=5119,
            anchors_disjoint=True, same_cache_for_all_treatments=True)
        stage = copy.deepcopy(contract['radar_stages']['train'])
        stage['velocity_mode'] = 'processed'
        processed_loader = PIPELINES.build(stage)
        stage['velocity_mode'] = 'zero'
        zero_loader = PIPELINES.build(stage)
        if processed_loader.cache.protocol_sha256 != zero_loader.cache.protocol_sha256:
            raise ValueError('Velocity arms use different cache protocols')
        report['paired_radar_views'] = []
        for index in INDICES:
            info = val.data_infos[index]
            inputs = dict(sample_idx=info['token'], timestamp=info['timestamp'] / 1e6)
            pair = paired_radar_views(processed_loader(copy.deepcopy(inputs)),
                                      zero_loader(copy.deepcopy(inputs)))
            report['paired_radar_views'].append(dict(index=index, token=info['token'], **pair))
    else:
        report['real_input_parity'] = gpu_contract(module, reference, cfg, contract)
    report['status'] = 'passed'
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    display = copy.deepcopy(report)
    display['initialization']['state_schema'] = {'tensor_count': len(initialization['state_schema'])}
    display.pop('real_input_parity', None)
    print(json.dumps(display, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
