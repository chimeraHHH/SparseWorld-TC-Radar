#!/usr/bin/env python3
"""Export at most 32 frozen A anchors without changing its prediction pipeline.

Run with the original b0c98 snapshot's Python environment. GPU capacity admission
and the existing per-card lock must be held by the caller. This script neither
starts training nor acquires/releases a training controller's lock. All artifacts
go into a new --output-dir; the immutable --code directory is read only.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


REVISION = 'b0c98b3e7768d263c43c6e1ed73299b065e2fb6a'
CONFIG = 'configs/sw-radar-forecast-transport.py'
FUTURE_FRAMES = [0, 2, 4, 6]
SCHEMA = 'frozen-a-diagnostics-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_selection(selection):
    indices, tokens = selection.get('indices'), selection.get('tokens')
    if not isinstance(indices, list) or not isinstance(tokens, list):
        raise ValueError('Selection requires indices and tokens lists')
    if not 1 <= len(indices) <= 32 or len(tokens) != len(indices):
        raise ValueError('Selection must contain 1..32 paired indices/tokens')
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError('Dataset indices must be nonnegative integers')
    if len(set(indices)) != len(indices) or len(set(tokens)) != len(tokens):
        raise ValueError('Selection indices and tokens must be unique')
    if any(not isinstance(token, str) or len(token) != 32 or
           any(char not in '0123456789abcdef' for char in token) for token in tokens):
        raise ValueError('Expected nuScenes 32-character hexadecimal sample tokens')
    return indices, tokens


def inspect_snapshot(code):
    manifest_path = code / 'code_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('git_revision') != REVISION:
        raise ValueError('This exporter only supports the original immutable A revision')
    entries = manifest.get('sha256', {})
    if not entries or CONFIG not in entries:
        raise ValueError('Snapshot manifest is missing source hashes or A config')
    for name, expected in entries.items():
        path = (code / name).resolve()
        if code not in path.parents or sha256(path) != expected:
            raise ValueError('Snapshot hash/path mismatch: ' + name)
    return {'git_revision': REVISION, 'code_manifest_sha256': sha256(manifest_path),
            'verified_source_files': len(entries)}


def unpack_metadata(value):
    """Copy only the small metadata tree, never mutate MMDataParallel inputs."""
    import torch
    from mmcv.parallel import DataContainer
    if isinstance(value, DataContainer):
        return unpack_metadata(value.data)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: unpack_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [unpack_metadata(item) for item in value]
    return value


def leaves(value):
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in leaves(child)]
    return [value]


def batch_metadata(data, expected_token):
    import numpy as np
    metadata = leaves(unpack_metadata(data['img_metas']))
    if len(metadata) != 1 or metadata[0].get('sample_idx') != expected_token:
        raise ValueError('Loaded batch token/order differs from requested selection')
    matrices = np.concatenate([
        np.asarray(item).reshape(-1, 4, 4)
        for item in leaves(unpack_metadata(data['fut2cur']))], axis=0)
    frames = np.concatenate([
        np.asarray(item).reshape(-1)
        for item in leaves(unpack_metadata(data['fut_list']))]).astype(np.int64)
    if matrices.shape != (4, 4, 4) or not np.isfinite(matrices).all():
        raise ValueError('Expected four finite input future-to-current ego matrices')
    if frames.tolist() != FUTURE_FRAMES:
        raise ValueError('Loaded batch horizons differ from original A protocol')
    return matrices.astype(np.float32), frames


class TargetMetadata:
    """Read timestamps/poses only; never load future occupancy or box targets."""
    def __init__(self, dataset, endpoint_cache):
        self.dataset = dataset
        self.root = endpoint_cache
        self.frame_hashes = {}
        if self.root is not None:
            manifest_path = self.root / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get('schema_version') != 'endpoint-segments-v1' or
                    manifest.get('horizons') != FUTURE_FRAMES):
                raise ValueError('Unsupported endpoint metadata manifest')
            anchors_path = self.root / 'val' / 'anchors.json'
            if sha256(anchors_path) != manifest['splits']['val']['anchors_sha256']:
                raise ValueError('Validation endpoint index checksum mismatch')
            self.anchors = json.loads(anchors_path.read_text())
            self.provenance = dict(source='endpoint_metadata_only', root=str(self.root),
                                   manifest_sha256=sha256(manifest_path),
                                   anchors_sha256=sha256(anchors_path))
        else:
            from loaders.pipelines.loading import get_nusc
            self.nusc = get_nusc(dataset.data_root)
            self.provenance = dict(source='original_nuscenes_metadata',
                                   data_root=str(dataset.data_root))

    def get(self, token, input_transforms):
        import numpy as np
        if self.root is not None:
            tokens = self.anchors[token]
            rows = []
            for target in tokens:
                path = self.root / 'val' / 'frames' / (target + '.npz')
                with np.load(path, allow_pickle=False) as frame:
                    rows.append({key: frame[key] for key in (
                        'timestamp_us', 'sample_timestamp_us', 'ego2global_rotation',
                        'ego2global_translation', 'scene_name')})
                self.frame_hashes[target] = sha256(path)
            if len(tokens) != 4 or tokens[0] != token:
                raise ValueError('Endpoint metadata target order mismatch')
            transforms = []
            R0, t0 = rows[0]['ego2global_rotation'], rows[0]['ego2global_translation']
            for row in rows:
                matrix = np.eye(4)
                matrix[:3, :3] = R0.T @ row['ego2global_rotation']
                matrix[:3, 3] = R0.T @ (row['ego2global_translation'] - t0)
                transforms.append(matrix)
            lidar_times = [int(row['timestamp_us']) for row in rows]
            sample_times = [int(row['sample_timestamp_us']) for row in rows]
            scenes = [str(row['scene_name']) for row in rows]
        else:
            from loaders.pipelines.loading import T_fut2cur
            chain = [self.nusc.get('sample', token)]
            for _ in range(max(FUTURE_FRAMES)):
                if not chain[-1]['next']:
                    raise ValueError('Incomplete future target chain')
                chain.append(self.nusc.get('sample', chain[-1]['next']))
            targets = [chain[frame] for frame in FUTURE_FRAMES]
            if len({target['scene_token'] for target in targets}) != 1:
                raise ValueError('Future target chain crosses scenes')
            tokens = [target['token'] for target in targets]
            lidar = [self.nusc.get('sample_data', target['data']['LIDAR_TOP'])
                     for target in targets]
            poses = [self.nusc.get('ego_pose', item['ego_pose_token']) for item in lidar]
            transforms = [T_fut2cur(poses[0], pose) for pose in poses]
            lidar_times = [int(item['timestamp']) for item in lidar]
            sample_times = [int(target['timestamp']) for target in targets]
            name = self.nusc.get('scene', targets[0]['scene_token'])['name']
            scenes = [name] * 4
        if len(set(scenes)) != 1 or not np.allclose(
                input_transforms, np.asarray(transforms), atol=5e-4, rtol=1e-5):
            raise ValueError('Diagnostic metadata scene/poses disagree with actual model inputs')
        lidar_times = np.asarray(lidar_times, dtype=np.int64)
        if np.any(np.diff(lidar_times) <= 0):
            raise ValueError('Future timestamps must be strictly increasing')
        return dict(future_tokens=tokens, scene_name=scenes[0],
                    lidar_timestamps_us=lidar_times,
                    sample_timestamps_us=np.asarray(sample_times, dtype=np.int64),
                    actual_seconds=(lidar_times - lidar_times[0]) / 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', required=True, help='Immutable original A snapshot')
    parser.add_argument('--checkpoint', required=True, help='Original A best_future.pth')
    parser.add_argument('--selection-json', required=True)
    parser.add_argument('--output-dir', required=True, help='Must not already exist')
    parser.add_argument('--endpoint-cache', help='Optional annotation-only endpoint cache')
    parser.add_argument('--raw', action='store_true', help='Also save unmodified last-layer points/logits')
    args = parser.parse_args()
    code = Path(args.code).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    selection_path = Path(args.selection_json).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    endpoint_cache = Path(args.endpoint_cache).expanduser().resolve() if args.endpoint_cache else None
    exporter_path = Path(__file__).resolve()
    if output.exists():
        raise FileExistsError('Refusing an existing output directory: ' + str(output))
    if code == output or code in output.parents:
        raise ValueError('Diagnostic artifacts must be outside the immutable snapshot')
    if checkpoint_path.name != 'best_future.pth':
        raise ValueError('This diagnostic is declared for A best_future.pth')
    indices, tokens = validate_selection(json.loads(selection_path.read_text()))
    source = inspect_snapshot(code)
    # Prevent Python imports from adding __pycache__ to the immutable snapshot.
    sys.dont_write_bytecode = True
    os.chdir(code)
    sys.path.insert(0, str(code))
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
    if Path(models.__file__).resolve().parent != code / 'models':
        raise RuntimeError('Models were imported from a different source tree')
    if Path(loaders.__file__).resolve().parent != code / 'loaders':
        raise RuntimeError('Loaders were imported from a different source tree')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Caller must expose exactly one admitted GPU')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = Config.fromfile(str(code / CONFIG))
    if list(cfg.future_frames) != FUTURE_FRAMES:
        raise ValueError('Config forecast horizons differ from original A')
    dataset = build_dataset(cfg.data.val)
    for index, token in zip(indices, tokens):
        if index >= len(dataset) or dataset.data_infos[index]['token'] != token:
            raise ValueError('Selected validation dataset index/token mismatch')
    metadata_reader = TargetMetadata(dataset, endpoint_cache)
    manifest = dict(schema=SCHEMA, state='preparing', created_at_utc=utc_now(),
        code=str(code), **source, config=CONFIG, config_sha256=sha256(code / CONFIG),
        checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
        selection_json=str(selection_path), selection_sha256=sha256(selection_path),
        indices=indices, tokens=tokens, future_frames=FUTURE_FRAMES,
        dataset_samples=len(dataset), batch_size=1, workers=2, shuffle=False,
        raw=bool(args.raw), samples=[], target_metadata=metadata_reader.provenance,
        exporter_sha256=sha256(exporter_path),
        coordinate_contract=dict(
            occ_loc='xyz voxel indices in each target future ego grid',
            raw_normalized_points='original decoder normalized coordinates in the same future ego grid',
            fut2cur='column-vector transform from each target future ego to anchor ego',
            horizon_suffix='0..3 indices corresponding to future_frames, not frame offsets'),
        class_names=list(cfg.get('occ_names', [])),
        protocol='Original offline FP16 inference; unchanged filtering, voxelization and padding; no training',
        limitations=['No predicted instance identity is inferred from query indices.',
                     'Each horizon output is a semantic occupancy union, not separate actor trajectories.'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'samples').mkdir()
    write_json(output / 'manifest.json', manifest)
    started = time.monotonic()
    hook = None
    try:
        module = build_model(cfg.model)
        module.init_weights()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = {}
        for key, value in checkpoint['state_dict'].items():
            name = key[7:] if key.startswith('module.') else key
            if name in state:
                raise ValueError('Duplicate checkpoint key after module prefix removal')
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite/non-tensor model state: ' + name)
            state[name] = value
        module.load_state_dict(state, strict=True)
        manifest['checkpoint_meta'] = {key: checkpoint.get('meta', {}).get(key)
                                       for key in ('epoch', 'iter')}
        manifest['finite_model_tensors'] = len(state)
        manifest['optimizer_restored'] = False
        del checkpoint, state
        module.requires_grad_(False)
        module.cuda().eval()
        wrap_fp16_model(module)
        model = MMDataParallel(module, [0])
        model.eval()
        # Identical selection of offline inference as evaluate_subset, without
        # replacing forward/get_occ or changing any postprocessing parameter.
        module.simple_test = module.simple_test_offline
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise AssertionError('Every model parameter must remain frozen')
        # Grid indices refer to the dataset's exact configured grid. Keep FP16
        # buffer rounding separately for decoding optional raw normalized points.
        manifest['pc_range'] = list(cfg.model.pts_bbox_head.pc_range)
        manifest['voxel_size'] = list(cfg.model.pts_bbox_head.voxel_size)
        manifest['effective_model_buffers'] = dict(
            pc_range=module.pts_bbox_head.pc_range.detach().cpu().tolist(),
            voxel_size=module.pts_bbox_head.voxel_size.detach().cpu().tolist())
        manifest['grid_shape'] = module.pts_bbox_head.voxel_num.detach().cpu().tolist()
        manifest['test_cfg'] = dict(module.pts_bbox_head.test_cfg)
        manifest['state'] = 'running'
        write_json(output / 'manifest.json', manifest)
        captured = {}
        if args.raw:
            def capture_head(_module, _inputs, result):
                if captured:
                    raise RuntimeError('More than one head forward for a single anchor')
                for name, tensor in (
                        ('raw_normalized_points', result['all_refine_pts'][-1]),
                        ('raw_logits', result['all_cls_scores'][-1])):
                    if not torch.isfinite(tensor).all():
                        raise FloatingPointError('Nonfinite last decoder output: ' + name)
                    captured[name] = tensor.detach().cpu().numpy().copy()
                # Returning None leaves the model output unchanged.
            hook = module.pts_bbox_head.register_forward_hook(capture_head)
        loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1,
            workers_per_gpu=2, dist=False, shuffle=False, seed=0, pin_memory=True)
        with torch.no_grad():
            for ordinal, data in enumerate(loader):
                index, token = indices[ordinal], tokens[ordinal]
                captured.clear()
                transforms, frames = batch_metadata(data, token)
                metadata = metadata_reader.get(token, transforms)
                torch.cuda.synchronize()
                prediction_started = time.monotonic()
                prediction = model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                inference_seconds = time.monotonic() - prediction_started
                if len(prediction) != 4:
                    raise ValueError('Expected four original-A horizon predictions')
                arrays = dict(fut2cur=transforms, fut_list=frames,
                    sample_token=np.asarray(token), future_tokens=np.asarray(metadata['future_tokens']),
                    lidar_timestamps_us=metadata['lidar_timestamps_us'],
                    sample_timestamps_us=metadata['sample_timestamps_us'],
                    actual_seconds=metadata['actual_seconds'])
                voxel_counts = []
                for horizon, result in enumerate(prediction):
                    if set(result) != {'occ_loc', 'sem_pred'}:
                        raise ValueError('Unexpected original-A prediction schema')
                    locations, labels = np.asarray(result['occ_loc']), np.asarray(result['sem_pred'])
                    if (locations.ndim != 2 or locations.shape[1] != 3 or
                            labels.shape != (len(locations),) or
                            not np.issubdtype(locations.dtype, np.integer) or
                            not np.issubdtype(labels.dtype, np.integer)):
                        raise ValueError('Invalid occupancy prediction shape/dtype')
                    if ((locations < 0).any() or (locations >= np.asarray(manifest['grid_shape'])).any()
                            or (labels < 0).any() or (labels >= 17).any()):
                        raise ValueError('Prediction coordinate/class is outside the declared grid')
                    arrays['occ_loc_' + str(horizon)] = locations
                    arrays['sem_pred_' + str(horizon)] = labels
                    voxel_counts.append(len(locations))
                if args.raw:
                    if set(captured) != {'raw_normalized_points', 'raw_logits'}:
                        raise RuntimeError('Raw head capture did not execute exactly once')
                    points, logits = captured['raw_normalized_points'], captured['raw_logits']
                    if (points.ndim != 4 or points.shape[0] != 4 or points.shape[-1] != 3 or
                            logits.shape != points.shape[:-1] + (17,)):
                        raise ValueError('Unexpected last-layer horizon/query/point shape')
                    arrays.update(captured)
                destination = output / 'samples' / (token + '.npz')
                save_started = time.monotonic()
                # No object arrays; consumers can always use allow_pickle=False.
                np.savez_compressed(destination, **arrays)
                record = dict(dataset_index=index, token=token,
                    file=str(destination.relative_to(output)),
                    future_tokens=metadata['future_tokens'], scene_name=metadata['scene_name'],
                    inference_seconds=inference_seconds,
                    save_seconds=time.monotonic() - save_started,
                    voxel_counts=voxel_counts, bytes=destination.stat().st_size,
                    sha256=sha256(destination))
                manifest['samples'].append(record)
                manifest['elapsed_seconds'] = time.monotonic() - started
                manifest['last_updated_at_utc'] = utc_now()
                write_json(output / 'manifest.json', manifest)
                if (ordinal + 1) % 4 == 0 or ordinal + 1 == len(indices):
                    print(json.dumps(dict(event='EXPORT_PROGRESS', completed=ordinal + 1,
                        total=len(indices), elapsed_seconds=manifest['elapsed_seconds']),
                        allow_nan=False), flush=True)
        if len(manifest['samples']) != len(indices):
            raise RuntimeError('Data loader did not yield every selected anchor')
        if metadata_reader.frame_hashes:
            manifest['target_metadata']['selected_frame_sha256'] = metadata_reader.frame_hashes
        manifest.update(state='complete', completed_at_utc=utc_now(),
                        elapsed_seconds=time.monotonic() - started,
                        inference_seconds=sum(row['inference_seconds'] for row in manifest['samples']))
        write_json(output / 'manifest.json', manifest)
        print(json.dumps(dict(event='EXPORT_COMPLETE', output_dir=str(output),
                              samples=len(indices), elapsed_seconds=manifest['elapsed_seconds'])), flush=True)
    except Exception as error:
        manifest.update(state='failed', failed_at_utc=utc_now(),
                        error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.monotonic() - started)
        write_json(output / 'manifest.json', manifest)
        raise
    finally:
        if hook is not None:
            hook.remove()


if __name__ == '__main__':
    main()
