"""Build a sealed, CPU-only latest-causal radar cache for train and validation.

Both geometry-only and processed-velocity arms read exactly these point arrays.
The geometry-only view clears velocity after loading, never before membership
selection. No image, occupancy label or future point cloud is read.
"""
import argparse
import collections
import datetime
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmcv import Config
import loaders
from loaders.pipelines.loading import get_nusc
from loaders.pipelines.radar import LoadSingleSweepRadar, RADAR_CHANNELS
from loaders.pipelines.radar_cache import json_digest, point_digest, sha256_file
from loaders.pipelines.single_sweep_cache import (
    SCHEMA, SingleSweepRadarCache, single_sweep_protocol,
    validate_single_sweep, write_single_sweep)


def generate(item):
    token, timestamp = item
    # Explicitly use the processed view for the one cache shared by all arms.
    result = online._load_online(dict(sample_idx=token, timestamp=timestamp))
    points = result['radar_points']
    validate_single_sweep(result, protocol['parameters'], RADAR_CHANNELS)
    if abs(result['radar_reference_timestamp_us'] / 1e6 - timestamp) > 1e-5:
        raise ValueError('Info timestamp does not match current LiDAR')
    path = staging / 'points' / (token + '.npz')
    write_single_sweep(path, result, protocol_hash)
    with np.load(path, allow_pickle=False) as stored:
        if (not np.array_equal(stored['points'], points)
                or not np.array_equal(stored['sensor_indices'], result['radar_point_sensor_indices'])
                or json.loads(str(stored['provenance_json'])) != result['radar_sweep_provenance']):
            raise ValueError('Single-sweep serialization mismatch')
    return token, dict(
        reference_timestamp_us=result['radar_reference_timestamp_us'],
        points=len(points), points_sha256=point_digest(points),
        bytes=path.stat().st_size, file_sha256=sha256_file(path),
        sensor_status=dict(collections.Counter(x['status'] for x in result['radar_sweep_provenance'])),
        invalid_velocity_points_before_cap=sum(x['invalid_velocity_points'] for x in result['radar_sweep_provenance']),
        geometry_valid_points_before_cap=sum(x['geometry_valid_points'] for x in result['radar_sweep_provenance']),
        max_abs_processed_velocity=float(np.abs(points[:, 3:5]).max()) if len(points) else 0.,
        max_processed_speed=float(np.linalg.norm(points[:, 3:5].astype(np.float64), axis=1).max()) if len(points) else 0.,
        max_abs_derived_radial=float(np.abs(points[:, 7]).max()) if len(points) else 0.)


def main():
    global online, protocol, protocol_hash, staging
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val'])
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.splits)) != len(args.splits):
        raise ValueError('Invalid workers or duplicate splits')
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('Cache generation must run without CUDA')
    output = Path(args.output).resolve()
    published = output.exists()
    if published and not args.verify_existing:
        raise FileExistsError('Published cache exists; use --verify-existing: ' + str(output))
    staging = output if published else output.with_name(output.name + '.building')
    staging.mkdir(parents=True, exist_ok=True)
    with (staging / 'builder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (staging / 'points').mkdir(exist_ok=True)
        cfg = Config.fromfile(args.config)
        items, split_tokens, split_scenes, annotations, common_params = [], {}, {}, {}, None
        for split in args.splits:
            settings = cfg.data[split]
            params = dict(next(x for x in settings.pipeline if x['type'] == 'LoadSingleSweepRadar'))
            params.pop('type')
            params.pop('cache_root', None)
            params['velocity_mode'] = 'processed'
            if common_params is not None and params != common_params:
                raise ValueError('Train and validation have different radar extraction parameters')
            common_params = params
            annotations[split] = dict(path=str(Path(settings.ann_file).resolve()),
                                      sha256=sha256_file(settings.ann_file))
            with open(settings.ann_file, 'rb') as stream:
                infos = pickle.load(stream)['infos']
            rows = [(row['token'], row['timestamp'] / 1e6) for row in infos]
            split_tokens[split] = sorted(token for token, _ in rows)
            split_scenes[split] = set(row['scene_name'] for row in infos)
            if len(rows) != len(set(split_tokens[split])):
                raise ValueError('Duplicate radar anchors within ' + split)
            items.extend(rows)
        if len(items) != len(set(token for token, _ in items)):
            raise ValueError('Train and validation radar anchors overlap')
        if len(split_scenes) == 2 and split_scenes['train'] & split_scenes['val']:
            raise ValueError('Train and validation radar scenes overlap')
        if not items:
            raise ValueError('No radar anchors in configured splits')
        online = LoadSingleSweepRadar(**common_params)
        protocol = single_sweep_protocol(online)
        protocol_hash = json_digest(protocol)
        protocol_path = staging / 'protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing single-sweep cache has a different protocol')
        if not args.verify_existing:
            protocol_path.write_text(json.dumps(protocol, indent=2))
        get_nusc(online.data_root)  # Fork shares SDK metadata without GPU contexts.
        started = time.monotonic()
        manifest_path = staging / 'manifest.json'
        if args.verify_existing:
            receipt = json.loads((staging / 'COMPLETE.json').read_text())
            manifest = json.loads(manifest_path.read_text())
            if (set(manifest) != set(token for token, _ in items)
                    or receipt['protocol_sha256'] != protocol_hash
                    or receipt['manifest_sha256'] != sha256_file(manifest_path)
                    or receipt['annotations'] != annotations
                    or receipt['split_token_sha256'] != {k: json_digest(v) for k, v in split_tokens.items()}):
                raise ValueError('Existing single-sweep cache receipt mismatch')
        else:
            manifest = {}
            with mp.get_context('fork').Pool(args.workers) as pool:
                for token, entry in pool.imap_unordered(generate, items, chunksize=16):
                    manifest[token] = entry
                    if len(manifest) % 1000 == 0 or len(manifest) == len(items):
                        print(json.dumps(dict(generated=len(manifest), total=len(items),
                                              seconds=time.monotonic() - started)), flush=True)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
            status_counts = collections.Counter()
            for row in manifest.values():
                status_counts.update(row['sensor_status'])
            receipt = dict(
                schema=SCHEMA, protocol_sha256=protocol_hash,
                builder_sha256=sha256_file(__file__),
                manifest_sha256=sha256_file(manifest_path), samples=len(items),
                split_samples={k: len(v) for k, v in split_tokens.items()},
                split_token_sha256={k: json_digest(v) for k, v in split_tokens.items()},
                annotations=annotations, cached_bytes=sum(row['bytes'] for row in manifest.values()),
                online_serialization_equal_samples=len(items),
                geometry_valid_points_before_cap=sum(row['geometry_valid_points_before_cap'] for row in manifest.values()),
                invalid_velocity_points_before_cap=sum(row['invalid_velocity_points_before_cap'] for row in manifest.values()),
                sensor_status=dict(status_counts),
                max_abs_processed_velocity=max(row['max_abs_processed_velocity'] for row in manifest.values()),
                max_processed_speed=max(row['max_processed_speed'] for row in manifest.values()),
                max_abs_derived_radial=max(row['max_abs_derived_radial'] for row in manifest.values()),
                generation_seconds=time.monotonic() - started)
            (staging / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2))
        cached = SingleSweepRadarCache(staging, online)
        for token, timestamp in items:
            cached.load(dict(sample_idx=token, timestamp=timestamp))
        indices = np.random.RandomState(20260927).choice(len(items), min(128, len(items)), replace=False)
        for index in indices:
            token, timestamp = items[index]
            actual = cached.load(dict(sample_idx=token, timestamp=timestamp))
            expected = online._load_online(dict(sample_idx=token, timestamp=timestamp))
            if (not np.array_equal(actual['radar_points'], expected['radar_points'])
                    or not np.array_equal(actual['radar_point_sensor_indices'], expected['radar_point_sensor_indices'])
                    or actual['radar_sweep_provenance'] != expected['radar_sweep_provenance']
                    or actual['radar_reference_timestamp_us'] != expected['radar_reference_timestamp_us']):
                raise ValueError('Independent online single-sweep recheck differs')
        receipt.update(production_reader_verified_samples=len(items),
                       independent_online_rechecks=len(indices),
                       completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if not published:
            (staging / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2))
            os.rename(staging, output)
        print('SINGLE_SWEEP_RADAR_CACHE_COMPLETE ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
