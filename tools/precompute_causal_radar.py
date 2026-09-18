"""Generate and verify a complete training cache, then publish it atomically.

Run on Linux with CUDA_VISIBLE_DEVICES='' and the training environment.
The online algorithm, filtering, coordinate frame and point cap are unchanged.
"""
import argparse
import datetime
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
import loaders
from loaders.pipelines.loading import get_nusc
from loaders.pipelines.radar import LoadCausalRadar
from loaders.pipelines.radar_cache import (
    cache_protocol, json_digest, point_digest, sha256_file, validate_points,
    write_points, RadarCache)


def generate(item):
    token, timestamp = item
    result = online(dict(sample_idx=token, timestamp=timestamp))
    points = result['radar_points']
    validate_points(points, protocol['parameters'])
    if abs(result['radar_reference_timestamp_us'] / 1e6 - timestamp) > 1e-5:
        raise ValueError('Dataset timestamp does not match current LiDAR')
    path = staging / 'points' / (token + '.npz')
    write_points(path, result, protocol_hash)
    # Verify serialization for every sample, not just the later random subset.
    with np.load(path, allow_pickle=False) as stored:
        if not np.array_equal(stored['points'], points):
            raise ValueError('Radar serialization mismatch')
    return token, dict(reference_timestamp_us=result['radar_reference_timestamp_us'],
                       points=len(points), points_sha256=point_digest(points),
                       bytes=path.stat().st_size, file_sha256=sha256_file(path))


def main():
    global online, protocol, protocol_hash, staging
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-existing', action='store_true',
                        help='Revalidate a fully generated but unpublished staging cache')
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    output = Path(args.output).resolve()
    staging = output.with_name(output.name + '.building')
    if output.exists():
        raise FileExistsError('Published cache already exists: ' + str(output))
    staging.mkdir(parents=True, exist_ok=True)
    with (staging / 'builder.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (staging / 'points').mkdir(exist_ok=True)
        cfg = Config.fromfile(args.config)
        params = dict(next(s for s in cfg.data.train.pipeline if s['type'] == 'LoadCausalRadar'))
        params.pop('type')
        params.pop('cache_root', None)
        online = LoadCausalRadar(**params)
        protocol = cache_protocol(online)
        protocol_hash = json_digest(protocol)
        protocol_path = staging / 'protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Staging cache has a different protocol')
        protocol_path.write_text(json.dumps(protocol, indent=2))
        dataset = build_dataset(cfg.data.train)
        items = [(row['token'], row['timestamp'] / 1e6) for row in dataset.data_infos]
        assert len(items) == len(set(t for t, _ in items)), 'Duplicate sample tokens'
        # Fork after preloading the SDK, sharing read-only table pages.
        get_nusc(online.data_root)
        started = time.monotonic()
        manifest_path = staging / 'manifest.json'
        if args.verify_existing:
            receipt = json.loads((staging / 'COMPLETE.json').read_text())
            manifest = json.loads(manifest_path.read_text())
            assert set(manifest) == set(token for token, _ in items)
            assert receipt['protocol_sha256'] == protocol_hash
            assert receipt['manifest_sha256'] == sha256_file(manifest_path)
            assert receipt['annotations_sha256'] == sha256_file(cfg.data.train.ann_file)
            assert receipt['online_serialization_equal_samples'] == len(items)
            print('REVERIFY_COMPLETE_STAGING_CACHE', flush=True)
        else:
            manifest = {}
            with mp.get_context('fork').Pool(args.workers) as pool:
                for token, entry in pool.imap_unordered(generate, items, chunksize=16):
                    manifest[token] = entry
                    if len(manifest) % 1000 == 0 or len(manifest) == len(items):
                        print(json.dumps(dict(generated=len(manifest), total=len(items),
                                              seconds=time.monotonic() - started)), flush=True)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
            receipt = dict(protocol_sha256=protocol_hash,
                           manifest_sha256=sha256_file(manifest_path), samples=len(items),
                           cached_bytes=sum(row['bytes'] for row in manifest.values()),
                           annotations_sha256=sha256_file(cfg.data.train.ann_file),
                           online_serialization_equal_samples=len(items),
                           generation_seconds=time.monotonic() - started)
            (staging / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2))
        cached = RadarCache(staging, online)
        # Exercise the production reader over the entire published manifest.
        for token, timestamp in items:
            cached.load(dict(sample_idx=token, timestamp=timestamp))
        indices = np.random.RandomState(1918).choice(len(items), min(128, len(items)), replace=False)
        for index in indices:
            token, timestamp = items[index]
            actual = cached.load(dict(sample_idx=token, timestamp=timestamp))
            expected = online(dict(sample_idx=token, timestamp=timestamp))
            assert actual.keys() == expected.keys()
            assert np.array_equal(actual['radar_points'], expected['radar_points'])
            assert actual['radar_reference_timestamp_us'] == expected['radar_reference_timestamp_us']
        receipt.update(production_reader_verified_samples=len(items),
                       independent_online_rechecks=len(indices),
                       completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (staging / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2))
        os.rename(staging, output)
        print('RADAR_CACHE_COMPLETE ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
