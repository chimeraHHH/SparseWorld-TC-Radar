"""Corruption and protocol checks for the real cache reader (CPU only)."""
import json
from pathlib import Path
import numpy as np
import pytest
from loaders.pipelines.radar import LoadCausalRadar
from loaders.pipelines.radar_cache import (
    RadarCache, cache_protocol, json_digest, point_digest, sha256_file, write_points)


@pytest.fixture
def cache(tmp_path):
    metadata = tmp_path / 'data' / 'v1.0-trainval'
    metadata.mkdir(parents=True)
    for name in ('sample', 'sample_data', 'ego_pose', 'calibrated_sensor', 'sensor'):
        (metadata / (name + '.json')).write_text('[]')
    loader = LoadCausalRadar(str(metadata.parent))
    root = tmp_path / 'cache'
    (root / 'points').mkdir(parents=True)
    protocol = cache_protocol(loader)
    protocol_hash = json_digest(protocol)
    (root / 'protocol.json').write_text(json.dumps(protocol))
    points = np.zeros((3, 10), np.float32)
    points[:, 6] = [0, .25, .5]
    result = dict(sample_idx='abc', radar_reference_timestamp_us=123000000,
                  radar_points=points)
    write_points(root / 'points' / 'abc.npz', result, protocol_hash)
    (root / 'manifest.json').write_text(json.dumps({'abc':dict(
        reference_timestamp_us=123000000, points=3, points_sha256=point_digest(points))}))
    (root / 'COMPLETE.json').write_text(json.dumps(dict(
        protocol_sha256=protocol_hash, manifest_sha256=sha256_file(root / 'manifest.json'), samples=1)))
    return root, loader, result, protocol_hash


def test_exact_points_and_audit_metadata(cache):
    root, loader, expected, _ = cache
    actual = RadarCache(root, loader).load(dict(sample_idx='abc', timestamp=123.))
    assert np.array_equal(actual['radar_points'], expected['radar_points'])
    assert actual['radar_reference_timestamp_us'] == 123000000


def test_wrong_parameters_and_source_tables_rejected(cache):
    root, loader, _, _ = cache
    loader.sweeps_num = 8
    with pytest.raises(ValueError, match='protocol'):
        RadarCache(root, loader)
    loader.sweeps_num = 5
    (Path(loader.data_root) / 'v1.0-trainval' / 'ego_pose.json').write_text('[{}]')
    with pytest.raises(ValueError, match='metadata'):
        RadarCache(root, loader)


@pytest.mark.parametrize('change', ['negative_age', 'too_old', 'nan', 'shape', 'dtype', 'too_many', 'range', 'checksum', 'token', 'protocol'])
def test_corrupted_points_rejected(cache, change):
    root, loader, result, protocol_hash = cache
    reader = RadarCache(root, loader)
    p = result['radar_points']
    if change == 'negative_age': p[0, 6] = -.01
    elif change == 'too_old': p[0, 6] = .6
    elif change == 'nan': p[0, 0] = np.nan
    elif change == 'shape': result['radar_points'] = p[:, :9]
    elif change == 'dtype': result['radar_points'] = p.astype(np.float64)
    elif change == 'too_many': result['radar_points'] = np.zeros((4097, 10), np.float32)
    elif change == 'range': p[0, 0] = 45
    elif change == 'checksum': p[0, 5] = 9
    elif change == 'token': result['sample_idx'] = 'another'
    elif change == 'protocol': protocol_hash = 'wrong'
    write_points(root / 'points' / 'abc.npz', result, protocol_hash)
    with pytest.raises(ValueError):
        reader.load(dict(sample_idx='abc', timestamp=123.))


def test_missing_and_wrong_timestamp_rejected(cache):
    root, loader, _, _ = cache
    reader = RadarCache(root, loader)
    with pytest.raises(ValueError, match='timestamp'):
        reader.load(dict(sample_idx='abc', timestamp=124.))
    with pytest.raises(KeyError):
        reader.load(dict(sample_idx='missing', timestamp=123.))
    (root / 'points' / 'abc.npz').unlink()
    with pytest.raises(FileNotFoundError):
        reader.load(dict(sample_idx='abc', timestamp=123.))


def test_unsealed_or_tampered_manifest_rejected(cache):
    root, loader, _, _ = cache
    (root / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest'):
        RadarCache(root, loader)
    (root / 'COMPLETE.json').unlink()
    with pytest.raises(FileNotFoundError):
        RadarCache(root, loader)
