"""Production-reader isolation, temporal provenance and shared velocity views."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from loaders.pipelines import radar
from loaders.pipelines.radar import LoadSingleSweepRadar, RADAR_CHANNELS
from loaders.pipelines.radar_cache import json_digest, point_digest, sha256_file
from loaders.pipelines.single_sweep_cache import (
    SCHEMA, SingleSweepRadarCache, single_sweep_protocol,
    validate_single_sweep, write_single_sweep)


@pytest.fixture
def cache(tmp_path):
    metadata = tmp_path / 'data/v1.0-trainval'
    metadata.mkdir(parents=True)
    for name in ('sample', 'sample_data', 'ego_pose', 'calibrated_sensor', 'sensor'):
        (metadata / (name + '.json')).write_text('[]')
    loader = LoadSingleSweepRadar(str(metadata.parent))
    root = tmp_path / 'cache'
    (root / 'points').mkdir(parents=True)
    protocol = single_sweep_protocol(loader)
    protocol_hash = json_digest(protocol)
    (root / 'protocol.json').write_text(json.dumps(protocol))
    points = np.zeros((5, 10), np.float32)
    points[:, 0] = np.arange(5)
    points[:, [3, 4, 7]] = [3, 4, 3]
    points[:, 6] = .1
    points[:, 8] = 1.
    provenance = [dict(channel=channel, token='sweep' + str(index),
                       timestamp_us=122900000, next_timestamp_us=123100000,
                       status='selected', geometry_valid_points=1,
                       invalid_velocity_points=0, kept_points=1)
                  for index, channel in enumerate(RADAR_CHANNELS)]
    result = dict(sample_idx='abc', radar_reference_timestamp_us=123000000,
                  radar_points=points, radar_sweep_provenance=provenance,
                  radar_point_sensor_indices=np.arange(5, dtype=np.int8))
    path = root / 'points/abc.npz'
    write_single_sweep(path, result, protocol_hash)
    (root / 'manifest.json').write_text(json.dumps({'abc': dict(
        reference_timestamp_us=123000000, points=5, points_sha256=point_digest(points),
        file_sha256=sha256_file(path))}))
    (root / 'COMPLETE.json').write_text(json.dumps(dict(
        protocol_sha256=protocol_hash, manifest_sha256=sha256_file(root / 'manifest.json'), samples=1)))
    return root, loader, result, protocol


def test_shared_cache_velocity_view_and_provenance(cache):
    root, loader, expected, protocol = cache
    processed = LoadSingleSweepRadar(loader.data_root, cache_root=root, velocity_mode='processed')
    zero = LoadSingleSweepRadar(loader.data_root, cache_root=root, velocity_mode='zero')
    assert single_sweep_protocol(processed) == single_sweep_protocol(zero) == protocol
    actual = processed(dict(sample_idx='abc', timestamp=123.))
    geometry = zero(dict(sample_idx='abc', timestamp=123.))
    assert np.array_equal(actual['radar_points'], expected['radar_points'])
    assert actual['radar_sweep_provenance'] == expected['radar_sweep_provenance']
    assert np.array_equal(actual['radar_point_sensor_indices'], geometry['radar_point_sensor_indices'])
    assert np.array_equal(actual['radar_points'][:, [0, 1, 2, 5, 6, 8, 9]],
                          geometry['radar_points'][:, [0, 1, 2, 5, 6, 8, 9]])
    assert not geometry['radar_points'][:, [3, 4, 7]].any()
    assert actual['radar_points'][:, [3, 4, 7]].any()


@pytest.mark.parametrize('change', ['future', 'not_latest', 'mixed_age', 'sensor', 'duplicates', 'count'])
def test_bad_temporal_or_sensor_provenance_fails(cache, change):
    _, _, result, protocol = cache
    result = copy.deepcopy(result)
    if change == 'future': result['radar_sweep_provenance'][0]['timestamp_us'] = 123001000
    if change == 'not_latest': result['radar_sweep_provenance'][0]['next_timestamp_us'] = 122999000
    if change == 'mixed_age': result['radar_points'][0, 6] = .2
    if change == 'sensor': result['radar_point_sensor_indices'][0] = 6
    if change == 'duplicates': result['radar_sweep_provenance'][0]['channel'] = RADAR_CHANNELS[1]
    if change == 'count': result['radar_sweep_provenance'][0]['kept_points'] = 2
    with pytest.raises(ValueError):
        validate_single_sweep(result, protocol['parameters'], protocol['channels'])


def test_source_metadata_parameters_and_file_tamper_rejected(cache):
    root, loader, _, _ = cache
    reader = SingleSweepRadarCache(root, loader)
    with pytest.raises(ValueError, match='protocol'):
        LoadSingleSweepRadar(loader.data_root, max_age=.25, cache_root=root)
    (Path(loader.data_root) / 'v1.0-trainval/ego_pose.json').write_text('[{}]')
    with pytest.raises(ValueError, match='metadata'):
        SingleSweepRadarCache(root, loader)
    with (root / 'points/abc.npz').open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='checksum'):
        reader.load(dict(sample_idx='abc', timestamp=123.))


def test_future_record_never_read_as_point_cloud(monkeypatch):
    reference = 1000000
    pose = dict(translation=[0, 0, 0], rotation=[1, 0, 0, 0])
    tables = {'sample': {'anchor': {'data': {'LIDAR_TOP': 'lidar'}}},
              'sample_data': {'lidar': dict(timestamp=reference, ego_pose_token='pose')},
              'ego_pose': {'pose': pose}, 'calibrated_sensor': {'cal': pose}}
    for index, channel in enumerate(RADAR_CHANNELS):
        prefix = str(index)
        tables['sample']['anchor']['data'][channel] = prefix + 'future'
        for suffix, timestamp, previous, nxt in (
                ('old', 700000, '', prefix + 'causal'),
                ('causal', 900000, prefix + 'old', prefix + 'future'),
                ('future', 1100000, prefix + 'causal', '')):
            token = prefix + suffix
            tables['sample_data'][token] = dict(
                token=token, timestamp=timestamp, prev=previous, next=nxt,
                calibrated_sensor_token='cal', ego_pose_token='pose', filename=token)
    nusc = SimpleNamespace(get=lambda table, token: tables[table][token])
    read_paths = []
    def read_points(path):
        read_paths.append(Path(path).name)
        raw = np.zeros((18, 2))
        raw[0] = [1, 2]
        raw[8] = [50, np.nan]
        return SimpleNamespace(points=raw)
    monkeypatch.setattr(radar, 'get_nusc', lambda root: nusc)
    monkeypatch.setattr(radar.RadarPointCloud, 'from_file', read_points)
    result = LoadSingleSweepRadar('/unused', max_points=8)(dict(sample_idx='anchor', timestamp=1.))
    assert read_paths == [str(i) + 'causal' for i in range(5)]
    assert len(result['radar_points']) == 8
    assert sum(x['invalid_velocity_points'] for x in result['radar_sweep_provenance']) == 5
    assert np.isfinite(result['radar_points']).all()
    assert len({x['token'] for x in result['radar_sweep_provenance']}) == 5
