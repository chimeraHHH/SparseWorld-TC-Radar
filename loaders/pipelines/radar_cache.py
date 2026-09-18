"""Versioned, fail-closed cache for deterministic causal radar point features."""
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud

SCHEMA = 'causal_lidar_ego_v2'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def point_digest(points):
    return hashlib.sha256(points.tobytes(order='C')).hexdigest()


def cache_protocol(loader):
    # Import here to avoid a circular import with the online loader.
    from .radar import transform_radar, RADAR_CHANNELS
    root = Path(loader.data_root).resolve()
    metadata = root / 'v1.0-trainval'
    return dict(
        schema=SCHEMA, data_root=str(root), version='v1.0-trainval',
        fields=['x', 'y', 'z', 'vx', 'vy', 'rcs', 'age', 'radial', 'los_x', 'los_y'],
        parameters=dict(sweeps_num=loader.sweeps_num, max_age=loader.max_age,
                        max_points=loader.max_points, xy_limit=loader.xy_limit),
        channels=list(RADAR_CHANNELS),
        radar_filters={key: list(getattr(RadarPointCloud, key)) for key in
                       ('invalid_states', 'dynprop_states', 'ambig_states')},
        implementation_sha256=hashlib.sha256((inspect.getsource(transform_radar)
            + inspect.getsource(type(loader)._load_online)
            + inspect.getsource(RadarPointCloud.from_file)).encode()).hexdigest(),
        metadata_sha256={name: sha256_file(metadata / (name + '.json')) for name in
                        ('sample', 'sample_data', 'ego_pose', 'calibrated_sensor', 'sensor')})


def validate_points(points, parameters):
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 10:
        raise ValueError('Radar cache must contain float32 N x 10 points')
    if len(points) > parameters['max_points'] or not np.isfinite(points).all():
        raise ValueError('Invalid radar cache point count or nonfinite features')
    # The online loader casts double precision age to float32 before caching.
    if np.any(points[:, 6] < 0) or np.any(points[:, 6] > np.float32(parameters['max_age'])):
        raise ValueError('Noncausal or out-of-window radar cache')
    if np.any(np.abs(points[:, :2]) > np.float32(parameters['xy_limit'])):
        raise ValueError('Out-of-range radar cache')


def write_points(path, result, protocol_sha256):
    path = Path(path)
    temporary = path.with_name(path.name + '.%d.partial' % os.getpid())
    with temporary.open('wb') as stream:
        np.savez(stream, schema=np.array(SCHEMA),
                 sample_token=np.array(result['sample_idx']),
                 reference_timestamp_us=np.int64(result['radar_reference_timestamp_us']),
                 protocol_sha256=np.array(protocol_sha256),
                 points=result['radar_points'])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RadarCache:
    def __init__(self, root, loader):
        self.root = Path(root)
        complete = json.loads((self.root / 'COMPLETE.json').read_text())
        protocol = json.loads((self.root / 'protocol.json').read_text())
        self.protocol_sha256 = json_digest(protocol)
        if complete['protocol_sha256'] != self.protocol_sha256 or protocol != cache_protocol(loader):
            raise ValueError('Radar cache protocol/source/metadata mismatch; regenerate the cache')
        manifest_path = self.root / 'manifest.json'
        if sha256_file(manifest_path) != complete['manifest_sha256']:
            raise ValueError('Radar cache manifest checksum mismatch')
        self.entries = json.loads(manifest_path.read_text())
        if len(self.entries) != complete['samples']:
            raise ValueError('Incomplete radar cache manifest')
        self.parameters = protocol['parameters']

    def load(self, results):
        token = results['sample_idx']
        entry = self.entries[token]  # Missing samples must never fall back silently.
        with np.load(self.root / 'points' / (token + '.npz'), allow_pickle=False) as data:
            if str(data['schema']) != SCHEMA or str(data['sample_token']) != token:
                raise ValueError('Wrong radar cache schema or sample token')
            if str(data['protocol_sha256']) != self.protocol_sha256:
                raise ValueError('Radar point file belongs to another protocol')
            reference_us = int(data['reference_timestamp_us'])
            points = data['points']
        if reference_us != entry['reference_timestamp_us'] or abs(reference_us / 1e6 - results['timestamp']) > 1e-5:
            raise ValueError('Radar/reference timestamp mismatch')
        validate_points(points, self.parameters)
        if len(points) != entry['points'] or point_digest(points) != entry['points_sha256']:
            raise ValueError('Radar point checksum mismatch')
        results['radar_points'] = points
        results['radar_reference_timestamp_us'] = reference_us
        return results
