"""Isolated cache with per-sensor temporal provenance and a shared velocity view."""
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
from .radar_cache import sha256_file, json_digest, point_digest, validate_points

SCHEMA = 'latest_causal_single_sweep_lidar_ego_v1'


def single_sweep_protocol(loader):
    from .radar import transform_radar, RADAR_CHANNELS, LoadSingleSweepRadar
    from . import single_sweep_radar
    root = Path(loader.data_root).resolve()
    metadata = root / 'v1.0-trainval'
    return dict(
        schema=SCHEMA, data_root=str(root), version='v1.0-trainval',
        fields=['x', 'y', 'z', 'vx_comp_rotated', 'vy_comp_rotated', 'rcs',
                'age', 'radial_derived_from_comp_velocity', 'los_x', 'los_y'],
        parameters=dict(sweeps_num=1, max_age=loader.max_age,
                        max_points=loader.max_points, xy_limit=loader.xy_limit),
        channels=list(RADAR_CHANNELS),
        selection='latest sample_data timestamp <= current LIDAR timestamp per sensor',
        velocity='processed nuScenes vx_comp/vy_comp; not raw Doppler',
        velocity_view='cache always stores processed; zero view applied after verification',
        invalid_velocity='preserve geometry-valid point; zero all velocity columns; count',
        radar_filters={key: list(getattr(RadarPointCloud, key)) for key in
                       ('invalid_states', 'dynprop_states', 'ambig_states')},
        implementation_sha256=hashlib.sha256((inspect.getsource(transform_radar)
            + inspect.getsource(LoadSingleSweepRadar._load_online)
            + inspect.getsource(single_sweep_radar)
            + inspect.getsource(RadarPointCloud.from_file)).encode()).hexdigest(),
        metadata_sha256={name: sha256_file(metadata / (name + '.json')) for name in
                        ('sample', 'sample_data', 'ego_pose', 'calibrated_sensor', 'sensor')})


def validate_single_sweep(result, parameters, channels):
    points = result['radar_points']
    validate_points(points, parameters)
    sensor_indices = result['radar_point_sensor_indices']
    if sensor_indices.dtype != np.int8 or sensor_indices.shape != (len(points),):
        raise ValueError('Invalid radar point sensor provenance')
    if np.any(sensor_indices < 0) or np.any(sensor_indices >= len(channels)):
        raise ValueError('Invalid radar sensor index')
    provenance = result['radar_sweep_provenance']
    if [item['channel'] for item in provenance] != list(channels):
        raise ValueError('Missing, duplicate or unordered radar sensor provenance')
    reference = result['radar_reference_timestamp_us']
    for index, item in enumerate(provenance):
        subset = points[sensor_indices == index]
        if len(subset) != item['kept_points']:
            raise ValueError('Radar sensor point count mismatch')
        next_us = item['next_timestamp_us']
        if next_us is not None and next_us <= reference:
            raise ValueError('Radar sweep is not the latest causal sweep')
        stamp = item['timestamp_us']
        if item['status'] != 'selected':
            if len(subset) or item['status'] not in ('missing', 'stale'):
                raise ValueError('Unselected radar sensor contains points')
            if item['status'] == 'missing' and (stamp is not None or item['token'] is not None):
                raise ValueError('Missing radar sensor has a selected token')
            if item['status'] == 'stale' and (stamp is None or (reference - stamp) / 1e6 <= parameters['max_age']):
                raise ValueError('Invalid stale radar sensor')
            continue
        if not item['token'] or stamp is None or not 0 <= reference - stamp <= parameters['max_age'] * 1e6:
            raise ValueError('Noncausal or stale selected radar sweep')
        age = np.float32((reference - stamp) / 1e6)
        if np.any(subset[:, 6] != age):
            raise ValueError('Radar point age differs from its selected sweep')
        if not 0 <= item['invalid_velocity_points'] <= item['geometry_valid_points']:
            raise ValueError('Invalid velocity audit count')
        if len(subset) > item['geometry_valid_points']:
            raise ValueError('Point cap added points')


def write_single_sweep(path, result, protocol_sha256):
    path = Path(path)
    temporary = path.with_name(path.name + '.%d.partial' % os.getpid())
    with temporary.open('wb') as stream:
        np.savez(stream, schema=np.array(SCHEMA),
                 sample_token=np.array(result['sample_idx']),
                 reference_timestamp_us=np.int64(result['radar_reference_timestamp_us']),
                 protocol_sha256=np.array(protocol_sha256), points=result['radar_points'],
                 sensor_indices=result['radar_point_sensor_indices'],
                 provenance_json=np.array(json.dumps(result['radar_sweep_provenance'], sort_keys=True)))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class SingleSweepRadarCache:
    def __init__(self, root, loader):
        self.root = Path(root)
        complete = json.loads((self.root / 'COMPLETE.json').read_text())
        protocol = json.loads((self.root / 'protocol.json').read_text())
        self.protocol_sha256 = json_digest(protocol)
        if complete['protocol_sha256'] != self.protocol_sha256 or protocol != single_sweep_protocol(loader):
            raise ValueError('Single-sweep radar cache protocol/source/metadata mismatch')
        manifest = self.root / 'manifest.json'
        if sha256_file(manifest) != complete['manifest_sha256']:
            raise ValueError('Single-sweep radar manifest checksum mismatch')
        self.entries = json.loads(manifest.read_text())
        if len(self.entries) != complete['samples']:
            raise ValueError('Incomplete single-sweep radar manifest')
        self.parameters, self.channels = protocol['parameters'], protocol['channels']

    def load(self, results):
        token = results['sample_idx']
        entry = self.entries[token]
        path = self.root / 'points' / (token + '.npz')
        # File digest binds sensor provenance as well as the point array.
        if sha256_file(path) != entry['file_sha256']:
            raise ValueError('Single-sweep radar file checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            if str(data['schema']) != SCHEMA or str(data['sample_token']) != token:
                raise ValueError('Wrong single-sweep radar schema or token')
            if str(data['protocol_sha256']) != self.protocol_sha256:
                raise ValueError('Single-sweep radar file belongs to another protocol')
            result = dict(radar_points=data['points'],
                          radar_point_sensor_indices=data['sensor_indices'],
                          radar_sweep_provenance=json.loads(str(data['provenance_json'])),
                          radar_reference_timestamp_us=int(data['reference_timestamp_us']))
        reference = result['radar_reference_timestamp_us']
        if reference != entry['reference_timestamp_us'] or abs(reference / 1e6 - results['timestamp']) > 1e-5:
            raise ValueError('Single-sweep radar reference timestamp mismatch')
        validate_single_sweep(result, self.parameters, self.channels)
        if len(result['radar_points']) != entry['points'] or point_digest(result['radar_points']) != entry['points_sha256']:
            raise ValueError('Single-sweep radar point checksum mismatch')
        results.update(result)
        return results
