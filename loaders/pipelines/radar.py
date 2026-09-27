"""Strictly causal nuScenes radar loading into the current LiDAR-time ego frame."""
import os
import logging
import numpy as np
from mmdet.datasets.builder import PIPELINES
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from .loading import get_nusc
from .radar_cache import RadarCache

RADAR_CHANNELS = ('RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
                  'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT')


def transform_radar(points, sensor_to_current):
    """Rotate velocities/LOS, translate positions; never translate velocity."""
    xyz = points[:3].T @ sensor_to_current[:3, :3].T + sensor_to_current[:3, 3]
    vel = np.c_[points[8:10].T, np.zeros(points.shape[1])]
    vel = vel @ sensor_to_current[:3, :3].T
    los_sensor = points[:3].T.copy()
    # nuScenes Doppler observes the sensor-plane component, not vertical motion.
    los_sensor[:, 2] = 0
    los_sensor /= np.linalg.norm(los_sensor, axis=-1, keepdims=True).clip(1e-6)
    los = los_sensor @ sensor_to_current[:3, :3].T
    radial = np.sum(vel * los, axis=-1)
    return xyz, vel, los, radial


@PIPELINES.register_module()
class LoadCausalRadar:
    def __init__(self, data_root, sweeps_num=5, max_age=0.5,
                 max_points=4096, xy_limit=44., cache_root=None):
        self.data_root = data_root
        self.sweeps_num = sweeps_num
        self.max_age = max_age
        self.max_points = max_points
        self.xy_limit = xy_limit
        self.cache_root = cache_root
        self.cache = RadarCache(cache_root, self) if cache_root else None
        if self.cache is not None:
            logging.info('RADAR_CACHE_ENABLED root=%s samples=%d protocol=%s',
                         cache_root, len(self.cache.entries), self.cache.protocol_sha256)

    def __call__(self, results):
        if self.cache is not None:
            return self.cache.load(results)
        return self._load_online(results)

    def _load_online(self, results):
        nusc = get_nusc(self.data_root)
        sample = nusc.get('sample', results['sample_idx'])
        ref_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ref_pose = nusc.get('ego_pose', ref_sd['ego_pose_token'])
        global_to_current = transform_matrix(ref_pose['translation'], Quaternion(ref_pose['rotation']), inverse=True)
        reference_us = ref_sd['timestamp']
        arrays = []
        for channel in RADAR_CHANNELS:
            sd = nusc.get('sample_data', sample['data'][channel])
            count = 0
            while True:
                age = (reference_us - sd['timestamp']) / 1e6
                if age > self.max_age or count >= self.sweeps_num:
                    break
                if age >= 0:
                    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
                    pose = nusc.get('ego_pose', sd['ego_pose_token'])
                    sensor_to_current = global_to_current @ transform_matrix(pose['translation'], Quaternion(pose['rotation'])) @ transform_matrix(cs['translation'], Quaternion(cs['rotation']))
                    raw = RadarPointCloud.from_file(os.path.join(self.data_root, sd['filename'])).points
                    xyz, vel, los, radial = transform_radar(raw, sensor_to_current)
                    features = np.c_[xyz, vel[:, :2], raw[5], np.full(raw.shape[1], age), radial, los[:, :2]]
                    keep = np.isfinite(features).all(1) & (np.abs(xyz[:, :2]) <= self.xy_limit).all(1)
                    arrays.append(features[keep].astype(np.float32))
                    count += 1
                if not sd['prev']:
                    break
                sd = nusc.get('sample_data', sd['prev'])
        points = np.concatenate(arrays) if arrays else np.empty((0, 10), dtype=np.float32)
        # Stable youngest-first cap; never random validation sampling.
        if len(points) > self.max_points:
            points = points[np.argsort(points[:, 6], kind='stable')[:self.max_points]]
        results['radar_points'] = points
        results['radar_reference_timestamp_us'] = reference_us
        return results


@PIPELINES.register_module()
class LoadSingleSweepRadar:
    """One latest causal sweep from each sensor; shared cache for both arms."""
    def __init__(self, data_root, sweeps_num=1, max_age=0.5,
                 max_points=4096, xy_limit=44., cache_root=None,
                 velocity_mode='processed'):
        from .single_sweep_cache import SingleSweepRadarCache
        if sweeps_num != 1:
            raise ValueError('LoadSingleSweepRadar requires exactly one sweep per sensor')
        if velocity_mode not in ('processed', 'zero'):
            raise ValueError('velocity_mode must be processed or zero')
        if max_age < 0 or max_points < 1 or xy_limit <= 0:
            raise ValueError('Invalid single-sweep radar limits')
        self.data_root, self.sweeps_num = data_root, 1
        self.max_age, self.max_points, self.xy_limit = max_age, max_points, xy_limit
        self.cache_root, self.velocity_mode = cache_root, velocity_mode
        self.cache = SingleSweepRadarCache(cache_root, self) if cache_root else None
        if self.cache is not None:
            logging.info('SINGLE_SWEEP_RADAR_CACHE root=%s samples=%d protocol=%s velocity=%s',
                         cache_root, len(self.cache.entries), self.cache.protocol_sha256, velocity_mode)

    def __call__(self, results):
        from .single_sweep_radar import velocity_view
        results = self.cache.load(results) if self.cache is not None else self._load_online(results)
        results['radar_points'] = velocity_view(results['radar_points'], self.velocity_mode)
        return results

    def _load_online(self, results):
        from .single_sweep_radar import latest_causal_sweep, geometry_filter_and_sanitize
        from .single_sweep_cache import validate_single_sweep
        nusc = get_nusc(self.data_root)
        sample = nusc.get('sample', results['sample_idx'])
        ref_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ref_pose = nusc.get('ego_pose', ref_sd['ego_pose_token'])
        global_to_current = transform_matrix(ref_pose['translation'], Quaternion(ref_pose['rotation']), inverse=True)
        reference_us = int(ref_sd['timestamp'])
        arrays, sensor_indices, provenance = [], [], []
        for index, channel in enumerate(RADAR_CHANNELS):
            sd, next_us = latest_causal_sweep(
                lambda token: nusc.get('sample_data', token), sample['data'].get(channel), reference_us)
            entry = dict(channel=channel, token=sd['token'] if sd else None,
                         timestamp_us=int(sd['timestamp']) if sd else None,
                         next_timestamp_us=next_us, status='missing',
                         geometry_valid_points=0, invalid_velocity_points=0, kept_points=0)
            if sd is not None:
                age = (reference_us - sd['timestamp']) / 1e6
                entry['status'] = 'stale' if age > self.max_age else 'selected'
                if entry['status'] == 'selected':
                    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
                    pose = nusc.get('ego_pose', sd['ego_pose_token'])
                    sensor_to_current = global_to_current @ transform_matrix(pose['translation'], Quaternion(pose['rotation'])) @ transform_matrix(cs['translation'], Quaternion(cs['rotation']))
                    raw = RadarPointCloud.from_file(os.path.join(self.data_root, sd['filename'])).points
                    xyz, vel, los, radial = transform_radar(raw, sensor_to_current)
                    features = np.c_[xyz, vel[:, :2], raw[5], np.full(raw.shape[1], age), radial, los[:, :2]]
                    points, invalid_velocity = geometry_filter_and_sanitize(features, self.xy_limit)
                    entry.update(geometry_valid_points=len(points), invalid_velocity_points=invalid_velocity)
                    arrays.append(points)
                    sensor_indices.append(np.full(len(points), index, dtype=np.int8))
            provenance.append(entry)
        points = np.concatenate(arrays) if arrays else np.empty((0, 10), dtype=np.float32)
        indices = np.concatenate(sensor_indices) if sensor_indices else np.empty(0, dtype=np.int8)
        if len(points) > self.max_points:
            selected = np.argsort(points[:, 6], kind='stable')[:self.max_points]
            points, indices = points[selected], indices[selected]
        for index, entry in enumerate(provenance):
            entry['kept_points'] = int(np.sum(indices == index))
        results.update(radar_points=points, radar_point_sensor_indices=indices,
                       radar_reference_timestamp_us=reference_us, radar_sweep_provenance=provenance)
        validate_single_sweep(results, dict(max_age=self.max_age, max_points=self.max_points,
                                           xy_limit=self.xy_limit), RADAR_CHANNELS)
        return results
