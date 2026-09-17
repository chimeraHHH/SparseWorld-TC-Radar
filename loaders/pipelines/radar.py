"""Strictly causal nuScenes radar loading into the current LiDAR-time ego frame."""
import os
import numpy as np
from mmdet.datasets.builder import PIPELINES
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from .loading import get_nusc

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

    def __call__(self, results):
        if self.cache_root:
            path = os.path.join(self.cache_root, results['sample_idx'] + '.npz')
            with np.load(path) as data:
                if str(data['schema']) != 'causal_lidar_ego_v1':
                    raise ValueError('Wrong radar cache schema')
                reference_us = int(data['reference_timestamp_us'])
                if abs(reference_us / 1e6 - results['timestamp']) > 1e-5:
                    raise ValueError('Radar/reference timestamp mismatch')
                points = data['points']
            if not np.isfinite(points).all() or np.any(points[:, 6] < 0):
                raise ValueError('Invalid/noncausal radar cache')
            results['radar_points'] = points
            return results
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
