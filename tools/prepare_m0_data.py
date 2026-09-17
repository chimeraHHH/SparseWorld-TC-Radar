"""Build full-split SparseWorld camera metadata from the official nuScenes tables.

Labels remain official Occ3D; no pseudo labels or split substitutions. Only anchors
with all 0/1/2/3-second targets are admitted. Camera histories stay in their scene.
"""
import argparse
import json
import os
import pickle
from pathlib import Path
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

CAMS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def camera_info(nusc, sd, root, global_to_lidar):
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    sensor_to_ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']))
    sensor_to_global = transform_matrix(pose['translation'], Quaternion(pose['rotation'])) @ sensor_to_ego
    sensor_to_lidar = global_to_lidar @ sensor_to_global
    return dict(data_path=str(root / sd['filename']), timestamp=sd['timestamp'],
        cam_intrinsic=np.array(cs['camera_intrinsic']),
        sensor2ego_rotation=sensor_to_ego[:3, :3], sensor2ego_translation=sensor_to_ego[:3, 3],
        sensor2global_rotation=sensor_to_global[:3, :3].T,
        sensor2global_translation=sensor_to_global[:3, 3],
        sensor2lidar_rotation=sensor_to_lidar[:3, :3],
        sensor2lidar_translation=sensor_to_lidar[:3, 3])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    root, out = Path(args.data_root).resolve(), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes('v1.0-trainval', str(root), verbose=True)
    split_scenes = create_splits_scenes()
    report = {'version': 'v1.0-trainval', 'horizons': [0, 2, 4, 6],
              'causal_reference': 'LIDAR_TOP sample_data timestamp', 'splits': {}}
    for split in ('train', 'val'):
        scenes = set(split_scenes[split])
        infos = []
        excluded_tail = 0
        for sample in nusc.sample:
            scene = nusc.get('scene', sample['scene_token'])
            if scene['name'] not in scenes:
                continue
            future = sample
            for _ in range(6):
                if not future['next']:
                    break
                future = nusc.get('sample', future['next'])
            else:
                future = None  # all six available
            if future is not None:
                excluded_tail += 1
                continue
            lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
            pose = nusc.get('ego_pose', lidar['ego_pose_token'])
            global_to_lidar = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=True) @ transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=True)
            current_cams = {c: nusc.get('sample_data', sample['data'][c]) for c in CAMS}
            info = dict(token=sample['token'], timestamp=lidar['timestamp'], scene_name=scene['name'],
                scene_token=scene['token'], prev=sample['prev'], next=sample['next'],
                lidar_path=str(root / lidar['filename']), lidar2ego_translation=cs['translation'],
                lidar2ego_rotation=cs['rotation'], ego2global_translation=pose['translation'],
                ego2global_rotation=pose['rotation'], gt_boxes=np.empty((0, 7), dtype=np.float32),
                gt_names=np.empty((0,), dtype=str), gt_velocity=np.empty((0, 2), dtype=np.float32),
                num_lidar_pts=np.empty(0, dtype=int), valid_flag=np.empty(0, dtype=bool), lidar_sweeps=[],
                cams={c: camera_info(nusc, sd, root, global_to_lidar) for c, sd in current_cams.items()})
            sweeps = []
            if sample['prev']:
                for _ in range(5):
                    if not all(sd['prev'] for sd in current_cams.values()):
                        break
                    current_cams = {c: nusc.get('sample_data', sd['prev']) for c, sd in current_cams.items()}
                    sweeps.append({c: camera_info(nusc, sd, root, global_to_lidar) for c, sd in current_cams.items()})
            info['cam_sweeps'] = sweeps
            infos.append(info)
        infos.sort(key=lambda x: x['timestamp'])
        target = out / ('nuscenes_infos_' + split + '_sweep_occ.pkl')
        with target.open('wb') as f:
            pickle.dump(dict(infos=infos, metadata=dict(version='v1.0-trainval')), f)
        report['splits'][split] = dict(samples=len(infos), excluded_scene_tail=excluded_tail,
            scene_names=sorted(set(i['scene_name'] for i in infos)), path=str(target))
        print(split, len(infos), 'excluded_tail', excluded_tail, flush=True)
    assert not set(report['splits']['train']['scene_names']) & set(report['splits']['val']['scene_names'])
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
