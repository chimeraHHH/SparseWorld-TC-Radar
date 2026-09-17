"""Read-only integrity gate before real training (all anchors and full label set)."""
import argparse
import json
import os
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
from nuscenes import NuScenes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--info-root', required=True)
    p.add_argument('--occ-root', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    nusc = NuScenes('v1.0-trainval', a.data_root, verbose=False)
    report = {'splits': {}, 'missing_files': [], 'label_probe': []}
    label_paths = set()
    sensors = Counter()
    for split in ('train', 'val'):
        with open(Path(a.info_root) / ('nuscenes_infos_'+split+'_sweep_occ.pkl'), 'rb') as f:
            infos = pickle.load(f)['infos']
        scenes = set()
        for info in infos:
            scenes.add(info['scene_name'])
            sample = nusc.get('sample', info['token'])
            for j in range(7):
                if j in (0, 2, 4, 6):
                    label_paths.add(str(Path(a.occ_root) / info['scene_name'] / sample['token'] / 'labels.npz'))
                if j < 6:
                    if not sample['next']:
                        raise ValueError('Metadata contains incomplete forecasting target')
                    sample = nusc.get('sample', sample['next'])
            # Coverage of all camera keyframes and supplied inter-keyframe sweeps.
            for cams in [info['cams']] + info['cam_sweeps']:
                for cam in cams.values():
                    if not os.path.isfile(cam['data_path']):
                        report['missing_files'].append(cam['data_path'])
            sample = nusc.get('sample', info['token'])
            for channel, token in sample['data'].items():
                if not channel.startswith('RADAR'):
                    continue
                sd = nusc.get('sample_data', token)
                count = 0
                while True:
                    age = (info['timestamp'] - sd['timestamp']) / 1e6
                    if age > .5 or count >= 5:
                        break
                    if age >= 0:
                        path = str(Path(a.data_root) / sd['filename'])
                        sensors[channel] += 1
                        if not os.path.isfile(path):
                            report['missing_files'].append(path)
                        count += 1
                    if not sd['prev']:
                        break
                    sd = nusc.get('sample_data', sd['prev'])
        report['splits'][split] = {'anchors': len(infos), 'scenes': sorted(scenes)}
    for path in sorted(label_paths):
        if not os.path.isfile(path):
            report['missing_files'].append(path)
    available = sorted(p for p in label_paths if os.path.isfile(p))
    for path in available[::max(1, len(available)//100)]:
        with np.load(path) as d:
            assert all(d[k].shape == (200, 200, 16) for k in ('semantics', 'mask_camera', 'mask_lidar'))
            assert d['semantics'].min() >= 0 and d['semantics'].max() <= 17
            report['label_probe'].append(path)
    assert not set(report['splits']['train']['scenes']) & set(report['splits']['val']['scenes'])
    report['unique_target_labels'] = len(label_paths)
    report['radar_sweep_checks'] = sensors
    report['missing_files'] = sorted(set(report['missing_files']))
    report['passed'] = not report['missing_files'] and len(available) > 30000
    Path(a.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('label_probe','splits','missing_files')}, indent=2))
    print('missing', len(report['missing_files']))
    if not report['passed']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
