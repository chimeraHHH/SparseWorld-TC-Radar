#!/usr/bin/env python3
"""Build annotation-only, split-isolated endpoint metadata. Never reads point clouds.

Occupancy targets are deliberately NOT cached here. The training loader extracts
observed, semantically compatible, uniquely assigned voxel centers at runtime.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

SCHEMA_VERSION = 'endpoint-segments-v1'
CATEGORY_TO_OCC = {
    'movable_object.barrier': 1, 'vehicle.bicycle': 2,
    'vehicle.bus.bendy': 3, 'vehicle.bus.rigid': 3, 'vehicle.car': 4,
    'vehicle.construction': 5, 'vehicle.motorcycle': 6,
    'human.pedestrian.adult': 7, 'human.pedestrian.child': 7,
    'human.pedestrian.construction_worker': 7, 'human.pedestrian.police_officer': 7,
    'movable_object.trafficcone': 8, 'vehicle.trailer': 9, 'vehicle.truck': 10,
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def iter_json_array(path, chunk_size=1024 * 1024):
    """Bounded-memory parsing of standard nuScenes top-level JSON arrays."""
    decoder = json.JSONDecoder()
    with open(path) as stream:
        buf = ''; begun = False; eof = False
        while True:
            buf = buf.lstrip()
            if not begun:
                if not buf and not eof:
                    chunk = stream.read(chunk_size); buf += chunk; eof = not chunk; continue
                if not buf.startswith('['):
                    raise ValueError('Expected JSON array: ' + str(path))
                begun = True; buf = buf[1:]; continue
            buf = buf.lstrip()
            if buf.startswith(','):
                buf = buf[1:].lstrip()
            if buf.startswith(']'):
                if buf[1:].strip() or stream.read().strip():
                    raise ValueError('Unexpected trailing JSON data')
                return
            try:
                value, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError('Incomplete JSON array: ' + str(path))
                chunk = stream.read(chunk_size); buf += chunk; eof = not chunk; continue
            yield value
            buf = buf[end:]


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Invalid quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def build(args):
    started = time.monotonic()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite nonempty endpoint cache: ' + str(out))
    tables = Path(args.data_root) / args.version
    samples = {x['token']: x for x in iter_json_array(tables / 'sample.json')}
    scenes = {x['token']: x['name'] for x in iter_json_array(tables / 'scene.json')}
    category = {x['token']: x['name'] for x in iter_json_array(tables / 'category.json')}
    instance_labels = {x['token']: CATEGORY_TO_OCC[category[x['category_token']]]
                       for x in iter_json_array(tables / 'instance.json')
                       if category[x['category_token']] in CATEGORY_TO_OCC}
    anchors = {}; selected = {}; source_files = [Path(__file__)]
    for split in args.splits:
        source = Path(args.infos_root) / ('nuscenes_infos_' + split + '_sweep_occ.pkl')
        source_files.append(source)
        with source.open('rb') as stream:
            infos = pickle.load(stream)['infos']
        mapping = {}; needed = set()
        for info in infos:
            chain = [info['token']]
            for _ in range(max(args.horizons)):
                nxt = samples[chain[-1]]['next']
                if not nxt or samples[nxt]['scene_token'] != samples[chain[0]]['scene_token']:
                    raise ValueError('Incomplete or cross-scene target window')
                chain.append(nxt)
            tokens = [chain[h] for h in args.horizons]
            mapping[info['token']] = tokens; needed.update(tokens)
        del infos
        anchors[split] = mapping; selected[split] = needed
    split_scenes = {s: {samples[t]['scene_token'] for t in ts} for s, ts in selected.items()}
    if len(split_scenes) > 1:
        a, b = split_scenes.values()
        if a & b:
            raise ValueError('Train/validation scenes overlap')
    needed = set().union(*selected.values())
    sensors = {x['token']: x['channel'] for x in iter_json_array(tables/'sensor.json')}
    lidar_calibrations = {x['token'] for x in iter_json_array(tables/'calibrated_sensor.json')
                          if sensors[x['sensor_token']] == 'LIDAR_TOP'}
    sample_data = {}
    for x in iter_json_array(tables/'sample_data.json'):
        if x['sample_token'] in needed and x['is_key_frame'] and x['calibrated_sensor_token'] in lidar_calibrations:
            if x['sample_token'] in sample_data:
                raise ValueError('Duplicate keyframe LIDAR_TOP record')
            sample_data[x['sample_token']] = {k:x[k] for k in ('timestamp','ego_pose_token')}
    if set(sample_data) != needed:
        raise ValueError('Missing keyframe LiDAR reference')
    pose_tokens = {x['ego_pose_token'] for x in sample_data.values()}
    poses = {x['token']: x for x in iter_json_array(tables/'ego_pose.json') if x['token'] in pose_tokens}
    annotations = collections.defaultdict(list)
    for x in iter_json_array(tables/'sample_annotation.json'):
        if x['sample_token'] in needed and x['instance_token'] in instance_labels:
            annotations[x['sample_token']].append(x)
    table_names = ['sample','scene','category','instance','sensor','calibrated_sensor','sample_data','ego_pose','sample_annotation']
    source_files.extend(tables/(name+'.json') for name in table_names)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=SCHEMA_VERSION, horizons=args.horizons,
                    created_unix=time.time(), data_root=str(Path(args.data_root).resolve()),
                    occ_root=str(Path(args.occ_root).resolve()), occupancy_labels_cached=False,
                    category_to_occ=CATEGORY_TO_OCC, splits={}, sources={})
    for split, needed_split in selected.items():
        directory = out/split/'frames'; directory.mkdir(parents=True)
        frames_digest = hashlib.sha256(); counts = []
        for tok in sorted(needed_split):
            pose = poses[sample_data[tok]['ego_pose_token']]
            R = rotation(pose['rotation']); trans = np.array(pose['translation'])
            rows = sorted(annotations[tok], key=lambda x:x['instance_token'])
            tokens = [a['instance_token'] for a in rows]
            if len(tokens) != len(set(tokens)):
                raise ValueError('Duplicate instance in sample')
            centers = np.array([(np.array(a['translation'])-trans)@R for a in rows], dtype=np.float32).reshape(-1,3)
            rotations = np.array([R.T@rotation(a['rotation']) for a in rows], dtype=np.float32).reshape(-1,3,3)
            sizes = np.array([np.array(a['size'])[[1,0,2]] for a in rows], dtype=np.float32).reshape(-1,3)
            path = directory/(tok+'.npz')
            np.savez(path, centers=centers, rotations=rotations, sizes=sizes,
                     labels=np.array([instance_labels[a['instance_token']] for a in rows], dtype=np.int64),
                     instance_tokens=np.array(tokens,dtype='U32'),
                     annotation_tokens=np.array([a['token'] for a in rows],dtype='U32'),
                     timestamp_us=np.int64(sample_data[tok]['timestamp']),
                     sample_timestamp_us=np.int64(samples[tok]['timestamp']),
                     ego2global_rotation=R.astype(np.float64), ego2global_translation=trans,
                     scene_name=np.array(scenes[samples[tok]['scene_token']]))
            frames_digest.update(tok.encode());frames_digest.update(bytes.fromhex(sha256(path)));counts.append(len(rows))
        anchor_path = out/split/'anchors.json'
        anchor_path.write_text(json.dumps(anchors[split],sort_keys=True,separators=(',',':'))+'\n')
        manifest['splits'][split] = dict(anchors=len(anchors[split]),frames=len(needed_split),
            scenes=len(split_scenes[split]),anchor_tokens=sorted(anchors[split]),
            anchors_sha256=sha256(anchor_path),frames_sha256=frames_digest.hexdigest(),
            annotation_boxes=sum(counts),mean_boxes_per_frame=float(np.mean(counts)))
    for path in source_files:
        manifest['sources'][str(path.resolve())] = dict(sha256=sha256(path),bytes=path.stat().st_size)
    manifest['elapsed_seconds'] = time.monotonic()-started
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(out),'elapsed_seconds':manifest['elapsed_seconds'],
                      'splits':{s:{k:v for k,v in x.items() if k!='anchor_tokens'} for s,x in manifest['splits'].items()}},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default='/storage/data/metaiot_data/public_dataset/nuscenes')
    p.add_argument('--infos-root',default='/storage/data/metaiot_data/huayiming/SparseWorld/datasets/infos')
    p.add_argument('--occ-root',default='/storage/data/metaiot_data/huayiming/SparseWorld/datasets/occ3d/gts')
    p.add_argument('--output',required=True)
    p.add_argument('--version',default='v1.0-trainval')
    p.add_argument('--splits',nargs='+',choices=['train','val'],default=['train','val'])
    p.add_argument('--horizons',nargs='+',type=int,default=[0,2,4,6])
    a=p.parse_args()
    if a.horizons[0]!=0 or sorted(set(a.horizons))!=a.horizons:
        p.error('Horizons must be unique increasing indices beginning with zero')
    build(a)

if __name__=='__main__':main()
