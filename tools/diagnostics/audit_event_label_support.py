#!/usr/bin/env python3
"""Bounded, read-only CPU audit of existing intermediate occupancy labels.

Run with the existing H200 environment's Python, optionally via stdin.  This
script never imports the model, creates labels, changes data, or runs inference.
Existing Occ3D files are reconstructed dataset labels, not independent physical
ground truth.  Pass --script-sha256 when executing via stdin for reproducibility.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import time


ROOT = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
SOURCE_RECEIPTS = {}


def file_receipt(path, digest=None):
    path = Path(path)
    stat = path.stat()
    if digest is None:
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                value.update(block)
        digest = value.hexdigest()
    return dict(path=str(path), resolved_path=str(path.resolve()),
                bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest)


def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCE_RECEIPTS[str(path)] = file_receipt(path, hashlib.sha256(data).hexdigest())
    return json.loads(data)


def iter_json_array(path, chunk_size=1024 * 1024):
    """Stream complete nuScenes tables while hashing their original UTF-8 bytes."""
    path = Path(path)
    decoder, digest = json.JSONDecoder(), hashlib.sha256()
    with path.open('r', encoding='utf-8', newline='') as stream:
        buf, begun, eof = '', False, False

        def read_chunk():
            text = stream.read(chunk_size)
            digest.update(text.encode('utf-8'))
            return text

        while True:
            buf = buf.lstrip()
            if not begun:
                if not buf and not eof:
                    chunk = read_chunk(); buf += chunk; eof = not chunk
                    continue
                if not buf.startswith('['):
                    raise ValueError('Expected JSON array: ' + str(path))
                begun, buf = True, buf[1:]
                continue
            if buf.startswith(','):
                buf = buf[1:].lstrip()
            if buf.startswith(']'):
                trailing = buf[1:]
                while not eof:
                    chunk = read_chunk(); trailing += chunk; eof = not chunk
                if trailing.strip():
                    raise ValueError('Trailing JSON data: ' + str(path))
                SOURCE_RECEIPTS[str(path)] = file_receipt(path, digest.hexdigest())
                return
            try:
                value, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError('Incomplete JSON array: ' + str(path))
                chunk = read_chunk(); buf += chunk; eof = not chunk
                continue
            yield value
            buf = buf[end:]


def subset_identity(payload):
    indices, tokens = payload['indices'], payload['tokens']
    if len(indices) != 256 or len(tokens) != 256:
        raise ValueError('Expected the original fixed 256-anchor validation cohort')
    if len(set(indices)) != 256 or len(set(tokens)) != 256:
        raise ValueError('Validation cohort has repeated indices or tokens')
    return list(zip(indices, tokens))


def summary(values):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), min=float(array.min()),
                median=float(np.median(array)), mean=float(array.mean()),
                max=float(array.max()), p95=float(np.percentile(array, 95)))


def audit_label(path):
    """Read each unique NPZ once; no occupancy arrays appear in the output."""
    path = Path(path)
    row = dict(path=str(path), exists=path.is_file())
    if not row['exists']:
        return row
    try:
        content = path.read_bytes()
        row.update(file_receipt(path, hashlib.sha256(content).hexdigest()))
        with np.load(io.BytesIO(content), allow_pickle=False) as label:
            row['keys'] = sorted(label.files)
            required = {'semantics', 'mask_camera', 'mask_lidar'}
            row['required_keys_present'] = required.issubset(label.files)
            if not row['required_keys_present']:
                return row
            semantics = label['semantics']
            camera, lidar = label['mask_camera'], label['mask_lidar']
            shapes = {name: list(label[name].shape) for name in sorted(required)}
            row['shapes'] = shapes
            row['same_3d_shape'] = (semantics.ndim == 3 and
                                   semantics.shape == camera.shape == lidar.shape)
            row['expected_occ3d_shape'] = semantics.shape == (200, 200, 16)
            row['semantics_dtype'] = str(semantics.dtype)
            legal = (np.issubdtype(semantics.dtype, np.integer) and
                     bool(np.all((semantics >= 0) & (semantics <= 17))))
            row['semantic_values_legal_0_to_17'] = legal
            semantic_values = np.unique(semantics)
            row['semantic_values'] = (semantic_values.tolist() if legal else
                                      [str(value) for value in semantic_values[:32]])
            row['semantic_unique_value_count'] = int(len(semantic_values))
            row['mask_values_binary'] = {
                name: bool(np.all((value == 0) | (value == 1)))
                for name, value in (('mask_camera', camera), ('mask_lidar', lidar))}
            if row['same_3d_shape']:
                known = camera.astype(bool)
                row['voxel_count'] = int(semantics.size)
                row['camera_known_voxels'] = int(known.sum())
                row['camera_known_fraction'] = float(known.mean())
                row['lidar_known_fraction'] = float(lidar.astype(bool).mean())
                row['camera_known_occupied_voxels'] = int(((semantics < 17) & known).sum())
                row['camera_known_free_voxels'] = int(((semantics == 17) & known).sum())
                row['camera_known_class_counts'] = (np.bincount(
                    semantics[known].astype(np.int64), minlength=18).tolist() if legal else None)
            row['usable_schema'] = bool(row['same_3d_shape'] and
                row['expected_occ3d_shape'] and legal and all(row['mask_values_binary'].values()))
        row['label_kind'] = 'existing reconstructed occupancy labels'
        row['interpolation_provenance'] = (
            'Not certified by NPZ existence or keys; no interpolation is performed by this audit.')
    except Exception as error:
        row['read_error'] = type(error).__name__ + ': ' + str(error)
    return row


def run(args):
    started = time.monotonic()
    cohort = subset_identity(read_json(args.subset))
    other = subset_identity(read_json(args.crosscheck_subset))
    if cohort != other:
        raise ValueError('A and censored-path fixed-256 cohorts do not match exactly')
    infos_path = Path(args.infos)
    with infos_path.open('rb') as stream:
        infos = pickle.load(stream)['infos']
    SOURCE_RECEIPTS[str(infos_path)] = file_receipt(infos_path)
    # mmdet3d NuScenesDataset.load_annotations sorts by timestamp; load_interval=1.
    infos = sorted(infos, key=lambda item: item['timestamp'])
    mismatches = [dict(index=i, subset_token=t, info_token=infos[i]['token']
                      if 0 <= i < len(infos) else None)
                  for i, t in cohort if not 0 <= i < len(infos) or infos[i]['token'] != t]
    if mismatches:
        raise ValueError('Dataset-index/token mismatch: ' + json.dumps(mismatches[:3]))

    tables = Path(args.tables)
    samples = {row['token']: row for row in iter_json_array(tables / 'sample.json')}
    scenes = {row['token']: row['name'] for row in iter_json_array(tables / 'scene.json')}
    # Deterministic scene-stratified selection; never select based on label availability.
    ranked = sorted(cohort, key=lambda pair: hashlib.sha256(pair[1].encode()).hexdigest())
    selected, seen_scenes = [], set()
    for pair in ranked:
        scene = samples[pair[1]]['scene_token']
        if scene not in seen_scenes and len(selected) < args.max_anchors:
            selected.append(pair); seen_scenes.add(scene)
    for pair in ranked:
        if len(selected) >= args.max_anchors:
            break
        if pair not in selected:
            selected.append(pair)
    selected.sort(key=lambda pair: pair[0])

    anchors, needed = [], set()
    for index, token in selected:
        first = samples[token]
        scene = scenes[first['scene_token']]
        if infos[index]['scene_name'] != scene:
            raise ValueError('Info/sample scene identity mismatch')
        anchor = dict(index=index, token=token, scene_name=scene,
                      anchor_sample_timestamp_us=first['timestamp'], frames=[])
        current = first
        for step in range(7):
            frame = dict(step=step, nominal_seconds=step / 2,
                         is_intermediate=step in (1, 3, 5), keyframe_exists=current is not None)
            if current is not None:
                if current['scene_token'] != first['scene_token']:
                    raise ValueError('Sample chain crossed a scene boundary')
                actual = (current['timestamp'] - first['timestamp']) / 1e6
                frame.update(token=current['token'], sample_timestamp_us=current['timestamp'],
                             actual_sample_seconds=actual,
                             sample_nominal_error_seconds=actual - step / 2)
                needed.add(current['token'])
                nxt = current.get('next')
                current = samples[nxt] if nxt else None
            anchor['frames'].append(frame)
        anchors.append(anchor)

    sensors = {row['token']: row['channel'] for row in iter_json_array(tables / 'sensor.json')}
    lidar_cals = {row['token'] for row in iter_json_array(tables / 'calibrated_sensor.json')
                  if sensors[row['sensor_token']] == 'LIDAR_TOP'}
    lidar = {}
    for row in iter_json_array(tables / 'sample_data.json'):
        if (row['sample_token'] in needed and row['is_key_frame'] and
                row['calibrated_sensor_token'] in lidar_cals):
            if row['sample_token'] in lidar:
                raise ValueError('Duplicate LIDAR_TOP keyframe record')
            lidar[row['sample_token']] = dict(timestamp_us=row['timestamp'], token=row['token'])

    labels = {}
    for anchor in anchors:
        first_lidar = lidar.get(anchor['token'])
        for frame in anchor['frames']:
            if not frame['keyframe_exists']:
                continue
            token = frame['token']
            frame['lidar_keyframe_exists'] = token in lidar
            if token in lidar and first_lidar is not None:
                actual = (lidar[token]['timestamp_us'] - first_lidar['timestamp_us']) / 1e6
                frame.update(lidar_timestamp_us=lidar[token]['timestamp_us'],
                             lidar_sample_data_token=lidar[token]['token'],
                             actual_lidar_seconds=actual,
                             lidar_nominal_error_seconds=actual - frame['nominal_seconds'],
                             lidar_minus_sample_seconds=(lidar[token]['timestamp_us'] -
                                                         frame['sample_timestamp_us']) / 1e6)
            label_path = Path(args.occ_root) / anchor['scene_name'] / token / 'labels.npz'
            key = str(label_path)
            if key not in labels:
                labels[key] = audit_label(label_path)
            frame['label_path'] = key

    by_horizon = {}
    for step in range(7):
        frames = [anchor['frames'][step] for anchor in anchors]
        present = [frame for frame in frames if frame['keyframe_exists']]
        rows = [labels[frame['label_path']] for frame in present]
        by_horizon[str(step / 2)] = dict(
            nominal_seconds=step / 2, checked_anchors=len(frames),
            sample_keyframes=sum(frame['keyframe_exists'] for frame in frames),
            lidar_keyframes=sum(frame.get('lidar_keyframe_exists', False) for frame in frames),
            existing_npz=sum(row['exists'] for row in rows),
            schema_valid_npz=sum(row.get('usable_schema', False) for row in rows),
            nonempty_camera_domain=sum(row.get('camera_known_voxels', 0) > 0 for row in rows),
            actual_sample_seconds=summary([frame['actual_sample_seconds'] for frame in present]),
            sample_abs_nominal_error_seconds=summary([
                abs(frame['sample_nominal_error_seconds']) for frame in present]),
            lidar_abs_nominal_error_seconds=summary([
                abs(frame['lidar_nominal_error_seconds']) for frame in present
                if 'lidar_nominal_error_seconds' in frame]),
            sample_error_over_100ms=sum(abs(frame['sample_nominal_error_seconds']) > .1
                                      for frame in present),
            camera_known_fraction=summary([row['camera_known_fraction'] for row in rows
                                           if 'camera_known_fraction' in row]))
    script_hash = args.script_sha256
    script_path = Path(globals().get('__file__', '<stdin>'))
    if script_path.is_file():
        script_hash = file_receipt(script_path)['sha256']
    return dict(schema_version='event-label-support-v1',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started,
        scope='CPU-only read; no model/GPU, inference, training, interpolation, or data writes',
        script_sha256=script_hash, script_hash_source=('local source file' if script_path.is_file()
                                                     else 'caller supplied for stdin'),
        paths=dict(infos=str(args.infos), tables=str(args.tables), occ_root=str(args.occ_root),
                   subset=str(args.subset), crosscheck_subset=str(args.crosscheck_subset)),
        source_receipts=SOURCE_RECEIPTS,
        selection=dict(indices=[row['index'] for row in anchors],
                       tokens=[row['token'] for row in anchors], order='dataset index ascending'),
        cohort=dict(original_size=len(cohort), A_and_censored_path_tokens_and_indices_match=True,
                    sorted_infos_indices_match=True, full_infos_count=len(infos),
                    selected_anchors=len(anchors), selected_scenes=len(seen_scenes),
                    selection='sha256(token), first one per scene then fill, max 32; no label-based selection',
                    indices=[row['index'] for row in anchors], tokens=[row['token'] for row in anchors]),
        coverage_by_nominal_horizon=by_horizon,
        unique_npz_checked=len(labels), anchors=anchors, labels=list(labels.values()),
        limitations=[
            'The existing reconstructed occupancy labels are not independent physical truth.',
            'Native sample-chain keyframes and existing NPZ files are distinguished; neither proves non-interpolated reconstruction provenance.',
            'This script does not generate or use interpolated labels, synthetic boxes, or hidden-region pseudo-targets.',
            'Step indices 0..6 are approximately 0..3 s, not exact timestamps; real sample and LiDAR-reference times are retained.',
            'Per-frame camera masks are known-domain proxies, not perfect visibility or evidence of physical absence.',
            'Mask fractions are measured in each target ego grid; no common-world registration or joint-known-domain test was performed.',
            'Finite snapshots bound some transitions but cannot determine exact event times, repeated hidden transitions, or persistent instance identity.',
            'This bounded subset audits availability/schema only; it does not establish learnability, model benefit, novelty, or dataset-wide coverage.',
            'These 256 samples have already been used for selection and analysis; intermediate targets are not an independent blind test set.'
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infos', type=Path, default=ROOT / 'datasets/infos/nuscenes_infos_val_sweep_occ.pkl')
    parser.add_argument('--tables', type=Path, default=Path('/storage/data/metaiot_data/public_dataset/nuscenes/v1.0-trainval'))
    parser.add_argument('--occ-root', type=Path, default=ROOT / 'datasets/occ3d/gts')
    parser.add_argument('--subset', type=Path, default=ROOT / 'work_dirs/radar_forecast_transport_seed0/validation_epoch_00_subset.json')
    parser.add_argument('--crosscheck-subset', type=Path, default=ROOT / 'work_dirs/radar_forecast_censored-path_seed0/validation_epoch_00_subset.json')
    parser.add_argument('--max-anchors', type=int, default=32)
    parser.add_argument('--script-sha256', default=None)
    args = parser.parse_args()
    if not 1 <= args.max_anchors <= 32:
        parser.error('--max-anchors must be between 1 and 32')
    if args.script_sha256 is not None and (len(args.script_sha256) != 64 or
            any(char not in '0123456789abcdef' for char in args.script_sha256)):
        parser.error('--script-sha256 must be a lowercase SHA256 digest')
    global np
    import numpy as np
    try:
        result = run(args)
        print(json.dumps(result, separators=(',', ':'), allow_nan=False))
    except Exception as error:
        print(json.dumps(dict(schema_version='event-label-support-v1', status='audit_failed',
                              error=type(error).__name__ + ': ' + str(error),
                              source_receipts=SOURCE_RECEIPTS), separators=(',', ':')))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
