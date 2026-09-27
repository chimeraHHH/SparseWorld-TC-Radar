#!/usr/bin/env python3
"""Read-only CPU diagnosis of reconstructed occupancy-label time sequences.

No model, GPU, inference, optimization, interpolation-label generation or output
files.  Requires the prior event_label_support JSON and the original label/pose
files.  JSON goes to stdout.  --self-test uses synthetic arrays only.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np


SHAPE = (200, 200, 16)
ORIGIN = np.array([-40.0, -40.0, -1.0])
VOXEL = 0.4
FREE = 17
MOVABLE = [2, 3, 4, 5, 6, 7, 9, 10]
SOURCES = {}


def receipt(path, digest):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), resolved_path=str(path.resolve()),
                sha256=digest, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path)] = receipt(path, hashlib.sha256(data).hexdigest())
    return json.loads(data)


def iter_json_array(path, chunk_size=1024 * 1024):
    """Stream and hash an entire original UTF-8 JSON table, preserving newlines."""
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
                SOURCES[str(path)] = receipt(path, digest.hexdigest())
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


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Invalid wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def grid_centers(shape=SHAPE, origin=ORIGIN, voxel=VOXEL):
    indices = np.indices(shape, dtype=np.float64).reshape(3, -1).T
    return (indices + 0.5) * voxel + np.asarray(origin)


def conservative_known(labels, known):
    """Native-grid 3x3x3 all-known and all-same-label center support."""
    result = np.zeros(labels.shape, dtype=bool)
    if any(size < 3 for size in labels.shape):
        return result
    center = labels[1:-1, 1:-1, 1:-1]
    valid = known[1:-1, 1:-1, 1:-1].copy()
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                sl = (slice(dx, dx + center.shape[0]),
                      slice(dy, dy + center.shape[1]),
                      slice(dz, dz + center.shape[2]))
                valid &= known[sl] & (labels[sl] == center)
    result[1:-1, 1:-1, 1:-1] = valid
    return result


def target_indices(anchor_points, anchor_pose, target_pose,
                   shape=SHAPE, origin=ORIGIN, voxel=VOXEL):
    """Row-vector anchor ego -> world -> target ego, then containing voxel."""
    anchor_R, anchor_t = anchor_pose
    target_R, target_t = target_pose
    transform = anchor_R.T @ target_R
    translation = (anchor_t - target_t) @ target_R
    target_points = anchor_points @ transform + translation
    coordinates = np.floor((target_points - np.asarray(origin)) / voxel).astype(np.int64)
    inside = np.all((coordinates >= 0) & (coordinates < np.asarray(shape)), axis=1)
    return coordinates, inside


def project_frame(labels, known, safe_known, anchor_points, anchor_pose, target_pose,
                  origin=ORIGIN, voxel=VOXEL):
    coordinates, inside = target_indices(anchor_points, anchor_pose, target_pose,
                                        labels.shape, origin, voxel)
    sem = np.full(len(anchor_points), 255, dtype=np.uint8)
    mask, safe = np.zeros(len(anchor_points), bool), np.zeros(len(anchor_points), bool)
    xyz = tuple(coordinates[inside].T)
    sem[inside], mask[inside], safe[inside] = labels[xyz], known[xyz], safe_known[xyz]
    return sem, mask, safe, int(inside.sum())


def sequence_counts(labels, known):
    domain = known.all(axis=0)
    occupied = labels < FREE
    semantic_runs = 1 + np.count_nonzero(labels[1:] != labels[:-1], axis=0)
    binary_runs = 1 + np.count_nonzero(occupied[1:] != occupied[:-1], axis=0)
    episodes = occupied[0].astype(np.int64) + np.count_nonzero(occupied[1:] & ~occupied[:-1], axis=0)
    strata = dict(all=domain, any_occupied=domain & occupied.any(axis=0),
                  any_movable=domain & np.isin(labels, MOVABLE).any(axis=0),
                  semantic_change=domain & (semantic_runs > 1),
                  binary_change=domain & (binary_runs > 1))
    result = dict(grid_cells=labels.shape[1], joint_known=int(domain.sum()),
                  any_frame_known=int(known.any(axis=0).sum()), strata={})
    for name, keep in strata.items():
        result['strata'][name] = dict(support=int(keep.sum()),
            semantic_runs_hist=np.bincount(semantic_runs[keep], minlength=8).tolist(),
            binary_state_runs_hist=np.bincount(binary_runs[keep], minlength=8).tolist(),
            occupied_episodes_hist=np.bincount(episodes[keep], minlength=8).tolist())
    return result


def triplet_counts(labels, known, times):
    left, middle, right = labels
    domain = known.all(axis=0)
    alpha = float((times[1] - times[0]) / (times[2] - times[0]))
    if not 0 < alpha < 1:
        raise ValueError('Non-increasing actual triplet timestamps')
    nearest = left if alpha <= .5 else right
    # Linear one-hot probabilities have only the two endpoint classes in support.
    # Their argmax equals nearest endpoint, with the same left-endpoint tie rule.
    events = dict(endpoints_same_middle_different=(left == right) & (middle != left),
                  free_occupied_free=(left == FREE) & (middle < FREE) & (right == FREE),
                  occupied_free_occupied=(left < FREE) & (middle == FREE) & (right < FREE))
    strata = dict(all=domain, any_occupied=domain & (labels < FREE).any(axis=0),
                  middle_occupied=domain & (middle < FREE),
                  any_movable=domain & np.isin(labels, MOVABLE).any(axis=0),
                  observed_semantic_change=domain & ((left != middle) | (middle != right)))
    result = dict(grid_cells=labels.shape[1], joint_known=int(domain.sum()),
                  any_frame_known=int(known.any(axis=0).sum()), strata={})
    for name, keep in strata.items():
        result['strata'][name] = dict(support=int(keep.sum()),
            left_hold_semantic_errors=int(((left != middle) & keep).sum()),
            nearest_endpoint_semantic_errors=int(((nearest != middle) & keep).sum()),
            linear_onehot_argmax_semantic_errors=int(((nearest != middle) & keep).sum()),
            middle_class_counts=np.bincount(middle[keep], minlength=18).tolist(),
            **{key: int((value & keep).sum()) for key, value in events.items()})
    return result


def add_counts(target, source):
    """Aggregate integer counts/histograms; never average ratios implicitly."""
    for key, value in source.items():
        if isinstance(value, dict):
            add_counts(target.setdefault(key, {}), value)
        elif isinstance(value, list):
            existing = target.setdefault(key, [0] * len(value))
            if len(existing) != len(value):
                raise ValueError('Histogram lengths differ')
            target[key] = [a + b for a, b in zip(existing, value)]
        else:
            target[key] = target.get(key, 0) + value


def readable(counts):
    """Derived micro-averages accompany raw counts and per-anchor evidence."""
    result = json.loads(json.dumps(counts))
    result['joint_known_grid_fraction'] = counts['joint_known'] / max(1, counts['grid_cells'])
    result['joint_over_any_known_fraction'] = (counts['joint_known'] /
                                             max(1, counts['any_frame_known']))
    occupied_support = counts['strata']['any_occupied']['support']
    movable_support = counts['strata']['any_movable']['support']
    occupied_fraction = occupied_support / max(1, counts['joint_known'])
    result['occupied_support_check'] = dict(
        any_occupied_voxels=occupied_support, any_movable_voxels=movable_support,
        any_occupied_fraction_of_joint_known=occupied_fraction,
        degenerate_for_occupied_analysis=(occupied_support < 100 or occupied_fraction < .001),
        movable_support_below_100_voxels=movable_support < 100,
        flag_rule='Descriptive warning if occupied support <100 voxels or <0.1% of common-known domain; not a statistical power criterion',
        consequence='A degenerate domain cannot establish absence of events or superiority of any reconstruction; 3D erosion may erase thin occupied surfaces')
    for name, raw in counts['strata'].items():
        row = result['strata'][name]
        denominator = raw['support']
        for key, value in raw.items():
            if key.endswith('_errors') or key in ('endpoints_same_middle_different',
                                                 'free_occupied_free', 'occupied_free_occupied'):
                row[key + '_fraction'] = value / denominator if denominator else None
            if key.endswith('_hist'):
                row[key.replace('_hist', '_mean')] = (sum(i * n for i, n in enumerate(value)) /
                                                      denominator if denominator else None)
        if 'occupied_episodes_hist' in raw:
            row['two_or_more_occupied_episodes_fraction'] = (
                sum(raw['occupied_episodes_hist'][2:]) / denominator if denominator else None)
    return result


def native_label(frame, expected):
    path = Path(frame['label_path'])
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected['sha256']:
        raise ValueError('Label changed since support audit: ' + str(path))
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        semantics, camera = archive['semantics'], archive['mask_camera']
        if semantics.shape != SHAPE or camera.shape != SHAPE:
            raise ValueError('Unexpected occupancy/mask shape')
        if not np.issubdtype(semantics.dtype, np.integer) or not np.all((semantics >= 0) & (semantics <= FREE)):
            raise ValueError('Illegal semantic values')
        if not np.all((camera == 0) | (camera == 1)):
            raise ValueError('Nonbinary camera mask')
        sem, known = semantics.astype(np.uint8), camera.astype(bool)
    return sem, known, receipt(path, digest)


def run(args):
    start = time.monotonic()
    support = read_json(args.support_json)
    if support.get('schema_version') != 'event-label-support-v1':
        raise ValueError('Unexpected support audit schema')
    anchors = support['anchors'][:args.max_anchors]
    if not anchors or len(anchors) > 32:
        raise ValueError('Need 1..32 support-audited anchors')
    if len(set(anchor['token'] for anchor in anchors)) != len(anchors):
        raise ValueError('Duplicate anchor token')
    selection = support['selection']
    if [(a['index'], a['token']) for a in anchors] != list(zip(selection['indices'], selection['tokens']))[:len(anchors)]:
        raise ValueError('Support selection metadata mismatch')
    tables = Path(args.tables or support['paths']['tables'])
    wanted = {}
    for anchor in anchors:
        if [f['step'] for f in anchor['frames']] != list(range(7)):
            raise ValueError('Need all seven audited steps')
        for frame in anchor['frames']:
            if not frame.get('keyframe_exists') or not frame.get('lidar_keyframe_exists'):
                raise ValueError('A required audited keyframe is missing')
            wanted[frame['lidar_sample_data_token']] = frame
    data_rows = {}
    for row in iter_json_array(tables / 'sample_data.json'):
        if row['token'] in wanted:
            frame = wanted[row['token']]
            if (not row['is_key_frame'] or row['sample_token'] != frame['token'] or
                    row['timestamp'] != frame['lidar_timestamp_us']):
                raise ValueError('Keyframe sample_data identity/timestamp mismatch')
            data_rows[row['token']] = row
    if set(data_rows) != set(wanted):
        raise ValueError('Missing sample_data references')
    original_receipt = support['source_receipts'].get(str(tables / 'sample_data.json'))
    if original_receipt and original_receipt['sha256'] != SOURCES[str(tables / 'sample_data.json')]['sha256']:
        raise ValueError('sample_data table changed since support audit')
    pose_tokens = {row['ego_pose_token'] for row in data_rows.values()}
    poses = {row['token']: row for row in iter_json_array(tables / 'ego_pose.json')
             if row['token'] in pose_tokens}
    if set(poses) != pose_tokens:
        raise ValueError('Missing ego pose')
    expected_labels = {row['path']: row for row in support['labels']}
    points = grid_centers()
    evidence, label_receipts, seven_totals, triple_totals = [], {}, {}, {}
    pose_rows = {}
    for anchor in anchors:
        frame_poses, timestamps = [], []
        for frame in anchor['frames']:
            data_row = data_rows[frame['lidar_sample_data_token']]
            pose = poses[data_row['ego_pose_token']]
            translation = np.asarray(pose['translation'], dtype=np.float64)
            if translation.shape != (3,) or not np.isfinite(translation).all():
                raise ValueError('Invalid ego translation')
            frame_poses.append((rotation(pose['rotation']), translation))
            timestamps.append(frame['lidar_timestamp_us'])
            pose_rows[pose['token']] = pose
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError('Non-increasing seven-frame timestamps')
        labels, masks, safe_masks, frames = [], [], [], []
        for frame, pose in zip(anchor['frames'], frame_poses):
            sem, known, label_receipt = native_label(frame, expected_labels[frame['label_path']])
            label_receipts[frame['label_path']] = label_receipt
            safe = conservative_known(sem, known)
            projected, mask, safe_mask, inside = project_frame(sem, known, safe, points, frame_poses[0], pose)
            labels.append(projected); masks.append(mask); safe_masks.append(safe_mask)
            pose_token = data_rows[frame['lidar_sample_data_token']]['ego_pose_token']
            frames.append(dict(step=frame['step'], sample_token=frame['token'],
                lidar_sample_data_token=frame['lidar_sample_data_token'], ego_pose_token=pose_token,
                timestamp_us=frame['lidar_timestamp_us'],
                actual_seconds=(frame['lidar_timestamp_us'] - timestamps[0]) / 1e6,
                label_path=frame['label_path'], native_known=int(known.sum()),
                native_conservative_known=int(safe.sum()), projected_in_roi=inside,
                projected_known=int(mask.sum()), projected_conservative_known=int(safe_mask.sum())))
        labels, masks, safe_masks = np.stack(labels), np.stack(masks), np.stack(safe_masks)
        row = dict(index=anchor['index'], token=anchor['token'], scene_name=anchor['scene_name'],
                   frames=frames, seven_frame={}, triplets={})
        for domain_name, domain_masks in (('camera_known', masks), ('conservative_same_label_3x3x3', safe_masks)):
            seq = sequence_counts(labels, domain_masks)
            row['seven_frame'][domain_name] = seq
            add_counts(seven_totals.setdefault(domain_name, {}), seq)
            for left in (0, 2, 4):
                steps = [left, left+1, left+2]
                times = [timestamps[s] for s in steps]
                name = str(left) + '_' + str(left+1) + '_' + str(left+2)
                triplet = triplet_counts(labels[steps], domain_masks[steps], times)
                triple_row = row['triplets'].setdefault(name, dict(
                    steps=steps, actual_seconds=[(t-timestamps[0])/1e6 for t in times],
                    actual_alpha=(times[1]-times[0])/(times[2]-times[0]), domains={}))
                triple_row['domains'][domain_name] = triplet
                add_counts(triple_totals.setdefault(name, {}).setdefault(domain_name, {}), triplet)
        evidence.append(row)
    script_hash = args.script_sha256
    source_path = Path(globals().get('__file__', '<stdin>'))
    if source_path.is_file():
        script_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return dict(schema_version='event-sequence-audit-v1',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
        script_sha256=script_hash,
        scope='Read-only CPU label reconstruction diagnostic; not a learned model prediction',
        support_audit=receipt(args.support_json, SOURCES[str(args.support_json)]['sha256']),
        source_receipts=SOURCES, label_receipts=list(label_receipts.values()),
        pose_records=[pose_rows[key] for key in sorted(pose_rows)],
        selection=dict(indices=[a['index'] for a in anchors], tokens=[a['token'] for a in anchors]),
        geometry=dict(anchor_grid_shape=list(SHAPE), anchor_ego_origin=ORIGIN.tolist(), voxel_size=VOXEL,
            registration='Anchor LiDAR-reference ego voxel centers -> global -> target LiDAR-reference ego containing voxel',
            conservative_domain='Each native frame: every 3x3x3 neighbor camera-known and same semantic label; then project and intersect',
            movable_semantic_class_ids=MOVABLE, movable_meaning='Traffic-actor semantic classes, not measured moving objects'),
        seven_frame_summary={name: readable(value) for name, value in seven_totals.items()},
        triplet_summary={name: {domain: readable(value) for domain, value in domains.items()}
                         for name, domains in triple_totals.items()},
        anchors=evidence,
        reconstruction=dict(inputs='True integer-step endpoint reconstructed GT labels, not A predictions',
            targets='Existing intermediate keyframe reconstructed GT labels',
            times='Actual LiDAR-reference timestamps; integer/half-second names are nominal only',
            left_hold='Use left endpoint semantic label',
            nearest_endpoint='Use closer actual endpoint; choose left at an exact tie',
            linear_onehot_argmax='Interpolate the two endpoint one-hot class vectors; argmax exactly equals nearest endpoint under the same left tie rule',
            unseen_middle_class='If the middle class is absent from both endpoint classes, these reconstructions cannot recover it'),
        definitions=dict(semantic_runs='1 + semantic label changes across seven known snapshots; includes free',
            binary_state_runs='1 + occupied/free changes across seven snapshots',
            occupied_episodes='Number of contiguous occupied runs; occupied semantic changes do not create a new episode',
            histogram_bins='Integer run/episode count; bin 0 retained, including all-free sequences'),
        limitations=[
            'Availability and these descriptive diagnostics do not establish novelty, learnability, calibrated event times or learned model benefit.',
            'Existing reconstructed occupancy labels are not independent physical truth; pipeline interpolation/reconstruction provenance is not certified.',
            'No instance identity exists in these dense labels or A query indices; runs describe Eulerian semantic states, not object lifetimes.',
            'Discrete snapshots provide lower bounds on transitions; matching endpoints do not rule out unobserved events between them.',
            'Seven-frame common-known support is selection-biased toward consistently visible areas; independent triplet domains are also reported.',
            'The conservative spatial domain can exclude thin objects and real boundary changes; compare coverage, class support and raw counts, not only its smaller error rate.',
            'Voxel containment after rigid ego registration has resampling/label noise; 3x3x3 sensitivity is not a proof of correct continuous geometry.',
            'GT-endpoint reconstruction is an information/representation diagnostic, not sensor forecasting performance; no oracle result is a model score.',
            'Linear one-hot argmax duplicates nearest-endpoint classification, so they are not independent baselines.',
            'Counts are micro-aggregated, spatially correlated and partly temporally overlapping; 32 scenes and previously reused validation data do not constitute a blind test.'
        ])


def self_test():
    identity = (np.eye(3), np.zeros(3))
    shape, origin, voxel = (5, 5, 5), np.zeros(3), 1.0
    points = grid_centers(shape, origin, voxel)
    expected = np.indices(shape).reshape(3, -1).T
    got, inside = target_indices(points, identity, identity, shape, origin, voxel)
    assert inside.all() and np.array_equal(got, expected)
    shifted = (np.eye(3), np.array([1.0, 0.0, 0.0]))
    got, inside = target_indices(points, identity, shifted, shape, origin, voxel)
    assert np.array_equal(got, expected - np.array([1, 0, 0]))
    assert inside.sum() == 4*5*5
    q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    r = rotation(q)
    p = np.array([[1.5, .5, .5]])
    world = p @ r.T + np.array([2., 3., 4.])
    recovered = (world - np.array([2., 3., 4.])) @ r
    assert np.allclose(recovered, p)
    got, inside = target_indices(p, (r, np.array([2., 3., 4.])), identity,
                                 (10, 10, 10), np.zeros(3), 1.)
    assert inside.all() and np.array_equal(got, np.floor(world).astype(np.int64))
    sem, known = np.full(shape, 4, np.uint8), np.ones(shape, bool)
    assert conservative_known(sem, known).sum() == 27
    known[2,2,2] = False
    assert conservative_known(sem, known).sum() == 0
    known[:] = True; sem[2,2,2] = FREE
    assert conservative_known(sem, known).sum() == 0
    # Columns: always free; one occupied episode; two episodes; class switch inside one episode.
    sequence = np.array([[17,17,4,17], [17,4,4,4], [17,4,17,7], [17,17,4,7],
                         [17,17,4,17], [17,17,17,17], [17,17,17,17]], dtype=np.uint8)
    counts = sequence_counts(sequence, np.ones_like(sequence, bool))
    assert counts['strata']['all']['occupied_episodes_hist'][:3] == [1,2,1]
    assert counts['strata']['all']['semantic_runs_hist'][1] == 1
    assert counts['strata']['all']['semantic_runs_hist'][4] == 2
    triple = np.array([[17,4,4,4], [4,17,7,7], [17,4,7,4]], np.uint8)
    stats = triplet_counts(triple, np.ones_like(triple, bool), [0, 4, 10])['strata']['all']
    assert stats['free_occupied_free'] == 1 and stats['occupied_free_occupied'] == 1
    assert stats['endpoints_same_middle_different'] == 3
    assert stats['left_hold_semantic_errors'] == 4
    stats2 = triplet_counts(triple, np.ones_like(triple, bool), [0, 6, 10])['strata']['all']
    assert stats2['nearest_endpoint_semantic_errors'] == 3
    assert stats2['linear_onehot_argmax_semantic_errors'] == 3
    reduced_known = np.ones_like(triple, bool); reduced_known[1,0] = False
    assert triplet_counts(triple, reduced_known, [0, 6, 10])['joint_known'] == 3
    doubled = {}; add_counts(doubled, counts); add_counts(doubled, counts)
    assert doubled['joint_known'] == 8
    assert doubled['strata']['all']['occupied_episodes_hist'][:3] == [2,4,2]
    print(json.dumps(dict(status='synthetic_checks_passed', checks=[
        'identity and translation registration', 'wxyz rotation and row-vector composition',
        'ROI bounds', '3x3x3 known and semantic erosion', 'semantic/binary/occupied run counting',
        'actual-time endpoint choice and onehot equivalence', 'unknown-domain exclusion',
        'integer histogram aggregation']), separators=(',', ':')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support-json', type=Path)
    parser.add_argument('--tables', type=Path, default=None)
    parser.add_argument('--max-anchors', type=int, default=32)
    parser.add_argument('--script-sha256', default=None)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if args.support_json is None:
        parser.error('--support-json is required unless --self-test')
    if not 1 <= args.max_anchors <= 32:
        parser.error('--max-anchors must be 1..32')
    if args.script_sha256 is not None and (len(args.script_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in args.script_sha256)):
        parser.error('--script-sha256 must be a lowercase SHA256 digest')
    try:
        print(json.dumps(run(args), separators=(',', ':'), allow_nan=False))
    except Exception as error:
        print(json.dumps(dict(schema_version='event-sequence-audit-v1', status='audit_failed',
            error=type(error).__name__ + ': ' + str(error), source_receipts=SOURCES), separators=(',', ':')))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
