#!/usr/bin/env python3
"""CPU-only diagnostics for exported, frozen A occupancy predictions.

Reads immutable sparse predictions, Occ3D labels and endpoint pose metadata.
No model is loaded, trained or changed. Static transport uses the actual ego
poses; it is NOT an estimate of object motion. Future labels are used only to
define diagnostic strata and explicitly labelled GT oracles.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


GRID_SHAPE = (200, 200, 16)
PC_RANGE = np.array([-40., -40., -1., 40., 40., 5.4])
VOXEL_SIZE = np.array([.4, .4, .4])
FUTURE_FRAMES = [0, 2, 4, 6]
FREE = 17
CLASSES = 18
MOVABLE = np.array([2, 3, 4, 5, 6, 7, 9, 10], dtype=np.int64)
CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free',
]
STRATA = [
    '1_outside_current_roi', '2_current_unknown', '3_semantic_persistent',
    '4_free_to_occupied', '5_occupied_to_free', '6_occupied_semantic_change',
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_export_selection(manifest, max_samples):
    if manifest.get('state') != 'complete':
        raise ValueError('Prediction export must have state=complete; partial exports are not analyzable')
    samples = manifest.get('samples')
    if not isinstance(samples, list) or not samples or len(samples) > max_samples:
        raise ValueError('Expected 1..max-samples explicitly exported anchors; refusing silent truncation')
    if any(type(s.get('dataset_index')) is not int or s['dataset_index'] < 0 or not isinstance(s.get('token'), str) for s in samples):
        raise ValueError('Every sample must contain a nonnegative integer dataset_index and string token')
    indices = [s['dataset_index'] for s in samples]
    tokens = [s['token'] for s in samples]
    if manifest.get('indices') != indices or manifest.get('tokens') != tokens:
        raise ValueError('Manifest indices/tokens must exactly match samples in order and length')
    if len(set(indices)) != len(samples) or len(set(tokens)) != len(samples):
        raise ValueError('Exported anchor tokens and dataset indices must be unique')
    return samples


def verify_selected_frame_hashes(manifest, samples, cache_root, split):
    metadata = manifest.get('target_metadata', {})
    if 'selected_frame_sha256' not in metadata:
        return None
    expected = metadata['selected_frame_sha256']
    selected = {token for sample in samples for token in sample['future_tokens']}
    if not isinstance(expected, dict) or set(expected) != selected:
        raise ValueError('Selected endpoint hashes must cover exactly the exported future tokens')
    for token in sorted(selected):
        if sha256(cache_root / split / 'frames' / (token + '.npz')) != expected[token]:
            raise ValueError('Selected endpoint frame checksum mismatch: ' + token)
    return len(selected)


def checked_integer_array(value, name, ndim=None):
    value = np.asarray(value)
    if value.dtype.kind not in 'iu' or (ndim is not None and value.ndim != ndim):
        raise ValueError(name + ' must have an integer dtype and expected rank')
    return value


def sparse_prediction(arrays, horizon):
    """Use the evaluation convention: unlisted voxels are predicted free."""
    loc = checked_integer_array(arrays['occ_loc_' + str(horizon)], 'occ_loc', 2)
    sem = checked_integer_array(arrays['sem_pred_' + str(horizon)], 'sem_pred', 1)
    if loc.shape != (len(sem), 3):
        raise ValueError('Sparse locations and semantics have inconsistent shapes')
    if ((loc < 0) | (loc >= np.asarray(GRID_SHAPE))).any():
        raise ValueError('Sparse prediction falls outside the fixed grid')
    if ((sem < 0) | (sem >= CLASSES)).any():
        raise ValueError('Predicted semantic labels must be in 0..17')
    dense = np.full(np.prod(GRID_SHAPE), FREE, dtype=np.uint8)
    if len(sem):
        flat = np.ravel_multi_index(loc.T, GRID_SHAPE)
        order = np.argsort(flat, kind='stable')
        ids, labels = flat[order], sem[order]
        if ((ids[1:] == ids[:-1]) & (labels[1:] != labels[:-1])).any():
            raise ValueError('Conflicting duplicate sparse voxels have no safe CPU decoding')
        dense[flat] = sem
    return dense.reshape(GRID_SHAPE)


def load_labels(path):
    with np.load(path, allow_pickle=False) as data:
        labels = checked_integer_array(data['semantics'], 'semantics', 3).copy()
        raw_mask = np.asarray(data['mask_camera'])
        if labels.shape != GRID_SHAPE or raw_mask.shape != GRID_SHAPE:
            raise ValueError('Occ3D grid must be exactly 200 x 200 x 16: ' + str(path))
        if not np.isin(raw_mask, [0, 1]).all():
            raise ValueError('Camera mask must contain only zero/one values')
        camera = raw_mask.astype(bool)
    valid = (labels >= 0) & (labels < CLASSES)
    return labels, camera & valid, int((camera & ~valid).sum())


def load_frame(path):
    with np.load(path, allow_pickle=False) as data:
        result = {
            'rotation': np.asarray(data['ego2global_rotation'], dtype=np.float64),
            'translation': np.asarray(data['ego2global_translation'], dtype=np.float64),
            'scene_name': str(np.asarray(data['scene_name']).item()),
            'timestamp_us': int(np.asarray(data['timestamp_us']).item()),
        }
    rotation, translation = result['rotation'], result['translation']
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError('Invalid ego pose shapes: ' + str(path))
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError('Nonfinite ego pose: ' + str(path))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
        raise ValueError('Ego pose rotation is not a proper orthogonal matrix')
    return result


def future_to_current(current, future):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = current['rotation'].T @ future['rotation']
    transform[:3, 3] = current['rotation'].T @ (future['translation'] - current['translation'])
    return transform


def backproject_indices(flat_indices, transform):
    """Future voxel centers -> global -> current ego -> nearest current center.

    Index floor((x-lower)/spacing) is nearest-center lookup; exact ties go to
    the higher index. Bounds are checked before any clipping for safe lookup.
    """
    indices = np.stack(np.unravel_index(flat_indices, GRID_SHAPE), axis=1)
    centers = (indices + .5) * VOXEL_SIZE + PC_RANGE[:3]
    current_centers = centers @ transform[:3, :3].T + transform[:3, 3]
    inside = ((current_centers >= PC_RANGE[:3]) & (current_centers < PC_RANGE[3:])).all(axis=1)
    current_indices = np.floor((current_centers - PC_RANGE[:3]) / VOXEL_SIZE).astype(np.int64)
    # The clipped indices are placeholders only; callers must apply inside.
    current_indices = np.clip(current_indices, 0, np.asarray(GRID_SHAPE) - 1)
    current_flat = np.ravel_multi_index(current_indices.T, GRID_SHAPE)
    return current_flat, inside


def neighborhood_stable(labels, known):
    """Full 3x3x3 neighbourhood must be known and share the center's label.

    Outer grid cells are excluded, as are any neighbourhoods touching unknown
    labels/masks. Two-frame stability is the intersection of this property in
    the future grid and at the mapped current cell; it does not assert motion.
    """
    if labels.shape != GRID_SHAPE or known.shape != GRID_SHAPE:
        raise ValueError('Stable-neighbourhood arrays violate the grid contract')
    stable = np.zeros(GRID_SHAPE, dtype=bool)
    middle = tuple(slice(1, n - 1) for n in GRID_SHAPE)
    center = labels[middle]
    interior = known[middle].copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                neighbor = tuple(slice(1 + d, n - 1 + d) for d, n in zip((dx, dy, dz), GRID_SHAPE))
                interior &= known[neighbor] & (labels[neighbor] == center)
    stable[middle] = interior
    return stable


def stratify(current_labels, future_labels, inside, current_known):
    result = np.empty(len(future_labels), dtype=np.uint8)
    result[~inside] = 0
    result[inside & ~current_known] = 1
    common = inside & current_known
    result[common & (current_labels == future_labels)] = 2
    result[common & (current_labels == FREE) & (future_labels != FREE)] = 3
    result[common & (current_labels != FREE) & (future_labels == FREE)] = 4
    result[common & (current_labels != FREE) & (future_labels != FREE) & (current_labels != future_labels)] = 5
    return result, common


def robust_enter_release_masks(layers, current_stable, future_stable):
    """Post-hoc boundary check: require stability only on the free side.

    Strata 3/4 (zero-based) already guarantee common-known free->occupied and
    occupied->free respectively. The occupied side may be a thin surface.
    Occupied semantic changes are deliberately excluded from this new check.
    """
    enter = (layers == 3) & current_stable
    release = (layers == 4) & future_stable
    return enter, release, enter | release


def confusion(gt, pred):
    if len(gt) == 0:
        return np.zeros((CLASSES, CLASSES), dtype=np.int64)
    return np.bincount(CLASSES * gt.astype(np.int64) + pred.astype(np.int64), minlength=CLASSES ** 2).reshape(CLASSES, CLASSES)


def finite_mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def metric_summary(hist, include_confusion=True):
    """Exactly match old_metrics.py's absent-GT-class exclusion convention."""
    support = hist.sum(axis=1)
    union = support + hist.sum(axis=0) - np.diag(hist)
    iou = np.full(CLASSES, np.nan)
    np.divide(np.diag(hist), union, out=iou, where=(union > 0) & (support > 0))
    iou[support == 0] = np.nan
    as_percent = [float(v * 100) if np.isfinite(v) else None for v in iou]
    count = int(hist.sum())
    occupied_tp = int(hist[:FREE, :FREE].sum())
    occupied_union = count - int(hist[FREE, FREE])
    result = {
        'voxels': count,
        'correct': int(np.trace(hist)),
        'errors': count - int(np.trace(hist)),
        'accuracy_percent': float(np.trace(hist) / count * 100) if count else None,
        'semantic_miou_percent': finite_mean(as_percent[:FREE]),
        'movable_class_miou_percent': finite_mean([as_percent[i] for i in MOVABLE]),
        'binary_occupied_iou_percent': float(occupied_tp / occupied_union * 100) if occupied_union and support[:FREE].sum() else None,
        'class_gt_support': support.tolist(),
        'class_pred_support': hist.sum(axis=0).tolist(),
        'class_iou_percent': as_percent,
    }
    if include_confusion:
        result['confusion_gt_rows_pred_columns'] = hist.tolist()
    return result


class HorizonAccumulator:
    def __init__(self):
        self.histograms = {}
        self.mapping_counts = {
            'future_camera_mask_invalid_label': 0,
            'future_camera_known': 0,
            'common_known': 0,
            'conservative_common_known': 0,
            'semantic_transition': 0,
            'conservative_semantic_transition': 0,
            'semantic_transition_future_occupied': 0,
            'semantic_transition_future_movable': 0,
            'conservative_semantic_transition_future_occupied': 0,
            'conservative_semantic_transition_future_movable': 0,
            'robust_enter': 0,
            'robust_release': 0,
            'robust_enter_release': 0,
        }
        self.seconds = []

    def add(self, domain, method, gt, pred, mask=None):
        key = (domain, method)
        if key not in self.histograms:
            self.histograms[key] = np.zeros((CLASSES, CLASSES), dtype=np.int64)
        if mask is not None:
            gt, pred = gt[mask], pred[mask]
        self.histograms[key] += confusion(gt, pred)

    def summary(self):
        domains = {}
        for (domain, method), hist in sorted(self.histograms.items()):
            domains.setdefault(domain, {})[method] = metric_summary(hist)
        for base, changed in [('future_camera_known', 'oracle_replace_transitions'), ('future_camera_known', 'oracle_replace_conservative_transitions'), ('future_camera_known', 'oracle_replace_robust_enter_release'), ('common_known', 'oracle_replace_robust_enter_release')]:
            baseline = domains[base]['A']['semantic_miou_percent']
            alternative = domains[base][changed]['semantic_miou_percent']
            domains[base][changed]['delta_vs_A_pp'] = alternative - baseline if baseline is not None and alternative is not None else None
        transition_composition = {}
        for prefix in ('', 'conservative_'):
            denominator = self.mapping_counts[prefix + 'semantic_transition']
            transition_composition[prefix + 'semantic_transition'] = {
                'voxels': denominator,
                'future_gt_occupied_fraction': self.mapping_counts[prefix + 'semantic_transition_future_occupied'] / denominator if denominator else None,
                'future_gt_movable_fraction': self.mapping_counts[prefix + 'semantic_transition_future_movable'] / denominator if denominator else None,
            }
        return {
            'anchors': len(self.seconds),
            'actual_seconds_min_max': [min(self.seconds), max(self.seconds)],
            'mapping_counts': self.mapping_counts,
            'transition_composition': transition_composition,
            'domains': domains,
        }


def analyze_sample(sample, manifest_root, occ_root, cache_root, split, anchors, accumulators, chunk_size):
    token = sample['token']
    tokens = list(sample['future_tokens'])
    if len(tokens) != len(FUTURE_FRAMES) or tokens[0] != token or anchors.get(token) != tokens:
        raise ValueError('Prediction future tokens disagree with the frozen endpoint anchor index')
    prediction_path = manifest_root / sample['file']
    prediction_sha256 = sha256(prediction_path)
    if sample.get('sha256') != prediction_sha256:
        raise ValueError('Exported prediction checksum mismatch: ' + str(prediction_path))
    frames = [load_frame(cache_root / split / 'frames' / (t + '.npz')) for t in tokens]
    scene = frames[0]['scene_name']
    if any(frame['scene_name'] != scene for frame in frames) or sample.get('scene_name', scene) != scene:
        raise ValueError('Anchor future sequence crosses scenes or disagrees with export')
    times = np.array([frame['timestamp_us'] for frame in frames], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError('Future timestamps are not increasing')
    transforms = np.stack([future_to_current(frames[0], frame) for frame in frames])
    labels0, known0, _ = load_labels(occ_root / scene / tokens[0] / 'labels.npz')
    stable0 = neighborhood_stable(labels0, known0).reshape(-1)
    labels0, known0 = labels0.reshape(-1), known0.reshape(-1)
    seconds = (times - times[0]) / 1e6
    sample_result = {
        'dataset_index': int(sample['dataset_index']), 'token': token,
        'future_tokens': tokens, 'scene_name': scene,
        'prediction_file': str(prediction_path.resolve()),
        'prediction_sha256': prediction_sha256,
        'actual_seconds': seconds.tolist(), 'horizons': {},
    }
    with np.load(prediction_path, allow_pickle=False) as data:
        if 'fut2cur' not in data or data['fut2cur'].shape != (4, 4, 4) or not np.allclose(data['fut2cur'], transforms, atol=5e-4, rtol=1e-5):
            raise ValueError('Exported model-input fut2cur disagrees with independent endpoint poses')
        if 'fut_list' in data and data['fut_list'].tolist() != FUTURE_FRAMES:
            raise ValueError('Exported future indices disagree with protocol')
        if 'future_tokens' in data and data['future_tokens'].tolist() != tokens:
            raise ValueError('NPZ future tokens disagree with manifest')
        if 'lidar_timestamps_us' in data and not np.array_equal(data['lidar_timestamps_us'], times):
            raise ValueError('Exported LIDAR timestamps disagree with endpoint cache')
        if 'actual_seconds' in data and not np.allclose(data['actual_seconds'], seconds, atol=1e-6, rtol=0):
            raise ValueError('Exported elapsed seconds disagree with endpoint cache')
        pred0 = sparse_prediction(data, 0).reshape(-1)
        for h, frame_index in enumerate(FUTURE_FRAMES):
            gt_grid, known_grid, invalid_count = load_labels(occ_root / scene / tokens[h] / 'labels.npz')
            stable_future = neighborhood_stable(gt_grid, known_grid).reshape(-1)
            gt = gt_grid.reshape(-1)
            pred = sparse_prediction(data, h).reshape(-1)
            target_indices = np.flatnonzero(known_grid.reshape(-1))
            if not len(target_indices):
                raise ValueError('No valid camera-known labels for target ' + tokens[h])
            acc = accumulators[h]
            acc.seconds.append(float(seconds[h]))
            acc.mapping_counts['future_camera_mask_invalid_label'] += invalid_count
            local_counts = np.zeros(len(STRATA), dtype=np.int64)
            local_errors = np.zeros(len(STRATA), dtype=np.int64)
            local_common = local_conservative = 0
            local_robust_counts = np.zeros(3, dtype=np.int64)
            local_robust_errors = np.zeros(3, dtype=np.int64)
            for start in range(0, len(target_indices), chunk_size):
                target = target_indices[start:start + chunk_size]
                source, inside = backproject_indices(target, transforms[h])
                gt_future, a = gt[target], pred[target]
                gt_current = labels0[source]
                layers, common = stratify(gt_current, gt_future, inside, known0[source])
                conservative = common & stable_future[target] & stable0[source]
                transition = common & (gt_current != gt_future)
                conservative_transition = transition & conservative
                robust_enter, robust_release, robust_union = robust_enter_release_masks(layers, stable0[source], stable_future[target])
                robust_domains = [('robust_enter', robust_enter), ('robust_release', robust_release), ('robust_enter_release', robust_union)]
                future_movable = np.isin(gt_future, MOVABLE)
                movable_related = future_movable | (common & np.isin(gt_current, MOVABLE))
                local_counts += np.bincount(layers, minlength=len(STRATA))
                local_errors += np.bincount(layers[a != gt_future], minlength=len(STRATA))
                local_common += int(common.sum())
                local_conservative += int(conservative.sum())
                acc.mapping_counts['future_camera_known'] += len(target)
                acc.mapping_counts['common_known'] += int(common.sum())
                acc.mapping_counts['conservative_common_known'] += int(conservative.sum())
                acc.mapping_counts['semantic_transition'] += int(transition.sum())
                acc.mapping_counts['conservative_semantic_transition'] += int(conservative_transition.sum())
                for i, (domain, selection) in enumerate(robust_domains):
                    selected_count = int(selection.sum())
                    acc.mapping_counts[domain] += selected_count
                    local_robust_counts[i] += selected_count
                    local_robust_errors[i] += int((selection & (a != gt_future)).sum())
                for prefix, selected in [('', transition), ('conservative_', conservative_transition)]:
                    acc.mapping_counts[prefix + 'semantic_transition_future_occupied'] += int((selected & (gt_future != FREE)).sum())
                    acc.mapping_counts[prefix + 'semantic_transition_future_movable'] += int((selected & future_movable).sum())
                acc.add('future_camera_known', 'A', gt_future, a)
                for domain, selection in [('common_known', common), ('conservative_common_known', conservative)] + robust_domains:
                    for method, candidate in [('A', a), ('transported_pred0', pred0[source]), ('oracle_transported_GT0', gt_current)]:
                        acc.add(domain, method, gt_future, candidate, selection)
                    for suffix, subset in [('future_gt_movable', future_movable), ('movable_related', movable_related)]:
                        for method, candidate in [('A', a), ('transported_pred0', pred0[source]), ('oracle_transported_GT0', gt_current)]:
                            acc.add(domain + '/' + suffix, method, gt_future, candidate, selection & subset)
                for i, name in enumerate(STRATA):
                    selection = layers == i
                    for suffix, subset in [('all', np.ones(len(target), dtype=bool)), ('future_gt_movable', future_movable), ('movable_related', movable_related), ('conservative', conservative), ('conservative_movable_related', conservative & movable_related)]:
                        acc.add('strata/' + name + '/' + suffix, 'A', gt_future, a, selection & subset)
                for name, selection in [('oracle_replace_transitions', transition), ('oracle_replace_conservative_transitions', conservative_transition), ('oracle_replace_robust_enter_release', robust_union)]:
                    corrected = np.where(selection, gt_future, a)
                    acc.add('future_camera_known', name, gt_future, corrected)
                    if name == 'oracle_replace_robust_enter_release':
                        acc.add('common_known', name, gt_future, corrected, common)
            if int(local_counts.sum()) != len(target_indices):
                raise AssertionError('Six diagnostic strata do not partition the evaluated domain')
            sample_result['horizons'][str(frame_index)] = {
                'known_voxels': len(target_indices), 'common_known_voxels': local_common,
                'conservative_common_known_voxels': local_conservative,
                'A_errors': int(local_errors.sum()),
                'stratum_voxels': dict(zip(STRATA, local_counts.tolist())),
                'stratum_A_errors': dict(zip(STRATA, local_errors.tolist())),
                'posthoc_robust_voxels': dict(zip(['robust_enter', 'robust_release', 'robust_enter_release'], local_robust_counts.tolist())),
                'posthoc_robust_A_errors': dict(zip(['robust_enter', 'robust_release', 'robust_enter_release'], local_robust_errors.tolist())),
            }
    return sample_result


def analyze(args):
    started = time.monotonic()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    samples = validate_export_selection(manifest, args.max_samples)
    if manifest.get('future_frames') != FUTURE_FRAMES:
        raise ValueError('Only frozen [0,2,4,6] frame-index protocol is supported')
    for key, expected in [('grid_shape', GRID_SHAPE), ('voxel_size', VOXEL_SIZE), ('pc_range', PC_RANGE)]:
        value = np.asarray(manifest.get(key, []))
        if value.shape != np.asarray(expected).shape or not np.allclose(value, expected, atol=1e-7, rtol=0):
            raise ValueError('Manifest ' + key + ' does not match the fixed Occ3D contract')
    cache = Path(args.endpoint_cache).resolve()
    cache_manifest = json.loads((cache / 'manifest.json').read_text())
    anchor_file = cache / args.split / 'anchors.json'
    if cache_manifest.get('schema_version') != 'endpoint-segments-v1' or cache_manifest.get('horizons') != FUTURE_FRAMES:
        raise ValueError('Endpoint cache uses an incompatible protocol')
    if sha256(anchor_file) != cache_manifest['splits'][args.split]['anchors_sha256']:
        raise ValueError('Endpoint anchor index checksum mismatch')
    checked_frame_hashes = verify_selected_frame_hashes(manifest, samples, cache, args.split)
    anchors = json.loads(anchor_file.read_text())
    accumulators = [HorizonAccumulator() for _ in FUTURE_FRAMES]
    per_sample = []
    for sample in samples:
        per_sample.append(analyze_sample(sample, manifest_path.parent, Path(args.occ_root).resolve(), cache, args.split, anchors, accumulators, args.chunk_size))
        print(json.dumps({'completed_anchors': len(per_sample), 'total_anchors': len(samples), 'token': sample['token']}, ensure_ascii=False), flush=True)
    horizons = {str(frame): acc.summary() for frame, acc in zip(FUTURE_FRAMES, accumulators)}
    future_means = {}
    for domain, methods in horizons['2']['domains'].items():
        if domain.startswith('strata/'):
            continue
        future_means[domain] = {}
        for method in methods:
            values = [horizons[str(frame)]['domains'][domain][method]['semantic_miou_percent'] for frame in FUTURE_FRAMES[1:]]
            valid_horizons = sum(value is not None for value in values)
            future_means[domain][method] = {
                'semantic_miou_percent': finite_mean(values) if valid_horizons == len(FUTURE_FRAMES) - 1 else None,
                'per_horizon_semantic_miou_percent': values,
                'valid_horizon_count': valid_horizons,
            }
    report = {
        'schema_version': 'frozen-a-static-transport-diagnostic-v3',
        'created_unix': time.time(), 'elapsed_seconds': time.monotonic() - started,
        'provenance_checks': {'export_state_complete': True, 'ordered_selection_matches': True,
                              'prediction_sha256_verified': len(samples),
                              'selected_endpoint_frame_sha256_verified': checked_frame_hashes},
        'sources': {'prediction_manifest': str(manifest_path), 'prediction_manifest_sha256': sha256(manifest_path),
                    'endpoint_manifest': str(cache / 'manifest.json'), 'endpoint_manifest_sha256': sha256(cache / 'manifest.json'),
                    'endpoint_anchor_sha256': sha256(anchor_file), 'occ_root': str(Path(args.occ_root).resolve()),
                    'analysis_script_sha256': sha256(__file__)},
        'protocol': {
            'grid_shape': list(GRID_SHAPE), 'pc_range': PC_RANGE.tolist(), 'voxel_size': VOXEL_SIZE.tolist(),
            'future_frames': FUTURE_FRAMES, 'nominal_seconds': [0, 1, 2, 3],
            'class_names': CLASS_NAMES, 'movable_class_ids': MOVABLE.tolist(),
            'free_label': FREE, 'evaluated_samples': len(samples),
            'known': 'mask_camera AND semantic label in 0..17; unknown is never free',
            'transport': 'Backward sample future voxel centers using real future-to-current ego poses and nearest current voxel center; this is static-world pose transport, not object motion.',
            'conservative': 'Both sampled current voxel and future voxel have complete known, same-label 3x3x3 neighbourhoods; grid boundaries excluded. Labels may differ between the two frames.',
            'posthoc_robust_enter_release': {
                'status': 'Post-hoc exploratory sensitivity analysis; original strata and strict conservative statistics are retained unchanged.',
                'reason': 'The strict two-sided same-label neighbourhood filter had almost no occupied GT support in the initial 32-anchor diagnostic, so its mIoU or oracle gain cannot reject a research direction.',
                'robust_enter': 'Common-known free current GT -> occupied future GT, with the current 3x3x3 neighbourhood fully known and free. No same-label neighbourhood requirement on the occupied future side.',
                'robust_release': 'Common-known occupied current GT -> free future GT, with the future 3x3x3 neighbourhood fully known and free. No same-label neighbourhood requirement on the occupied current side.',
                'oracle': 'Replace A only on the union of robust enter/release with exact future GT; report both the unchanged future-camera-known domain and the matched common-known domain.',
                'excluded': 'Occupied-to-different-occupied semantic changes are not included.',
            },
            'movable_related': 'Future GT is a listed movable class OR common-known transported current GT is; this includes released voxels. It does not establish that an entity moved.',
            'metric': 'Confusions pooled across anchors separately for each horizon/domain. IoU is NaN/null when GT class support is zero, matching old_metrics.py, and semantic mIoU averages classes 0..16. Domain mIoUs are not additive.',
            'future_mean': 'Arithmetic mean of all three horizon mIoUs; null if any horizon has no supported occupied class. Never a mean of stratum mIoUs. Class supports are reported; support differences can make subset mIoUs incomparable.',
            'common_domain': 'A, transported_pred0 and oracle_transported_GT0 are compared only where both frames are camera-known after pose lookup; no invented labels outside current ROI.',
        },
        'limitations': [
            'This is a small exported diagnostic subset, not the full 5119 evaluation and not evidence of a new model benefit.',
            'Nearest-voxel pose resampling, camera visibility, annotations and discretization can create apparent semantic changes. The conservative subset reduces boundary sensitivity but does not prove physical change and underrepresents small/thin objects.',
            'A future query has no guaranteed instance identity. Semantic union occupancy does not identify collision, interaction order or individual worldlines; no collision rate is computed.',
            'GT current occupancy is unavailable to deployed prediction: oracle_transported_GT0 is a privileged diagnostic, not a fair initialization baseline.',
            'Oracle replacement uses exact future labels only in identified common-known transition regions, retaining A elsewhere. It is an optimistic upper bound for edits restricted to that region, only for rejection/localization; it cannot positively establish the value of ordering, a learned transition model, or causal motion.',
            'Semantic persistence includes free persistence and movement of indistinguishable same-class instances. Enter/release refer to per-voxel label transitions, not object birth/death.',
            'Robust enter/release is a post-hoc exploratory check designed after observing sparse occupied support under the original strict filter. Free-side neighbourhood stability reduces one-voxel boundary sensitivity without discarding thin occupied surfaces, but is neither a guarantee against pose/resampling errors nor evidence of physical motion or interaction. These data-adaptive results are not confirmatory.',
        ],
        'per_horizon': horizons, 'future_horizon_mean': future_means, 'per_sample': per_sample,
    }
    output = Path(args.output)
    if not output.parent.exists():
        raise FileNotFoundError('Create the report parent directory explicitly first: ' + str(output.parent))
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output.resolve()), 'anchors': len(samples), 'elapsed_seconds': report['elapsed_seconds'], 'future_horizon_mean': future_means}, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Frozen prediction export manifest JSON')
    parser.add_argument('--occ-root', default='datasets/occ3d/gts')
    parser.add_argument('--endpoint-cache', default='/home/huayiming/Workspace/SparseWorld-cache/endpoint_segments_v1_20260926')
    parser.add_argument('--split', choices=['val'], default='val')
    parser.add_argument('--output', required=True, help='New report JSON; existing files are never overwritten')
    parser.add_argument('--chunk-size', type=int, default=65536)
    parser.add_argument('--max-samples', type=int, default=32)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.max_samples < 1:
        parser.error('chunk-size and max-samples must be positive')
    if Path(args.output).exists():
        parser.error('output already exists; preserve evidence and choose a new path')
    analyze(args)


if __name__ == '__main__':
    main()
