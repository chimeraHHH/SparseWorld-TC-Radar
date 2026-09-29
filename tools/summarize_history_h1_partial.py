"""Validate and summarize the completed H1 half of the history/velocity experiment.

This is an offline analysis of saved JSON/scene confusion matrices. It neither
loads model weights nor launches training. The four-arm scientific decisions
remain unavailable until the matched H8 results exist.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_forecast_results import HORIZONS, MOVABLE, compare, load_confusions, semantic_iou

ARMS = ('h1-geometry', 'h1-velocity')
CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
           'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')
SAMPLES = 2000
SEED = 20260927
# This only covers algebraic float64 rounding when recomputing the SAME saved
# confusion matrices, not the separately authorized inference reproducibility gate.
RECOMPUTE_ATOL_PP = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label):
    require(np.allclose(actual, expected, rtol=0, atol=RECOMPUTE_ATOL_PP, equal_nan=True),
            'Saved confusion/JSON mismatch: ' + label)


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def remember(root, path, provenance):
    provenance[str(path.relative_to(root))] = sha256(path)


def load_evaluation(root, directory, mode, samples, provenance):
    path = root / directory / (mode + '.json')
    record = read_json(path)
    remember(root, path, provenance)
    require(record['checkpoint_meta'] == {'epoch': 10, 'iter': 29920},
            str(directory) + ': not the frozen epoch 10 checkpoint')
    require(record['mode'] == mode and record['samples'] == samples,
            str(directory) + ': wrong mode/sample count')
    indices = np.asarray(record['indices'])
    require(indices.shape == (samples,) and len(np.unique(indices)) == samples,
            str(directory) + ': duplicate/missing anchors')
    require(np.issubdtype(indices.dtype, np.integer), 'Anchors must be integer indices')
    require(np.all((indices >= 0) & (indices < 5119)), 'Anchor outside validation set')
    if samples == 5119:
        require(np.array_equal(indices, np.arange(5119)), 'Full anchors must be ordered 0..5118')
    elif samples == 256:
        expected = np.sort(np.random.RandomState(20260918).choice(5119, 256, replace=False))
        require(np.array_equal(indices, expected), 'Intervention anchors differ from frozen fixed 256')
    conf_dir = root / directory / ('confusions_' + mode)
    # Reuse the campaign loader, then perform explicit validation as its internal
    # checks use assertions, which can be disabled by python -O.
    arrays, names, anchors = load_confusions(conf_dir)
    require(arrays.shape == (4, len(names), 18, 18), 'Invalid semantic histogram shape')
    require(np.issubdtype(arrays.dtype, np.integer) and np.all(arrays >= 0),
            'Confusions must contain nonnegative integer counts')
    require(len(names) == len(np.unique(names)) and len(names) > 0, 'Invalid scene list')
    require(np.array_equal(indices, anchors), 'JSON/NPZ anchors differ')
    for number, horizon in enumerate(HORIZONS):
        npz_path = conf_dir / (horizon + '.npz')
        remember(root, npz_path, provenance)
        with np.load(npz_path, allow_pickle=False) as data:
            require(int(data['horizon']) == number * 2, 'Wrong future-frame index')
            require(np.array_equal(data['scene_names'], names), 'Horizon scene lists differ')
            require(np.array_equal(data['sample_indices'], anchors), 'Horizon anchor lists differ')
    require(set(record['metrics']) == set(HORIZONS), 'Metric horizon schema differs')
    totals = arrays.sum(1)
    class_iou = semantic_iou(totals)
    miou = np.nanmean(class_iou, axis=-1)
    binary = np.zeros((4, 2, 2), dtype=np.int64)
    binary[:, 0, 0] = totals[:, :17, :17].sum((1, 2))
    binary[:, 0, 1] = totals[:, :17, 17].sum(1)
    binary[:, 1, 0] = totals[:, 17, :17].sum(1)
    binary[:, 1, 1] = totals[:, 17, 17]
    binary_iou = semantic_iou(binary)[:, 0]
    metric_keys = {'Semantic mIoU', 'Binary IoU', 'evaluated_samples'} | {c + '_IoU' for c in CLASSES}
    for number, horizon in enumerate(HORIZONS):
        metric = record['metrics'][horizon]
        require(set(metric) == metric_keys, 'Metric class schema differs')
        require(metric['evaluated_samples'] == samples, 'Per-horizon sample count differs')
        close(class_iou[number], [metric[c + '_IoU'] for c in CLASSES], horizon + ' classes')
        close(miou[number], metric['Semantic mIoU'], horizon + ' mIoU')
        close(binary_iou[number], metric['Binary IoU'], horizon + ' binary IoU')
    close(np.nanmean(miou[1:]), record['future_mean_miou'], 'future mIoU')
    require(np.isfinite(record['elapsed_seconds']) and record['elapsed_seconds'] >= 0,
            'Invalid evaluation elapsed time')
    summary = {
        'checkpoint_name': Path(record['checkpoint']).name,
        'checkpoint_meta': record['checkpoint_meta'],
        'evaluation_git_revision': record['git_revision'],
        'mode': mode, 'samples': samples, 'scenes': len(names),
        'miou_by_horizon': miou, 'binary_iou_by_horizon': binary_iou,
        'class_iou_by_horizon': class_iou,
        'future_mean_miou': float(np.nanmean(miou[1:])),
        'evaluation_elapsed_seconds': record['elapsed_seconds'],
        'recomputed_metrics_match': True,
    }
    return arrays, names, anchors, summary


def load_checkpoint_audit(root, arm, provenance):
    path = root / (arm + '_recovery_checkpoint_audit.json')
    audit = read_json(path)
    remember(root, path, provenance)
    final = audit['final_audit']
    require(final['epoch'] == 10 and final['iterations'] == 29920,
            arm + ': saved checkpoint budget differs')
    require(final['finite_model_tensors'] == 771 and final['optimizer_finite'] is True
            and final['state_schema_matches_cpu'] is True, arm + ': invalid checkpoint audit')
    require(0 < final['optimizer_steps'] <= final['iterations'], 'Invalid optimizer step count')
    hashes = audit['checkpoint_sha256']
    require(hashes['model_state']['epoch_10.pth'] == hashes['model_state']['best_future.pth'],
            arm + ': best/final model content differs')
    init = audit['initialization']
    require(init['loaded_tensors'] == 669 and init['new_radar_tensors'] == 102
            and init['all_camera_tensors_exact'] is True and init['optimizer_restored'] is False
            and init['epoch_reset_to'] == 0, arm + ': initialization audit differs')
    return {
        'original_training_git_revision': audit['original_git_revision'],
        'evaluation_git_revision': audit['evaluation_git_revision'],
        'epoch': final['epoch'], 'iterations': final['iterations'],
        'optimizer_steps': final['optimizer_steps'],
        'finite_model_tensors': final['finite_model_tensors'],
        'optimizer_finite': final['optimizer_finite'],
        'state_schema_matches_cpu': final['state_schema_matches_cpu'],
        'best_and_final_model_content_equal': True,
        'checkpoint_sha256': hashes,
        'official_initialization_sha256': init['sha256'],
    }


def paired_details(reference, candidate):
    # The established compare function supplies the principal H1 contrast. Its
    # legacy engineering-threshold field is irrelevant to this frozen factorial
    # protocol and is deliberately not exported.
    principal = compare(reference, candidate, samples=SAMPLES, seed=SEED)
    principal.pop('engineering_threshold_pass')
    n = reference.shape[1]
    counts = np.random.RandomState(SEED).multinomial(n, np.full(n, 1 / n), size=SAMPLES)
    left = semantic_iou(np.einsum('sn,hnij->shij', counts, reference, optimize=True))
    right = semantic_iou(np.einsum('sn,hnij->shij', counts, candidate, optimize=True))
    delta = right - left
    horizon_delta = np.nanmean(right, axis=-1) - np.nanmean(left, axis=-1)
    future_delta = np.nanmean(horizon_delta[:, 1:], axis=-1)
    close(np.percentile(future_delta, [2.5, 97.5]),
          principal['future_delta_scene_bootstrap_ci95'], 'paired bootstrap future interval')
    principal.update({
        'contrast': 'h1-velocity minus h1-geometry',
        'miou_delta_pp_by_horizon': np.asarray(principal['candidate_miou_by_horizon'])
                                    - np.asarray(principal['reference_miou_by_horizon']),
        'miou_delta_ci95_by_horizon': np.percentile(horizon_delta, [2.5, 97.5], axis=0).T,
        'class_iou_delta_ci95_by_horizon': np.percentile(delta, [2.5, 97.5], axis=0).transpose(1, 2, 0),
        'future_class_mean_delta_pp': np.nanmean(semantic_iou(candidate.sum(1))[1:], axis=0)
                                      - np.nanmean(semantic_iou(reference.sum(1))[1:], axis=0),
        'future_class_mean_delta_ci95': np.percentile(np.nanmean(delta[:, 1:], axis=1),
                                                     [2.5, 97.5], axis=0).T,
        'descriptive_future_movable_delta_ci95': np.percentile(
            np.nanmean(right[:, 1:, MOVABLE], axis=(1, 2))
            - np.nanmean(left[:, 1:, MOVABLE], axis=(1, 2)), [2.5, 97.5]),
        'class_and_horizon_intervals_are_descriptive_unadjusted': True,
    })
    return principal


def summarize(root):
    provenance, arms, loaded = {}, {}, {}
    for arm in ARMS:
        array, names, anchors, summary = load_evaluation(
            root, arm + '_final_full', 'normal', 5119, provenance)
        require(summary['checkpoint_name'] == 'epoch_10.pth', 'Full evaluation must use epoch 10')
        summary['checkpoint_audit'] = load_checkpoint_audit(root, arm, provenance)
        require(summary['evaluation_git_revision'] == summary['checkpoint_audit']['evaluation_git_revision'],
                'Evaluation and checkpoint audit revisions differ')
        arms[arm] = summary
        loaded[arm] = (array, names, anchors)
    left, left_names, left_anchors = loaded[ARMS[0]]
    right, right_names, right_anchors = loaded[ARMS[1]]
    require(np.array_equal(left_names, right_names) and np.array_equal(left_anchors, right_anchors),
            'H1 comparison requires identical ordered scenes/anchors')
    require(np.array_equal(left.sum(-1), right.sum(-1)), 'Target counts differ between arms')
    require(arms[ARMS[0]]['evaluation_git_revision'] == arms[ARMS[1]]['evaluation_git_revision'],
            'Full evaluation revisions differ between arms')
    interventions, costs = {}, {}
    for arm in ARMS:
        directory = arm + '_interventions'
        modes = ('normal', 'drop', 'zero_velocity', 'shuffle_velocity') if arm.endswith('velocity') else ('normal', 'drop')
        available = [mode for mode in modes if (root / directory / (mode + '.json')).exists()]
        entry = {'available_modes': available, 'pending_modes': [m for m in modes if m not in available],
                 'scope': 'fixed 256 subset; descriptive input interventions on trained weights', 'modes': {}}
        base = None
        for mode in available:
            array, names, anchors, summary = load_evaluation(root, directory, mode, 256, provenance)
            require(summary['checkpoint_name'] == 'best_future.pth', 'Intervention checkpoint differs')
            require(summary['evaluation_git_revision'] == arms[arm]['evaluation_git_revision'],
                    'Intervention and full evaluation revisions differ')
            if base is None:
                require(mode == 'normal', 'Available interventions require the normal reference')
                base = (array, names, anchors, summary)
            require(np.array_equal(base[1], names) and np.array_equal(base[2], anchors),
                    'Intervention scene/anchor lists differ')
            require(np.array_equal(base[0].sum(-1), array.sum(-1)), 'Intervention targets differ')
            summary['future_delta_pp_vs_same_subset_normal'] = summary['future_mean_miou'] - base[3]['future_mean_miou']
            summary['current_delta_pp_vs_same_subset_normal'] = summary['miou_by_horizon'][0] - base[3]['miou_by_horizon'][0]
            entry['modes'][mode] = summary
        interventions[arm] = entry
        receipt = root / (arm + '_result.json')
        if not receipt.exists():
            receipt = root / (arm + '_status.json')
        cost = {'full_evaluation_elapsed_seconds': arms[arm]['evaluation_elapsed_seconds'],
                'completed_recovery_stage_seconds': {},
                'scope': 'Evaluation recovery only; excludes original training, earlier failed evaluations, pre-subprocess GPU/host admission queue time, and diagnostic/preflight work. Any lock waits inside a subprocess remain included. Not total experiment cost.'}
        if receipt.exists():
            record = read_json(receipt)
            remember(root, receipt, provenance)
            cost['completed_recovery_stage_seconds'] = record.get('stage_seconds', {})
            require(all(np.isfinite(v) and v >= 0 for v in cost['completed_recovery_stage_seconds'].values()),
                    'Invalid recovery stage timing')
        cost['completed_intervention_evaluation_seconds_by_mode'] = {
            mode: report['evaluation_elapsed_seconds'] for mode, report in entry['modes'].items()}
        cost['timing_note'] = 'Stage wall times include subprocess overhead; evaluation elapsed time is nested inside and must not be added to stage wall time.'
        costs[arm] = cost
    return {
        'schema_version': 1, 'partial_result': True, 'campaign_complete': False,
        'primary_scope': 'fixed epoch 10, complete 5119 validation anchors, H1 contrast only',
        'horizons': HORIZONS, 'class_names': CLASSES,
        'movable_class_names': [CLASSES[i] for i in MOVABLE],
        'validation': {'samples': 5119, 'scenes': len(left_names),
                       'all_horizons_and_arms_ordered_scene_anchor_lists_identical': True,
                       'target_histogram_marginals_identical_between_arms': True,
                       'all_json_metrics_recomputed': True, 'same_histogram_float64_rounding_atol_pp': RECOMPUTE_ATOL_PP},
        'arms': arms, 'velocity_gain_short': paired_details(left, right),
        'pending_four_arm_decisions': {
            'h8_results': 'pending; this summary contains no H8 evidence',
            'interaction_short_minus_long': {'assessable': False, 'delta_pp': None, 'ci95': None},
            'short_velocity_minus_long_velocity': {'assessable': False, 'delta_pp': None, 'ci95': None,
                                                   'prespecified_noninferiority_lower_bound_pp': -0.3},
            'history_replacement_claim_supported': False,
        },
        'interventions_fixed256': interventions, 'recovery_evaluation_costs': costs,
        'limitations': [
            'Single seed; scene bootstrap conditions on these fixed trained checkpoints and does not estimate retraining uncertainty.',
            'Fixed 256 selection/intervention anchors are within the 5119 validation anchors; this is reused validation, not blinded test evidence.',
            'H1 encodes current six camera views and repeats features/projections into eight slots; it is not a true eight-frame input.',
            'Velocity channels are nuScenes SDK compensated vx/vy and derived radial velocity from one causal sweep, not raw independent scalar Doppler.',
            'Positive H1 velocity gain alone cannot establish reduced dependence on visual history; matched H8 contrasts and costs remain required.',
            'Per-class and per-horizon confidence intervals are descriptive and have no multiplicity correction.',
        ],
        'source_files_sha256': dict(sorted(provenance.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = safe(summarize(Path(args.evidence_dir)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'partial_result': True,
                      'future_mean_miou': {a: result['arms'][a]['future_mean_miou'] for a in ARMS},
                      'delta_pp': result['velocity_gain_short']['future_mean_delta_pp'],
                      'ci95': result['velocity_gain_short']['future_delta_scene_bootstrap_ci95'],
                      'h8_decisions_assessable': False}, indent=2))


if __name__ == '__main__':
    main()
