"""Audit exported, completed history-budget evidence on CPU only.

This does not load checkpoints, execute a model, or run bootstrap comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_forecast_results import HORIZONS, load_confusions, semantic_iou

CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
           'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
           'vegetation')
REVISION = '0f33492d1b639da716897d8faa4a1df293354c49'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label, atol=1e-10):
    require(np.allclose(actual, expected, atol=atol, rtol=0, equal_nan=False), label)


def audit_metrics(record_path, confusion_dir, samples, indices):
    record = read(record_path)
    require(record['samples'] == samples and record['indices'] == indices,
            'Wrong sample count/order: ' + record_path.name)
    require(set(record['metrics']) == set(HORIZONS), 'Wrong horizon schema')
    arrays, names, anchors = load_confusions(confusion_dir)
    require(np.array_equal(anchors, indices) and len(np.unique(names)) == len(names),
            'Wrong anchor/scene identity')
    require(arrays.shape == (4, len(names), 18, 18)
            and np.issubdtype(arrays.dtype, np.integer) and np.all(arrays >= 0),
            'Invalid confusion arrays')
    if samples == 5119:
        require(len(names) == 150, 'Full validation needs 150 scenes')
    totals = arrays.sum(1)
    classes = semantic_iou(totals)
    miou = np.mean(classes, axis=-1)
    binary = np.zeros((4, 2, 2), dtype=np.int64)
    binary[:, 0, 0] = totals[:, :17, :17].sum((1, 2))
    binary[:, 0, 1] = totals[:, :17, 17].sum(1)
    binary[:, 1, 0] = totals[:, 17, :17].sum(1)
    binary[:, 1, 1] = totals[:, 17, 17]
    binary_iou = semantic_iou(binary)[:, 0]
    schema = {'Semantic mIoU', 'Binary IoU', 'evaluated_samples'} | {
        c + '_IoU' for c in CLASSES}
    for h, horizon in enumerate(HORIZONS):
        metric = record['metrics'][horizon]
        require(set(metric) == schema and metric['evaluated_samples'] == samples,
                'Wrong metric schema/count')
        close(classes[h], [metric[c + '_IoU'] for c in CLASSES], 'Class IoU mismatch')
        close(miou[h], metric['Semantic mIoU'], 'Semantic mIoU mismatch')
        close(binary_iou[h], metric['Binary IoU'], 'Binary IoU mismatch')
        with np.load(confusion_dir / (horizon + '.npz'), allow_pickle=False) as data:
            require(int(data['horizon']) == h * 2, 'Wrong horizon index')
    close(np.mean(miou[1:]), record['future_mean_miou'], 'Future mean mismatch')
    return arrays, names, dict(
        samples=samples, scenes=len(names), future_mean_miou=float(np.mean(miou[1:])),
        miou_by_horizon=miou.tolist(), binary_iou_by_horizon=binary_iou.tolist(),
        class_iou_by_horizon=classes.tolist(), elapsed_seconds=record['elapsed_seconds'],
        metrics_recomputed_from_saved_matrices=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('h8-velocity', 'h8-geometry',
                                        'h2-velocity', 'h2-geometry'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, arm = args.root, args.arm
    manifest = read(root / 'MANIFEST.json')
    require(not manifest.get('missing_required') and not manifest.get('missing_optional'),
            'Incomplete completed-arm export')
    for name, expected in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe member')
        path = root / relative
        require(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
                'Evidence file hash/size mismatch: ' + name)
    campaign = root / 'analysis/history_budget_mainline_20261001'
    work = root / ('work_dirs/history_budget_' + arm + '_20261001_seed0')
    receipt = read(campaign / (arm + '_result.json'))
    require(receipt['arm'] == arm and receipt['git_revision'] == REVISION,
            'Wrong result arm/scientific revision')
    for key, epoch in [('final_audit', 10), ('best_audit', receipt['best_epoch'])]:
        audit = receipt[key]
        require(audit['epoch'] == epoch and audit['iterations'] == epoch * 2992,
                'Wrong checkpoint epoch/iteration budget')
        require(type(audit['optimizer_steps']) is int
                and 0 < audit['optimizer_steps'] <= audit['iterations'],
                'Invalid actual optimizer count')
        require(audit['finite_model_tensors'] == 771
                and audit['optimizer_finite'] is True
                and audit['state_schema_matches_cpu'] is True, 'Checkpoint audit failed')
    initial = receipt['official_initialization']
    require(initial == read(work / 'official_initialization.json'), 'Initialization changed')
    require(initial['loaded_tensors'] == 669 and initial['new_radar_tensors'] == 102
            and initial['all_camera_tensors_exact'] is True
            and initial['optimizer_restored'] is False and initial['epoch_reset_to'] == 0,
            'Wrong official/fresh initialization')
    full_record = read(work / 'validation_epoch_10_full.json')
    require(full_record['epoch'] == 10 and full_record['scope'] == 'full'
            and full_record['dataset_samples'] == 5119, 'Wrong final evaluation')
    tokens = full_record['tokens']
    require(len(tokens) == len(set(tokens)) == 5119, 'Wrong full token identity')
    final, scenes, final_metrics = audit_metrics(
        work / 'validation_epoch_10_full.json', work / 'confusions_epoch_10_full',
        5119, list(range(5119)))
    best_meta = read(work / 'best_future.json')
    require(best_meta['epoch'] == receipt['best_epoch'] and best_meta['samples'] == 256,
            'Wrong best selection metadata')
    if receipt['best_epoch'] == 10:
        best, best_scenes, best_metrics = final, scenes, dict(final_metrics)
    else:
        folder = campaign / (arm + '_best_full')
        record = read(folder / 'normal.json')
        require(record['git_revision'] == REVISION
                and record['checkpoint_meta'] == {'epoch': receipt['best_epoch'],
                                                 'iter': receipt['best_epoch'] * 2992},
                'Wrong best evaluation checkpoint/revision')
        best, best_scenes, best_metrics = audit_metrics(
            folder / 'normal.json', folder / 'confusions_normal', 5119, list(range(5119)))
    require(np.array_equal(scenes, best_scenes)
            and np.array_equal(final.sum(-1), best.sum(-1)), 'Full truth pairing mismatch')
    # Fixed subset and best-selection metrics have their original schema/order.
    subset = read(work / 'validation_epoch_10_subset.json')
    indices = subset['indices']
    require(subset['samples'] == 256 and len(set(indices)) == 256
            and [tokens[i] for i in indices] == subset['tokens'], 'Subset token mismatch')
    selection = read(work / ('validation_epoch_%02d_subset.json' % receipt['best_epoch']))
    require(selection['scope'] == 'subset' and selection['samples'] == 256
            and selection['dataset_samples'] == 5119 and selection['indices'] == indices
            and selection['tokens'] == subset['tokens']
            and set(selection['metrics']) == set(HORIZONS), 'Best subset identity/schema changed')
    close(selection['future_mean_miou'], best_meta['future_mean_miou'], 'Best metadata changed')
    modes = ('normal', 'drop', 'zero_velocity', 'shuffle_velocity') if arm.endswith(
        'velocity') else ('normal', 'drop')
    interventions = {}
    truth, subset_scenes = None, None
    for mode in modes:
        folder = campaign / (arm + '_interventions')
        record = read(folder / (mode + '.json'))
        require(record['mode'] == mode and record['git_revision'] == REVISION
                and record['checkpoint_meta'] == {'epoch': receipt['best_epoch'],
                                                 'iter': receipt['best_epoch'] * 2992},
                'Wrong intervention checkpoint/mode/revision')
        arrays, names, metrics = audit_metrics(
            folder / (mode + '.json'), folder / ('confusions_' + mode), 256, indices)
        if truth is None:
            truth, subset_scenes = arrays.sum(-1), names
        require(np.array_equal(truth, arrays.sum(-1))
                and np.array_equal(subset_scenes, names), 'Intervention truth pairing mismatch')
        interventions[mode] = metrics
    close(interventions['normal']['future_mean_miou'], best_meta['future_mean_miou'],
          'Normal independent aggregate exceeds authorized tolerance', atol=.001)
    normal_metrics = interventions['normal']
    for h, horizon in enumerate(HORIZONS):
        metric = selection['metrics'][horizon]
        require(set(metric) == {'Semantic mIoU', 'Binary IoU', 'evaluated_samples'} | {
            c + '_IoU' for c in CLASSES} and metric['evaluated_samples'] == 256,
            'Wrong selected-subset metric schema')
        close(normal_metrics['miou_by_horizon'][h], metric['Semantic mIoU'],
              'Normal semantic aggregate exceeds authorized tolerance', atol=.001)
        close(normal_metrics['binary_iou_by_horizon'][h], metric['Binary IoU'],
              'Normal binary aggregate exceeds authorized tolerance', atol=.001)
        close(normal_metrics['class_iou_by_horizon'][h], [metric[c + '_IoU'] for c in CLASSES],
              'Normal classes exceed authorized tolerance', atol=.01)
    normal = interventions['normal']['future_mean_miou']
    for metrics in interventions.values():
        metrics['future_delta_pp_from_normal'] = metrics['future_mean_miou'] - normal
    # Absolute weight file SHA is absent in these receipts: do not invent one.
    clean_audits = {key: {k: v for k, v in receipt[key].items() if k != 'path'}
                    for key in ('final_audit', 'best_audit')}
    report = dict(at_utc=manifest['at_utc'], arm=arm, science_revision=REVISION,
                  scheduler_revision=receipt['scheduler_revision'], arm_complete=True,
                  verified_original_files=len(manifest['files']), final=final_metrics,
                  best=best_metrics, best_epoch=receipt['best_epoch'],
                  interventions=interventions, checkpoint_audits=clean_audits,
                  checkpoint_file_sha256=None,
                  checkpoint_audit_basis='SHA-verified controller receipts; no independent weight reload or full weight-file download',
                  official_initialization={k: v for k, v in initial.items() if k != 'path'},
                  full_scene_anchor_ground_truth_pairing=True,
                  intervention_scene_anchor_ground_truth_pairing=True,
                  normal_independent_minus_selection_pp=normal-best_meta['future_mean_miou'],
                  normal_selection_tolerance_pp=dict(aggregate=.001, classes=.01, rtol=0),
                  stage_seconds=receipt['stage_seconds'],
                  stage_timing_basis=receipt['stage_timing_basis'],
                  comparative_statistics_run=False, cost_benchmark_run=False,
                  source_file_sha256={k: v['sha256'] for k, v in manifest['files'].items()})
    if arm == 'h8-velocity':
        report['adopted_exit'] = read(campaign / 'h8-velocity_adopted_exit.json')
        require(report['adopted_exit']['returncode'] is None, 'Invented adopted exit code')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(arm=arm, files=report['verified_original_files'],
                          final=final_metrics['future_mean_miou'],
                          best=best_metrics['future_mean_miou'],
                          interventions={m: v['future_mean_miou'] for m, v in interventions.items()})))


if __name__ == '__main__':
    main()
