"""CPU-only independent audit of saved four-arm evaluations. No inference/training.

Inputs are immutable evidence exports; all metrics are recomputed from integer
scene confusion matrices. Public output deliberately omits host/process paths.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile

import numpy as np

ARMS = ('h1-velocity', 'h1-geometry', 'h8-velocity', 'h8-geometry')
HORIZONS = ('0.0s', '1.0s', '2.0s', '3.0s')
CLASSES = ('barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
           'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
           'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')
# The stored official schema also includes the leading "others" class.
CLASSES = ('others',) + CLASSES
CONTRASTS = {
    'velocity_gain_short': (1, -1, 0, 0),
    'velocity_gain_long': (0, 0, 1, -1),
    'interaction_short_minus_long': (1, -1, -1, 1),
    'short_velocity_minus_long_velocity': (1, 0, -1, 0),
    'short_geometry_minus_long_geometry': (0, 1, 0, -1),
}
REV = 'e80bca6ece3ceafbb582d8ca21595b2e4128691c'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, message, atol=1e-10):
    require(np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True), message)


def verify_manifest(root):
    manifest = read(root / 'MANIFEST.json')
    require(not manifest['missing'], 'Missing exported evidence')
    for name, expected in manifest['files'].items():
        path = root / name
        require(sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes'],
                'Evidence hash/size mismatch: ' + name)
    return manifest


def measures(total):
    """Independent row=GT, column=prediction implementation; empty class is 17."""
    diag = np.diagonal(total, axis1=-2, axis2=-1)
    gt = total.sum(-1)
    denom = gt + total.sum(-2) - diag
    ratios = np.full(diag.shape, np.nan)
    np.divide(100. * diag, denom, out=ratios, where=(denom > 0) & (gt > 0))
    classes = ratios[..., :17]
    occupied_tp = total[..., :17, :17].sum((-2, -1))
    occupied_union = occupied_tp + total[..., :17, 17].sum(-1) + total[..., 17, :17].sum(-1)
    binary = 100. * occupied_tp / occupied_union
    return dict(class_iou=classes, miou=np.nanmean(classes, axis=-1), binary_iou=binary)


def load(record_path, matrix_dir, samples):
    record = read(record_path)
    require(record['samples'] == samples and set(record['metrics']) == set(HORIZONS), 'Wrong scope/schema')
    if samples == 5119:
        require(np.array_equal(record['indices'], np.arange(5119)), 'Full anchor order mismatch')
    arrays, names, indices = [], None, None
    provenance = {str(record_path): sha(record_path)}
    for i, horizon in enumerate(HORIZONS):
        path = matrix_dir / (horizon + '.npz')
        provenance[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            require(int(data['horizon']) == i * 2, 'Wrong physical horizon index')
            if names is None:
                names, indices = data['scene_names'].copy(), data['sample_indices'].copy()
            require(np.array_equal(names, data['scene_names']) and np.array_equal(indices, data['sample_indices']),
                    'Horizon scene/anchor mismatch')
            arrays.append(data['histograms'].copy())
    array = np.stack(arrays)
    scenes = 150 if samples == 5119 else 126
    require(array.shape == (4, scenes, 18, 18) and np.issubdtype(array.dtype, np.integer)
            and np.all(array >= 0) and len(np.unique(names)) == scenes, 'Invalid scene matrices')
    require(np.array_equal(record['indices'], indices), 'JSON/matrix anchor mismatch')
    metric = measures(array.sum(1))
    schema = {'Semantic mIoU', 'Binary IoU', 'evaluated_samples'} | {c + '_IoU' for c in CLASSES}
    for i, h in enumerate(HORIZONS):
        logged = record['metrics'][h]
        require(set(logged) == schema and logged['evaluated_samples'] == samples, 'Metric schema/sample mismatch')
        close(metric['class_iou'][i], [logged[c + '_IoU'] for c in CLASSES], 'Class metrics mismatch')
        close(metric['miou'][i], logged['Semantic mIoU'], 'Semantic metrics mismatch')
        close(metric['binary_iou'][i], logged['Binary IoU'], 'Binary metrics mismatch')
    future = float(metric['miou'][1:].mean())
    close(future, record['future_mean_miou'], 'Future mean mismatch')
    summary = dict(samples=samples, scenes=scenes, future_mean_miou=future,
        miou_by_horizon=metric['miou'].tolist(), binary_iou_by_horizon=metric['binary_iou'].tolist(),
        class_iou_by_horizon=metric['class_iou'].tolist(),
        future_class_iou=metric['class_iou'][1:].mean(0).tolist(),
        elapsed_seconds=record['elapsed_seconds'], checkpoint_meta=record.get('checkpoint_meta'),
        epoch=record.get('epoch'), recomputed_all_metrics=True)
    return array, names, indices, summary, provenance


def bootstrap(arrays, independent=False):
    n = next(iter(arrays.values())).shape[1]
    counts = np.random.RandomState(20260927).multinomial(n, np.full(n, 1 / n), size=2000)
    points, draws = {}, {}
    for a, array in arrays.items():
        points[a] = measures(array.sum(1))
        draws[a] = measures(np.einsum('sn,hnij->shij', counts, array, optimize=True))
        if independent:
            # Alternate integer accumulation checks every draw, not merely CI endpoints.
            alternative = np.stack([np.sum(array * row[None, :, None, None], axis=1) for row in counts])
            check = measures(alternative)
            for key in draws[a]:
                close(draws[a][key], check[key], 'Independent 2000-draw accumulation differs')
    return points, draws, counts


def contrast(weights, points, draws):
    def diff(key, source):
        return sum(w * source[a][key] for a, w in zip(ARMS, weights) if w)
    cp, cd = diff('class_iou', points), diff('class_iou', draws)
    hp, hd = diff('miou', points), diff('miou', draws)
    bp, bd = diff('binary_iou', points), diff('binary_iou', draws)
    ci = lambda values: np.percentile(values, [2.5, 97.5], axis=0).tolist()
    return dict(delta_pp=float(hp[1:].mean()), ci95=ci(hd[:, 1:].mean(1)),
        miou_delta_pp_by_horizon=hp.tolist(), horizon_ci95=np.asarray(ci(hd)).T.tolist(),
        class_delta_pp_by_horizon=cp.tolist(), class_ci95_by_horizon=np.moveaxis(np.asarray(ci(cd)), 0, -1).tolist(),
        future_class_delta_pp=cp[1:].mean(0).tolist(),
        future_class_ci95=np.asarray(ci(cd[:, 1:].mean(1))).T.tolist(),
        binary_delta_pp_by_horizon=bp.tolist(), binary_ci95=np.asarray(ci(bd)).T.tolist())


def cost_ledger(original_path, recovery_path, windows_path, receipts):
    original, recovery, windows = read(original_path), read(recovery_path), read(windows_path)
    require(sha(original_path) == recovery['original_cost_sha256'], 'Original cost ledger changed')
    registry = dict(original['source_registry'])
    for row in recovery['additional_independent_outer_stages']:
        name = row['archive']
        registry.setdefault(name, dict(sha256=row['archive_sha256'], members_used={}))
        registry[name]['members_used'][row['member']] = dict(sha256=row['member_sha256'])
    generation_values = set()
    for row in recovery['cache_generation_references']:
        require(row['archive'] in registry, 'Cache archive absent from source registry')
        registry[row['archive']]['members_used'][row['member']] = dict(sha256=row['member_sha256'])
        generation_values.update(row['generation_seconds_values'])
    require(len(generation_values) == 1, 'Repeated cache receipts do not describe one generation')
    for name, expected in registry.items():
        path = original_path.parent / name
        require(sha(path) == expected['sha256'], 'Cost source archive changed')
        with tarfile.open(path) as archive:
            for member, value in expected['members_used'].items():
                raw = archive.extractfile(member).read()
                require(hashlib.sha256(raw).hexdigest() == value['sha256'], 'Cost source member changed')
                if 'bytes' in value:
                    require(len(raw) == value['bytes'], 'Cost source size changed')
            for row in recovery['additional_independent_outer_stages']:
                if row['archive'] != name:
                    continue
                value = json.loads(archive.extractfile(row['member']).read())
                for key in row['json_pointer'].strip('/').split('/'):
                    value = value[key]
                close(value, row['seconds'], 'Outer-stage cost differs from raw receipt')
    h1 = {}
    for a, entry in original['arms'].items():
        wall, fail, timing = entry['formal_process_wall_clock'], entry['failed_full_evaluation'], entry['logged_optimizer_time_estimate']
        h1[a] = dict(formal_process_wall_bounds_seconds=[wall['duration_lower_bound_seconds'], wall['duration_upper_bound_seconds']],
            observed_before_epoch10_save_seconds=entry['original_training_portion']['first_log_to_epoch10_checkpoint_save_begin_seconds'],
            logged_iteration_time_seconds=timing['weighted_logged_seconds'], logged_iteration_coverage=timing['covered_iterations'],
            unlogged_iteration_tails=timing['missing_tail_iterations'],
            failed_full_eval_lifetime_bounds_seconds=[fail['evaluation_lifetime_lower_bound_seconds'], fail['evaluation_lifetime_upper_bound_seconds']],
            failed_full_last_generated_samples=fail['last_progress_completed_samples'],
            original_admission_stage_seconds=entry['original_admission_completed_stage_seconds'],
            recovery_stage_seconds=receipts[a]['stage_seconds'],
            recorded_device_peak_memory_mib=entry['gpu_telemetry']['peak_device_memory_used_mib'])
    h8 = {}
    for a in ('h8-velocity', 'h8-geometry'):
        train = windows['arms'][a]['training']
        require(not train['invalid_train_metadata'] and not train['duplicate_global_steps']
                and not train['nonmonotonic_global_step_events'], 'Invalid final training windows')
        for source in train['sources']:
            require(not source['malformed_json_lines'], 'Malformed final training log')
        h8[a] = dict(stage_seconds=receipts[a]['stage_seconds'],
            outer_stage_sum_seconds=sum(receipts[a]['stage_seconds'].values()),
            final_train_window_seconds_per_iteration={k: v['time']['weighted_mean_seconds_per_step'] for k, v in train['windows'].items()},
            final_train_window_coverage={k: v['time']['covered_steps'] for k, v in train['windows'].items()},
            recovered_gradient_windows=len(train['nonfinite_loss_grad_events']),
            nonfinite_loss_grad_fields=train['nonfinite_loss_grad_fields'])
    old_failed = [dict(id=r['id'], seconds=r['seconds']) for r in recovery['additional_independent_outer_stages'] if r['id'].startswith('old_recovery')]
    diagnostic = {r['id']: r['seconds'] for r in recovery['additional_independent_outer_stages'] if not r['id'].startswith('old_recovery')}
    gpu_independent = sum(v['outer_stage_sum_seconds'] for v in h8.values())
    gpu_independent += sum(sum(r['original_admission_stage_seconds'].values()) + sum(r['recovery_stage_seconds'].values()) for r in h1.values())
    gpu_independent += sum(r['seconds'] for r in old_failed) + diagnostic['diagnostic_subprocess']
    bounds = [gpu_independent + sum(r['formal_process_wall_bounds_seconds'][i] for r in h1.values()) for i in (0, 1)]
    return dict(status='Completed observed-cost ledger; exact all-inclusive campaign bill unavailable',
        h1_original_and_recovery=h1, h8=h8, failed_strict_recovery_parity=old_failed,
        diagnostic_and_shared_admission_seconds=diagnostic,
        single_cached_generation_seconds=generation_values.pop(),
        known_single_GPU_stage_and_process_wall_subtotal_bounds_seconds=bounds,
        known_single_GPU_stage_and_process_wall_subtotal_bounds_hours=[s / 3600 for s in bounds],
        audit_sources_sha256={name: row['sha256'] for name, row in registry.items()},
        exact_energy_kwh=None, measured_deployment_latency_ms=None, measured_deployment_FPS=None,
        scope='Sum of observed single-GPU stages/process lifetimes, including CPU work, validation, storage and possible internal waits. This is neither campaign elapsed time, GPU-active time nor billable GPU hours.',
        accounting_rules=['H1 failed evaluation is already inside original formal-process bounds; displayed separately but never added again.',
            'H8 final-full evaluator elapsed is inside train stage; do not add to train.',
            'H1 best equals epoch10 and reuses the same full evaluation; no fictitious extra best-full cost.',
            'Diagnostic inner elapsed, CPU/cache checks inside admission, and repeated cached generation receipts are not double counted.',
            'Pre-subprocess admission queue excluded by stage timers; subprocess-internal host-lock waits may remain included.',
            'Parallel-arm sums are resource-stage sums, not elapsed user time; missing logger tails are not extrapolated.'],
        unresolved=['Exact original/v2 shared CPU/cache preparation and first 74e1929 helper-failure outer duration',
            'Complete controller gaps/queue occupancy, exact H1 spawn/SIGKILL timestamps and GPU-active energy',
            'Deployment streaming-cache latency, peak memory, I/O, startup/recovery and KD training cost benchmarks not run'])


def main():
    parser = argparse.ArgumentParser()
    for name in ('h1-root', 'h8-velocity-root', 'completion-root', 'original-cost', 'recovery-cost',
                 'training-windows', 'velocity-best-subset', 'out', 'public-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    completion = args.completion_root
    c = completion / 'analysis/history_doppler_recovery_v3_20260929'
    manifests = [verify_manifest(completion), verify_manifest(args.h8_velocity_root)]
    receipts = {a: read(c / (a + '_result.json')) for a in ARMS}
    h1_archive = args.h1_root.parent / 'history_doppler_h1_complete_raw_20260929.tar.gz'
    with tarfile.open(h1_archive) as archive:
        h1_manifest = json.loads(archive.extractfile('manifest.json').read())
        for name, expected in h1_manifest['files'].items():
            raw = archive.extractfile(name).read()
            digest = expected if isinstance(expected, str) else expected['sha256']
            require(hashlib.sha256(raw).hexdigest() == digest, 'H1 archive member changed')
            if isinstance(expected, dict):
                require(len(raw) == expected['bytes'], 'H1 archive member size changed')
            # Earlier local extraction retained only analysis inputs; verify all
            # archive members and every retained local copy without requiring
            # unrelated CPU stdout files to have been extracted.
            if (args.h1_root / name).exists():
                require(sha(args.h1_root / name) == digest, 'H1 extracted source changed: ' + name)
    selections = {}
    with tarfile.open(args.original_cost.parent / 'history_doppler_failure_raw_20260928.tar.gz') as archive:
        for a in ('h1-velocity', 'h1-geometry'):
            selections[a] = json.loads(archive.extractfile('work/' + a + '/validation_epoch_10_subset.json').read())
    require(selections['h1-velocity']['tokens'] == selections['h1-geometry']['tokens']
            and selections['h1-velocity']['indices'] == selections['h1-geometry']['indices'], 'Original H1 frozen tokens differ')
    exit_record = read(completion / 'PROCESS_EXIT_EVIDENCE.json')
    require(all(not r['alive'] for r in exit_record['known_processes'].values()), 'Original PID still live')
    for gpu in (0, 1):
        worker = read(c / ('gpu%d_worker.json' % gpu))
        require(worker['stage'] == 'worker_queue_finished' and worker['state'] == 'complete'
                and worker['child_returncode'] == 0, 'Worker completion missing')
    for a, receipt in receipts.items():
        require(receipt['arm'] == a and receipt['evaluation_git_revision'] == REV, 'Wrong arm/version')
        audit = receipt['final_audit']
        require((audit['epoch'], audit['iterations']) == (10, 29920), 'Wrong final budget')
        require(audit['optimizer_steps'] == (29913 if a.startswith('h1') else 29914), 'Wrong exact optimizer steps')
        for tag in ('final_audit', 'best_audit'):
            audit = receipt[tag]
            require(audit['finite_model_tensors'] == 771 and audit['optimizer_finite']
                    and audit['state_schema_matches_cpu'], 'Nonfinite/schema checkpoint')
        require(read(c / (a + '_status.json'))['state'] == 'complete', 'Arm incomplete')
        initial = receipt['official_initialization']
        require(initial['loaded_tensors'] == 669 and initial['new_radar_tensors'] == 102
                and initial['all_camera_tensors_exact'] and not initial['optimizer_restored']
                and initial['epoch_reset_to'] == 0, 'Initialization mismatch')
        if a.startswith('h1'):
            require(sha(c / (a + '_result.json')) == sha(args.h1_root / (a + '_result.json')), 'H1 receipt changed')
        elif a == 'h8-velocity':
            require(sha(c / (a + '_result.json')) == sha(args.h8_velocity_root / 'analysis/history_doppler_recovery_v3_20260929' / (a + '_result.json')), 'Velocity receipt changed')
    report = dict(schema_version=1, completed_at_utc=read(c / 'gpu0_worker.json')['at_utc'],
        evaluation_revision=REV, primary_scope='final epoch10/full5119', secondary_scope='fixed256-selected best/full5119',
        horizons=list(HORIZONS), classes=list(CLASSES), bootstrap_samples=2000, bootstrap_seed=20260927,
        all_original_workers_and_trainers_exited=True, all_arms_complete=True, source_files_sha256={})
    report['evidence_archive_sha256'] = {h1_archive.name: sha(h1_archive),
        'history_h8_velocity_full_raw_20261001_complete.tar.gz': sha(args.h8_velocity_root.parent / 'history_h8_velocity_full_raw_20261001_complete.tar.gz'),
        'history_four_arm_completion_raw_20261001_retry1.tar.gz': sha(completion.parent / 'history_four_arm_completion_raw_20261001_retry1.tar.gz')}
    evidence = args.public_dir / 'matrices'
    evidence.mkdir(parents=True, exist_ok=True)
    token = scene = gt = None
    raw_comparison = read(c / 'comparison.json')
    for scope in ('final', 'best'):
        arrays, metrics = {}, {}
        for a in ARMS:
            if a.startswith('h1'):
                folder = args.h1_root / (a + '_final_full')
                record, matrix = folder / 'normal.json', folder / 'confusions_normal'
            else:
                root = args.h8_velocity_root if a.endswith('velocity') else completion
                if scope == 'final':
                    folder = root / ('work_dirs/history_' + a + '_seed0')
                    record, matrix = folder / 'validation_epoch_10_full.json', folder / 'confusions_epoch_10_full'
                else:
                    folder = root / 'analysis/history_doppler_recovery_v3_20260929' / (a + '_best_full')
                    record, matrix = folder / 'normal.json', folder / 'confusions_normal'
            array, names, indices, entry, provenance = load(record, matrix, 5119)
            if scene is None:
                scene, gt = names, array.sum(-1)
            require(np.array_equal(names, scene) and np.array_equal(array.sum(-1), gt), 'Full per-scene GT mismatch')
            arrays[a], metrics[a] = array, entry
            report['source_files_sha256'][scope + '/' + a] = {Path(p).name: digest for p, digest in provenance.items()}
            for h in HORIZONS:
                dest = evidence / (scope + '_' + a) / (h + '.npz')
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(matrix / (h + '.npz'), dest)
                require(sha(dest) == provenance[str(matrix / (h + '.npz'))], 'Public matrix copy mismatch')
        points, draws, counts = bootstrap(arrays, independent=True)
        contrasts = {key: contrast(w, points, draws) for key, w in CONTRASTS.items()}
        for a in ARMS:
            close(metrics[a]['future_mean_miou'], raw_comparison[scope]['future_mean_miou'][a], 'Controller point differs')
        for key in CONTRASTS:
            if key not in raw_comparison[scope]:
                continue
            close(contrasts[key]['delta_pp'], raw_comparison[scope][key]['delta_pp'], 'Controller contrast differs')
            close(contrasts[key]['ci95'], raw_comparison[scope][key]['ci95'], 'Controller bootstrap CI differs')
        report[scope] = dict(metrics=metrics, contrasts=contrasts, scenes=150,
            identical_scene_GT_and_order=True, independent_all_draw_recompute=True,
            counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest(), controller_result_reproduced=True)
    interventions = {}
    repeat_inference = {}
    subset_tokens = selections['h1-velocity']['tokens']
    subset_indices = subset_names = subset_gt = None
    for a in ARMS:
        root = args.h1_root if a.startswith('h1') else (
            args.h8_velocity_root / 'analysis/history_doppler_recovery_v3_20260929' if a.endswith('velocity') else c)
        folder = root / (a + '_interventions')
        modes = ('normal', 'drop', 'zero_velocity', 'shuffle_velocity') if a.endswith('velocity') else ('normal', 'drop')
        local_arrays, entries = {}, {}
        for mode in modes:
            record, matrix = folder / (mode + '.json'), folder / ('confusions_' + mode)
            array, names, indices, entry, provenance = load(record, matrix, 256)
            logged = read(record)
            require(logged['mode'] == mode and logged['git_revision'] == REV
                    and logged['checkpoint_meta'] == {'epoch': receipts[a]['best_audit']['epoch'], 'iter': receipts[a]['best_audit']['iterations']}, 'Intervention version/weight mismatch')
            if subset_indices is None:
                subset_indices, subset_names, subset_gt = indices, names, array.sum(-1)
            require(np.array_equal(indices, subset_indices) and np.array_equal(names, subset_names)
                    and np.array_equal(array.sum(-1), subset_gt), 'Intervention anchors/scene GT mismatch')
            local_arrays[mode], entries[mode] = array, entry
            report['source_files_sha256']['interventions/' + a + '/' + mode] = {Path(p).name: d for p, d in provenance.items()}
            for h in HORIZONS:
                dest = evidence / ('interventions_' + a + '_' + mode) / (h + '.npz')
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(matrix / (h + '.npz'), dest)
                require(sha(dest) == provenance[str(matrix / (h + '.npz'))], 'Public intervention copy mismatch')
        # Same frozen selection anchors and tokens across all four arms.
        if a.startswith('h1'):
            selection = read(args.h1_root / (a + '_recovery_subset_parity/normal.json'))
        else:
            root = args.h8_velocity_root if a.endswith('velocity') else completion
            work = root / ('work_dirs/history_' + a + '_seed0')
            selection = read(work / 'validation_epoch_10_subset.json')
            if subset_tokens is None:
                subset_tokens = selection['tokens']
            require(selection['tokens'] == subset_tokens, 'Frozen subset token order differs')
            best = read(work / 'best_future.json')
            close(entries['normal']['future_mean_miou'], best['future_mean_miou'], 'Independent normal exceeds .001pp', .001)
            best_subset = read(work / ('validation_epoch_%02d_subset.json' % best['epoch'])) if a.endswith('geometry') else read(args.velocity_best_subset)
            require(best_subset['epoch'] == best['epoch'], 'Best subset epoch mismatch')
            differences, aggregate_pass, class_pass = {}, True, True
            for i, h in enumerate(HORIZONS):
                d = dict(semantic_pp=entries['normal']['miou_by_horizon'][i] - best_subset['metrics'][h]['Semantic mIoU'],
                    binary_pp=entries['normal']['binary_iou_by_horizon'][i] - best_subset['metrics'][h]['Binary IoU'],
                    class_pp=(np.asarray(entries['normal']['class_iou_by_horizon'][i]) -
                        [best_subset['metrics'][h][cl + '_IoU'] for cl in CLASSES]).tolist())
                differences[h] = d
                aggregate_pass &= max(abs(d['semantic_pp']), abs(d['binary_pp'])) <= .001
                class_pass &= max(abs(v) for v in d['class_pp']) <= .01
            repeat_inference[a] = dict(future_mean_difference_pp=entries['normal']['future_mean_miou'] - best['future_mean_miou'],
                differences_by_horizon=differences, aggregate_tolerance_pp=.001, class_tolerance_pp=.01, rtol=0,
                all_aggregate_within_H1_recovery_tolerance=bool(aggregate_pass),
                all_classes_within_H1_recovery_tolerance=bool(class_pass),
                scope='Additional post-training H8 comparison of separate inference runs. H1 migration admission is separately frozen and passed. Each run independently reproduces its own integer confusion metrics exactly; no threshold relaxation or re-evaluation performed.')
        require(np.array_equal(selection['indices'], subset_indices), 'Frozen selection anchors differ')
        points, draws, _ = bootstrap(local_arrays)
        for mode in modes:
            m = entries[mode]
            hp = points[mode]['miou'] - points['normal']['miou']
            hd = draws[mode]['miou'] - draws['normal']['miou']
            cp = points[mode]['class_iou'] - points['normal']['class_iou']
            cd = draws[mode]['class_iou'] - draws['normal']['class_iou']
            m.update(delta_pp_from_normal=float(hp[1:].mean()),
                delta_ci95=np.percentile(hd[:, 1:].mean(1), [2.5, 97.5]).tolist(),
                miou_delta_by_horizon=hp.tolist(),
                horizon_delta_ci95=np.percentile(hd, [2.5, 97.5], axis=0).T.tolist(),
                class_delta_by_horizon=cp.tolist(),
                future_class_delta=cp[1:].mean(0).tolist(),
                future_class_delta_ci95=np.percentile(cd[:, 1:].mean(1), [2.5, 97.5], axis=0).T.tolist())
        interventions[a] = entries
    report['interventions'] = interventions
    report['additional_H8_repeat_inference_check'] = repeat_inference
    report['checkpoint_receipts'] = {a: {k: {kk: vv for kk, vv in r[k].items() if kk != 'path'}
        for k in ('final_audit', 'best_audit')} for a, r in receipts.items()}
    report['checkpoint_file_SHA_independently_obtained'] = False
    report['weight_audit_scope'] = 'Controller finite/schema/optimizer/meta audits; no new checkpoint-file SHA or independent weight reload.'
    report['stage_wall_seconds'] = {a: r['stage_seconds'] for a, r in receipts.items()}
    report['offline_cost'] = cost_ledger(args.original_cost, args.recovery_cost, args.training_windows, receipts)
    report['noninferiority_margin_pp'] = -.3
    report['noninferiority_pass'] = report['final']['contrasts']['short_velocity_minus_long_velocity']['ci95'][0] > -.3
    report['positive_interaction_CI'] = report['final']['contrasts']['interaction_short_minus_long']['ci95'][0] > 0
    report['statistical_scope'] = 'Paired scene-cluster bootstrap conditional on fixed trained models and reused validation scenes; not multi-seed training uncertainty. Horizon/class/intervention intervals exploratory, unadjusted for multiplicity.'
    report['limitations'] = ['SDK processed velocity, not raw Doppler; shared upstream filters may preserve motion priors.',
        'H1 encodes current six images then duplicates features/projections/timestamps into eight slots; sampler cost retained.',
        'Shared frozen T/G sampling layout mismatch; synthetic impact verified, real accuracy effect unknown.',
        'One seed, 256 reused for selection and contained in the 5119 validation set; not blind testing.',
        'No native short-history, deployment latency, streaming-cache, energy, visibility/change-domain or speed-coverage benchmark added.']
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    args.public_dir.mkdir(parents=True, exist_ok=True)
    dest = args.public_dir / 'audit.json'
    dest.write_text(json.dumps(report, indent=2, allow_nan=False))
    require(not any(p in dest.read_text() for p in ('/Users/', '/storage/', '/home/')), 'Private path in public output')
    print(json.dumps(dict(complete=True, final={k: {x: v[x] for x in ('delta_pp', 'ci95')} for k, v in report['final']['contrasts'].items()},
        noninferiority_pass=report['noninferiority_pass']), indent=2))


if __name__ == '__main__':
    main()
