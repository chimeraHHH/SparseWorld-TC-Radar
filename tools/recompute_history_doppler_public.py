"""Recompute published four-arm statistics from saved matrices, CPU only."""
import argparse
import json
from pathlib import Path
import numpy as np

from audit_history_doppler_completion import ARMS, HORIZONS, CONTRASTS, bootstrap, close, contrast, measures, require, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evidence-dir', type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.evidence_dir / 'audit.json').read_text())
    reference_names = reference_indices = reference_gt = None
    verified = 0
    for scope in ('final', 'best', 'interventions'):
        arrays = {}
        for arm in ARMS:
            modes = tuple(report['interventions'][arm]) if scope == 'interventions' else (None,)
            for mode in modes:
                key = arm if mode is None else arm + '/' + mode
                folder = args.evidence_dir / 'matrices' / (scope + '_' + arm + ('' if mode is None else '_' + mode))
                expected = report[scope]['metrics'][arm] if mode is None else report[scope][arm][mode]
                payload = []
                names = indices = None
                for i, h in enumerate(HORIZONS):
                    path = folder / (h + '.npz')
                    require(sha(path) == report['source_files_sha256'][scope + '/' + key][path.name], 'Published matrix hash differs')
                    with np.load(path, allow_pickle=False) as data:
                        require(int(data['horizon']) == 2 * i, 'Horizon mismatch')
                        if names is None:
                            names, indices = data['scene_names'].copy(), data['sample_indices'].copy()
                        require(np.array_equal(names, data['scene_names']) and np.array_equal(indices, data['sample_indices']), 'Horizon pairing differs')
                        payload.append(data['histograms'])
                    verified += 1
                array = np.stack(payload)
                require(np.issubdtype(array.dtype, np.integer) and np.all(array >= 0), 'Invalid integer matrices')
                if reference_names is None:
                    reference_names, reference_indices, reference_gt = names, indices, array.sum(-1)
                require(np.array_equal(reference_names, names) and np.array_equal(reference_indices, indices)
                        and np.array_equal(reference_gt, array.sum(-1)), 'Scene ground truth/order differs')
                metric = measures(array.sum(1))
                for key_metric, summary_key in [('class_iou', 'class_iou_by_horizon'), ('miou', 'miou_by_horizon'), ('binary_iou', 'binary_iou_by_horizon')]:
                    close(metric[key_metric], expected[summary_key], 'Published metric differs')
                close(metric['miou'][1:].mean(), expected['future_mean_miou'], 'Future metric differs')
                arrays[key] = array
        if scope != 'interventions':
            points, draws, _ = bootstrap(arrays)
            for key, weights in CONTRASTS.items():
                actual = contrast(weights, points, draws)
                for field in actual:
                    close(actual[field], report[scope]['contrasts'][key][field], 'Published paired statistics differ')
        else:
            for arm in ARMS:
                local = {m: arrays[arm + '/' + m] for m in report[scope][arm]}
                points, draws, _ = bootstrap(local)
                for mode, expected in report[scope][arm].items():
                    delta = points[mode]['miou'] - points['normal']['miou']
                    samples = draws[mode]['miou'] - draws['normal']['miou']
                    close(delta[1:].mean(), expected['delta_pp_from_normal'], 'Intervention delta differs')
                    close(np.percentile(samples[:, 1:].mean(1), [2.5, 97.5]), expected['delta_ci95'], 'Intervention CI differs')
        # Full scopes share anchors; subset scopes require their own scene population.
        if scope == 'best':
            reference_names = reference_indices = reference_gt = None
    print(json.dumps(dict(verified=True, matrices=verified, samples_full=5119, scenes_full=150,
        samples_subset=256, scenes_subset=126, resamples=2000, seed=20260927,
        scope='Saved predictions only; no GPU, model loading, training or inference.'), indent=2))


if __name__ == '__main__':
    main()
