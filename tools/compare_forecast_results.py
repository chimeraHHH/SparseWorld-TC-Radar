"""Paired scene-cluster bootstrap on saved semantic confusion matrices."""
import argparse
import json
from pathlib import Path

import numpy as np

HORIZONS = ('0.0s', '1.0s', '2.0s', '3.0s')
MOVABLE = (2, 3, 4, 5, 6, 7, 9, 10)


def semantic_iou(hist):
    diagonal = np.diagonal(hist, axis1=-2, axis2=-1)
    target = hist.sum(-1)
    union = target + hist.sum(-2) - diagonal
    iou = np.divide(diagonal, union, out=np.full(diagonal.shape, np.nan), where=union > 0)
    iou[target == 0] = np.nan
    return iou[..., :17] * 100.


def load_confusions(directory):
    arrays = []
    names = indices = None
    for h in HORIZONS:
        with np.load(Path(directory) / (h + '.npz'), allow_pickle=False) as data:
            if names is None:
                names, indices = data['scene_names'], data['sample_indices']
            assert np.array_equal(names, data['scene_names'])
            assert np.array_equal(indices, data['sample_indices'])
            arrays.append(data['histograms'])
    return np.stack(arrays), names, indices


def compare(reference, candidate, samples=2000, seed=20260920):
    if reference.shape != candidate.shape or reference.ndim != 4:
        raise ValueError('Expected matching horizon, scene, class, class histograms')
    n = reference.shape[1]
    counts = np.random.RandomState(seed).multinomial(n, np.full(n, 1/n), size=samples)
    left = semantic_iou(np.einsum('sn,hnij->shij', counts, reference, optimize=True))
    right = semantic_iou(np.einsum('sn,hnij->shij', counts, candidate, optimize=True))
    future_deltas = (np.nanmean(np.nanmean(right[:, 1:], axis=-1), axis=-1) -
                     np.nanmean(np.nanmean(left[:, 1:], axis=-1), axis=-1))
    left_point = semantic_iou(reference.sum(1))
    right_point = semantic_iou(candidate.sum(1))
    delta = np.nanmean(np.nanmean(right_point[1:], axis=-1)) - np.nanmean(np.nanmean(left_point[1:], axis=-1))
    current_delta = np.nanmean(right_point[0]) - np.nanmean(left_point[0])
    return dict(scenes=n, bootstrap_samples=samples, seed=seed,
                reference_miou_by_horizon=np.nanmean(left_point, axis=1).tolist(),
                candidate_miou_by_horizon=np.nanmean(right_point, axis=1).tolist(),
                future_mean_delta_pp=float(delta), current_delta_pp=float(current_delta),
                future_delta_scene_bootstrap_ci95=np.percentile(future_deltas, [2.5, 97.5]).tolist(),
                future_movable_mean_delta_pp=float(np.nanmean(right_point[1:, MOVABLE]) -
                                                   np.nanmean(left_point[1:, MOVABLE])),
                class_iou_delta_by_horizon=(right_point-left_point).tolist(),
                engineering_threshold_pass=bool(delta >= .3 and current_delta >= -.3),
                interpretation='Conditional on fixed selected checkpoints; not multi-seed or held-out-test evidence.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    left, ln, li = load_confusions(args.reference)
    right, rn, ri = load_confusions(args.candidate)
    assert np.array_equal(ln, rn) and np.array_equal(li, ri), 'Comparison requires identical scenes and anchors'
    result = compare(left, right)
    result.update(reference=args.reference, candidate=args.candidate, anchors=len(li))
    # Missing classes in tiny diagnostic fixtures are represented as null.
    def safe(x):
        if isinstance(x, list): return [safe(v) for v in x]
        if isinstance(x, dict): return {k:safe(v) for k,v in x.items()}
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    Path(args.out).write_text(json.dumps(safe(result), indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='class_iou_delta_by_horizon'}, indent=2))


if __name__ == '__main__':
    main()
