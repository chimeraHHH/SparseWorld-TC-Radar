"""Causality and velocity-independent membership of the single-sweep experiment."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# These utilities deliberately need no training/CUDA stack.
_path = Path(__file__).resolve().parents[1] / 'loaders/pipelines/single_sweep_radar.py'
_spec = importlib.util.spec_from_file_location('single_sweep_radar_utils', _path)
utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utils)


def chain(timestamps):
    return {str(i): dict(token=str(i), timestamp=stamp,
                         prev=str(i-1) if i else '', next=str(i+1) if i+1 < len(timestamps) else '')
            for i, stamp in enumerate(timestamps)}


@pytest.mark.parametrize('start', ['0', '1', '2', '3'])
def test_latest_causal_selection_from_either_side(start):
    records = chain([100, 200, 300, 400])
    selected, next_us = utils.latest_causal_sweep(records.__getitem__, start, 250)
    assert selected['token'] == '1'
    assert next_us == 300


def test_exact_boundary_and_no_accumulation():
    records = chain([100, 200, 300])
    selected, next_us = utils.latest_causal_sweep(records.__getitem__, '0', 200)
    assert selected['timestamp'] == 200
    assert next_us == 300
    selected, next_us = utils.latest_causal_sweep(records.__getitem__, '2', 50)
    assert selected is None and next_us == 100
    selected, next_us = utils.latest_causal_sweep(records.__getitem__, '0', 350)
    assert selected['timestamp'] == 300 and next_us is None


def test_invalid_chain_fails_closed():
    records = chain([100, 90])
    with pytest.raises(ValueError, match='Nonmonotone'):
        utils.latest_causal_sweep(records.__getitem__, '0', 250)
    records = chain([100, 200])
    records['1']['next'] = '0'
    with pytest.raises(ValueError, match='cyclic'):
        utils.latest_causal_sweep(records.__getitem__, '0', 250)


def test_velocity_does_not_change_membership_or_geometry():
    features = np.zeros((5, 10), dtype=np.float64)
    features[:, 0] = [1, 2, 3, 4, 45]
    features[:, 3] = [2, np.nan, np.inf, 999, 0]
    features[:, 4] = [3, 4, 5, 999, 0]
    features[:, 7] = [2, np.nan, np.inf, 999, 0]
    actual, invalid = utils.geometry_filter_and_sanitize(features, 44.)
    velocity_removed = features.copy()
    velocity_removed[:, [3, 4, 7]] = 0
    expected, _ = utils.geometry_filter_and_sanitize(velocity_removed, 44.)
    assert len(actual) == len(expected) == 4
    assert invalid == 2
    assert np.array_equal(actual[:, [0, 1, 2, 5, 6, 8, 9]], expected[:, [0, 1, 2, 5, 6, 8, 9]])
    assert np.isfinite(actual).all()
    assert actual[3, 3] == 999  # No motion-magnitude filtering.
    assert np.array_equal(utils.velocity_view(actual, 'zero'), expected)
    assert np.any(actual[:, [3, 4, 7]] != 0)  # zero view does not mutate shared input.


def test_nonfinite_geometry_rejected_in_both_views():
    features = np.zeros((3, 10), dtype=np.float64)
    features[0, 0] = np.nan
    features[1, 5] = np.inf
    processed, _ = utils.geometry_filter_and_sanitize(features, 44.)
    assert processed.shape == (1, 10)
    with pytest.raises(ValueError, match='velocity_mode'):
        utils.velocity_view(processed, 'raw_doppler')
