import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).parents[1] / 'tools/compare_forecast_results.py'
spec = importlib.util.spec_from_file_location('comparison', path)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def fixture():
    hist = np.repeat((np.eye(18, dtype=np.int64)*20)[None, None], 4, axis=0)
    return np.repeat(hist, 3, axis=1)


def test_identical_scene_predictions_have_exact_zero_paired_interval():
    hist = fixture()
    result = comparison.compare(hist, hist, samples=100)
    assert result['future_mean_delta_pp'] == 0.
    assert result['future_delta_scene_bootstrap_ci95'] == [0., 0.]


def test_correcting_future_errors_improves_forecast_but_not_current():
    perfect = fixture()
    bad = perfect.copy()
    bad[1:, :, 4, 4] -= 10
    bad[1:, :, 4, 17] += 10
    result = comparison.compare(bad, perfect, samples=100)
    assert result['current_delta_pp'] == 0.
    assert result['future_mean_delta_pp'] > 0.
    assert result['future_delta_scene_bootstrap_ci95'][0] > 0.
    assert result['future_movable_mean_delta_pp'] > 0.
