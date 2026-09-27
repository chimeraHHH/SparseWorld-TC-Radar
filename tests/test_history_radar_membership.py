"""Velocity ablations must not change transport's pre-association membership."""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('history_radar', Path(__file__).parents[1] / 'models/radar_fusion.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_no_speed_filter_keeps_identical_returns_after_zeroing():
    points = torch.zeros(1, 4, 10)
    points[..., 6] = .2
    points[0, :, 3] = torch.tensor([0., 2., 36., 100.])
    geometry = points.clone()
    geometry[..., 3:5] = 0
    geometry[..., 7] = 0
    _, original = module.advect_radar(points, 3.)
    assert original.tolist() == [[True, True, False, False]]
    moved, full = module.advect_radar(points, 3., speed_filter=False)
    stationary, zero = module.advect_radar(geometry, 3., speed_filter=False)
    assert torch.equal(full, zero) and full.all()
    assert torch.equal(stationary[..., :2], geometry[..., :2])
    torch.testing.assert_close(moved[..., 0], points[..., 3] * 3.2)


def test_disabling_speed_filter_does_not_disable_causality():
    points = torch.zeros(1, 3, 10)
    points[0, :, 6] = torch.tensor([-.1, .51, .1])
    _, valid = module.advect_radar(points, 0., speed_filter=False)
    assert valid.tolist() == [[False, False, True]]
