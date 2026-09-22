"""Behavioral contracts for A's soft association and temporal reliability."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('radar_fusion', Path(__file__).parents[1]/'models/radar_fusion.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def network(**kwargs):
    return module.RadarQueryFusion(embed_dims=8, mode='transport',
        velocity_consistency=True, temporal_reliability=True, **kwargs)


def points():
    radar = torch.zeros(2, 5, 10)
    radar[..., 0] = torch.tensor([-.1, 0., .1, .2, .3])
    radar[..., 3] = torch.tensor([1., 1., 1., 1., 8.])
    radar[..., 7] = radar[..., 3]
    radar[..., 8] = 1.
    return radar


def test_inconsistent_same_location_return_gets_lower_soft_weight():
    net = network()
    radar = points()
    radar[..., :3] = 0.
    score = net.velocity_agreement(radar, torch.ones(2, 5))
    assert torch.all(score[:, :4] > score[:, 4:])
    assert torch.all(.25 + .75 * score >= .25)
    score.sum().backward()
    assert net.velocity_scale.grad.abs() > 0


def test_leave_one_out_singleton_fallback_and_masked_peer_exclusion():
    net = network()
    radar = points()[:, :2]
    radar[..., 7] = 100.  # would reject if self-consistency were tested
    assert torch.equal(net.velocity_agreement(radar[:, :1], torch.ones(2, 1)), torch.ones(2, 1))
    score = net.velocity_agreement(radar, torch.tensor([[1., 0.], [1., 0.]]))
    assert torch.equal(score[:, 0], torch.ones(2))


def test_projection_respects_different_lines_of_sight():
    radar = points()[:, :2]
    radar[..., 3:5] = torch.tensor([2., 3.])
    radar[:, 0, 8:10] = torch.tensor([1., 0.])
    radar[:, 1, 8:10] = torch.tensor([0., 1.])
    radar[..., 7] = torch.tensor([2., 3.])
    torch.testing.assert_close(network().velocity_agreement(radar, torch.ones(2, 2)), torch.ones(2, 2))


def test_gate_starts_at_A_decay_and_can_learn_quality_dependence():
    net = network()
    quality = torch.rand(2, 3, 6)
    for t in (0., 1., 2., 3.):
        torch.testing.assert_close(net.temporal_factor(quality, t), torch.full((2, 3, 1), math.exp(-t/3)))
    net.temporal_factor(quality, 3.).sum().backward()
    assert net.reliability_gate[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        net.reliability_gate[-1].weight.fill_(1.)
    assert torch.all(net.temporal_factor(quality, 3.) > 0)
    assert torch.all(net.temporal_factor(quality, 3.) < 1)
    assert not torch.equal(net.temporal_factor(quality, 3.), net.temporal_factor(quality + 1, 3.))


@pytest.mark.parametrize('empty', [False, True])
def test_invalid_or_empty_radar_identity_finite_gradients(empty):
    net = network()
    radar = points()[:, :0] if empty else points()
    if not empty:
        radar[..., 3] = float('nan')
    q = torch.randn(2, 3, 8, requires_grad=True)
    result = net(q, torch.zeros(2, 3, 3), radar, torch.ones(radar.shape[:2], dtype=torch.bool), 3.)
    assert torch.equal(result, q)
    result.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_official_zero_residual_parity_all_horizons():
    net = network(radius=40.)
    torch.nn.init.zeros_(net.output.weight)
    q, centers, radar = torch.randn(2, 3, 8), torch.zeros(2, 3, 3), points()
    for t in (0., 1., 2., 3.):
        assert torch.equal(net(q, centers, radar, torch.ones(2, 5, dtype=torch.bool), t), q)


def test_permutation_scene_isolation_and_new_gradients():
    net = network(radius=40.)
    q, centers, radar = torch.randn(2, 3, 8), torch.zeros(2, 3, 3), points()
    valid = torch.ones(2, 5, dtype=torch.bool)
    result = net(q, centers, radar, valid, 2.)
    order = torch.tensor([4, 2, 1, 0, 3])
    torch.testing.assert_close(result, net(q, centers, radar[:, order], valid[:, order], 2.))
    for b in range(2):
        torch.testing.assert_close(result[b:b+1], net(q[b:b+1], centers[b:b+1], radar[b:b+1], valid[b:b+1], 2.))
    result.square().sum().backward()
    assert net.velocity_scale.grad.abs() > 0
    assert net.reliability_gate[-1].weight.grad.abs().sum() > 0


def test_extensions_preserve_A_parameter_initialization_and_rng():
    torch.manual_seed(17)
    baseline = module.RadarQueryFusion(embed_dims=8, mode='transport')
    rng = torch.get_rng_state()
    torch.manual_seed(17)
    candidate = network()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
