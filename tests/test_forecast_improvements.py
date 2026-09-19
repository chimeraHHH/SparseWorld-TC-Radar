import importlib.util
from pathlib import Path

import pytest
import torch


def load(name):
    path = Path(__file__).parents[1] / 'models' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


radar_module = load('radar_fusion')
objective = load('forecast_objective')


def test_transport_uses_age_plus_horizon_and_compensated_velocity():
    radar = torch.zeros(1, 2, 10)
    radar[0, 0, :3] = torch.tensor([1., 2., 3.])
    radar[0, 0, 3:5] = torch.tensor([2., -1.])
    radar[0, 0, 6] = .25
    moved, valid = radar_module.advect_radar(radar, 3.)
    torch.testing.assert_close(moved[0, 0, :3], torch.tensor([7.5, -1.25, 3.]))
    torch.testing.assert_close(moved[..., 3:], radar[..., 3:])
    torch.testing.assert_close(radar[0, 0, :3], torch.tensor([1., 2., 3.]))
    assert valid.all()
    with pytest.raises(ValueError):
        radar_module.advect_radar(radar, -1.)


def test_query_coordinate_transform_includes_ego_rotation_and_translation():
    centers = torch.tensor([[[1., 2., 3.]]])
    matrix = torch.tensor([[[0., -1., 0., 10.], [1., 0., 0., 20.],
                            [0., 0., 1., 4.], [0., 0., 0., 1.]]])
    torch.testing.assert_close(radar_module.centers_to_current(centers, matrix),
                               torch.tensor([[[8., 21., 7.]]]))


def test_transport_gives_future_query_evidence_beyond_current_radius():
    torch.manual_seed(4)
    net = radar_module.RadarQueryFusion(embed_dims=8, mode='transport',
                                        radius=1., max_offset=0.)
    query = torch.zeros(1, 1, 8, requires_grad=True)
    centers = torch.tensor([[[10., 0., 0.]]])
    radar = torch.zeros(1, 1, 10)
    radar[..., 0] = 1.
    radar[..., 3] = 3.
    valid = torch.ones(1, 1, dtype=torch.bool)
    future = net(query, centers, radar, valid, horizon_seconds=3.)
    assert not torch.equal(future, query)
    assert torch.equal(net(query, centers, radar, valid, horizon_seconds=0.), query)
    stopped = radar.clone()
    stopped[..., 3:5] = 0
    assert torch.equal(net(query, centers, stopped, valid, horizon_seconds=3.), query)
    future.square().sum().backward()
    assert net.output.weight.grad.abs().sum() > 0


def test_unreliable_and_padding_returns_are_identity_with_finite_backward():
    net = radar_module.RadarQueryFusion(embed_dims=8, mode='transport')
    query = torch.randn(1, 2, 8, requires_grad=True)
    centers = torch.zeros(1, 2, 3)
    radar = torch.zeros(1, 4, 10)
    radar[0, 0, 3] = 1000.
    radar[0, 1, 6] = -.01
    radar[0, 2, 6] = .6
    radar[0, 3, 3] = float('nan')
    output = net(query, centers, radar, torch.ones(1, 4, dtype=torch.bool), 3.)
    assert torch.equal(output, query)
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_zero_residual_preserves_pretrained_queries_at_every_horizon():
    net = radar_module.RadarQueryFusion(embed_dims=8, mode='transport')
    torch.nn.init.zeros_(net.output.weight)
    query = torch.randn(2, 3, 8)
    centers = torch.zeros(2, 3, 3)
    radar = torch.zeros(2, 4, 10)
    radar[..., 6] = .25
    valid = torch.ones(2, 4, dtype=torch.bool)
    for seconds in (0., 1., 2., 3.):
        assert torch.equal(net(query, centers, radar, valid, seconds), query)


def test_category_weighting_preserves_mean_scale_and_emphasizes_movable():
    labels = torch.tensor([4, 7, 11, 15])
    weights = objective.normalized_movable_weights(labels, 2.)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.))
    torch.testing.assert_close(weights[:2], 2 * weights[2:])
    assert torch.equal(objective.normalized_movable_weights(labels, 1.), torch.ones(4))
    # All-vehicle and all-static batches must not change overall loss scale.
    for category in (4, 11):
        assert torch.equal(objective.normalized_movable_weights(torch.full((4,), category), 2.),
                           torch.ones(4))


def test_horizon_gradient_budget_shifts_to_future_without_global_rescaling():
    losses = torch.ones(4, requires_grad=True)
    total = objective.weighted_horizon_mean(list(losses), [.25, 1., 1.25, 1.5])
    assert total.item() == 1.
    total.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([.0625, .25, .3125, .375]))
    with pytest.raises(ValueError):
        objective.weighted_horizon_mean(list(losses), [1., 1.])
