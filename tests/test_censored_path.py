"""Small mechanistic checks without importing the detection framework."""
import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location('censored_path', Path(__file__).parents[1] / 'models/censored_path.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(batch=2):
    torch.manual_seed(17)
    net = m.SharedCensoredPath(embed_dims=8, hidden_dims=12, pc_range=(-10., -20., -2., 10., 20., 2.))
    features = torch.randn(4 * batch, 3, 8)
    points = torch.rand(4 * batch, 3, 5, 3)
    scores = torch.randn(4 * batch, 3, 5, 17)
    transforms = [torch.eye(4).repeat(batch, 1, 1) for _ in range(4)]
    frames = [torch.full((batch,), f) for f in (0, 2, 4, 6)]
    return net, features, points, scores, transforms, frames


def test_initialization_and_current_bypass_are_exact():
    net, features, points, scores, transforms, frames = fixture()
    output = net(features, points, scores, transforms, frames)
    for mode in range(2):
        assert torch.equal(output['mode_points'][mode], points)
        assert torch.equal(output['mode_scores'][mode], scores)
    assert torch.equal(output['mode_probability'], torch.full((2, 2), .5))
    assert output['mode_separation_m'] == 0
    assert not torch.equal(net.path_encoder['mode_embedding'].weight[0], net.path_encoder['mode_embedding'].weight[1])
    torch.nn.init.normal_(net.path_head.weight)
    shifted = net(features, points, scores, transforms, frames)
    assert torch.equal(shifted['mode_points'][:, :2], points[:2].expand(2, -1, -1, -1, -1))
    assert not torch.equal(shifted['mode_points'][:, 2:], output['mode_points'][:, 2:])
    net.zero_residual_outputs()
    restored = net(features, points, scores, transforms, frames)
    assert torch.equal(restored['mode_points'], output['mode_points'])


def test_real_mmcv_checkpoint_keys_match_torch_and_exclude_scene_extent():
    checkpoint = pytest.importorskip('mmcv.runner.checkpoint')
    net, features, points, scores, transforms, frames = fixture(batch=1)
    wrapper = torch.nn.Module()
    wrapper.pts_bbox_head = torch.nn.Module()
    wrapper.pts_bbox_head.censored_path = net
    extent = net.scene_extent.clone()
    for convert in (wrapper.float, wrapper.half):
        convert()
        native = wrapper.state_dict()
        mmcv_state = checkpoint.get_state_dict(wrapper)
        assert set(mmcv_state) == set(native)
        assert not any(name.endswith('scene_extent') for name in mmcv_state)
        assert 'scene_extent' not in dict(net.named_buffers())
        assert net.scene_extent.dtype == torch.float32
        assert torch.equal(net.scene_extent, extent)
        for name in native:
            assert torch.equal(mmcv_state[name], native[name]), name
    # CPU half kernels need not exist: restore network float before forwarding.
    wrapper.float()
    output = net(features, points, scores, transforms, frames)
    assert torch.equal(output['mode_points'], points.unsqueeze(0).expand(2, *points.shape))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required')
def test_plain_extent_is_moved_explicitly_in_half_cuda_forward():
    net, features, points, scores, transforms, frames = fixture(batch=1)
    net = net.cuda().half()
    # Plain configuration tensors intentionally do not participate in .cuda().
    assert net.scene_extent.device.type == 'cpu' and net.scene_extent.dtype == torch.float32
    features, points, scores = (value.cuda().half() for value in (features, points, scores))
    transforms = [value.cuda() for value in transforms]
    frames = [value.cuda() for value in frames]
    output = net(features, points, scores, transforms, frames)
    assert torch.equal(output['mode_points'], points.unsqueeze(0).expand(2, *points.shape))
    with torch.no_grad():
        net.path_head.bias[0] = .5
    output = net(features, points, scores, transforms, frames)
    assert output['mode_points'].device.type == 'cuda'
    assert output['mode_points'].dtype == torch.float16
    assert torch.isfinite(output['mode_points']).all()
    assert not torch.equal(output['mode_points'][:, 1:], points[None, 1:].expand(2, -1, -1, -1, -1))


def test_displacement_rotation_ignores_translation_and_preserves_horizon_order():
    net, features, points, scores, transforms, frames = fixture()
    with torch.no_grad():
        net.path_head.bias[0] = torch.atanh(torch.tensor(.5))
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transforms[-1][0, :3, :3] = rotation
    transforms[-1][:, :3, 3] = 1000.
    output = net(features, points, scores, transforms, frames)['mode_points']
    # At u=1, a current-frame +3m x residual is a future-frame -3m y.
    torch.testing.assert_close(output[:, 6] - points[6], torch.tensor([0., -3. / 40., 0.]).expand(2, 3, 5, 3))
    torch.testing.assert_close(output[:, 7] - points[7], torch.tensor([3. / 20., 0., 0.]).expand(2, 3, 5, 3))
    torch.testing.assert_close(output[:, 2] - points[2], torch.tensor([1. / 20., 0., 0.]).expand(2, 3, 5, 3))


def test_sequence_energy_is_normalized_and_cannot_switch_modes_at_each_endpoint():
    logits = torch.tensor([2., -3.], requires_grad=True)
    equal = torch.tensor([4., 4.], requires_grad=True)
    torch.testing.assert_close(m.censored_sequence_energy(logits, equal), torch.tensor(4.))
    endpoint_costs = torch.tensor([[0., 10.], [10., 0.]])
    whole = m.censored_sequence_energy(torch.zeros(2), endpoint_costs.mean(-1))
    independent = m.censored_sequence_energy(torch.zeros(2, 2), endpoint_costs.T).mean()
    assert whole > independent + 4.
    loss = m.censored_sequence_energy(logits, torch.tensor([1., 3.]))
    loss.backward()
    assert logits.grad[0] < 0 and logits.grad[1] > 0
    torch.testing.assert_close(loss, m.censored_sequence_energy(logits.detach() + 100., torch.tensor([1., 3.])))


def endpoints(points, observed=None):
    count, frames, sample, _ = points.shape
    if observed is None:
        observed = torch.ones(count, frames, dtype=torch.bool)
    return {'points': points, 'point_valid': observed[..., None].expand(count, frames, sample).clone(),
            'observed': observed, 'labels': torch.zeros(count, dtype=torch.long)}


def test_track_assignment_cannot_swap_queries_between_frames():
    # Two stationary annotated identities; queries exchange their locations.
    truth = torch.zeros(2, 2, 1, 3)
    truth[1, :, :, 0] = 4.
    prediction = torch.zeros(2, 2, 2, 1, 3, requires_grad=True)
    with torch.no_grad():
        prediction[:, 0, 1, 0, 0] = 4.
        prediction[:, 1, 0, 0, 0] = 4.
    scores = torch.zeros(2, 2, 2, 1, 17, requires_grad=True)
    energy, detail = m.tracker_matching_energy(prediction, scores, endpoints(truth), semantic_weight=0., return_details=True)
    torch.testing.assert_close(energy, torch.full((2,), 8.))
    assert all(len(columns.unique()) == 2 for _, columns in detail['assignments'])
    energy.sum().backward()
    assert prediction.grad.abs().sum() > 0 and torch.isfinite(prediction.grad).all()


def test_unknown_coordinates_ignored_and_t0_not_required():
    prediction = torch.zeros(2, 3, 1, 1, 3, requires_grad=True)
    scores = torch.zeros(2, 3, 1, 1, 17, requires_grad=True)
    truth = torch.ones(1, 3, 1, 3)
    truth[:, 0] = float('nan')
    target = endpoints(truth, torch.tensor([[False, True, True]]))
    energy = m.tracker_matching_energy(prediction, scores, target, semantic_weight=0.)
    torch.testing.assert_close(energy, torch.full((2,), 3.))
    target['points'][:, 0] = 1e20
    torch.testing.assert_close(energy, m.tracker_matching_energy(prediction, scores, target, semantic_weight=0.))
    target['observed'][:, 1] = False
    empty = m.tracker_matching_energy(prediction, scores, target)
    assert torch.equal(empty, torch.zeros(2))
    empty.sum().backward()
    assert prediction.grad is not None and scores.grad is not None


def test_allocation_cap_and_invalid_observations_fail_loudly():
    prediction = torch.zeros(2, 2, 1, 1, 3)
    scores = torch.zeros(2, 2, 1, 1, 17)
    target = endpoints(torch.full((1, 2, 1, 3), float('nan')))
    with pytest.raises(ValueError, match='nonfinite'):
        m.tracker_matching_energy(prediction, scores, target)
    with pytest.raises(ValueError, match='cap'):
        m.tracker_matching_energy(prediction, scores, endpoints(torch.zeros(33, 2, 1, 3)))


def test_semantic_matching_uses_the_observed_class():
    prediction = torch.zeros(2, 2, 1, 1, 3)
    scores = torch.zeros(2, 2, 1, 1, 17)
    scores[0, :, :, :, 0] = 5.
    scores[1, :, :, :, 0] = -5.
    energy = m.tracker_matching_energy(prediction, scores, endpoints(torch.zeros(1, 2, 1, 3)))
    assert energy[0] < energy[1]


def test_new_parameter_groups_receive_real_updates_after_zero_init():
    net, features, points, scores, transforms, frames = fixture(batch=1)
    features.requires_grad_()
    before = {name: value.detach().clone() for name, value in net.named_parameters()}
    optimizer = torch.optim.AdamW(net.parameters(), lr=.02, weight_decay=0.)
    for _ in range(4):
        optimizer.zero_grad()
        output = net(features, points, scores, transforms, frames)
        # Fixed observation wants a displaced endpoint; its target is mode-free.
        errors = (output['mode_points'][:, 1:, ..., 0] - points[None, 1:, ..., 0] - .1).square().mean((1, 2, 3))
        loss = m.censored_sequence_energy(output['mode_logits'][0], errors, tau=.01)
        loss.backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in net.parameters() if parameter.grad is not None)
        optimizer.step()
    for group in ('path_encoder', 'path_head', 'mixture_head'):
        assert any(not torch.equal(before[name], value) for name, value in net.named_parameters() if name.startswith(group))
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert net(features, points, scores, transforms, frames)['mode_separation_m'] > 0
