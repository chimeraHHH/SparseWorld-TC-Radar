import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('radar_fusion', Path(__file__).parents[1] / 'models/radar_fusion.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RadarQueryFusion = module.RadarQueryFusion


def fixture():
    torch.manual_seed(7)
    net = RadarQueryFusion(embed_dims=16)
    q = torch.randn(2, 5, 16, requires_grad=True)
    xyz = torch.zeros(2, 5, 3)
    radar = torch.randn(2, 13, 10)
    radar[..., 6] = radar[..., 6].abs() * .1
    valid = torch.ones(2, 13, dtype=torch.bool)
    return net, q, xyz, radar, valid


def test_empty_is_identity_and_backward_finite():
    net, q, xyz, radar, valid = fixture()
    out = net(q, xyz, radar[:, :0], valid[:, :0])
    assert torch.equal(out, q)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_padding_and_no_neighborhood_are_exact_identity():
    net, q, xyz, radar, valid = fixture()
    valid[0] = False
    xyz[1] = 1000
    assert torch.equal(net(q, xyz, radar, valid), q)


def test_return_permutation_invariance():
    net, q, xyz, radar, valid = fixture()
    order = torch.randperm(radar.shape[1])
    torch.testing.assert_close(net(q, xyz, radar, valid), net(q, xyz, radar[:, order], valid[:, order]))


def test_real_motion_features_affect_output_and_learn():
    net, q, xyz, radar, valid = fixture()
    out = net(q, xyz, radar, valid)
    changed = radar.clone()
    changed[..., 3:5] += 10
    assert not torch.allclose(out, net(q, xyz, changed, valid))
    out.square().mean().backward()
    for name in ('point_encoder.0.weight', 'offset.weight', 'gate.weight'):
        grad = dict(net.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_batch_scene_isolation():
    net, q, xyz, radar, valid = fixture()
    combined = net(q, xyz, radar, valid)
    for b in range(2):
        torch.testing.assert_close(combined[b:b+1], net(q[b:b+1], xyz[b:b+1], radar[b:b+1], valid[b:b+1]))
