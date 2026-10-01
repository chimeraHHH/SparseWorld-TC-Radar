"""Exercise the real image-budget code on CPU, without detection/CUDA imports.

Only framework imports and decorators are omitted when loading the production
class. Its feature extraction, selection and replication bodies execute intact.
Independent expected tensors are built directly rather than via its helpers.
"""
import ast
import copy
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def source_module():
    source = Path(__file__).parents[1] / 'models/sparse_world.py'
    tree = ast.parse(source.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == '_VISUAL_VIEW_META_KEYS' for t in node.targets):
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                    child.decorator_list = []
            selected.append(node)
    namespace = dict(torch=torch, np=np, copy=copy, queue=queue,
                     MVXTwoStageDetector=torch.nn.Module)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return SimpleNamespace(**namespace)


m = source_module()


class CountBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 2, 1, bias=False)
        self.batch_sizes = []
        with torch.no_grad():
            self.layer.weight.fill_(.25)

    def forward(self, value):
        self.batch_sizes.append(value.shape[0])
        return [self.layer(value)]


def model(budget):
    net = m.SparseWorld.__new__(m.SparseWorld)
    torch.nn.Module.__init__(net)
    net.visual_history_frames = budget
    net.data_aug = None
    net.stop_prev_grad = 0
    net.use_grid_mask = False
    net.with_img_neck = False
    net.img_backbone = CountBackbone()
    net.memory = {'old_anchor': 'must_not_be_read'}
    net.queue = queue.Queue()
    return net


def metadata(batch=2, images=48):
    return [dict(
        filename=[f'b{b}_image{i}' for i in range(images)],
        img_timestamp=[100. - i // 6 for i in range(images)],
        lidar2img=[np.eye(4) * (i + 1) for i in range(images)],
        intrinsics=[np.eye(4) * (i + 2) for i in range(images)],
        extrinsics=[np.eye(4) * (i + 3) for i in range(images)],
        lidar2cam=[np.eye(4) * (i + 4) for i in range(6)],
        img_shape=[(3, 4, 3)] * images,
        ori_shape=[(3, 4, 3)] * images,
        pad_shape=[(3, 4, 3)] * images,
        ego2lidar=np.eye(4),
        radar_points=np.zeros((6, 8)),
        fut2cur=[np.eye(4) * (i + 20) for i in range(6)],
        unrelated=[i for i in range(48)],
    ) for b in range(batch)]


@pytest.mark.parametrize('images', [6, 48])
def test_h1_encodes_current_only_and_replication_preserves_view_and_batch_order(images):
    net = model(1)
    x = torch.arange(2 * images * 3 * 3 * 4, dtype=torch.float32).reshape(2, images, 3, 3, 4)
    metas = metadata(images=images)
    original = copy.deepcopy(metas)
    expected_current = torch.nn.functional.conv2d(
        x[:, :6].reshape(12, 3, 3, 4), net.img_backbone.layer.weight).reshape(2, 6, 2, 3, 4)
    actual = net.extract_feat(x, metas)[0]
    assert net.img_backbone.batch_sizes == [12]
    assert actual.shape == (2, 48, 2, 3, 4)
    for b in range(2):
        for slot in range(8):
            torch.testing.assert_close(actual[b, slot * 6:(slot + 1) * 6], expected_current[b])
            for view in range(6):
                i = slot * 6 + view
                assert metas[b]['filename'][i] == original[b]['filename'][view]
                assert metas[b]['img_timestamp'][i] == original[b]['img_timestamp'][view]
                np.testing.assert_equal(metas[b]['lidar2img'][i], original[b]['lidar2img'][view])
        assert metas[b]['visual_history_encoded_images'] == 6
        assert metas[b]['visual_history_slots'] == 8
        assert len(metas[b]['fut2cur']) == 6
        assert len(metas[b]['unrelated']) == 48
        np.testing.assert_equal(metas[b]['radar_points'], original[b]['radar_points'])
        np.testing.assert_equal(metas[b]['ego2lidar'], original[b]['ego2lidar'])
    # Replication must not alias mutable per-view calibration entries.
    metas[0]['lidar2img'][6][0, 0] = -100
    assert metas[0]['lidar2img'][0][0, 0] == 1


def test_h1_cannot_use_history_pixels_or_history_projection_and_has_correct_gradient():
    net = model(1)
    x = torch.randn(2, 48, 3, 3, 4, requires_grad=True)
    metas = metadata()
    first = net.extract_feat(x, metas)[0]
    altered = x.detach().clone()
    altered[:, 6:] = float('nan')
    altered_meta = metadata()
    for record in altered_meta:
        for projection in record['lidar2img'][6:]:
            projection[:] = float('nan')
        record['img_timestamp'][6:] = [1e30] * 42
    second = net.extract_feat(altered, altered_meta)[0]
    assert torch.equal(first, second)
    assert np.isfinite(np.asarray(altered_meta[0]['lidar2img'])).all()
    first.sum().backward()
    assert torch.equal(x.grad[:, 6:], torch.zeros_like(x.grad[:, 6:]))
    # Eight copies, each read by two output channels with weight 0.25.
    torch.testing.assert_close(x.grad[:, :6], torch.full_like(x.grad[:, :6], 4.))


def test_h8_and_default_have_identical_features_and_parameter_shapes():
    default, full, short = model(None), model(8), model(1)
    x = torch.randn(2, 48, 3, 3, 4)
    meta_default, meta_full = metadata(), metadata()
    expected = default.extract_feat(x, meta_default)[0]
    actual = full.extract_feat(x, meta_full)[0]
    assert torch.equal(actual, expected)
    assert full.img_backbone.batch_sizes == default.img_backbone.batch_sizes == [96]
    assert meta_full[0]['filename'] == meta_default[0]['filename']
    assert meta_full[0]['img_timestamp'] == meta_default[0]['img_timestamp']
    assert set(default.state_dict()) == set(full.state_dict()) == set(short.state_dict())
    for key in default.state_dict():
        assert torch.equal(default.state_dict()[key], short.state_dict()[key])


def test_default_selection_is_noop_even_for_online_six_view_input():
    x, metas = torch.randn(1, 6, 3, 3, 4), metadata(batch=1, images=6)
    filename = metas[0]['filename']
    assert m.select_visual_history_input(x, metas, None) is x
    assert metas[0]['filename'] is filename
    assert 'visual_history_slots' not in metas[0]


@pytest.mark.parametrize('budget,images', [(1, 12), (8, 6), (8, 42), (2, 6), (True, 6)])
def test_bad_visual_budgets_or_counts_fail_instead_of_silent_padding(budget, images):
    with pytest.raises(ValueError):
        m.select_visual_history_input(torch.zeros(1, images, 3, 3, 4), metadata(1, images), budget)


@pytest.mark.parametrize('key', ['filename', 'img_timestamp', 'lidar2img'])
def test_missing_or_misaligned_required_metadata_fails(key):
    x, metas = torch.randn(1, 6, 3, 3, 4), metadata(1, 6)
    metas[0][key] = metas[0][key][:5]
    with pytest.raises(ValueError, match=key):
        m.select_visual_history_input(x, metas, 1)


def test_h1_online_cannot_read_previous_anchor_cache():
    net = model(1).eval()
    net.simple_test_pts = lambda features, meta, *args, **kwargs: (features, meta)
    result, metas = net.simple_test_online(metadata(1, 6), [], [], torch.randn(1, 6, 3, 3, 4))
    assert net.img_backbone.batch_sizes == [6]
    assert result[0].shape[1] == 48
    assert net.memory == {'old_anchor': 'must_not_be_read'}
    assert metas[0]['visual_history_frames'] == 1


def test_h1_gradient_partition_does_not_reencode_replicated_slots():
    net = model(1)
    net.stop_prev_grad = 1
    x = torch.randn(2, 6, 3, 3, 4, requires_grad=True)
    net.extract_feat(x, metadata(2, 6))[0].sum().backward()
    assert net.img_backbone.batch_sizes == [12]
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_h8_online_entry_requires_the_complete_offline_budget():
    net = model(8).eval()
    net.simple_test_pts = lambda features, meta, *args, **kwargs: features
    features = net.simple_test_online(metadata(1, 48), [], [], torch.randn(1, 48, 3, 3, 4))
    assert net.img_backbone.batch_sizes == [48]
    assert features[0].shape[1] == 48
    assert net.memory == {'old_anchor': 'must_not_be_read'}
    with pytest.raises(ValueError, match='H8'):
        net.simple_test_online(metadata(1, 6), [], [], torch.randn(1, 6, 3, 3, 4))


def test_current_selection_precedes_color_augmentation_and_shape_updates():
    net = model(1)
    seen = []
    net.color_aug = lambda value: seen.append(value.shape[0]) or value
    net.data_aug = dict(img_color_aug=True, img_norm_cfg=dict(mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=False))
    metas = metadata(2, 48)
    features = net.extract_feat(torch.randn(2, 48, 3, 3, 4), metas)
    assert seen == [12]
    assert net.img_backbone.batch_sizes == [12]
    assert features[0].shape[1] == 48
    assert metas[0]['img_shape'] == [(3, 4, 3)] * 48
    assert metas[0]['ori_shape'] == [(3, 4, 3)] * 48
