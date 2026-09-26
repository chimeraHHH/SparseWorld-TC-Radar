"""Observed/unknown support and strict new-module initialization contracts."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from official_init import load_camera_state


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / ('models/' + name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_observed_free_is_background_and_unknown_is_not_free():
    m = module('censored_supervision')
    semantics = torch.tensor([0, 17, 4, 16]).reshape(4, 1, 1)
    mask = torch.tensor([True, True, False, False]).reshape(4, 1, 1)
    points = torch.tensor([[.5,.5,.5], [1.5,.5,.5], [2.5,.5,.5], [3.5,.5,.5], [-.1,.5,.5], [4.,.5,.5]])
    args = (torch.tensor([0.,0.,0.,4.,1.,1.]), torch.ones(3), 17)
    labels, known = m.observed_prediction_targets(points, semantics, mask, *args)
    assert labels[:2].tolist() == [0,17]
    assert known.tolist() == [True,True,False,False,False,False]
    semantics[2:] = 17
    altered, altered_known = m.observed_prediction_targets(points, semantics, mask, *args)
    assert torch.equal(known, altered_known)
    assert torch.equal(labels[known], altered[known])


def test_unknown_invalid_class_is_ignored_and_lookup_detached():
    m = module('censored_supervision')
    semantics = torch.tensor([255]).reshape(1,1,1)
    points = torch.tensor([[.5,.5,.5]], requires_grad=True)
    _, known = m.observed_prediction_targets(points, semantics, torch.ones_like(semantics),
        torch.tensor([0.,0.,0.,1.,1.,1.]), torch.ones(3),17)
    assert not known.any() and not known.requires_grad


def test_official_initialization_accepts_only_explicit_censored_path_prefix():
    path = module('censored_path')
    net = nn.Module()
    net.pts_bbox_head = nn.Module()
    net.pts_bbox_head.camera = nn.Linear(3,4)
    net.pts_bbox_head.censored_path = path.SharedCensoredPath(embed_dims=8)
    state = {name: torch.randn_like(value) for name,value in net.state_dict().items()
             if not name.startswith('pts_bbox_head.censored_path.')}
    report = load_camera_state(net, {'state_dict': state})
    assert report['all_camera_tensors_exact'] and report['new_path_tensors'] > 0
    assert net.pts_bbox_head.censored_path.path_head.weight.count_nonzero() == 0
    assert net.pts_bbox_head.censored_path.path_encoder['mode_embedding'].weight.count_nonzero() > 0
    net.accidental_missing_module = nn.Linear(1,1)
    with pytest.raises(ValueError):
        load_camera_state(net, {'state_dict': state})
