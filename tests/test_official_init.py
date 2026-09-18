import copy
import pytest
import torch
from torch import nn
from official_init import load_camera_state


def pair():
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.camera = nn.Linear(3, 4)
    model.decoder.radar_fusion = nn.Module()
    model.decoder.radar_fusion.output = nn.Linear(4, 4, bias=False)
    model.decoder.radar_fusion.encoder = nn.Linear(3, 4)
    state = {k: torch.randn_like(v) for k, v in model.state_dict().items()
             if '.radar_fusion.' not in k}
    return model, state


def test_all_pretrained_tensors_loaded_and_residual_zero():
    model, state = pair()
    result = load_camera_state(model, {'state_dict': state, 'optimizer': {'ignored': 1}})
    assert result['all_camera_tensors_exact']
    assert model.decoder.radar_fusion.output.weight.count_nonzero() == 0
    assert model.decoder.radar_fusion.encoder.weight.count_nonzero() > 0


@pytest.mark.parametrize('bad', ['missing', 'extra', 'shape', 'nonfinite', 'radar'])
def test_incompatible_pretrained_state_is_rejected(bad):
    model, state = pair()
    key = next(iter(state))
    if bad == 'missing': del state[key]
    elif bad == 'extra': state['unrelated.weight'] = torch.ones(1)
    elif bad == 'shape': state[key] = torch.ones(1)
    elif bad == 'nonfinite': state[key].fill_(float('nan'))
    elif bad == 'radar': state['decoder.radar_fusion.output.weight'] = torch.ones(4, 4)
    with pytest.raises(ValueError):
        load_camera_state(model, {'state_dict': state})
