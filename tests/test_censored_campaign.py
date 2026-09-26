"""Safety and scientific audit contracts for the isolated new-arm controller."""
import json
from types import SimpleNamespace

import pytest

from tools.launch_censored_path_campaign import reserve_submission
from tools.run_censored_path_experiment import (
    audit_checkpoint, audit_learning_log, require_fresh_initialization,
    require_full_result, validate_state_schema,
)


COMPONENTS = ['path_encoder', 'path_head', 'mixture_head']


def learning_log(path_values=None):
    lines = []
    for index, iteration in enumerate(range(4, 25, 4)):
        lines.append('RADAR_LEARNING microstep_grad_norm=1.0 parameter_delta=2e-4')
        pretrained = {name: dict(gradient_norm=1., parameter_delta=.1)
                      for name in ('img_backbone', 'img_neck', 'pts_bbox_head')}
        lines.append('JOINT_LEARNING ' + json.dumps(pretrained))
        value = (0. if index < 4 else 1.) if path_values is None else path_values[index]
        components = {name: dict(gradient_norm=value, parameter_delta=value)
                      for name in COMPONENTS}
        lines.append('CENSORED_PATH_LEARNING iter=%d %s' % (iteration, json.dumps(components)))
    return '\n'.join(lines)


def test_failed_or_empty_campaign_cannot_be_resubmitted(tmp_path):
    campaign = tmp_path / 'new_campaign'
    receipt = reserve_submission(campaign, dict(state='preflight', launcher_pid=123))
    before = receipt.read_text()
    with pytest.raises(FileExistsError):
        reserve_submission(campaign, dict(state='preflight', launcher_pid=456))
    assert receipt.read_text() == before
    empty = tmp_path / 'empty_existing_campaign'
    empty.mkdir()
    with pytest.raises(FileExistsError):
        reserve_submission(empty, {})


def test_schema_checks_keys_and_shapes_not_just_tensor_count():
    state = {'official.weight': SimpleNamespace(shape=(2, 3)),
             'path.weight': SimpleNamespace(shape=(4,))}
    validate_state_schema(state, {'official.weight': [2, 3], 'path.weight': [4]})
    with pytest.raises(ValueError):
        validate_state_schema(state, {'official.weight': [2, 3], 'other.weight': [4]})
    with pytest.raises(ValueError):
        validate_state_schema(state, {'official.weight': [3, 2], 'path.weight': [4]})


def test_zero_initialized_components_may_warm_up_but_must_eventually_learn():
    report = audit_learning_log(learning_log(), COMPONENTS)
    assert report['radar_audit_windows'] == report['pretrained_audit_windows'] == 6
    assert len(report['worldline_audit_windows']) == 6
    assert all(value == 2. for component in report['last_two_worldline_windows_totals'].values()
               for value in component.values())
    # One genuine update in the final two windows is sufficient; first-window
    # zero gradients from the identity initialization do not create a false fail.
    audit_learning_log(learning_log([0, 0, 0, 0, 0, 1]), COMPONENTS)


def test_early_updates_do_not_hide_a_dead_component_at_end_of_smoke():
    with pytest.raises(ValueError, match='last two'):
        audit_learning_log(learning_log([1, 1, 1, 1, 0, 0]), COMPONENTS)


def test_missing_component_and_nonfinite_audits_fail():
    with pytest.raises(ValueError, match='component audit'):
        audit_learning_log(learning_log(), COMPONENTS + ['missing_component'])
    with pytest.raises(ValueError, match='Invalid learning audit'):
        audit_learning_log(learning_log([0, 0, 0, 0, float('nan'), 1]), COMPONENTS)
    with pytest.raises(ValueError, match='Invalid learning audit'):
        audit_learning_log(learning_log().replace('parameter_delta=2e-4', 'parameter_delta=inf', 1), COMPONENTS)


def test_audit_requires_real_increasing_smoke_iterations():
    with pytest.raises(ValueError, match='increase'):
        audit_learning_log(learning_log().replace('iter=24', 'iter=20'), COMPONENTS)
    with pytest.raises(ValueError, match='outside'):
        audit_learning_log(learning_log().replace('iter=24', 'iter=28'), COMPONENTS)


def test_pretrained_and_radar_updates_are_not_optional():
    lines = learning_log().splitlines()
    with pytest.raises(ValueError, match='six radar'):
        audit_learning_log('\n'.join(line for line in lines if not line.startswith('RADAR_LEARNING')), COMPONENTS)
    with pytest.raises(ValueError, match='Pretrained component'):
        audit_learning_log(learning_log().replace('"parameter_delta": 0.1', '"parameter_delta": 0.0', 1), COMPONENTS)


def test_full_evaluation_requires_all_horizons_and_5119_anchors(tmp_path):
    result = tmp_path / 'normal.json'
    confusions = tmp_path / 'confusions'
    confusions.mkdir()
    result.write_text(json.dumps(dict(samples=5119)))
    for name in ('0.0s', '1.0s', '2.0s'):
        (confusions / (name + '.npz')).touch()
    with pytest.raises(FileNotFoundError):
        require_full_result(result, confusions)
    (confusions / '3.0s.npz').touch()
    require_full_result(result, confusions)
    result.write_text(json.dumps(dict(samples=256)))
    with pytest.raises(ValueError, match='5119'):
        require_full_result(result, confusions)


def test_formal_initialization_cannot_restore_optimizer_or_use_other_checkpoint(tmp_path):
    path = tmp_path / 'official_initialization.json'
    payload = dict(loaded_tensors=669, optimizer_restored=False, epoch_reset_to=0,
                   all_camera_tensors_exact=True, sha256='official_hash')
    path.write_text(json.dumps(payload))
    require_fresh_initialization(path, dict(sha256='official_hash'))
    with pytest.raises(ValueError, match='checkpoint'):
        require_fresh_initialization(path, dict(sha256='smoke_hash'))
    payload['optimizer_restored'] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='optimizer_restored'):
        require_fresh_initialization(path, dict(sha256='official_hash'))


def test_checkpoint_audit_detects_nonfinite_optimizer_and_incomplete_smoke(tmp_path):
    torch = pytest.importorskip('torch')
    path = tmp_path / 'iter_24.pth'
    payload = dict(state_dict={'path.weight': torch.ones(2)},
                   optimizer={'state': {0: dict(step=torch.tensor(24.), exp_avg=torch.ones(2))}},
                   meta=dict(epoch=0, iter=24))
    torch.save(payload, path)
    audit_checkpoint(path, {'path.weight': [2]}, exact_steps=24)
    payload['optimizer']['state'][0]['step'] = torch.tensor(23.)
    torch.save(payload, path)
    with pytest.raises(ValueError, match='24 optimizer'):
        audit_checkpoint(path, {'path.weight': [2]}, exact_steps=24)
    payload['optimizer']['state'][0]['step'] = torch.tensor(24.)
    payload['optimizer']['state'][0]['exp_avg'][0] = float('inf')
    torch.save(payload, path)
    with pytest.raises(FloatingPointError, match='optimizer'):
        audit_checkpoint(path, {'path.weight': [2]}, exact_steps=24)
