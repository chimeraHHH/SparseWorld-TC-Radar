"""Scientific input controls and independent-reference admission failures."""
import copy

import pytest

from tools.check_history_doppler_contracts import (
    CACHE_ROOT, configuration_contract, paired_radar_views, repeat_reference_metadata, tensor_parity,
)


def configurations(history=1, mode='zero'):
    train = [dict(type='LoadMultiViewImageFromFiles'),
             dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=7),
             dict(type='LoadCausalRadar', data_root='/dataset', sweeps_num=5,
                  max_age=.5, max_points=4096, cache_root='/old/cache'),
             dict(type='LoadOccFromFile', load_labels=True)]
    val = copy.deepcopy(train)
    val[1].update(test_mode=True, force_offline=True)
    val[-1]['load_labels'] = False
    baseline = dict(batch_size=8, total_epochs=10, seed=0,
        optimizer=dict(type='AdamW', lr=2e-5), optimizer_config=dict(cumulative_iters=1),
        resume_from=None, resume_epoch_boundary=False, official_finetune=True,
        revise_keys=None, load_from='/official.pth', future_frames=[0, 2, 4, 6],
        model=dict(samplewise_loss=True, pts_bbox_head=dict(transformer=dict(
            num_frames=8, radar_cfg=dict(mode='transport', max_speed=35.)))),
        data=dict(workers_per_gpu=12,
            train=dict(ann_file='train.pkl', pipeline=train),
            val=dict(ann_file='val.pkl', pipeline=val),
            test=dict(ann_file='val.pkl', pipeline=copy.deepcopy(val))),
        custom_hooks=[dict(type='FiniteTrainingLossHook', priority='HIGH'),
            dict(type='RadarLearningAuditHook', interval=10),
            dict(type='JointLearningAuditHook', interval=10),
            dict(type='FinetuneValidationHook', config='configs/A.py', samples=256,
                 full_at_end=True, keep_best_trained=True, save_scene_confusions=True)])
    cfg = copy.deepcopy(baseline)
    cfg['model']['visual_history_frames'] = history
    cfg['model']['pts_bbox_head']['transformer']['radar_cfg']['speed_filter'] = False
    for split in ('train', 'val', 'test'):
        pipeline = cfg['data'][split]['pipeline']
        if history == 1:
            pipeline[:] = [stage for stage in pipeline if stage['type'] != 'LoadMultiViewImageFromMultiSweeps']
        radar = next(stage for stage in pipeline if stage['type'] == 'LoadCausalRadar')
        radar.update(type='LoadSingleSweepRadar', sweeps_num=1,
                     cache_root=CACHE_ROOT, velocity_mode=mode)
    cfg['custom_hooks'][-1]['config'] = 'configs/candidate.py'
    return cfg, baseline


@pytest.mark.parametrize('history', [1, 8])
@pytest.mark.parametrize('mode', ['zero', 'processed'])
def test_only_four_declared_input_treatments_are_admitted(history, mode):
    cfg, baseline = configurations(history, mode)
    before = copy.deepcopy(cfg)
    contract = configuration_contract(cfg, baseline, 'configs/candidate.py')
    assert contract['physical_images_per_anchor'] == history * 6
    assert contract['velocity_mode'] == mode
    assert cfg == before, 'Auditing must not mutate the training configuration'


@pytest.mark.parametrize('factor,value', [('batch_size', 4), ('total_epochs', 2), ('seed', 3)])
def test_budget_changes_are_rejected(factor, value):
    cfg, baseline = configurations()
    cfg[factor] = value
    with pytest.raises(ValueError, match='configuration difference'):
        configuration_contract(cfg, baseline)


def test_h1_cannot_load_real_history_or_reduce_official_parameter_slots():
    cfg, baseline = configurations()
    cfg['data']['train']['pipeline'].insert(1, dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=7))
    with pytest.raises(ValueError, match='History image loader'):
        configuration_contract(cfg, baseline)
    cfg, baseline = configurations()
    cfg['model']['pts_bbox_head']['transformer']['num_frames'] = 1
    with pytest.raises(ValueError, match='model except'):
        configuration_contract(cfg, baseline)


def test_velocity_condition_cannot_change_between_training_and_evaluation():
    cfg, baseline = configurations()
    radar = next(s for s in cfg['data']['val']['pipeline'] if s['type'] == 'LoadSingleSweepRadar')
    radar['velocity_mode'] = 'processed'
    with pytest.raises(ValueError, match='between dataset splits'):
        configuration_contract(cfg, baseline)


def test_speed_filter_and_hidden_data_changes_are_rejected():
    cfg, baseline = configurations()
    cfg['model']['pts_bbox_head']['transformer']['radar_cfg']['speed_filter'] = True
    with pytest.raises(ValueError, match='velocity-dependent'):
        configuration_contract(cfg, baseline)
    cfg, baseline = configurations()
    cfg['data']['train']['ann_file'] = 'different_train.pkl'
    with pytest.raises(ValueError, match='data except'):
        configuration_contract(cfg, baseline)


def test_validation_must_use_the_arm_own_configuration():
    cfg, baseline = configurations()
    with pytest.raises(ValueError, match='own treatment'):
        configuration_contract(cfg, baseline, 'configs/wrong.py')


def test_reference_repeats_current_calibration_and_preserves_nonimage_metadata():
    radar = [[float(x) for x in range(10)] for _ in range(6)]
    item = dict(filename=['cam%d' % x for x in range(6)],
                img_timestamp=[100. + x / 100. for x in range(6)],
                lidar2img=[[[x]] for x in range(6)], radar_points=radar,
                ego2lidar=[[1, 0], [0, 1]])
    original = copy.deepcopy(item)
    repeat_reference_metadata([item])
    for key in ('filename', 'img_timestamp', 'lidar2img'):
        assert item[key] == original[key] * 8
    assert item['radar_points'] == radar and item['ego2lidar'] == original['ego2lidar']
    item['lidar2img'][0][0][0] = -1
    assert item['lidar2img'][6] == original['lidar2img'][0]


def test_reference_rejects_missing_current_projections():
    with pytest.raises(ValueError, match='lacks repeated'):
        repeat_reference_metadata([dict(filename=['camera'] * 6)])


def test_parity_records_exactness_and_rejects_nonfinite_or_dtype_drift():
    torch = pytest.importorskip('torch')
    left = torch.tensor([1., 2.])
    assert tensor_parity(left, left.clone(), 'equal')['exact']
    result = tensor_parity(left, left + 1e-5, 'small roundoff')
    assert not result['exact'] and result['max_abs_difference'] > 0
    with pytest.raises(ValueError, match='dtype'):
        tensor_parity(left, left.double(), 'wrong dtype')
    with pytest.raises(FloatingPointError):
        tensor_parity(left, torch.tensor([float('nan'), 2.]), 'nonfinite')
    with pytest.raises(AssertionError):
        tensor_parity(left, left + .01, 'meaningfully different')


def test_velocity_zeroing_must_preserve_points_order_geometry_and_time():
    np = pytest.importorskip('numpy')
    processed = dict(radar_points=np.arange(20, dtype=np.float32).reshape(2, 10),
        radar_point_sensor_indices=np.array([0, 1], dtype=np.int8),
        radar_reference_timestamp_us=123, radar_sweep_provenance=[dict(channel='front')])
    zero = copy.deepcopy(processed)
    zero['radar_points'][:, [3, 4, 7]] = 0
    assert paired_radar_views(processed, zero)['points'] == 2
    zero['radar_points'][0, 0] += 1
    with pytest.raises(ValueError, match='geometry or membership'):
        paired_radar_views(processed, zero)
    zero = copy.deepcopy(processed)
    with pytest.raises(ValueError, match='retained velocity'):
        paired_radar_views(processed, zero)
