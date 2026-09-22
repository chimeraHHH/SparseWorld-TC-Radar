# A + velocity-consistency soft association + learned temporal reliability.
_base_ = './sw-radar-forecast-transport.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/radar_forecast_transport-reliable_seed0'
model = dict(pts_bbox_head=dict(transformer=dict(radar_cfg=dict(
    velocity_consistency=True, temporal_reliability=True))))
custom_hooks = [dict(type='FiniteTrainingLossHook', priority='HIGH'),
                dict(type='RadarLearningAuditHook', components=True, transport_components=True, interval=10),
                dict(type='JointLearningAuditHook', interval=10),
                dict(type='FinetuneValidationHook', config='configs/sw-radar-forecast-transport-reliable.py',
                     samples=256, full_at_end=True, keep_best_trained=True,
                     save_scene_confusions=True, priority='LOW')]
