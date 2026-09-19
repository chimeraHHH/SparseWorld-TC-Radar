# Arm A: direct, causal, motion-compensated radar evidence at every horizon.
_base_ = './sw-radar-m0-official-ft.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/radar_forecast_transport_seed0'
model = dict(pts_bbox_head=dict(transformer=dict(radar_cfg=dict(
    mode='transport', max_speed=35., age_decay=.5, horizon_decay=3.))))
custom_hooks = [dict(type='FiniteTrainingLossHook', priority='HIGH'),
                dict(type='RadarLearningAuditHook', interval=10),
                dict(type='JointLearningAuditHook', interval=10),
                dict(type='FinetuneValidationHook', config='configs/sw-radar-forecast-transport.py',
                     samples=256, full_at_end=True, keep_best_trained=True,
                     save_scene_confusions=True, priority='LOW')]
