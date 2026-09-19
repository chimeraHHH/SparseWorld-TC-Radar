# Arm B: original M0 architecture with horizon/category-balanced supervision.
_base_ = './sw-radar-m0-official-ft.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/radar_forecast_balanced_seed0'
model = dict(samplewise_loss=True, pts_bbox_head=dict(forecast_objective=dict(
    horizon_weights=[.25, 1., 1.25, 1.5], movable_weight=2.)))
custom_hooks = [dict(type='FiniteTrainingLossHook', priority='HIGH'),
                dict(type='RadarLearningAuditHook', interval=10),
                dict(type='JointLearningAuditHook', interval=10),
                dict(type='FinetuneValidationHook', config='configs/sw-radar-forecast-balanced.py',
                     samples=256, full_at_end=True, keep_best_trained=True,
                     save_scene_confusions=True, priority='LOW')]
