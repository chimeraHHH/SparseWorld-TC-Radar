# Shared reference-time radar belief.
_base_ = './sw-radar-m0-official-ft.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/radar_forecast_belief_seed0'
model = dict(pts_bbox_head=dict(transformer=dict(radar_cfg=dict(_delete_=True, mode="belief", neighbors=16, radius=6., heldout_weight=.05))))
custom_hooks = [dict(type='FiniteTrainingLossHook', priority='HIGH'),
                dict(type='RadarLearningAuditHook', components=True, interval=10),
                dict(type='JointLearningAuditHook', interval=10),
                dict(type='FinetuneValidationHook', config='configs/sw-radar-forecast-belief.py',
                     samples=256, full_at_end=True, keep_best_trained=True,
                     save_scene_confusions=True, priority='LOW')]
