# Matched official-initialized camera-only fine-tuning control.
_base_ = './sw-radar-m0-official-ft.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/radar_forecast_camera_seed0'
model = dict(pts_bbox_head=dict(transformer=dict(radar_cfg=None)))
custom_hooks = [dict(type='FiniteTrainingLossHook', priority='HIGH'),
                dict(type='JointLearningAuditHook', interval=10),
                dict(type='FinetuneValidationHook', config='configs/sw-radar-forecast-camera.py',
                     samples=256, full_at_end=True, keep_best_trained=True,
                     save_scene_confusions=True, priority='LOW')]
