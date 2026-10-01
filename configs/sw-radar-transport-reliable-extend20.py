"""Epoch-10 continuation: ten extra epochs with saved per-group learning rates."""
_base_ = './sw-radar-forecast-transport-reliable.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/transport_reliable_extend20_20261001'
resume_from = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/radar_forecast_transport-reliable_seed0/epoch_10.pth'
load_from = None
resume_epoch_boundary = False
total_epochs = 20
lr_config = dict(_delete_=True, policy='CheckpointConstant', by_epoch=True, warmup=None)
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
custom_hooks = [
    dict(type='FiniteTrainingLossHook', priority='HIGH'),
    dict(type='RadarLearningAuditHook', components=True, transport_components=True, interval=10),
    dict(type='JointLearningAuditHook', interval=10),
    dict(type='ExtensionResumeAuditHook', checkpoint=resume_from, smoke=False, priority='NORMAL'),
    dict(type='ExtensionValidationHook', config='configs/sw-radar-transport-reliable-extend20.py',
         samples=256, full_at_end=True, keep_best_trained=True,
         save_scene_confusions=True, priority='LOW'),
]
