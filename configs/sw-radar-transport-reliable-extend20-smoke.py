_base_ = './sw-radar-transport-reliable-extend20.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/transport_reliable_extend20_smoke_v2_20261001'
max_iters = 29944
lr_config = dict(by_epoch=False)
checkpoint_config = dict(interval=24, by_epoch=False, max_keep_ckpts=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook', interval=1, reset_flag=True, by_epoch=False)])
custom_hooks = [
    dict(type='FiniteTrainingLossHook', priority='HIGH'),
    dict(type='RadarLearningAuditHook', components=True, transport_components=True, interval=4),
    dict(type='JointLearningAuditHook', interval=4),
    dict(type='ExtensionResumeAuditHook',
         checkpoint='/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/radar_forecast_transport-reliable_seed0/epoch_10.pth',
         smoke=True, priority='NORMAL'),
]
