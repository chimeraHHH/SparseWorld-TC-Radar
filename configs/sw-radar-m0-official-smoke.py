_base_ = './sw-radar-m0-official-ft.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/m0_official_smoke_seed0'
max_iters = 24
lr_config = dict(by_epoch=False, warmup_iters=4)
checkpoint_config = dict(interval=24, by_epoch=False, max_keep_ckpts=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook', interval=1, reset_flag=True, by_epoch=False)])
custom_hooks = [dict(type='RadarLearningAuditHook', interval=4),
                dict(type='JointLearningAuditHook', interval=4)]
