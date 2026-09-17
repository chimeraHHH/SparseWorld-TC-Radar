_base_ = './sw-radar-m0.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/m0_smoke'
max_iters = 16
lr_config = dict(by_epoch=False, warmup_iters=8)
checkpoint_config = dict(interval=16, by_epoch=False, max_keep_ckpts=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook', interval=1, reset_flag=True, by_epoch=False)])
