# Fresh model-only initialization, followed by joint camera/radar fine-tuning.
_base_ = './sw-radar-m0-single-bs8-cache.py'
project_root = '/storage/data/metaiot_data/huayiming/SparseWorld'
work_dir = project_root + '/work_dirs/m0_official_ft_seed0'
official_finetune = True
load_from = project_root + '/checkpoints/sw-tc-small.pth'
resume_from = None
resume_epoch_boundary = False
revise_keys = None
total_epochs = 10
batch_size = 8
optimizer = dict(_delete_=True, type='AdamW', lr=2e-5, weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
        'radar_fusion': dict(lr_mult=10.0)}))
optimizer_config = dict(_delete_=True, type='GradientCumulativeFp16OptimizerHook',
    cumulative_iters=1, loss_scale=dict(init_scale=512., growth_interval=2000),
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(_delete_=True, policy='CosineAnnealing', by_epoch=True,
    warmup='linear', warmup_iters=300, warmup_ratio=0.1, min_lr_ratio=0.1)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
custom_hooks = [dict(type='RadarLearningAuditHook', interval=10),
                dict(type='JointLearningAuditHook', interval=10),
                dict(type='FinetuneValidationHook', config='configs/sw-radar-m0-official-ft.py',
                     samples=256, full_at_end=True, priority='LOW')]
