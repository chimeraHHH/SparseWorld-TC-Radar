_base_ = './sw-radar-m0.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/m0_seed0_dual'
resume_from = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/m0_seed0/epoch_1.pth'
resume_epoch_boundary = True
resume_previous_iters_per_epoch = 23930
# Four scenes per GPU; two GPUs; one optimizer step per iteration = effective8.
batch_size = 8
model = dict(samplewise_loss=True, img_backbone=dict(with_cp=False))
find_unused_parameters = False
preload_nuscenes = True
data = dict(workers_per_gpu=12)
dataloader_options = dict(pin_memory=True, persistent_workers=True, prefetch_factor=2)
optimizer_config = dict(cumulative_iters=1)
# Original warmup already ended in epoch1; retain its sample-count equivalent.
lr_config = dict(warmup_iters=500)
custom_hooks = [dict(type='RadarLearningAuditHook',interval=10)]
