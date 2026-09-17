_base_ = './sw-tc.py'

# H200, huayiming: all durable data, weights, caches and output on the NAS.
project_root = '/storage/data/metaiot_data/huayiming/SparseWorld'
dataset_root = '/storage/data/metaiot_data/public_dataset/nuscenes/'
occ_root = project_root + '/datasets/occ3d/gts/'
info_root = project_root + '/datasets/infos/'
work_dir = project_root + '/work_dirs/m0_seed0'

model = dict(pts_bbox_head=dict(transformer=dict(radar_cfg=dict(
    input_dims=10, num_samples=4, neighbors=8, radius=4.0, max_offset=2.0))))
future_frames = [0, 2, 4, 6]
input_modality = dict(use_lidar=False, use_camera=True, use_radar=True, use_map=False, use_external=True)
object_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
ida_aug_conf = dict(resize_lim=(0.38, 0.55), final_dim=(256, 704), bot_pct_lim=(0., 0.), rot_lim=(0., 0.), H=900, W=1600, rand_flip=True)
meta_keys = ('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar', 'lidar2cam', 'intrinsics', 'extrinsics', 'sample_idx', 'radar_points')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=7),
    dict(type='LoadCausalRadar', data_root=dataset_root, sweeps_num=5, max_age=0.5, max_points=4096),
    dict(type='LoadOccFromFile', occ_root=occ_root, data_root=dataset_root, future_frames=future_frames),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=object_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera', 'fut2cur', 'fut_list'], meta_keys=meta_keys)]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=7, test_mode=True, force_offline=True),
    dict(type='LoadCausalRadar', data_root=dataset_root, sweeps_num=5, max_age=0.5, max_points=4096),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='LoadOccFromFile', occ_root=occ_root, data_root=dataset_root, future_frames=future_frames, load_labels=False),
    dict(type='MultiScaleFlipAug3D', img_scale=(1600, 900), pts_scale_ratio=1, flip=False, transforms=[
        dict(type='DefaultFormatBundle3D', class_names=object_names, with_label=False),
        dict(type='Collect3D', keys=['img', 'fut2cur', 'fut_list'], meta_keys=meta_keys)])]
data = dict(workers_per_gpu=4,
    train=dict(data_root=dataset_root, occ_root=occ_root, ann_file=info_root+'nuscenes_infos_train_sweep_occ.pkl', pipeline=train_pipeline, modality=input_modality),
    val=dict(data_root=dataset_root, occ_root=occ_root, ann_file=info_root+'nuscenes_infos_val_sweep_occ.pkl', pipeline=test_pipeline, modality=input_modality),
    test=dict(data_root=dataset_root, occ_root=occ_root, ann_file=info_root+'nuscenes_infos_val_sweep_occ.pkl', pipeline=test_pipeline, modality=input_modality))
# Start with one scene/GPU. Preserve official effective batch 8 via accumulation.
batch_size = 1
optimizer = dict(lr=2e-4)
lr_config = dict(warmup_iters=4000)  # 500 optimizer updates at accumulation=8
optimizer_config = dict(type='GradientCumulativeFp16OptimizerHook', cumulative_iters=8, loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
load_from = project_root + '/checkpoints/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('^backbone\\.', 'img_backbone.')]
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook', interval=10, reset_flag=True), dict(type='MyTensorboardLoggerHook', interval=100, reset_flag=True)])
seed = 0

custom_hooks = [dict(type="RadarLearningAuditHook", interval=8)]
