# Offline deterministic radar features; architecture and global batch stay fixed.
_base_ = './sw-radar-m0-single-bs8.py'
project_root = '/storage/data/metaiot_data/huayiming/SparseWorld'
dataset_root = '/storage/data/metaiot_data/public_dataset/nuscenes/'
occ_root = project_root + '/datasets/occ3d/gts/'
radar_cache_root = '/home/huayiming/Workspace/SparseWorld-cache/radar_m0_v2_20260918'
work_dir = project_root + '/work_dirs/m0_seed0_single_bs8_cache'
resume_from = project_root + '/work_dirs/m0_seed0_single_bs8/epoch_4.pth'
resume_epoch_boundary = True
resume_previous_iters_per_epoch = 2992
future_frames = [0, 2, 4, 6]
object_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
ida_aug_conf = dict(resize_lim=(0.38, 0.55), final_dim=(256, 704), bot_pct_lim=(0., 0.), rot_lim=(0., 0.), H=900, W=1600, rand_flip=True)
meta_keys = ('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar', 'lidar2cam', 'intrinsics', 'extrinsics', 'sample_idx', 'radar_points')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=7),
    dict(type='LoadCausalRadar', data_root=dataset_root, sweeps_num=5, max_age=0.5, max_points=4096, cache_root=radar_cache_root),
    dict(type='LoadOccFromFile', occ_root=occ_root, data_root=dataset_root, future_frames=future_frames),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=object_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera', 'fut2cur', 'fut_list'], meta_keys=meta_keys)]
data = dict(train=dict(pipeline=train_pipeline))
