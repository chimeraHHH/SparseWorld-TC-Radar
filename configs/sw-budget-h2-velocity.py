# Corrected T/G layout; explicit real H2/H8 sources; fresh official initialization.
# Controlled history × processed radar-velocity experiment; official parameters stay 8-slot.
_base_ = './sw-radar-forecast-transport.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/history_budget_h2-velocity_20261001_seed0'
model = dict(visual_history_frames=2, pts_bbox_head=dict(transformer=dict(
    radar_cfg=dict(speed_filter=False))))
radar_cache_root = '/home/huayiming/Workspace/SparseWorld-cache/radar_single_sweep_v1_20260927'
dataset_root = '/storage/data/metaiot_data/public_dataset/nuscenes/'
occ_root = '/storage/data/metaiot_data/huayiming/SparseWorld/datasets/occ3d/gts/'
future_frames = [0, 2, 4, 6]
object_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
ida_aug_conf = dict(resize_lim=(0.38, 0.55), final_dim=(256, 704), bot_pct_lim=(0., 0.), rot_lim=(0., 0.), H=900, W=1600, rand_flip=True)
meta_keys = ('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar', 'lidar2cam', 'intrinsics', 'extrinsics', 'sample_idx', 'radar_points', 'reference_timestamp_us', 'img_timestamp_us', 'visual_source_kind', 'visual_source_id', 'visual_time_delta_s', 'visual_history_choices')
train_pipeline = [dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadBudgetedVisualHistory', frames=2),
    dict(type='LoadSingleSweepRadar', data_root=dataset_root, sweeps_num=1, max_age=0.5, max_points=4096, cache_root=radar_cache_root, velocity_mode='processed'),
    dict(type='LoadOccFromFile', occ_root=occ_root, data_root=dataset_root, future_frames=future_frames),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=object_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera', 'fut2cur', 'fut_list'], meta_keys=meta_keys)]
test_pipeline = [dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadBudgetedVisualHistory', frames=2, test_mode=True),
    dict(type='LoadSingleSweepRadar', data_root=dataset_root, sweeps_num=1, max_age=0.5, max_points=4096, cache_root=radar_cache_root, velocity_mode='processed'),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='LoadOccFromFile', occ_root=occ_root, data_root=dataset_root, future_frames=future_frames, load_labels=False),
    dict(type='MultiScaleFlipAug3D', img_scale=(1600, 900), pts_scale_ratio=1, flip=False, transforms=[
        dict(type='DefaultFormatBundle3D', class_names=object_names, with_label=False),
        dict(type='Collect3D', keys=['img', 'fut2cur', 'fut_list'], meta_keys=meta_keys)])]
data = dict(train=dict(pipeline=train_pipeline), val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))
custom_hooks = [dict(type='FiniteTrainingLossHook', priority='HIGH'),
    dict(type='RadarLearningAuditHook', interval=10), dict(type='JointLearningAuditHook', interval=10),
    dict(type='FinetuneValidationHook', config='configs/sw-budget-h2-velocity.py', samples=256,
         full_at_end=True, keep_best_trained=True, save_scene_confusions=True, priority='LOW')]
