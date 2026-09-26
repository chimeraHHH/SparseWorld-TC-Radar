# Shared complete-path hypotheses trained from observed endpoints only.
# Retain A's radar transport, official initialization and single-card BS8 budget.
_base_ = './sw-radar-forecast-transport.py'
project_root = '/storage/data/metaiot_data/huayiming/SparseWorld'
work_dir = project_root + '/work_dirs/radar_forecast_censored-path_seed0'
model = dict(pts_bbox_head=dict(censored_path=dict(
    num_modes=2, max_residual=6., temperature=.25, track_weight=.2)))
optimizer = dict(paramwise_cfg=dict(custom_keys={
    'censored_path': dict(lr_mult=10.)}))

# This extra supervision is available only to the training loss. Validation and
# inference inherit A's unchanged pipeline and do not load endpoint segments.
dataset_root = '/storage/data/metaiot_data/public_dataset/nuscenes/'
occ_root = project_root + '/datasets/occ3d/gts/'
radar_cache_root = '/home/huayiming/Workspace/SparseWorld-cache/radar_m0_v2_20260918'
endpoint_cache_root = '/home/huayiming/Workspace/SparseWorld-cache/endpoint_segments_v1_20260926'
future_frames = [0, 2, 4, 6]
object_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
ida_aug_conf = dict(resize_lim=(0.38, 0.55), final_dim=(256, 704),
                    bot_pct_lim=(0., 0.), rot_lim=(0., 0.), H=900, W=1600,
                    rand_flip=True)
train_meta_keys = ('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img',
                   'img_timestamp', 'ego2lidar', 'lidar2cam', 'intrinsics',
                   'extrinsics', 'sample_idx', 'radar_points', 'endpoint_segments')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=7),
    dict(type='LoadCausalRadar', data_root=dataset_root, sweeps_num=5, max_age=0.5,
         max_points=4096, cache_root=radar_cache_root),
    dict(type='LoadOccFromFile', occ_root=occ_root, data_root=dataset_root,
         future_frames=future_frames),
    dict(type='LoadEndpointSegments', cache_root=endpoint_cache_root, split='train'),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=object_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera', 'fut2cur', 'fut_list'],
         meta_keys=train_meta_keys),
]
data = dict(train=dict(pipeline=train_pipeline))
custom_hooks = [
    dict(type='FiniteTrainingLossHook', priority='HIGH'),
    dict(type='RadarLearningAuditHook', interval=10),
    dict(type='JointLearningAuditHook', interval=10),
    dict(type='CensoredPathLearningAuditHook', interval=10),
    dict(type='FinetuneValidationHook', config='configs/sw-radar-forecast-censored-path.py',
         samples=256, full_at_end=True, keep_best_trained=True,
         save_scene_confusions=True, priority='LOW'),
]
