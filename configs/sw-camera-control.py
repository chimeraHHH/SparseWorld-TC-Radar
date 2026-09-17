_base_ = './sw-radar-m0.py'
model = dict(pts_bbox_head=dict(transformer=dict(radar_cfg=None)))
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/camera_control_seed0'
# Identical data, optimizer, initialization, horizons and shared upstream fixes.
# Radar is loaded for matched pipeline/coverage but cannot enter this model.
