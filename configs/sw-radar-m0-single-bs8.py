# User-selected single-H200 continuation; preserve global and effective BS=8.
_base_ = './sw-radar-m0-dual.py'
work_dir = '/storage/data/metaiot_data/huayiming/SparseWorld/work_dirs/m0_seed0_single_bs8'
batch_size = 8
data = dict(workers_per_gpu=12)
optimizer_config = dict(cumulative_iters=1)
