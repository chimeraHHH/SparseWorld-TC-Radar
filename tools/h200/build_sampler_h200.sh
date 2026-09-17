#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/data/metaiot_data/huayiming/SparseWorld
export PATH="$ROOT/envs/hym_sparseworld/bin:/usr/local/cuda-11.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=3
export TMPDIR="$ROOT/cache/tmp"
cd /home/huayiming/Workspace/SparseWorld-TC/models/csrc
python setup.py build_ext --inplace --build-temp "$ROOT/cache/build_sampler"
echo SAMPLER_BUILD_COMPLETE
