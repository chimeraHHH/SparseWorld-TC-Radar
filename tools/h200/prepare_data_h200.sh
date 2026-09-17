#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/data/metaiot_data/huayiming/SparseWorld
ENV_DIR="$ROOT/envs/hym_sparse_prepare"
export PIP_CACHE_DIR="$ROOT/cache/pip_prepare"
export TMPDIR="$ROOT/cache/tmp"
export OPENBLAS_NUM_THREADS=1
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$ROOT/envs/hym_sparseworld/bin/python" -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/pip" install 'numpy==1.23.5' 'opencv-python==4.8.0.76' 'nuscenes-devkit==1.1.10' 'scipy==1.10.1' 'scikit-learn==1.3.2' 'matplotlib==3.5.2'
cd /home/huayiming/Workspace/SparseWorld-TC
"$ENV_DIR/bin/python" tools/prepare_m0_data.py --data-root /storage/data/metaiot_data/public_dataset/nuscenes --out-dir "$ROOT/datasets/infos"
"$ENV_DIR/bin/python" tools/preflight_m0.py --data-root /storage/data/metaiot_data/public_dataset/nuscenes --info-root "$ROOT/datasets/infos" --occ-root "$ROOT/datasets/occ3d/gts" --report "$ROOT/logs/data_preflight.json"
echo DATA_PREPARATION_COMPLETE
