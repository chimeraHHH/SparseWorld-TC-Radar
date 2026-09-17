#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/data/metaiot_data/huayiming/SparseWorld
ENV_DIR="$ROOT/envs/hym_sparseworld"
export OPENBLAS_NUM_THREADS=1
cd /home/huayiming/Workspace/SparseWorld-TC
"$ENV_DIR/bin/python" tools/prepare_m0_data.py --data-root /storage/data/metaiot_data/public_dataset/nuscenes --out-dir "$ROOT/datasets/infos"
"$ENV_DIR/bin/python" tools/preflight_m0.py --data-root /storage/data/metaiot_data/public_dataset/nuscenes --info-root "$ROOT/datasets/infos" --occ-root "$ROOT/datasets/occ3d/gts" --report "$ROOT/logs/data_preflight.json"
echo DATA_PREPARATION_COMPLETE
