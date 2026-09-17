#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/data/metaiot_data/huayiming/SparseWorld
mkdir -p "$ROOT/checkpoints" "$ROOT/datasets/occ3d" "$ROOT/datasets/downloads"
WEIGHT=cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
curl -fL --retry 5 --connect-timeout 15 -C - "https://hf-mirror.com/MrPicklesGG/SparseWorld-TC/resolve/main/$WEIGHT" -o "$ROOT/checkpoints/$WEIGHT.partial"
WEIGHT_SHA=$(sha256sum "$ROOT/checkpoints/$WEIGHT.partial" | cut -d ' ' -f1)
[[ "$WEIGHT_SHA" == 4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e ]]
mv "$ROOT/checkpoints/$WEIGHT.partial" "$ROOT/checkpoints/$WEIGHT"
echo OFFICIAL_PRETRAIN_VERIFIED
