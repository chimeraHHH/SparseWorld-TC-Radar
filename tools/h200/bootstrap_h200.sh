#!/usr/bin/env bash
set -euo pipefail
umask 027
ROOT=/storage/data/metaiot_data/huayiming/SparseWorld
ENV_DIR="$ROOT/envs/hym_sparseworld"
export CONDA_PKGS_DIRS="$ROOT/cache/conda"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export TMPDIR="$ROOT/cache/tmp"
mkdir -p "$TMPDIR"
[[ $(realpath "$ROOT") == /storage/data/metaiot_data/huayiming/SparseWorld ]]
[[ $(findmnt -n -o FSTYPE -T "$ROOT") == nfs* ]]
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  /home/metaiot_guest/miniconda3/bin/conda create -y --override-channels -c conda-forge -p "$ENV_DIR" python=3.10 pip
fi
"$ENV_DIR/bin/python" -m pip install 'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' --index-url https://download.pytorch.org/whl/cu118
"$ENV_DIR/bin/python" -m pip install 'numpy==1.23.5' 'opencv-python==4.8.0.76' 'setuptools<70' wheel ninja packaging psutil
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=6
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
"$ENV_DIR/bin/python" -m pip install --no-build-isolation 'mmcv-full==1.7.1' 'numpy==1.23.5' 'opencv-python==4.8.0.76'
"$ENV_DIR/bin/python" -m pip install 'numpy==1.23.5' 'opencv-python==4.8.0.76' 'mmdet==2.28.2' 'mmsegmentation==0.30.0' 'nuscenes-devkit==1.1.10' 'numba==0.57.1' 'scipy==1.10.1' 'yapf==0.40.1' 'setuptools<70' tensorboard mmengine fvcore 'timm==0.9.5' einops ninja plyfile trimesh scikit-image lyft-dataset-sdk 'huggingface-hub<1' gdown pytest
"$ENV_DIR/bin/python" -m pip install --no-deps --no-build-isolation 'mmdet3d==1.0.0rc6'
"$ENV_DIR/bin/python" -m pip freeze > "$ROOT/logs/environment.freeze.txt"
echo BOOTSTRAP_COMPLETE
