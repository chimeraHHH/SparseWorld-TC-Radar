#!/usr/bin/env bash
set -euo pipefail
umask 027
ROOT=/storage/data/metaiot_data/huayiming/SparseWorld
ENV_DIR="$ROOT/envs/hym_sparseworld"
export PATH="$ENV_DIR/bin:/usr/local/cuda-11.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_VISIBLE_DEVICES=GPU-000b6236-3632-a001-9667-1f02cbb61c8b
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
export TMPDIR="$ROOT/cache/tmp"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
cd "$(dirname "$0")/.."
[[ $(id -un) == huayiming ]]
[[ $(hostname) == WHUServer-H200 ]]
[[ $(realpath "$ROOT") == /storage/data/metaiot_data/huayiming/SparseWorld ]]
[[ $(findmnt -n -o FSTYPE -T "$ROOT") == nfs* ]]
exec 9> "$ROOT/training.lock"
flock -n 9 || { echo 'Another SparseWorld launch holds the lock'; exit 1; }
python - "$1" <<'PY'
import json,os,subprocess,sys
from pathlib import Path
from mmcv import Config
root=Path('/storage/data/metaiot_data/huayiming/SparseWorld')
config=Config.fromfile(sys.argv[1])
work=Path(config.work_dir).resolve()
assert work.is_relative_to(root/'work_dirs')
if work.exists() and any(work.iterdir()):
 raise RuntimeError('Work directory already has outputs; inspect state before any resume')
assert json.loads((root/'logs/data_preflight.json').read_text())['passed']
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert os.environ['CUDA_VISIBLE_DEVICES'] not in apps, 'Selected GPU already has a compute process'
from models.csrc.wrapper import MSMV_CUDA
assert MSMV_CUDA, 'CUDA sampler must be compiled before full-resolution training'
work.mkdir(parents=True,exist_ok=True)
(work/'launch.json').write_text(json.dumps(dict(config=sys.argv[1],gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getppid()),indent=2))
print('LAUNCH_PREFLIGHT_PASSED',work,flush=True)
PY
python train.py --config "$1"
