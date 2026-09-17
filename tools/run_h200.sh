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
import json,os,subprocess,sys,hashlib
from pathlib import Path
from mmcv import Config
root=Path('/storage/data/metaiot_data/huayiming/SparseWorld')
config=Config.fromfile(sys.argv[1])
manifest=json.loads(Path('code_manifest.json').read_text())
for path, expected in manifest['sha256'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path

work=Path(config.work_dir).resolve()
assert work.is_relative_to(root/'work_dirs')
if work.exists() and any(work.iterdir()):
 raise RuntimeError('Work directory already has outputs; inspect state before any resume')
assert json.loads((root/'logs/data_preflight.json').read_text())['passed']
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert os.environ['CUDA_VISIBLE_DEVICES'] not in apps, 'Selected GPU already has a compute process'
from models.csrc.wrapper import MSMV_CUDA
assert MSMV_CUDA, 'CUDA sampler must be compiled before full-resolution training'
import torch,re
from mmdet3d.models import build_model
model=build_model(config.model)
checkpoint=torch.load(config.load_from,map_location='cpu')
state=checkpoint.get('state_dict',checkpoint)
for pattern,replacement in config.revise_keys:
 state={re.sub(pattern,replacement,k):v for k,v in state.items()}
backbone={k:v for k,v in model.state_dict().items() if k.startswith('img_backbone.') and not k.endswith('num_batches_tracked')}
missing=[k for k,v in backbone.items() if k not in state or state[k].shape != v.shape]
assert not missing, missing
print('PRETRAIN_BACKBONE_COVERAGE',len(backbone),'of',len(backbone),flush=True)
work.mkdir(parents=True,exist_ok=True)
(work/'launch.json').write_text(json.dumps(dict(config=sys.argv[1],gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getppid(),git_revision=manifest['git_revision']),indent=2))
(work/'code_manifest.json').write_text(json.dumps(manifest,indent=2))
print('LAUNCH_PREFLIGHT_PASSED',work,flush=True)
PY
python train.py --config "$1"
