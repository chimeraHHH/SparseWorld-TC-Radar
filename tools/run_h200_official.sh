#!/usr/bin/env bash
set -euo pipefail
umask 027
ROOT=/storage/data/metaiot_data/huayiming/SparseWorld
ENV_DIR="$ROOT/envs/hym_sparseworld"
export PATH="$ENV_DIR/bin:/usr/local/cuda-11.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_VISIBLE_DEVICES=GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
runtime_tmp=$(mktemp -d /tmp/huayiming-sparseworld-official.XXXXXX)
export TMPDIR="$runtime_tmp"
trap 'rm -rf -- "$runtime_tmp"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
cd "$(dirname "$0")/.."
[[ $(id -un) == huayiming && $(hostname) == WHUServer-H200 ]]
exec 9> "$ROOT/training.lock"
flock -n 9 || { echo 'Another SparseWorld launch holds the lock'; exit 1; }
python - "$1" <<'PY'
import json,os,sys,subprocess,hashlib
from pathlib import Path
from mmcv import Config
from official_init import sha256_file, OFFICIAL_SHA256
cfg=Config.fromfile(sys.argv[1]);manifest=json.loads(Path('code_manifest.json').read_text())
for path,expected in manifest['sha256'].items():assert sha256_file(path)==expected,path
root=Path('/storage/data/metaiot_data/huayiming/SparseWorld');work=Path(cfg.work_dir).resolve()
assert work.is_relative_to(root/'work_dirs')
assert not work.exists() or not any(work.iterdir()),'Output already exists; inspect before resuming'
assert cfg.batch_size==8 and cfg.optimizer_config.cumulative_iters==1
assert cfg.official_finetune and cfg.resume_from is None
assert sha256_file(cfg.load_from)==OFFICIAL_SHA256
audit=json.loads(Path('official_cpu_audit.json').read_text())
assert audit['status']=='passed' and audit['sha256']==OFFICIAL_SHA256
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert os.environ['CUDA_VISIBLE_DEVICES'] not in apps,'Selected GPU occupied'
ready=json.loads((Path(cfg.radar_cache_root)/'COMPLETE.json').read_text())
assert ready['samples']==ready['production_reader_verified_samples']==23930
from models.csrc.wrapper import MSMV_CUDA
assert MSMV_CUDA
work.mkdir(parents=True,exist_ok=True)
(work/'launch.json').write_text(json.dumps(dict(config=sys.argv[1],gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],launcher_pid=os.getppid(),git_revision=manifest['git_revision'],initialization=cfg.load_from,checkpoint_sha256=OFFICIAL_SHA256,optimizer_reset=True,batch_size=8,total_epochs=cfg.total_epochs),indent=2))
(work/'code_manifest.json').write_text(json.dumps(manifest,indent=2))
print('OFFICIAL_FINETUNE_LAUNCH_PREFLIGHT_PASSED',work,flush=True)
PY
python train.py --config "$1"
