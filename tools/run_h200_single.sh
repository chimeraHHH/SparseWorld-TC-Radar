#!/usr/bin/env bash
set -euo pipefail
umask 027
ROOT=/storage/data/metaiot_data/huayiming/SparseWorld
ENV_DIR="$ROOT/envs/hym_sparseworld"
export PATH="$ENV_DIR/bin:/usr/local/cuda-11.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_VISIBLE_DEVICES=GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
runtime_tmp=$(mktemp -d /tmp/huayiming-sparseworld-single.XXXXXX)
export TMPDIR="$runtime_tmp"
trap 'rm -rf -- "$runtime_tmp"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
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
cfg=Config.fromfile(sys.argv[1]);manifest=json.loads(Path('code_manifest.json').read_text())
for path,expected in manifest['sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
root=Path('/storage/data/metaiot_data/huayiming/SparseWorld');work=Path(cfg.work_dir).resolve()
assert work.is_relative_to(root/'work_dirs')
assert not work.exists() or not any(work.iterdir()),'Output already exists; inspect before resuming'
assert cfg.batch_size*cfg.optimizer_config.cumulative_iters==8
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert all(gpu not in apps for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')),'Selected GPU occupied'
import torch
c=torch.load(cfg.resume_from,map_location='cpu')
assert c['meta']['epoch'] >= 1,c['meta']
assert c['meta']['iter'] == c['meta']['epoch'] * cfg.resume_previous_iters_per_epoch,c['meta']
assert 'optimizer' in c and 'fp16' in c['meta']
assert all(torch.isfinite(v).all() for v in c['state_dict'].values())
steps=sorted(set(float(state['step']) for state in c['optimizer']['state'].values()))
assert len(steps)==1 and steps[0]>0,steps
if cfg.get('radar_cache_root'):
    ready=json.loads((Path(cfg.radar_cache_root)/'COMPLETE.json').read_text())
    assert ready['samples']==ready['production_reader_verified_samples']
    assert ready['independent_online_rechecks']>=128
work.mkdir(parents=True,exist_ok=True)
(work/'launch.json').write_text(json.dumps(dict(config=sys.argv[1],gpu_uuids=os.environ['CUDA_VISIBLE_DEVICES'].split(','),launcher_pid=os.getppid(),git_revision=manifest['git_revision'],resume_from=cfg.resume_from,previous_meta=c['meta'],optimizer_steps=steps,radar_cache_root=cfg.get('radar_cache_root')),indent=2))
(work/'code_manifest.json').write_text(json.dumps(manifest,indent=2))
print('SINGLE_BS8_LAUNCH_PREFLIGHT_PASSED',work,flush=True)
PY
python train.py --config "$1"
