#!/usr/bin/env python3
"""One-shot frozen-A export under the existing GPU1 lock and memory gate."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
CODE = Path('/home/huayiming/Workspace/SparseWorld-TC-forecast-b0c98b3e7768')
GPU = 'GPU-000b6236-3632-a001-9667-1f02cbb61c8b'


def main():
    campaign = Path(sys.argv[1]).resolve()
    selection = campaign / 'selection.json'
    exporter = Path(__file__).resolve().with_name('export_frozen_a_diagnostics.py')
    campaign.mkdir(parents=True, exist_ok=True)
    # Exclusive receipt prevents duplicate submissions, including after failure.
    with (campaign / 'submission.json').open('x') as stream:
        json.dump(dict(controller_pid=os.getpid(), gpu=GPU, code=str(CODE),
                       purpose='Frozen A best epoch9; 32 anchors; no training',
                       exporter_sha256=hashlib.sha256(exporter.read_bytes()).hexdigest(),
                       controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest()), stream, indent=2)
    state = dict(controller_pid=os.getpid(), gpu=GPU, output=str(campaign / 'A_frozen32'))

    def record(**values):
        state.update(values, at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        temporary = campaign / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(campaign / 'status.json')
        print(json.dumps(state), flush=True)

    try:
        record(state='waiting_for_gpu_lock')
        with (ROOT / 'forecast_gpu1.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            since = None
            while True:
                raw = subprocess.check_output(['nvidia-smi', '-i', GPU,
                    '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True)
                free, total = [int(item.strip()) for item in raw.strip().split(',')]
                now = time.monotonic()
                since = (now if since is None else since) if free >= 132000 else None
                stable = 0 if since is None else now - since
                record(state='waiting_for_gpu_memory', free_mib=free, total_mib=total,
                       minimum_free_mib=132000, required_stable_seconds=60, stable_seconds=stable)
                if stable >= 60:
                    break
                time.sleep(15)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', PYTHONPATH=str(CODE),
                       PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TMPDIR='/tmp',
                       TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/torch_extensions'))
            command = [str(ROOT / 'envs/hym_sparseworld/bin/python'), str(exporter),
                '--code', str(CODE), '--checkpoint',
                str(ROOT / 'work_dirs/radar_forecast_transport_seed0/best_future.pth'),
                '--selection-json', str(selection), '--output-dir', str(campaign / 'A_frozen32'),
                '--endpoint-cache', '/home/huayiming/Workspace/SparseWorld-cache/endpoint_segments_v1_20260926']
            with (campaign / 'export.log').open('xb') as log:
                child = subprocess.Popen(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
                record(state='running', child_pid=child.pid, command=command)
                while True:
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        record(state='running')
                record(state='complete' if code == 0 else 'failed', child_pid=None, returncode=code)
                if code:
                    raise RuntimeError('Frozen export failed; preserve receipt and export.log')
            manifest = json.loads((campaign / 'A_frozen32/manifest.json').read_text())
            if manifest['state'] != 'complete' or len(manifest['samples']) != 32:
                raise RuntimeError('Missing complete 32-anchor export')
            record(state='complete', exported_samples=32,
                   checkpoint_sha256=manifest['checkpoint_sha256'])
    except Exception as error:
        record(state='failed', error=type(error).__name__ + ': ' + str(error))
        raise


if __name__ == '__main__':
    main()
