#!/usr/bin/env python3
"""Read-only inspection of the bounded frozen-A diagnostic export."""
import datetime
import json
from pathlib import Path
import subprocess

campaign = Path('/storage/data/metaiot_data/huayiming/SparseWorld/analysis/frozen_A_diagnostics_20260927')
result = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
for name in ('submission', 'status'):
    path = campaign / (name + '.json')
    result[name] = json.loads(path.read_text()) if path.exists() else None
for name in ('controller', 'export'):
    path = campaign / (name + '.log')
    if path.exists():
        result[name + '_tail'] = path.read_text(errors='replace')[-4000:]
manifest = campaign / 'A_frozen32/manifest.json'
if manifest.exists():
    value = json.loads(manifest.read_text())
    result['manifest'] = {key: value.get(key) for key in ('state', 'git_revision', 'checkpoint_sha256', 'checkpoint_meta', 'finite_model_tensors', 'elapsed_seconds', 'error')}
    result['manifest']['samples'] = len(value.get('samples', []))
result['processes'] = {}
for key in ('controller_pid', 'child_pid'):
    pid = (result.get('status') or {}).get(key)
    if pid:
        path = Path('/proc') / str(pid)
        try:
            result['processes'][key] = dict(pid=pid, command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(), cwd=str((path / 'cwd').resolve()), stat=(path / 'stat').read_text())
        except FileNotFoundError:
            result['processes'][key] = dict(pid=pid, exited=True)
result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
print(json.dumps(result, indent=2))
