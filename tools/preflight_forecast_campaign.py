"""CPU verification for the exact immutable source used by both GPU queues."""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preflight must hide GPUs'
code = Path(__file__).resolve().parents[1]
os.chdir(code)
root = Path('/storage/data/metaiot_data/huayiming/SparseWorld/analysis/forecast_campaign_20260920')
root.mkdir(parents=True, exist_ok=True)
revision = json.loads((code/'code_manifest.json').read_text())['git_revision']
report = dict(state='running', git_revision=revision,
              at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/'preflight.json').write_text(json.dumps(report, indent=2))
try:
    with (root/'cpu_tests.log').open('wb') as log:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_forecast_improvements.py',
                        'tests/test_forecast_comparison.py', 'tests/test_radar_fusion.py',
                        'tests/test_official_init.py', 'tests/test_radar_cache.py',
                        'tests/test_m0_contracts.py', '--junitxml='+str(root/'cpu_tests.xml')],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    for arm in ('transport', 'balanced'):
        with (root/('cpu_'+arm+'.log')).open('wb') as log:
            subprocess.run([sys.executable, 'tools/check_forecast_contracts.py', '--device', 'cpu',
                            '--config', 'configs/sw-radar-forecast-'+arm+'.py',
                            '--out', str(root/('cpu_'+arm+'.json'))],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    report['state'] = 'passed'
except BaseException as error:
    report.update(state='failed', error=repr(error))
    raise
finally:
    report['at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root/'preflight.json').write_text(json.dumps(report, indent=2))
print('FORECAST_CPU_PREFLIGHT_PASSED', revision, flush=True)
