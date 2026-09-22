"""Exclusive CPU preflight and submission for the authorized A extension.

Run once from an immutable deployment with CUDA hidden. Failed submissions are
preserved and never retried automatically; existing A/belief/camera are untouched.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    code = Path(__file__).resolve().parents[1]
    os.chdir(code)
    root = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
    campaign = root/'analysis/transport_reliable_campaign_20260922'
    campaign.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((code/'code_manifest.json').read_text())
    receipt = campaign/'submission.json'
    # Reserve before any work, including CPU preflight, to prevent duplicates.
    with receipt.open('x') as stream:
        json.dump(dict(state='preflight', launcher_pid=os.getpid(), code=str(code),
                       git_revision=manifest['git_revision']), stream, indent=2)
    report = dict(state='running', git_revision=manifest['git_revision'])
    try:
        for name, expected in manifest['sha256'].items():
            assert hashlib.sha256((code/name).read_bytes()).hexdigest() == expected, name
        assert not (campaign/'transport-reliable_status.json').exists()
        with (campaign/'cpu_tests.log').open('wb') as log:
            subprocess.run([sys.executable, '-m', 'pytest', '-q',
                'tests/test_transport_reliability.py', 'tests/test_radar_belief.py',
                'tests/test_forecast_improvements.py', 'tests/test_forecast_comparison.py',
                'tests/test_radar_fusion.py', 'tests/test_official_init.py',
                'tests/test_radar_cache.py', 'tests/test_m0_contracts.py', 'tests/test_gpu_capacity.py',
                '--junitxml='+str(campaign/'cpu_tests.xml')], stdout=log,
                stderr=subprocess.STDOUT, check=True)
        with (campaign/'cpu_transport-reliable.log').open('wb') as log:
            subprocess.run([sys.executable, 'tools/check_forecast_contracts.py', '--device', 'cpu',
                '--config', 'configs/sw-radar-forecast-transport-reliable.py',
                '--out', str(campaign/'cpu_transport-reliable.json')], stdout=log,
                stderr=subprocess.STDOUT, check=True)
        report.update(state='passed', at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (campaign/'preflight.json').write_text(json.dumps(report, indent=2))
        with (campaign/'transport-reliable_controller.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, 'tools/run_belief_experiment.py',
                '--arm', 'transport-reliable', '--gpu', '1'], cwd=code,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result = dict(state='submitted', code=str(code), git_revision=manifest['git_revision'],
                      controller_pid=process.pid, gpu=1, at_utc=report['at_utc'])
        receipt.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        report.update(state='failed', error=repr(error), traceback=traceback.format_exc(),
                      at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (campaign/'preflight.json').write_text(json.dumps(report, indent=2))
        raise


if __name__ == '__main__':
    main()
