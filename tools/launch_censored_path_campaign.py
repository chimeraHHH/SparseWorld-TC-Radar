"""Reserve and preflight a new immutable censored-path campaign once."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

if __package__:
    from .run_censored_path_experiment import ARM, CAMPAIGN, CONFIG, ROOT, write_json
else:
    from run_censored_path_experiment import ARM, CAMPAIGN, CONFIG, ROOT, write_json


# Include every new implementation/data/init regression here before deployment.
CPU_TEST_FILES = (
    'tests/test_censored_path.py', 'tests/test_endpoint_segments.py',
    'tests/test_censored_supervision.py',
    'tests/test_censored_campaign.py',
    'tests/test_transport_reliability.py', 'tests/test_radar_belief.py',
    'tests/test_forecast_improvements.py', 'tests/test_forecast_comparison.py',
    'tests/test_radar_fusion.py', 'tests/test_official_init.py',
    'tests/test_radar_cache.py', 'tests/test_m0_contracts.py', 'tests/test_gpu_capacity.py',
)


def reserve_submission(campaign, payload):
    """Refuse even an empty old campaign; failed submissions are not reusable."""
    campaign.mkdir(parents=True, exist_ok=False)
    receipt = campaign / 'submission.json'
    with receipt.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    return receipt


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU launcher requires CUDA_VISIBLE_DEVICES to be empty')
    code = Path(__file__).resolve().parents[1]
    os.chdir(code)
    manifest = json.loads((code / 'code_manifest.json').read_text())
    campaign = ROOT / 'analysis' / CAMPAIGN
    for path in (ROOT / 'work_dirs/radar_forecast_censored-path_seed0',
                 ROOT / 'work_dirs/radar_forecast_censored-path_smoke_seed0'):
        if path.exists():
            raise FileExistsError('New-arm output path already exists: ' + str(path))
    initial = dict(state='preflight', launcher_pid=os.getpid(), code=str(code),
                   git_revision=manifest['git_revision'])
    receipt = reserve_submission(campaign, initial)
    report = dict(state='running', git_revision=manifest['git_revision'])
    try:
        for name, expected in manifest['sha256'].items():
            if hashlib.sha256((code / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Immutable snapshot hash differs: ' + name)
        with (campaign / 'cpu_tests.log').open('xb') as log:
            subprocess.run([sys.executable, '-m', 'pytest', '-q', *CPU_TEST_FILES,
                '--junitxml=' + str(campaign / 'cpu_tests.xml')], cwd=code,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        with (campaign / ('cpu_' + ARM + '.log')).open('xb') as log:
            subprocess.run([sys.executable, 'tools/check_censored_path_contracts.py',
                '--device', 'cpu', '--config', CONFIG,
                '--out', str(campaign / ('cpu_' + ARM + '.json'))], cwd=code,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        contract = json.loads((campaign / ('cpu_' + ARM + '.json')).read_text())
        if (contract['status'] != 'passed' or contract['git_revision'] != manifest['git_revision']
                or not contract['initialization']['state_schema'] or not contract['worldline_components']):
            raise ValueError('Incomplete or mismatched CPU contract')
        report.update(state='passed', at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(campaign / 'preflight.json', report)
        with (campaign / (ARM + '_controller.log')).open('xb') as log:
            process = subprocess.Popen([sys.executable, 'tools/run_censored_path_experiment.py',
                '--gpu', '1'], cwd=code, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        result = dict(state='submitted', code=str(code), git_revision=manifest['git_revision'],
                      controller_pid=process.pid, gpu=1, at_utc=report['at_utc'])
        write_json(receipt, result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        report.update(state='failed', error=repr(error), traceback=traceback.format_exc(),
                      at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(campaign / 'preflight.json', report)
        write_json(receipt, dict(initial, state='failed', error=repr(error), at_utc=report['at_utc']))
        raise


if __name__ == '__main__':
    main()
