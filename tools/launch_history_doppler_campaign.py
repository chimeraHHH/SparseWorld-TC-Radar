"""Reserve one immutable four-arm campaign, CPU-test and cache before GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_history_doppler_experiment import ARMS, ROOT, CAMPAIGN, CACHE, utcnow, write_json, verify_snapshot

TESTS = ['tests/test_visual_history_budget.py', 'tests/test_single_sweep_radar.py',
         'tests/test_single_sweep_cache.py', 'tests/test_history_radar_membership.py', 'tests/test_history_doppler_contracts.py',
         'tests/test_history_doppler_campaign.py', 'tests/test_forecast_improvements.py',
         'tests/test_forecast_comparison.py', 'tests/test_radar_fusion.py',
         'tests/test_official_init.py', 'tests/test_m0_contracts.py', 'tests/test_gpu_capacity.py']


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Launcher must hide all GPUs')
    code = Path(__file__).resolve().parents[1]
    os.chdir(code)
    manifest = json.loads((code / 'code_manifest.json').read_text())
    verify_snapshot(code, manifest)
    campaign = ROOT / 'analysis' / CAMPAIGN
    campaign.mkdir(parents=True, exist_ok=False)
    report = dict(state='preflight', launcher_pid=os.getpid(), code=str(code),
                  git_revision=manifest['git_revision'], at_utc=utcnow())
    write_json(campaign / 'submission.json', report)
    try:
        for arm in ARMS:
            for suffix in ('_seed0', '_smoke_seed0'):
                if (ROOT / ('work_dirs/history_' + arm + suffix)).exists():
                    raise FileExistsError('Existing arm outputs: ' + arm + suffix)
        def run(stage, command):
            report.update(stage=stage, at_utc=utcnow())
            write_json(campaign / 'preflight.json', report)
            with (campaign / (stage + '.log')).open('xb') as log:
                subprocess.run([sys.executable] + command, cwd=code, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        run('cpu_tests', ['-m', 'pytest', '-q', *TESTS, '--junitxml=' + str(campaign/'cpu_tests.xml')])
        run('single_sweep_cache', ['tools/precompute_single_sweep_radar.py',
            '--config', 'configs/sw-history-h1-velocity.py', '--output', CACHE, '--workers', '8'])
        for arm in ARMS:
            run('cpu_' + arm, ['tools/check_history_doppler_contracts.py',
                '--config', 'configs/sw-history-' + arm + '.py', '--device', 'cpu',
                '--out', str(campaign / ('cpu_' + arm + '.json'))])
            contract = json.loads((campaign / ('cpu_' + arm + '.json')).read_text())
            if contract['status'] != 'passed' or contract['git_revision'] != manifest['git_revision']:
                raise ValueError('Arm contract mismatch: ' + arm)
            if not contract['initialization']['state_schema']:
                raise ValueError('Missing exact checkpoint schema')
        report.update(state='passed', at_utc=utcnow())
        write_json(campaign / 'preflight.json', report)
        controllers = {}
        for gpu in (1, 0):
            log_path = campaign / ('gpu%d_controller.log' % gpu)
            with log_path.open('xb') as log:
                child = subprocess.Popen([sys.executable, 'tools/run_history_doppler_experiment.py', '--gpu', str(gpu)],
                    cwd=code, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            controllers[str(gpu)] = dict(pid=child.pid, log=str(log_path))
            report.update(state='submitting', controllers=dict(controllers), at_utc=utcnow())
            write_json(campaign / 'submission.json', report)
        report.update(state='submitted', controllers=controllers, at_utc=utcnow())
        write_json(campaign / 'submission.json', report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        report.update(state='failed', error=repr(error), traceback=traceback.format_exc(), at_utc=utcnow())
        write_json(campaign / 'preflight.json', report)
        write_json(campaign / 'submission.json', report)
        raise


if __name__ == '__main__':
    main()
