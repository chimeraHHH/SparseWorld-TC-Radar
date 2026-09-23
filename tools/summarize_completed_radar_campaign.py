"""Read-only consolidation of completed arms, explicitly excluding canceled B.

Run on H200 with CUDA hidden; emit JSON to stdout for local evidence storage.
No training, evaluations, checkpoint loads, or remote writes are performed.
"""
import datetime
import json
import math
import os
from pathlib import Path
import re
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
sys.path.insert(0, '/home/huayiming/Workspace/SparseWorld-TC-forecast-2fc2feaf5edb/tools')
import numpy as np
from compare_forecast_results import compare, load_confusions

ROOT = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
CAMPAIGNS = {'transport': 'forecast_campaign_20260920',
             'belief': 'belief_campaign_20260920', 'camera': 'belief_campaign_20260920',
             'transport-reliable': 'transport_reliable_campaign_20260922'}


def read(path):
    return json.loads(Path(path).read_text())


def metrics(path):
    data = read(path)
    return {k:v for k,v in data.items() if k not in ('indices', 'tokens')}


def costs(work, status):
    lines = (work/'train.log').read_text(errors='replace').splitlines()
    dated = [(line[:25], line) for line in lines if re.match(r'^\[\d{4}-\d\d-\d\d ', line)]
    def timestamp(line):
        match = re.match(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\]', line)
        return datetime.datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=datetime.timezone.utc)
    start = timestamp(dated[0][1])
    finish = datetime.datetime.fromisoformat(status['at_utc'])
    training_lines = [line for _,line in dated if 'Epoch [' in line and 'time:' in line]
    last_step = timestamp(training_lines[-1])
    rows = []
    for path in sorted(work.glob('*.log.json')):
        for line in path.read_text().splitlines():
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            if 'loss' in row: rows.append(row)
    last = (rows[-1]['epoch']-1)*2992+rows[-1]['iter']
    previous, window, logged_seconds = 0, [], 0.
    for row in rows:
        step = (row['epoch']-1)*2992+row['iter']
        logged_seconds += (step-previous)*row['time']
        count = max(0, step-max(previous, last-1000))
        if count: window.append((count,row))
        previous = step
    n = sum(count for count,row in window)
    return dict(formal_log_start_utc=start.isoformat(), completion_utc=finish.isoformat(),
        formal_start_to_all_evaluations_hours=(finish-start).total_seconds()/3600,
        formal_start_to_last_logged_training_step_hours=(last_step-start).total_seconds()/3600,
        weighted_logged_iteration_hours=logged_seconds/3600,
        last1000_seconds_per_step=sum(count*row['time'] for count,row in window)/n,
        last1000_data_time=sum(count*row['data_time'] for count,row in window)/n,
        nonfinite_loss_windows=sum(any(not math.isfinite(v) for k,v in row.items() if 'loss' in k) for row in rows),
        nonfinite_gradient_windows=sum(not math.isfinite(row['grad_norm']) for row in rows),
        note='Wall and logged iteration time include I/O; not measured active GPU compute, energy, or a matched speed benchmark.')


def main():
    report = dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  arms={}, references={}, comparisons={})
    b = read(ROOT/'analysis/forecast_campaign_20260920/balanced_status.json')
    assert b['state'] == 'stopped_by_user'
    report['canceled_B'] = dict(state=b['state'], reason=b.get('stop_reason'),
                              interpretation='User-canceled; excluded from completed-arm comparisons.')
    for arm, name in CAMPAIGNS.items():
        campaign = ROOT/'analysis'/name
        status = read(campaign/(arm+'_status.json'))
        assert status['state'] == 'complete', (arm, status['state'])
        result = read(campaign/(arm+'_result.json'))
        work = ROOT/('work_dirs/radar_forecast_'+arm+'_seed0')
        entry = dict(result=result, best=metrics(result['best_json']), final=metrics(result['final_json']),
                     costs=costs(work,status), interventions={})
        assert entry['best']['samples'] == entry['final']['samples'] == 5119
        for path in sorted((campaign/(arm+'_counterfactual')).glob('*.json')):
            entry['interventions'][path.stem] = metrics(path)
        report['arms'][arm] = entry
    directories = {'official': ROOT/'analysis/forecast_campaign_20260920/reference_official',
                   'M0_final': ROOT/'analysis/forecast_campaign_20260920/reference_m0_final',
                   'M0_best': ROOT/'analysis/radar_forecast_20260920/m0_best_full'}
    for key,path in directories.items():
        data = metrics(path/'normal.json')
        assert data['samples'] == 5119
        report['references'][key] = data
    cache = {}
    def paired(reference, candidate):
        for path in (reference,candidate):
            if str(path) not in cache: cache[str(path)] = load_confusions(path)
        left,ln,li = cache[str(reference)]
        right,rn,ri = cache[str(candidate)]
        assert np.array_equal(ln,rn) and np.array_equal(li,ri) and len(li)==5119
        value = compare(left,right)
        value.update(reference=str(reference),candidate=str(candidate),anchors=len(li))
        return value
    for scope in ('best','final'):
        for arm in ('transport','belief','transport-reliable'):
            report['comparisons'][arm+'_vs_camera_'+scope] = paired(
                report['arms']['camera']['result'][scope+'_confusions'],
                report['arms'][arm]['result'][scope+'_confusions'])
        report['comparisons']['transport-reliable_vs_A_'+scope] = paired(
            report['arms']['transport']['result'][scope+'_confusions'],
            report['arms']['transport-reliable']['result'][scope+'_confusions'])
    for arm in CAMPAIGNS:
        for key in ('official','M0_final'):
            report['comparisons'][arm+'_best_vs_'+key] = paired(
                directories[key]/'confusions_normal', report['arms'][arm]['result']['best_confusions'])
    report['limitations'] = [
        'One seed; scene bootstrap is conditional on these trained checkpoints and does not measure training-seed variance.',
        'Fixed256 selection anchors are inside full5119; best-checkpoint results are not held-out test estimates.',
        'Best and final results are both reported; the subset-selected camera best need not be best on full5119.',
        'Combined velocity association and temporal gating; no component-training ablation or equal-capacity control.',
        'Radar interventions are inference dependence, not replacements for matched camera training controls.',
        'Movable categories include stationary objects; this is not an actual-moving-object evaluation.',
        'Compensated radial proxy and future ego conditioning remain unchanged; no raw-Doppler or unconditioned-driving claim.',
        'Full per-class/horizon metrics and paired class differences are in this JSON; classwise significance is not established.',
        'Wall times include validation, checkpointing, I/O and queue waits; cost figures do not establish a speedup.'
    ]
    def safe(x):
        if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
        if isinstance(x,list):return [safe(v) for v in x]
        if isinstance(x,float) and not math.isfinite(x):return None
        return x
    print(json.dumps(safe(report),indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
