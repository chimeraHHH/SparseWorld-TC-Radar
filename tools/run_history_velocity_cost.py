"""Isolated GPU1 cost queue; no training, geometry, or original claim mutation."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REVISION = '0f33492d1b639da716897d8faa4a1df293354c49'
ARMS = ('h8-velocity', 'h2-velocity')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    raw = json.dumps(value, indent=2, allow_nan=False)
    if exclusive:
        with Path(path).open('x') as stream:
            stream.write(raw)
    else:
        temp = Path(str(path)+'.tmp')
        temp.write_text(raw)
        temp.replace(path)


def schedule():
    return [(0, ARMS[0]), (0, ARMS[1]), (1, ARMS[1]),
            (1, ARMS[0]), (2, ARMS[0]), (2, ARMS[1])]


def receipt(path, arm):
    row = json.loads(Path(path).read_text())
    if row['arm'] != arm or row['git_revision'] != REVISION:
        raise ValueError('Wrong scientific arm or revision')
    audit = row['final_audit']
    if (audit['epoch'] != 10 or audit['iterations'] != 29920 or
            audit['finite_model_tensors'] != 771 or
            audit['optimizer_finite'] is not True or
            audit['state_schema_matches_cpu'] is not True):
        raise ValueError('Require completed original final checkpoint audit')
    if not 0 < audit['optimizer_steps'] <= 29920:
        raise ValueError('Invalid actual optimizer count')
    for key in ('final_json', 'best_json', 'final_confusions', 'best_confusions', 'interventions'):
        if not Path(row[key]).exists():
            raise ValueError('Incomplete original result: '+key)
    if not Path(audit['path']).is_file():
        raise ValueError('Original final checkpoint absent')
    return row


def process(pid):
    root = Path('/proc')/str(pid)
    stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttime=int(stat[19]), state=stat[0],
                command=(root/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                cwd=os.readlink(root/'cwd'))


def exclude_unpaired_cache(combined):
    """Never combine cached timings from only the replays that passed."""
    complete = all(
        not combined['arms'][arm].get('cache_rejections') and
        len(combined['arms'][arm]['cache_parity']) == 48 and
        all(p.get('voxels_exact') and p.get('warm_voxels_exact') and p.get('warm_reuse_without_extraction')
            for p in combined['arms'][arm]['cache_parity']) and
        {r['replay'] for r in combined['arms'][arm]['replays'] if r['mode']=='chronological_feature_cache'} == {0, 1, 2}
        for arm in ARMS)
    combined['feature_cache_claim_eligible'] = complete
    if not complete:
        for row in combined['arms'].values():
            row['excluded_cache_replays'] = [r for r in row['replays'] if r['mode']=='chronological_feature_cache']
            row['replays'] = [r for r in row['replays'] if r['mode']!='chronological_feature_cache']
    return combined


def main():
    parser = argparse.ArgumentParser()
    for key in ('science', 'source-campaign', 'out', 'gpu-uuid', 'python', 'full-eval-lock'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--legacy-source', help='Preserved source for one zero-update repair diagnostic before the preliminary block')
    args = parser.parse_args()
    science, source, out = Path(args.science), Path(args.source_campaign), Path(args.out)
    code = Path(__file__).resolve().parent
    manifest = json.loads((code.parent/'velocity_cost_manifest.json').read_text())
    original = json.loads((science/'code_manifest.json').read_text())
    assert original['git_revision'] == manifest['science_revision'] == REVISION
    for name, sha in original['sha256'].items():
        assert manifest['science_sha256'][name] == digest(science/name) == sha, name
    for name, sha in manifest['sha256'].items():
        assert digest(code.parent/name) == sha, name
    sys.path.insert(0, str(science/'tools'))
    from gpu_capacity import CapacityWindow, memory_snapshot
    root = source.parents[1]
    out.mkdir(exist_ok=True)
    status = dict(state='prepared', controller_pid=os.getpid(), controller_identity=process(os.getpid()),
                  gpu=1, gpu_uuid=args.gpu_uuid, science_revision=REVISION,
                  cost_revision=manifest['cost_revision'], completed_blocks=[], stage_seconds={},
                  deferred_arms=['h8-geometry'], training_submitted=False, child_pid=None,
                  source_campaign=str(source), at_utc=now())
    write(out/'submission.json', status, True)

    def record(**values):
        status.update(values, at_utc=now())
        write(out/'status.json', status)

    def wait_results(arms):
        while True:
            missing = [arm for arm in arms if not (source/(arm+'_result.json')).exists()]
            if not missing:
                rows = {arm: receipt(source/(arm+'_result.json'), arm) for arm in arms}
                record(result_receipt_sha256={arm: digest(source/(arm+'_result.json')) for arm in arms})
                return rows
            record(state='waiting_for_velocity_result', waiting_for_arms=missing, child_pid=None)
            time.sleep(30)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid,
               PYTHONPATH=str(science)+os.pathsep+str(science/'tools'), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               TORCH_EXTENSIONS_DIR=str(root/'cache/torch_extensions'))

    def block(replay, arm, diagnostic=False):
        name = 'cache_batching_diagnostic' if diagnostic else ('preliminary_' if replay == -1 else 'paired_')+arm+'_replay'+str(replay)
        record(stage=name)
        wait_results((arm,))
        with (root/'forecast_gpu1.lock').open('a') as gpu_lock:
            record(state='waiting_for_original_gpu1_lock')
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            with Path(args.full_eval_lock).open('a') as host_lock:
                record(state='waiting_for_original_full_eval_lock')
                fcntl.flock(host_lock, fcntl.LOCK_EX)
                window = CapacityWindow()
                while True:
                    snap = memory_snapshot(args.gpu_uuid)
                    command = ['nvidia-smi']+(['-i', args.gpu_uuid] if replay == -1 else [])
                    compute = subprocess.check_output(command+['--query-compute-apps=pid,process_name',
                        '--format=csv,noheader'], text=True).strip()
                    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                         if line.startswith('MemAvailable:')))*1024
                    tick = time.monotonic()
                    accepted = window.observe(snap['free_mib'] if not compute and available >= 64*1024**3 else 0, tick)
                    record(state='waiting_for_resources', admission=dict(**snap, compute=compute,
                        host_available_bytes=available, stable_seconds=tick-window.sufficient_since if window.sufficient_since else 0,
                        compute_scope='GPU1 only; GPU0 training may coexist' if replay == -1 else 'all GPUs quiet'))
                    if accepted:
                        break
                    time.sleep(15)
                context = dict(at_utc=now(), gpu_snapshot=subprocess.check_output(['nvidia-smi',
                    '--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu',
                    '--format=csv,noheader'], text=True), host_meminfo=Path('/proc/meminfo').read_text(),
                    role='preliminary; concurrent GPU0 training; excluded from primary paired costs' if replay == -1 else 'paired primary after both velocity arms complete')
                write(out/(name+'_context.json'), context, True)
                cmd = [args.python, str(code/('diagnose_velocity_cache_batching.py' if diagnostic else 'benchmark_history_velocity_block.py')),
                       '--campaign', str(source), '--out', str(out/(name+'.json'))]
                cmd += ['--legacy-source', args.legacy_source] if diagnostic else ['--arm', arm, '--replay', str(replay)]
                started = time.monotonic()
                with (out/(name+'.log')).open('x') as stream:
                    child = subprocess.Popen(cmd, cwd=science, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    record(state='running', child_pid=child.pid, child_identity=process(child.pid), command=cmd,
                           log=str(out/(name+'.log')), output=str(out/(name+'.json')), child_returncode=None)
                    while child.poll() is None:
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            record()
                elapsed = time.monotonic()-started
                status['stage_seconds'][name] = elapsed
                record(child_pid=None, child_returncode=child.returncode)
                if child.returncode:
                    raise RuntimeError(name+' failed; preserve all partial evidence; no automatic retry')
                result = json.loads((out/(name+'.json')).read_text())
                if diagnostic:
                    assert result['status'] == 'repaired_parity_passed' and result['optimizer_updates'] == 0
                    status['repair_diagnostic'] = dict(path=str(out/(name+'.json')), sha256=digest(out/(name+'.json')))
                else:
                    assert result['status'] == 'complete' and set(result['arms']) == {arm}
                    assert len(result['arms'][arm]['replays']) in (2, 3)
                status['completed_blocks'].append(name)
                record(state='block_complete')

    try:
        if args.legacy_source:
            block(-1, 'h8-velocity', diagnostic=True)
        block(-1, 'h8-velocity')
        wait_results(ARMS)
        summary_cmd = [args.python, str(code/'summarize_history_velocity.py'), '--campaign', str(source),
                       '--out', str(out/'comparison.json')]
        with (out/'velocity_comparison.log').open('x') as stream:
            record(state='running_cpu_comparison', stage='two_velocity_bootstrap')
            completed = subprocess.Popen(summary_cmd, cwd=science, env=dict(env, CUDA_VISIBLE_DEVICES=''),
                                         stdout=stream, stderr=subprocess.STDOUT)
            record(child_pid=completed.pid, child_identity=process(completed.pid), command=summary_cmd,
                   child_returncode=None, log=str(out/'velocity_comparison.log'))
            while completed.poll() is None:
                try:completed.wait(timeout=30)
                except subprocess.TimeoutExpired:record()
            record(child_pid=None, child_returncode=completed.returncode)
        if completed.returncode:
            raise RuntimeError('Two-velocity bootstrap failed; preserve evidence')
        for replay, arm in schedule():
            block(replay, arm)
        combined = dict(status='complete', science_revision=REVISION,
                        cost_revision=manifest['cost_revision'], measured_gpu_uuid=args.gpu_uuid,
                        schedule=schedule(), preliminary_excluded=True, arms={})
        for replay, arm in schedule():
            result = json.loads((out/('paired_'+arm+'_replay'+str(replay)+'.json')).read_text())
            if 'protocol' not in combined:
                combined['protocol'] = result['protocol']
                combined['protocol']['replays'] = 3
            assert result['protocol']['tokens'] == combined['protocol']['tokens']
            row = result['arms'][arm]
            target = combined['arms'].setdefault(arm, dict(checkpoint_sha256=row['checkpoint_sha256'], replays=[], cache_parity=[]))
            assert target['checkpoint_sha256'] == row['checkpoint_sha256']
            target['replays'].extend(row['replays'])
            target['cache_parity'].extend(row['cache_parity'])
            if 'cache_rejected_reason' in row:
                target.setdefault('cache_rejections', []).append(dict(replay=replay, reason=row['cache_rejected_reason']))
        write(out/'cost_benchmark.json', exclude_unpaired_cache(combined), True)
        record(state='complete', stage='velocity_cost_complete', result=str(out/'cost_benchmark.json'))
    except BaseException as error:
        record(state='failed_evidence_preserved', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
