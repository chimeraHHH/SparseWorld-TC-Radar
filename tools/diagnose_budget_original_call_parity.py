"""Preserve frozen evidence before the original H8 geometry comparisons.

This is a zero-optimization diagnostic, never an admission or a retry queue.
The scientific gpu_contract is compiled with one CPU-only persistence callback
before its comparisons. Every original statement and native model/get_occ call
remains unchanged. Persistence changes elapsed time between anchors; it cannot
establish why a prior, unrecorded mismatch occurred.
"""
import argparse
import ast
import copy
import datetime
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import traceback

REVISION = '0f33492d1b639da716897d8faa4a1df293354c49'
SCIENCE = Path('/home/huayiming/Workspace/SparseWorld-TC-forecast-0f33492d1b63')
ROOT = Path('/storage/data/metaiot_data/huayiming/SparseWorld')
CAMPAIGN = ROOT / 'analysis/history_budget_mainline_20261001'
UUID = 'GPU-000b6236-3632-a001-9667-1f02cbb61c8b'
FAILURE_BINDINGS = {
    'h8-geometry_claim.json': '96324a96c8658cc35a32cff6b43252b65b90c7ad392f45ef2d8675664870c57f',
    'parallel_gpu1_failed_h8_geometry_status_20261002.json': '7df7524180687728d931e6e7411fd9f117b98eae551a08ccf7cb53f977c417f3',
}
CALLBACK = '_persist_original_frozen_pair'


def instrument_source(source):
    """Reject an unexpected checker shape; add exactly one capture statement."""
    original = ast.parse(textwrap.dedent(source))
    if len(original.body) != 1 or not isinstance(original.body[0], ast.FunctionDef):
        raise ValueError('Expected one original gpu_contract function')
    if original.body[0].name != 'gpu_contract':
        raise ValueError('Unexpected original function name')
    if any(isinstance(n, ast.Name) and n.id == CALLBACK for n in ast.walk(original)):
        raise ValueError('Persistence callback already present')
    tree = copy.deepcopy(original)
    candidates = []
    for node in ast.walk(tree):
        for _field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for i, statement in enumerate(value):
                if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == 'tensors'
                        and isinstance(statement.value, ast.ListComp)):
                    candidates.append((value, i))
    if len(candidates) != 1:
        raise ValueError('Original raw comparison insertion point is not unique')
    body, position = candidates[0]
    capture = ast.parse(CALLBACK + '(index, captures, left, right)').body[0]
    body.insert(position, capture)
    # Removing the one inserted statement must recover the complete original
    # AST. This check covers model calls, order, comparisons and exception flow.
    stripped = copy.deepcopy(tree)
    removed = 0
    for node in ast.walk(stripped):
        for _field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for statement in list(value):
                    if (isinstance(statement, ast.Expr)
                            and isinstance(statement.value, ast.Call)
                            and isinstance(statement.value.func, ast.Name)
                            and statement.value.func.id == CALLBACK):
                        value.remove(statement)
                        removed += 1
    if removed != 1 or ast.dump(stripped) != ast.dump(original):
        raise ValueError('Diagnostic changed an original scientific statement')
    return ast.fix_missing_locations(tree)


def instrument_contract(function, persist):
    namespace = dict(function.__globals__)
    namespace[CALLBACK] = persist
    source = inspect.getsource(function)
    exec(compile(instrument_source(source), '<original-contract-with-cpu-capture>', 'exec'), namespace)
    return namespace['gpu_contract'], hashlib.sha256(source.encode()).hexdigest()


def failure_evidence():
    evidence = {}
    for name, expected in FAILURE_BINDINGS.items():
        raw = (CAMPAIGN / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Original failed evidence changed: ' + name)
        evidence[name] = dict(sha256=expected, bytes=len(raw))
    # Never signal these PIDs. Conservatively reject any extant /proc entry.
    for pid in (3321425, 3323298):
        if Path('/proc/%d' % pid).exists():
            raise RuntimeError('Original failed worker/child must be absent')
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='diagnostic_preparing', optimization_steps=0,
        science_revision=REVISION, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Original native calls once per model/anchor; CPU persistence before unchanged raw/voxel checks; not admission, training or scientific accuracy.',
        timing_limit='CPU file I/O alters timing between anchors; a non-reproduction does not explain the earlier failure.',
        anchors=[], artifacts={})
    os.chdir(SCIENCE)
    sys.path[:0] = [str(SCIENCE), str(SCIENCE / 'tools')]
    from run_history_doppler_experiment import verify_snapshot, write_json
    from gpu_capacity import CapacityWindow, memory_snapshot

    def save():
        write_json(output / 'diagnostic.json', report)

    save()
    try:
        manifest = json.loads((SCIENCE / 'code_manifest.json').read_text())
        if manifest['git_revision'] != REVISION:
            raise ValueError('Wrong frozen science manifest')
        verify_snapshot(SCIENCE, manifest)
        report['failed_evidence_binding'] = failure_evidence()
        with (ROOT / 'forecast_gpu1.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            window = CapacityWindow()
            while True:
                snapshot = memory_snapshot(UUID)
                compute = subprocess.check_output(['nvidia-smi', '-i', UUID,
                    '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:'))) * 1024
                now = time.monotonic()
                admitted = window.observe(snapshot['free_mib'] if not compute and available >= 64 * 1024**3 else 0, now)
                report.update(status='diagnostic_resource_wait', admission=dict(**snapshot,
                    other_compute=compute, host_available_bytes=available,
                    stable_seconds=now - window.sufficient_since if window.sufficient_since else 0))
                save()
                if admitted:
                    break
                time.sleep(15)
            failure_evidence()
            os.environ['CUDA_VISIBLE_DEVICES'] = UUID
            import numpy as np
            import torch
            from mmcv import Config
            from mmdet3d.models import build_model
            import models, loaders  # noqa: F401
            import tools.check_history_budget_contracts as checker
            from official_init import initialize_official
            torch.set_num_threads(2)
            torch.manual_seed(0)
            cfg = Config.fromfile('configs/sw-budget-h8-geometry.py')
            baseline = Config.fromfile('configs/sw-radar-forecast-transport.py')
            contract = checker.configuration_contract(cfg, baseline, 'configs/sw-budget-h8-geometry.py')
            module = build_model(copy.deepcopy(cfg.model))
            module.init_weights()
            initialization = initialize_official(module, cfg.load_from)
            if not (initialization['loaded_tensors'] == 669 and initialization['new_radar_tensors'] == 102
                    and initialization['new_path_tensors'] == 0 and len(initialization['zero_residual_outputs']) == 6
                    and initialization['all_camera_tensors_exact'] and not initialization['optimizer_restored']):
                raise ValueError('Original official initialization contract failed')
            reference_cfg = copy.deepcopy(baseline.model)
            reference_cfg['visual_history_frames'] = None
            reference_cfg['pts_bbox_head']['transformer']['radar_cfg'] = None
            reference = build_model(reference_cfg)
            reference.init_weights()
            reference_initialization = initialize_official(reference, cfg.load_from)
            if len(module.state_dict()) != 771 or len(reference.state_dict()) != 669:
                raise ValueError('Model schema count differs')
            if not all(torch.isfinite(tensor).all() for tensor in module.state_dict().values()):
                raise FloatingPointError('Nonfinite official model')
            for key, tensor in reference.state_dict().items():
                if not torch.equal(tensor, module.state_dict()[key]):
                    raise ValueError('Reference official tensor differs: ' + key)
            report.update(initialization=initialization, reference_initialization=reference_initialization,
                          status='diagnostic_running')

            def persist(index, captures, left, right):
                started = time.monotonic()
                if index not in checker.INDICES or index in [row['index'] for row in report['anchors']]:
                    raise ValueError('Unexpected/duplicate captured anchor')
                raw = {tag: captures[tag][0] for tag in ('new', 'camera_reference')}
                if any(len(values) != 13 or any(tensor.device.type != 'cpu' for tensor in values)
                       for values in raw.values()):
                    raise ValueError('Persistence requires all 13 existing CPU-captured tensors')
                tensors = []
                for a, b in zip(raw['new'], raw['camera_reference']):
                    same_schema = a.shape == b.shape and a.dtype == b.dtype
                    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                    tensors.append(dict(left_shape=list(a.shape), right_shape=list(b.shape),
                        left_dtype=str(a.dtype), right_dtype=str(b.dtype),
                        both_finite=finite,
                        exact=bool(torch.equal(a, b)),
                        max_abs_difference=float((a.float() - b.float()).abs().max())
                            if same_schema and finite and a.numel()
                            else (0. if same_schema and finite else None)))
                raw_path = output / ('anchor_%04d_raw.pt' % index)
                voxel_path = output / ('anchor_%04d_native_voxels.npz' % index)
                torch.save(raw, raw_path)  # Already CPU; no extra model/get_occ/GPU call.
                np.savez_compressed(voxel_path, **{f'{tag}_{h}_{key}': np.asarray(values[key])
                    for tag, result in (('new', left), ('camera_reference', right))
                    for h, values in enumerate(result) for key in ('occ_loc', 'sem_pred')})
                for path in (raw_path, voxel_path):
                    report['artifacts'][path.name] = dict(bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                report['anchors'].append(dict(index=index, tensors=tensors,
                    native_result_horizons=[len(left), len(right)],
                    persisted_before_original_comparisons=True,
                    persistence_seconds=time.monotonic() - started))
                save()  # A subsequent original assertion cannot discard this evidence.

            instrumented, original_sha = instrument_contract(checker.gpu_contract, persist)
            report['original_gpu_contract_source_sha256'] = original_sha
            report['original_statements_unchanged_except_cpu_capture'] = True
            report['original_contract_invoked'] = True
            save()
            report['original_contract_result'] = instrumented(module, reference, cfg, contract)
            report['original_contract_completed'] = True
        report['status'] = 'diagnostic_complete_not_admission'
    except Exception as error:
        report.update(error_phase=report['status'], status='diagnostic_evidence_preserved', original_contract_completed=False,
                      error=repr(error), traceback=traceback.format_exc())
    finally:
        for path in output.iterdir():
            if path.is_file() and path.name != 'diagnostic.json':
                report['artifacts'][path.name] = dict(bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('anchors', 'initialization', 'reference_initialization', 'original_contract_result')}, indent=2))
    return 0 if report.get('original_contract_completed') else 1


if __name__ == '__main__':
    sys.exit(main())
