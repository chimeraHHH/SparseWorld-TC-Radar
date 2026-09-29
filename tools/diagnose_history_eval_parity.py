"""Read-only diagnosis of H1 evaluation reproducibility; never approves parity.

The caller must hold the existing GPU lock and perform GPU/host admission.
This tool never trains, changes a checkpoint, or changes model/configuration code.
It emits observations, including failures of exact equality, not a relaxed gate.
"""
import argparse
import ast
import collections
import copy
import gc
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np


ORIGINAL_REVISION = '1e95cf1d53c8607efea04341e3d864099a2e48cc'


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(chunk)
    return result.hexdigest()


def select_indices(length, samples=16, explicit=None):
    """Select uniformly spaced positions inside the already-frozen subset."""
    fixed = sorted(np.random.RandomState(20260918).choice(
        length, min(256, length), replace=False).tolist())
    if explicit is not None:
        if (not isinstance(explicit, list) or not 1 <= len(explicit) <= 16
                or any(type(index) is not int for index in explicit)
                or len(set(explicit)) != len(explicit)
                or not set(explicit).issubset(fixed)):
            raise ValueError('Explicit indices must be 1 to 16 unique integers inside fixed256')
        return fixed, [fixed.index(index) for index in explicit], list(explicit)
    if not 1 <= samples <= min(16, len(fixed)):
        raise ValueError('Diagnostic sample count must be between 1 and 16')
    positions = np.linspace(0, len(fixed) - 1, samples, dtype=int).tolist()
    return fixed, positions, [fixed[p] for p in positions]


def array_exact(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind in 'fc':
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def exact_tree(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(exact_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(exact_tree(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return array_exact(left, right)
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return type(left) is type(right) and left == right


def array_hash(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def compare_predictions(left, right, dense_shape):
    """Separate sparse row ordering, coordinate collisions and actual semantics."""
    if len(left) != len(right):
        return dict(horizon_count_equal=False, left=len(left), right=len(right))
    result = []
    for reference, observed in zip(left, right):
        views = []
        for item in (reference, observed):
            points, labels = np.asarray(item['occ_loc']), np.asarray(item['sem_pred'])
            if points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),):
                raise ValueError('Invalid sparse prediction shapes')
            if points.dtype.kind not in 'iu' or labels.dtype.kind not in 'iu':
                raise ValueError('Occupancy comparison expects integer coordinates/classes')
            if len(points) and ((points < 0).any() or (points >= np.asarray(dense_shape)).any()):
                raise ValueError('Out-of-grid output')
            order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
            dense = np.full(dense_shape, 17, dtype=labels.dtype)
            dense[points[:, 0], points[:, 1], points[:, 2]] = labels
            unique = len(np.unique(points, axis=0))
            views.append(dict(points=points, labels=labels, order=order, dense=dense,
                              count=len(points), unique=unique))
        a, b = views
        result.append(dict(raw_arrays_exact=array_exact(a['points'], b['points'])
                           and array_exact(a['labels'], b['labels']),
            canonical_sorted_exact=array_exact(a['points'][a['order']], b['points'][b['order']])
                and array_exact(a['labels'][a['order']], b['labels'][b['order']]),
            dense_exact=array_exact(a['dense'], b['dense']),
            changed_dense_voxels=int(np.count_nonzero(a['dense'] != b['dense'])),
            reference_points=a['count'], observed_points=b['count'],
            reference_duplicate_coordinates=a['count'] - a['unique'],
            observed_duplicate_coordinates=b['count'] - b['unique'],
            reference_dense_sha256=array_hash(a['dense']),
            observed_dense_sha256=array_hash(b['dense'])))
    return dict(horizon_count_equal=True, horizons=result)


def compare_npz(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        keys_equal = set(a.files) == set(b.files)
        fields = {key: array_exact(a[key], b[key]) for key in set(a.files) & set(b.files)}
    return dict(keys_equal=keys_equal, fields=fields,
                exact=keys_equal and all(fields.values()))


def numpy_tree(value, prefix=''):
    if isinstance(value, dict):
        return {p: v for k, child in value.items()
                for p, v in numpy_tree(child, prefix + '/' + str(k)).items()}
    if isinstance(value, (tuple, list)):
        return {p: v for k, child in enumerate(value)
                for p, v in numpy_tree(child, prefix + '/' + str(k)).items()}
    if isinstance(value, np.ndarray):
        return {prefix: value}
    if hasattr(value, 'detach'):
        return {prefix: value.detach().cpu().numpy()}
    return {}


def compare_raw(reference, observed):
    a, b = numpy_tree(reference), numpy_tree(observed)
    details = {}
    for name in a.keys() & b.keys():
        shape_dtype = a[name].shape == b[name].shape and a[name].dtype == b[name].dtype
        max_abs = None
        if shape_dtype:
            diff = np.abs(a[name].astype(np.float64) - b[name].astype(np.float64))
            max_abs = float(diff.max()) if diff.size else 0.
        details[name] = dict(exact=array_exact(a[name], b[name]), max_abs_difference=max_abs,
                             shape_equal=a[name].shape == b[name].shape,
                             dtype_equal=a[name].dtype == b[name].dtype)
    return dict(keys_equal=a.keys() == b.keys(), tensors=details,
                all_exact=a.keys() == b.keys() and all(x['exact'] for x in details.values()))


def raw_manifest(value):
    return {name: dict(shape=list(array.shape), dtype=str(array.dtype), sha256=array_hash(array))
            for name, array in numpy_tree(value).items()}


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, payload):
    temporary = Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def runtime_flags(torch):
    return dict(cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=(torch.get_float32_matmul_precision()
                                 if hasattr(torch, 'get_float32_matmul_precision') else None),
        autocast_enabled=torch.is_autocast_enabled(),
        autocast_gpu_dtype=str(torch.get_autocast_gpu_dtype()),
        torch_threads=torch.get_num_threads(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def fp16_state(module):
    return dict(flags={name: bool(child.fp16_enabled) for name, child in module.named_modules()
                       if hasattr(child, 'fp16_enabled')},
                parameter_dtypes=dict(collections.Counter(str(p.dtype) for p in module.parameters())),
                buffer_dtypes=dict(collections.Counter(str(p.dtype) for p in module.buffers())))


def rng_snapshot(torch):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(torch, state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def clone_tree(torch, value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tree(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_tree(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(torch, v) for v in value)
    return copy.deepcopy(value)


def input_manifest(data):
    """Record tensors/arrays and scalar metadata, unwrapping DataContainers."""
    result = {}

    def visit(value, name):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, name + '/' + str(key))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, name + '/' + str(index))
        elif isinstance(value, np.ndarray) or hasattr(value, 'detach'):
            result.update(raw_manifest({name: value}))
        elif hasattr(value, 'cpu_only') and hasattr(value, 'data'):
            visit(value.data, name + '/DataContainer')
        elif value is None or isinstance(value, (str, int, float, bool, np.generic)):
            result[name] = json_ready(value)
        else:
            raise TypeError('Unrecorded input type: %s' % type(value))

    visit(data, 'batch')
    return result


def extract_original_evaluator(path, namespace):
    tree = ast.parse(Path(path).read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_subset')
    result = dict(namespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), result)
    return result['evaluate_subset']


class ReplayModel:
    """Replay fixed final arrays to isolate storage/metric code from GPU kernels."""
    def __init__(self, predictions):
        self.predictions, self.position = predictions, 0
        self.module, self.training, self.fp16_enabled = self, False, False
        self.simple_test = self.simple_test_offline = None

    def modules(self):
        return [self]

    def eval(self):
        self.training = False

    def train(self, mode):
        self.training = mode

    def __call__(self, **kwargs):
        if self.position >= len(self.predictions):
            raise ValueError('Replay model received too many anchors')
        output = copy.deepcopy(self.predictions[self.position])
        self.position += 1
        return output


def verify_scientific_source(code, original):
    manifest = json.loads((original / 'code_manifest.json').read_text())
    if manifest['git_revision'] != ORIGINAL_REVISION:
        raise ValueError('Unexpected original immutable source')
    checked = {}
    for name, digest in manifest['sha256'].items():
        if name.startswith(('models/', 'loaders/', 'configs/')):
            if sha256_file(original / name) != digest or sha256_file(code / name) != digest:
                raise ValueError('Scientific source changed: ' + name)
            checked[name] = digest
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/sw-history-h1-geometry.py')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--original-code', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--samples', default=16, type=int)
    parser.add_argument('--indices-json', type=Path,
                        help='Optional JSON list of 1–16 unique dataset indices inside fixed256')
    parser.add_argument('--repeats', default=5, type=int)
    parser.add_argument('--workers', default=0, type=int)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 5:
        raise ValueError('Use two to five bounded diagnostic repeats')
    output = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.original_code = args.original_code.resolve()
    explicit_indices = json.loads(args.indices_json.read_text()) if args.indices_json else None
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code))
    os.chdir(code)
    scientific_sources = verify_scientific_source(code, args.original_code)
    if args.config != 'configs/sw-history-h1-geometry.py':
        raise ValueError('This diagnostic is frozen to H1 geometry')

    import torch
    import mmcv
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import EpochBasedRunner, build_optimizer, wrap_fp16_model
    from mmcv.runner.hooks import HOOKS
    from mmcv.utils import build_from_cfg
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from torch.utils.data import Subset
    import models  # noqa: F401 -- register model types
    import loaders  # noqa: F401 -- register dataset/pipeline types
    import finetune_hooks
    from loaders.builder import build_dataloader

    if not torch.cuda.is_available():
        raise RuntimeError('This real-sample diagnosis requires the externally admitted CUDA device')
    started = time.monotonic()
    original_evaluator = extract_original_evaluator(args.original_code / 'finetune_hooks.py',
        dict(torch=torch, np=np, random=random, Path=Path, Subset=Subset, build_dataloader=build_dataloader))
    cfg = Config.fromfile(args.config)
    torch.set_num_threads(2)  # Matches the existing independent evaluation script.
    dataset = build_dataset(cfg.data.val)
    if len(dataset) != 5119:
        raise ValueError('Expected the frozen 5119-anchor validation dataset')
    fixed, positions, indices = select_indices(len(dataset), args.samples, explicit_indices)
    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = {k.removeprefix('module.'): v for k, v in checkpoint['state_dict'].items()}
    if checkpoint['meta']['epoch'] != 10 or checkpoint['meta']['iter'] != 29920:
        raise ValueError('Expected completed epoch10 checkpoint')
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Nonfinite model checkpoint')
    revision = (json.loads((code / 'code_manifest.json').read_text())['git_revision']
                if (code / 'code_manifest.json').exists() else
                subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    report = dict(status='running', purpose='diagnose; does not approve or relax recovery parity',
        config=args.config, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        checkpoint_meta={k: checkpoint['meta'][k] for k in ('epoch', 'iter')},
        source_revision=revision, original_revision=ORIGINAL_REVISION,
        original_evaluator_sha256=sha256_file(args.original_code / 'finetune_hooks.py'),
        current_evaluator_sha256=sha256_file(code / 'finetune_hooks.py'),
        scientific_source_sha256=scientific_sources,
        fixed256_indices=fixed, selection_positions=positions, indices=indices,
        selection_strategy=('explicit_diagnostic_indices' if explicit_indices is not None
                            else 'uniformly_spaced_fixed256_positions'),
        sampling_limitation='A null finding on these anchors cannot exclude rare nondeterminism',
        tokens=[dataset.data_infos[i]['token'] for i in indices],
        versions=dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      mmcv=mmcv.__version__, numpy=np.__version__),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_name=torch.cuda.get_device_name(0), initial_runtime=runtime_flags(torch),
        workers=args.workers, repeats=args.repeats, policies={})
    del checkpoint
    write_json(output / 'diagnostic.json', report)
    baseline_policies = {}
    try:
        # Offline first: manual_seed must not inherit deterministic flags from
        # a previously executed training setup in this diagnostic process.
        for policy in ('offline_manual_seed', 'training_deterministic_seed'):
            folder = output / policy
            folder.mkdir()
            before = runtime_flags(torch)
            if policy == 'offline_manual_seed':
                torch.manual_seed(0)
            else:
                set_random_seed(cfg.get('seed', 0), deterministic=True)
            module = build_model(cfg.model)
            module.init_weights()
            module.load_state_dict(state, strict=True)
            module.cuda()
            optimizer = runner = optimizer_hook = None
            if policy == 'offline_manual_seed':
                module.eval()
                wrap_fp16_model(module)
                model = MMDataParallel(module, [0])
                wrapping = 'mmcv.runner.wrap_fp16_model'
            else:
                module.train()
                model = MMDataParallel(module, [0])
                optimizer = build_optimizer(model, cfg.optimizer)
                def forbid_optimizer_step(*_args, **_kwargs):
                    raise AssertionError('Read-only diagnostic forbids optimizer.step')
                optimizer.step = forbid_optimizer_step
                optimizer_state_before = len(optimizer.state)
                runner = EpochBasedRunner(model, optimizer=optimizer, work_dir=str(folder),
                    max_epochs=cfg.total_epochs, meta={}, logger=logging.getLogger(__name__))
                optimizer_hook = build_from_cfg(dict(cfg.optimizer_config), HOOKS)
                optimizer_hook.before_run(runner)  # No runner.run, backward or step.
                wrapping = type(optimizer_hook).__module__ + '.' + type(optimizer_hook).__name__
            model.eval()
            module.simple_test = module.simple_test_offline
            policy_report = dict(runtime_before=before, runtime_after=runtime_flags(torch),
                precision=fp16_state(module), fp16_setup=wrapping,
                voxelization_deterministic=getattr(module.pts_bbox_head.voxel_generator, 'deterministic', None),
                voxelization_max_num_points=getattr(module.pts_bbox_head.voxel_generator, 'max_num_points', None),
                anchors=[])
            if optimizer_hook is not None:
                policy_report['optimizer_step_guard_installed'] = optimizer.step is forbid_optimizer_step
                policy_report['optimizer_state_entries_before_setup'] = optimizer_state_before
                policy_report['optimizer_state_entries_after_setup'] = len(optimizer.state)
                policy_report['optimizer_hook_before_run_source_sha256'] = hashlib.sha256(
                    inspect.getsource(type(optimizer_hook).before_run).encode()).hexdigest()
            report['policies'][policy] = policy_report
            captured = {}
            voxel_events = []

            def capture(_head, inputs, outputs):
                captured['raw'] = clone_tree(torch, outputs)
                captured['metadata'] = copy.deepcopy(inputs[1][0])

            def voxel_capture(_voxel, inputs, outputs):
                coordinates = outputs[1].detach().cpu().numpy()
                count = len(coordinates)
                unique = len(np.unique(coordinates, axis=0))
                voxel_events.append(dict(returned_voxels=count,
                    unique_voxels=unique, duplicate_coordinates=count - unique))

            handle = module.pts_bbox_head.register_forward_hook(capture)
            voxel_handle = module.pts_bbox_head.voxel_generator.register_forward_hook(voxel_capture)
            predictions = []
            loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1,
                workers_per_gpu=args.workers, dist=False, shuffle=False, seed=0, pin_memory=True)
            dense_shape = tuple(int(n) for n in module.pts_bbox_head.voxel_num.tolist())
            try:
                with torch.no_grad():
                    for anchor_number, data in enumerate(loader):
                        input_record = input_manifest(data)
                        forward_rng = rng_snapshot(torch)
                        restore_rng(torch, forward_rng)
                        voxel_events.clear()
                        reference_prediction = model(return_loss=False, rescale=True, **copy.deepcopy(data))
                        frozen = captured.pop('raw')
                        metadata = captured.pop('metadata')
                        entry = dict(index=indices[anchor_number], token=report['tokens'][anchor_number],
                            input_manifest=input_record, raw_reference=raw_manifest(frozen),
                            first_forward_voxelization=copy.deepcopy(voxel_events),
                            repeated_raw_forward=[], frozen_get_occ=[])
                        reference_rng = rng_snapshot(torch)
                        frozen_hash_before = raw_manifest(frozen)
                        for repeat in range(args.repeats):
                            restore_rng(torch, reference_rng)
                            voxel_events.clear()
                            observed = module.pts_bbox_head.get_occ(frozen, copy.deepcopy(metadata), rescale=True)
                            entry['frozen_get_occ'].append(dict(repeat=repeat,
                                prediction_comparison=compare_predictions(reference_prediction, observed, dense_shape),
                                voxelization=copy.deepcopy(voxel_events)))
                        entry['frozen_raw_unchanged'] = frozen_hash_before == raw_manifest(frozen)
                        for repeat in range(args.repeats - 1):
                            restore_rng(torch, forward_rng)
                            voxel_events.clear()
                            observed = model(return_loss=False, rescale=True, **copy.deepcopy(data))
                            repeated_raw = captured.pop('raw')
                            captured.pop('metadata')
                            entry['repeated_raw_forward'].append(dict(repeat=repeat + 1,
                                raw_comparison=compare_raw(frozen, repeated_raw),
                                prediction_comparison=compare_predictions(reference_prediction, observed, dense_shape),
                                voxelization=copy.deepcopy(voxel_events)))
                            del repeated_raw
                        del frozen, observed
                        predictions.append(reference_prediction)
                        policy_report['anchors'].append(entry)
                        write_json(output / 'diagnostic.json', report)
                        print('PARITY_DIAGNOSTIC_PROGRESS', policy, anchor_number + 1, len(indices), flush=True)
            finally:
                handle.remove()
                voxel_handle.remove()
            # Exact serialization of the same predictions; no second inference.
            with tempfile.TemporaryDirectory(prefix='parity-spool-',
                    dir=os.environ.get('SPARSEWORLD_EVAL_TMPDIR')) as temporary:
                store = finetune_hooks._DiskPredictionStore(temporary)
                for value in predictions:
                    store.append(value)
                policy_report['spool_arrays_exact'] = all(
                    exact_tree(predictions[i][h], store.horizon(h)[i])
                    for i in range(len(indices)) for h in range(len(dataset.future_frames)))
            # Run both full hook bodies with frozen outputs. This isolates list
            # versus disk storage while retaining original dataset.evaluate.
            list_dir, disk_dir = folder / 'list_confusions', folder / 'disk_confusions'
            old_replay, new_replay = ReplayModel(predictions), ReplayModel(predictions)
            reference_metrics = original_evaluator(old_replay, dataset, indices,
                workers=args.workers, confusion_dir=list_dir)
            observed_metrics = finetune_hooks.evaluate_subset(new_replay, dataset, indices,
                workers=args.workers, confusion_dir=disk_dir)
            if old_replay.position != len(indices) or new_replay.position != len(indices):
                raise ValueError('Replay did not consume every frozen anchor exactly once')
            npz_results = {str(h * .5) + 's': compare_npz(list_dir / ('%.1fs.npz' % (h * .5)),
                                                      disk_dir / ('%.1fs.npz' % (h * .5)))
                           for h in dataset.future_frames}
            policy_report['fixed_output_storage_comparison'] = dict(
                list_metrics=reference_metrics, disk_metrics=observed_metrics,
                metrics_exact=exact_tree(reference_metrics, observed_metrics), scene_npz=npz_results,
                all_scene_arrays_exact=all(item['exact'] for item in npz_results.values()),
                replay_calls=dict(original=old_replay.position, current=new_replay.position))
            policy_report['runtime_after_observations'] = runtime_flags(torch)
            policy_report['precision_after_observations'] = fp16_state(module)
            if optimizer is not None:
                policy_report['optimizer_state_entries_after_observations'] = len(optimizer.state)
                if optimizer.step is not forbid_optimizer_step or optimizer.state:
                    raise AssertionError('Optimizer guard/state changed during read-only diagnosis')
            baseline_policies[policy] = predictions
            write_json(output / 'diagnostic.json', report)
            del loader, model, module, optimizer, runner, optimizer_hook, old_replay, new_replay
            gc.collect()
            torch.cuda.empty_cache()
        offline = report['policies']['offline_manual_seed']['anchors']
        trained = report['policies']['training_deterministic_seed']['anchors']
        report['cross_policy'] = [dict(index=indices[i],
            inputs_exact=exact_tree(offline[i]['input_manifest'], trained[i]['input_manifest']),
            raw_reference_hashes_exact=exact_tree(offline[i]['raw_reference'], trained[i]['raw_reference']),
            predictions=compare_predictions(baseline_policies['offline_manual_seed'][i],
                baseline_policies['training_deterministic_seed'][i], dense_shape)) for i in range(len(indices))]
        if sha256_file(args.checkpoint) != checkpoint_hash:
            raise ValueError('Checkpoint changed during read-only diagnosis')
        report.update(status='complete_observations_only', elapsed_seconds=time.monotonic() - started,
                      optimizer_steps_executed=0, checkpoint_unchanged=True,
                      parity_gate_relaxed=False)
        write_json(output / 'diagnostic.json', report)
    except BaseException as error:
        report.update(status='diagnostic_failed', error=repr(error), elapsed_seconds=time.monotonic() - started)
        write_json(output / 'diagnostic.json', report)
        raise


if __name__ == '__main__':
    main()
