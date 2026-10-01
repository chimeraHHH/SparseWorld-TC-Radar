"""CPU regression of disk-backed evaluation against the existing metric code.

Run with ``python -m unittest discover -s tests -p test_finetune_prediction_spool.py``.
Only framework imports are replaced: the production storage/evaluation function,
NuScenes eval_miou, Metric_mIoU and sparse2dense execute their original bodies.
No detection framework, Torch installation, GPU or dataset download is required.
"""
import ast
import contextlib
import copy
import fcntl
import gc
import io
import math
import os
import pickle
import random
import sys
import tempfile
import threading
import types
import unittest
import weakref
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(relative, names, namespace):
    source = ROOT / relative
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    for node in selected:
        node.decorator_list = []
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)


class FakeTorch:
    Tensor = type('Tensor', (), {})

    def __init__(self):
        self.state = 123
        self.cuda_state = [456]
        self.cuda = types.SimpleNamespace(get_rng_state_all=lambda: self.cuda_state[:],
            set_rng_state_all=lambda value: setattr(self, 'cuda_state', value[:]))

    def get_rng_state(self):
        return self.state

    def set_rng_state(self, value):
        self.state = value

    def no_grad(self):
        return contextlib.nullcontext()


class FakeSubset:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices


def production_evaluator():
    torch = FakeTorch()
    namespace = dict(Path=Path, np=np, torch=torch, random=random, Sequence=Sequence,
        pickle=pickle, os=os, tempfile=tempfile, fcntl=fcntl, ExitStack=contextlib.ExitStack,
        Subset=FakeSubset,
        build_dataloader=lambda subset, **kwargs: ({'anchor': i} for i in subset.indices))
    load_definitions('tools/transport_extension_spool.py',
        ['_DiskPredictionView', '_DiskPredictionStore', 'evaluate_subset'], namespace)
    return namespace


class FakeModel:
    def __init__(self, torch, prediction, fail_at=None):
        self.torch, self.prediction, self.fail_at = torch, prediction, fail_at
        self.training = True
        self.fp16_enabled = False
        self.simple_test = 'original-online-method'
        self.simple_test_offline = 'offline-method'
        self.module = self
        self.calls = 0

    def modules(self):
        return [self]

    def eval(self):
        self.training = False

    def train(self, training):
        self.training = training

    def __call__(self, anchor, **kwargs):
        assert not self.training and self.simple_test == self.simple_test_offline
        random.random()
        np.random.rand()
        self.torch.state += 1
        self.torch.cuda_state[0] += 1
        self.fp16_enabled = True
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('inference failed')
        return self.prediction(anchor)


class DummyDataset:
    future_frames = [0, 2, 4, 6]

    def __init__(self, count, evaluate=None):
        self.count = count
        self.callback = evaluate

    def __len__(self):
        return self.count

    def evaluate(self, results, horizon, **kwargs):
        if self.callback:
            return self.callback(results, horizon, **kwargs)
        for prediction in results:
            assert set(prediction) == {'occ_loc', 'sem_pred'}
        return {'count': len(results), 'horizon': horizon}


def prediction(anchor):
    return [dict(occ_loc=np.array([[0, 0, 0], [1, 0, 0], [2, 2, 1]], dtype=np.int64),
                 sem_pred=np.array([(anchor + h) % 17, 4, 16], dtype=np.int64))
            for h in range(4)]


class PredictionSpoolTests(unittest.TestCase):
    def setUp(self):
        self.namespace = production_evaluator()
        self.torch = self.namespace['torch']
        self.evaluate = self.namespace['evaluate_subset']
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {
            'SPARSEWORLD_EVAL_TMPDIR': str(self.root),
            'SPARSEWORLD_FULL_EVAL_LOCK': str(self.root / 'full.lock')})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def assert_state_restored(self, model, before):
        self.assertEqual(random.getstate(), before[0])
        after = np.random.get_state()
        self.assertEqual(after[0], before[1][0])
        np.testing.assert_array_equal(after[1], before[1][1])
        self.assertEqual(after[2:], before[1][2:])
        self.assertEqual(self.torch.state, before[2])
        self.assertEqual(self.torch.cuda_state, before[3])
        self.assertEqual(model.training, before[4])
        self.assertFalse(model.fp16_enabled)
        self.assertEqual(model.simple_test, 'original-online-method')
        self.assertEqual(list(self.root.glob('sparseworld-eval-*')), [])
        # A failed or successful full evaluation must release the host lock.
        with (self.root / 'full.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def initial_state(self, model):
        return (random.getstate(), np.random.get_state(), self.torch.state,
                self.torch.cuda_state[:], model.training)

    def test_store_exact_dtype_roundtrip_and_lazy_indexing(self):
        store = self.namespace['_DiskPredictionStore'](self.root)
        expected = []
        for i in range(3):
            item = prediction(i)
            item[0]['sem_pred'] = np.array([1., np.nan, -0.], dtype='>f8')
            item[1]['occ_loc'] = item[1]['occ_loc'][:, ::-1]
            item[2]['sem_pred'] = np.array([], dtype=np.int16)
            item[2]['occ_loc'] = np.empty((0, 3), dtype=np.int32)
            expected.append(copy.deepcopy(item))
            store.append(item)
            item[0]['sem_pred'][0] = 999
        for h in range(4):
            view = store.horizon(h)
            self.assertIsInstance(view, Sequence)
            self.assertEqual(len(view), 3)
            for actual, original in zip(view, [x[h] for x in expected]):
                for key in original:
                    self.assertEqual(actual[key].dtype, original[key].dtype)
                    np.testing.assert_array_equal(actual[key], original[key])
            np.testing.assert_array_equal(view[-1]['occ_loc'], expected[-1][h]['occ_loc'])
            np.testing.assert_array_equal(view[::-1][1]['occ_loc'], expected[1][h]['occ_loc'])
            with self.assertRaises(IndexError):
                view[3]

    def test_inference_releases_previous_anchor_arrays_and_restores_state(self):
        references = []

        def make(anchor):
            gc.collect()
            self.assertTrue(all(r() is None for r in references))
            value = prediction(anchor)
            references.extend(weakref.ref(array) for item in value for array in item.values())
            return value

        model = FakeModel(self.torch, make)
        model.training = False  # Preserve evaluation callers as well as trainers.
        before = self.initial_state(model)
        metrics = self.evaluate(model, DummyDataset(9), [8, 1, 5, 0])
        self.assertEqual([v['count'] for v in metrics.values()], [4] * 4)
        gc.collect()
        self.assertTrue(all(r() is None for r in references))
        self.assert_state_restored(model, before)

    def test_inference_and_horizon_failures_cleanup(self):
        for bad_horizons in [False, True]:
            with self.subTest(bad_horizons=bad_horizons):
                model = FakeModel(self.torch,
                    (lambda _: prediction(0)[:3]) if bad_horizons else prediction,
                    fail_at=None if bad_horizons else 2)
                before = self.initial_state(model)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.evaluate(model, DummyDataset(3), [0, 1, 2])
                self.assert_state_restored(model, before)

    def test_partial_disk_write_failure_cleanup(self):
        def broken_dump(value, stream, **kwargs):
            stream.write(b'partial-file')
            raise OSError('disk full')

        model = FakeModel(self.torch, prediction)
        before = self.initial_state(model)
        with mock.patch.object(pickle, 'dump', side_effect=broken_dump):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.evaluate(model, DummyDataset(1), [0])
        self.assert_state_restored(model, before)

    def test_metric_failure_cleans_spool_and_keeps_lock_until_metric_finishes(self):
        def fail(results, horizon, **kwargs):
            self.assertTrue(list(self.root.glob('sparseworld-eval-*')))
            with (self.root / 'full.lock').open('a') as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise RuntimeError('metric failed')

        model = FakeModel(self.torch, prediction)
        before = self.initial_state(model)
        with self.assertRaisesRegex(RuntimeError, 'metric failed'):
            self.evaluate(model, DummyDataset(2, fail), [0, 1])
        self.assert_state_restored(model, before)

    def test_full_evaluation_waits_for_shared_host_lock(self):
        waiting = threading.Event()
        errors = []
        model = FakeModel(self.torch, prediction)
        before = self.initial_state(model)

        class Logger:
            def info(self, text, *args):
                if 'WAITING_FOR_HOST_LOCK' in text:
                    waiting.set()

        def run():
            try:
                self.evaluate(model, DummyDataset(1), [0], logger=Logger())
            except BaseException as error:
                errors.append(error)

        with (self.root / 'full.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(waiting.wait(3))
                self.assertEqual(model.calls, 0)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(model.calls, 1)
        self.assert_state_restored(model, before)

    def test_original_metrics_and_scene_confusions_exactly_match_memory_list(self):
        # Execute the original scientific metric bodies against generated labels.
        namespace = dict(np=np, torch=self.torch, osp=os.path,
                         __package__='_spool_test_dataset')
        load_definitions('loaders/old_metrics.py', ['Metric_mIoU'], namespace)
        load_definitions('models/utils.py', ['sparse2dense'], namespace)
        source = ROOT / 'loaders/nuscenes_occ_dataset.py'
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'NuScenesOccDataset')
        cls.bases, cls.decorator_list = [], []
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and node.name in {'evaluate', 'eval_miou'}]
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
        dataset = namespace['NuScenesOccDataset']()
        dataset.future_frames = [0, 2, 4, 6]
        dataset.data_root, dataset.occ_root = 'unused', str(self.root / 'labels')
        dataset.data_infos = [dict(scene_name='scene-%d' % (i % 2), token='a%d-t0' % i)
                              for i in range(4)]
        samples = {}
        for i, info in enumerate(dataset.data_infos):
            for h in range(7):
                token = 'a%d-t%d' % (i, h)
                samples[token] = dict(scene_token=info['scene_name'],
                    token=token, next='a%d-t%d' % (i, h + 1) if h < 6 else '')
                folder = Path(dataset.occ_root) / info['scene_name'] / token
                folder.mkdir(parents=True)
                labels = (np.arange(18).reshape(3, 3, 2) + i + h) % 18
                mask = np.ones_like(labels, dtype=bool)
                mask[2, 0, 0] = False
                np.savez(folder / 'labels.npz', semantics=labels,
                         mask_camera=mask, mask_lidar=np.ones_like(mask))
        loading = types.ModuleType('_spool_test_dataset.pipelines.loading')
        loading.get_nusc = lambda root: types.SimpleNamespace(get=lambda table, token: samples[token])
        tqdm = types.ModuleType('tqdm')
        tqdm.tqdm = lambda values: values
        # Dummy length is only used for full-lock classification by evaluate_subset.
        dataset.__class__.__len__ = lambda self: len(self.data_infos)
        indices = [3, 0, 2]
        baseline = self.root / 'baseline'
        actual = self.root / 'actual'
        baseline.mkdir()
        all_predictions = [p for i in indices for p in prediction(i)]
        model = FakeModel(self.torch, prediction)
        before = self.initial_state(model)
        with mock.patch.dict(sys.modules, {loading.__name__: loading, 'tqdm': tqdm}), \
                contextlib.redirect_stdout(io.StringIO()), np.errstate(divide='ignore', invalid='ignore'):
            reference = {str(h * .5) + 's': dataset.evaluate(
                all_predictions[j::4], h, sample_indices=indices,
                confusion_path=str(baseline / ('%.1fs.npz' % (h * .5))))
                for j, h in enumerate(dataset.future_frames)}
            observed = self.evaluate(model, dataset, indices, confusion_dir=actual)
        for horizon, metrics in reference.items():
            self.assertEqual(metrics.keys(), observed[horizon].keys())
            for key, value in metrics.items():
                np.testing.assert_equal(observed[horizon][key], value)
            with np.load(baseline / (horizon + '.npz')) as original, \
                    np.load(actual / (horizon + '.npz')) as saved:
                self.assertEqual(original.files, saved.files)
                for key in original.files:
                    np.testing.assert_array_equal(original[key], saved[key])
        self.assert_state_restored(model, before)


if __name__ == '__main__':
    unittest.main()
