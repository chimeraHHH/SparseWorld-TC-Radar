"""Lossless prediction spooling for the original frozen transport-reliable model.

Copied from the audited e80 storage implementation. Only storage changes; the
original deployment supplies model, data loader and metric implementations.
"""
import fcntl
import os
import pickle
import random
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from loaders.builder import build_dataloader

class _DiskPredictionView(Sequence):
    """One horizon of a disk-backed prediction sequence, without a RAM cache."""
    def __init__(self, store, horizon, anchors=None):
        self.store = store
        self.horizon = horizon
        self.anchors = range(len(store)) if anchors is None else anchors

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _DiskPredictionView(self.store, self.horizon, self.anchors[index])
        anchor = self.anchors[index]
        # Only files written by this invocation are read. Pickle preserves the
        # exact NumPy arrays (including dtype) instead of quantizing predictions.
        with self.store.path(anchor).open('rb') as stream:
            return pickle.load(stream)[self.horizon]

class _DiskPredictionStore:
    """Spool one anchor at a time; retain only a counter in process memory."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.count = 0

    def __len__(self):
        return self.count

    def path(self, anchor):
        return self.directory / ('%08d.pkl' % anchor)

    def append(self, prediction):
        with self.path(self.count).open('xb') as stream:
            pickle.dump(prediction, stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.count += 1

    def horizon(self, index):
        return _DiskPredictionView(self, index)

def evaluate_subset(model, dataset, indices, workers=4, logger=None, confusion_dir=None,
                    prediction_tmp_dir=None):
    """Evaluate with bounded prediction RAM and restore training state/RNG.

    Set SPARSEWORLD_EVAL_TMPDIR (or prediction_tmp_dir) to a local scratch disk.
    The default uses tempfile's normal location. Full-dataset evaluations share
    SPARSEWORLD_FULL_EVAL_LOCK (default: a lock in that scratch root). All spooled
    predictions and lock handles are cleaned up on success or Python exceptions;
    scientific evaluation is unchanged. SIGKILL cannot run Python cleanup.
    """
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all())
    module = model.module
    training = module.training
    original_test = module.simple_test
    fp16 = [(m, m.fp16_enabled) for m in module.modules() if hasattr(m, 'fp16_enabled')]
    try:
        model.eval()
        # Use this configuration's complete offline input. Online inference
        # changes FP16 flags and caches features across optimizer updates.
        module.simple_test = module.simple_test_offline
        scratch = prediction_tmp_dir or os.environ.get('SPARSEWORLD_EVAL_TMPDIR')
        with ExitStack() as cleanup:
            if len(indices) == len(dataset):
                lock_path = os.environ.get('SPARSEWORLD_FULL_EVAL_LOCK') or str(
                    Path(scratch or tempfile.gettempdir()) / 'sparseworld-full-eval.lock')
                lock = cleanup.enter_context(open(lock_path, 'a'))
                if logger:
                    logger.info('FINETUNE_FULL_EVAL_WAITING_FOR_HOST_LOCK %s', lock_path)
                fcntl.flock(lock, fcntl.LOCK_EX)
                # Linux loader workers can inherit this open description at
                # fork. Explicitly unlock before closing the parent's handle.
                cleanup.callback(fcntl.flock, lock, fcntl.LOCK_UN)
                if logger:
                    logger.info('FINETUNE_FULL_EVAL_HOST_LOCK_ACQUIRED %s', lock_path)
            directory = cleanup.enter_context(tempfile.TemporaryDirectory(
                prefix='sparseworld-eval-', dir=scratch))
            results = _DiskPredictionStore(directory)
            loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1,
                                      workers_per_gpu=workers, dist=False,
                                      shuffle=False, seed=0, pin_memory=True)
            with torch.no_grad():
                for i, data in enumerate(loader):
                    prediction = model(return_loss=False, rescale=True, **data)
                    if len(prediction) != len(dataset.future_frames):
                        raise ValueError('Expected one result per forecast horizon')
                    results.append(prediction)
                    del prediction
                    if logger and ((i + 1) % 64 == 0 or i + 1 == len(indices)):
                        logger.info('FINETUNE_VALIDATION_PROGRESS %d/%d', i + 1, len(indices))
            if confusion_dir is not None:
                Path(confusion_dir).mkdir(parents=True, exist_ok=True)
            metrics = {str(h * .5) + 's': dataset.evaluate(
                results.horizon(j), h, sample_indices=indices,
                confusion_path=(str(Path(confusion_dir) / ('%.1fs.npz' % (h * .5)))
                                if confusion_dir is not None else None))
                for j, h in enumerate(dataset.future_frames)}
        return metrics
    finally:
        module.simple_test = original_test
        for m, enabled in fp16:
            m.fp16_enabled = enabled
        model.train(training)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
