"""CPU checks for diagnostic equality semantics; no CUDA/framework dependencies."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'diagnose_history_eval_parity.py'
SPEC = importlib.util.spec_from_file_location('history_eval_diagnostic', TOOL)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class DiagnosticHelpersTest(unittest.TestCase):
    def test_uniform_and_explicit_subset_selection(self):
        fixed, positions, indices = diagnostic.select_indices(5119)
        expected = sorted(np.random.RandomState(20260918).choice(5119, 256, replace=False).tolist())
        self.assertEqual(fixed, expected)
        self.assertEqual(len(set(indices)), 16)
        self.assertEqual(positions, np.linspace(0, 255, 16, dtype=int).tolist())
        self.assertEqual(indices, [fixed[p] for p in positions])
        requested = [fixed[99], fixed[1]]
        self.assertEqual(diagnostic.select_indices(5119, explicit=requested)[1:], ([99, 1], requested))
        for invalid in ([fixed[0], fixed[0]], [], [True], [5119], [float(fixed[0])], fixed[:17], 'bad'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                diagnostic.select_indices(5119, explicit=invalid)

    def test_exact_arrays_preserve_dtype_shape_and_nan(self):
        a = np.array([1., np.nan], dtype='float32')
        self.assertTrue(diagnostic.array_exact(a, a.copy()))
        self.assertFalse(diagnostic.array_exact(a, a.astype('float64')))
        self.assertFalse(diagnostic.array_exact(a, a.reshape(1, 2)))
        changed = a.copy()
        changed[0] = np.nextafter(changed[0], np.float32(2.))
        self.assertFalse(diagnostic.array_exact(a, changed))
        self.assertTrue(diagnostic.exact_tree({'m': [float('nan'), a]}, {'m': [float('nan'), a.copy()]}))
        self.assertFalse(diagnostic.exact_tree({'m': 1.}, {'m': 1}))

    @staticmethod
    def prediction(points, labels):
        return [dict(occ_loc=np.array(points, dtype='int64'), sem_pred=np.array(labels, dtype='int64'))]

    def test_canonical_comparison_separates_order_and_semantics(self):
        a = self.prediction([[0, 0, 0], [1, 1, 1]], [4, 5])
        reordered = self.prediction([[1, 1, 1], [0, 0, 0]], [5, 4])
        result = diagnostic.compare_predictions(a, reordered, (2, 2, 2))['horizons'][0]
        self.assertFalse(result['raw_arrays_exact'])
        self.assertTrue(result['canonical_sorted_exact'])
        self.assertTrue(result['dense_exact'])
        self.assertEqual(result['changed_dense_voxels'], 0)
        changed = self.prediction([[1, 1, 1], [0, 0, 0]], [6, 4])
        result = diagnostic.compare_predictions(a, changed, (2, 2, 2))['horizons'][0]
        self.assertFalse(result['dense_exact'])
        self.assertEqual(result['changed_dense_voxels'], 1)
        duplicate = self.prediction([[0, 0, 0], [0, 0, 0]], [4, 5])
        result = diagnostic.compare_predictions(duplicate, duplicate, (2, 2, 2))['horizons'][0]
        self.assertEqual(result['reference_duplicate_coordinates'], 1)

    def test_raw_comparison_detects_single_ulp_dtype_and_missing_keys(self):
        a = {'all_cls_scores': [np.array([1.], dtype='float32')]}
        b = {'all_cls_scores': [np.array([np.nextafter(np.float32(1.), np.float32(2.))])]}
        self.assertTrue(diagnostic.compare_raw(a, a)['all_exact'])
        result = diagnostic.compare_raw(a, b)
        self.assertFalse(result['all_exact'])
        self.assertGreater(result['tensors']['/all_cls_scores/0']['max_abs_difference'], 0)
        self.assertFalse(diagnostic.compare_raw(a, {'all_cls_scores': [np.array([1.], dtype='float64')]})['all_exact'])
        self.assertFalse(diagnostic.compare_raw(a, {})['keys_equal'])

    def test_npz_compares_arrays_not_archive_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.npz', Path(directory) / 'b.npz'
            hist = np.arange(8, dtype='int64').reshape(2, 2, 2)
            scenes = np.array(['scene-a', 'scene-b'])
            np.savez(a, scene_names=scenes, histograms=hist)
            np.savez_compressed(b, histograms=hist, scene_names=scenes)
            self.assertNotEqual(diagnostic.sha256_file(a), diagnostic.sha256_file(b))
            self.assertTrue(diagnostic.compare_npz(a, b)['exact'])
            hist[0, 0, 0] += 1
            np.savez(b, scene_names=scenes, histograms=hist)
            result = diagnostic.compare_npz(a, b)
            self.assertFalse(result['exact'])
            self.assertTrue(result['fields']['scene_names'])
            self.assertFalse(result['fields']['histograms'])

    def test_replay_returns_copies_and_refuses_extra_anchors(self):
        prediction = self.prediction([[0, 0, 0]], [4])
        replay = diagnostic.ReplayModel([prediction])
        observed = replay(return_loss=False)
        observed[0]['sem_pred'][0] = 5
        self.assertEqual(prediction[0]['sem_pred'][0], 4)
        self.assertEqual(replay.position, 1)
        with self.assertRaises(ValueError):
            replay()

    def test_ast_extracts_only_requested_original_function(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'original.py'
            source.write_text('raise RuntimeError("top level must not execute")\n'
                              'def evaluate_subset(value):\n    return supplied + value\n')
            function = diagnostic.extract_original_evaluator(source, {'supplied': 3})
            self.assertEqual(function(4), 7)


if __name__ == '__main__':
    unittest.main()
