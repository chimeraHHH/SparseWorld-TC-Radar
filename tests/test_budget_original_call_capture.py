"""Diagnostic must preserve original failure flow and native call order."""
import ast
import importlib.util
import inspect
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('capture_diagnostic', ROOT / 'tools/diagnose_budget_original_call_parity.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def fixture(raw_fails, voxel_fails, events):
    def native(tag):
        events.append(tag)
        return {'native': tag}

    def parity(a, b):
        events.append('raw_compare')
        if raw_fails:
            raise AssertionError('raw mismatch')
        return True

    def voxels(left, right):
        events.append('voxel_compare')
        if voxel_fails:
            raise AssertionError('voxel mismatch')
        return 4

    namespace = dict(native=native, parity=parity, voxels=voxels)
    exec('''def gpu_contract():
    results = []
    for index in [2048]:
        captures = {'new': [[1]], 'camera_reference': [[1]]}
        left = native('new_native')
        right = native('reference_native')
        tensors = [parity(a, b) for a, b in zip(captures['new'][0], captures['camera_reference'][0])]
        results.append(voxels(left, right))
    return results
''', namespace)
    return namespace


class CaptureIntegrity(unittest.TestCase):
    def test_actual_checker_only_one_cpu_capture_is_added(self):
        spec = importlib.util.spec_from_file_location('original_checker', ROOT / 'tools/check_history_budget_contracts.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        tree = diagnostic.instrument_source(inspect.getsource(checker.gpu_contract))
        callbacks = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == diagnostic.CALLBACK]
        self.assertEqual(len(callbacks), 1)
        self.assertEqual([arg.id for arg in callbacks[0].args], ['index', 'captures', 'left', 'right'])
        with self.assertRaisesRegex(ValueError, 'already present'):
            diagnostic.instrument_source(ast.unparse(tree))

    def test_evidence_exists_before_each_original_failure_and_no_extra_native_call(self):
        for raw_fails, voxel_fails in [(True, False), (False, True), (False, False)]:
            with self.subTest(raw_fails=raw_fails, voxel_fails=voxel_fails), tempfile.TemporaryDirectory() as folder:
                events = []
                namespace = fixture(raw_fails, voxel_fails, events)
                source = '''def gpu_contract():
    results = []
    for index in [2048]:
        captures = {'new': [[1]], 'camera_reference': [[1]]}
        left = native('new_native')
        right = native('reference_native')
        tensors = [parity(a, b) for a, b in zip(captures['new'][0], captures['camera_reference'][0])]
        results.append(voxels(left, right))
    return results
'''
                evidence = Path(folder) / 'frozen-pair.txt'

                def persist(index, captures, left, right):
                    evidence.write_text(repr((index, captures, left, right)))
                    events.append('persist')

                namespace[diagnostic.CALLBACK] = persist
                exec(compile(diagnostic.instrument_source(source), '<fixture>', 'exec'), namespace)
                if raw_fails or voxel_fails:
                    with self.assertRaisesRegex(AssertionError, 'raw mismatch' if raw_fails else 'voxel mismatch'):
                        namespace['gpu_contract']()
                else:
                    self.assertEqual(namespace['gpu_contract'](), [4])
                self.assertTrue(evidence.is_file())
                self.assertEqual(events[:3], ['new_native', 'reference_native', 'persist'])
                self.assertEqual(events.count('new_native'), 1)
                self.assertEqual(events.count('reference_native'), 1)
                self.assertEqual(events[3:], ['raw_compare'] if raw_fails else ['raw_compare', 'voxel_compare'])

    def test_unknown_comparison_layout_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'not unique'):
            diagnostic.instrument_source('def gpu_contract():\n    return []\n')
        with self.assertRaisesRegex(ValueError, 'not unique'):
            diagnostic.instrument_source('def gpu_contract():\n    tensors = [x for x in []]\n    tensors = [y for y in []]\n')


if __name__ == '__main__':
    unittest.main()
