"""Protect the scope change and preserve the original cost measurement body."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import json
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('velocity_cost', ROOT/'tools/run_history_velocity_cost.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class VelocityCostScope(unittest.TestCase):
    def test_alternating_schedule_has_only_three_replays_of_two_velocities(self):
        expected = [(0, 'h8-velocity'), (0, 'h2-velocity'), (1, 'h2-velocity'),
                    (1, 'h8-velocity'), (2, 'h8-velocity'), (2, 'h2-velocity')]
        self.assertEqual(c.schedule(), expected)

    def test_exclusive_claim_does_not_replace_original_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'submission.json'
            c.write(path, {'original': True}, True)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                c.write(path, {'duplicate': True}, True)
            self.assertEqual(path.read_bytes(), before)

    def test_cost_rejects_incomplete_or_wrong_checkpoint_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for name in ('checkpoint', 'full', 'best', 'conf', 'bestconf', 'interventions'):
                (base/name).touch()
            row = dict(arm='h8-velocity', git_revision=c.REVISION,
                       final_audit=dict(epoch=10, iterations=29920, finite_model_tensors=771,
                       optimizer_finite=True, state_schema_matches_cpu=True, optimizer_steps=29914,
                       path=str(base/'checkpoint')), final_json=str(base/'full'), best_json=str(base/'best'),
                       final_confusions=str(base/'conf'), best_confusions=str(base/'bestconf'), interventions=str(base/'interventions'))
            path=base/'receipt.json'; path.write_text(json.dumps(row))
            self.assertEqual(c.receipt(path, 'h8-velocity'), row)
            for field, value in [('epoch', 9), ('iterations', 29918), ('optimizer_finite', False), ('finite_model_tensors', 770)]:
                changed=json.loads(json.dumps(row));changed['final_audit'][field]=value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):c.receipt(path, 'h8-velocity')

    def test_cache_and_summary_helpers_match_original_ast(self):
        original=ast.parse((ROOT/'tools/benchmark_history_budget.py').read_text())
        added=ast.parse((ROOT/'tools/benchmark_history_velocity_block.py').read_text())
        def helpers(tree):
            return {x.name:ast.dump(x, include_attributes=False) for x in tree.body
                    if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name!='main'}
        left,right=helpers(original),helpers(added)
        for name,value in left.items():self.assertEqual(right[name], value, name)

    def test_no_training_command_or_geometry_config_in_new_queue(self):
        script=(ROOT/'tools/run_history_velocity_cost.py').read_text()
        self.assertNotIn("'train.py'", script)
        self.assertNotIn('sw-budget-h8-geometry.py', script)
        self.assertNotIn('sw-budget-h2-geometry.py', script)

    def test_bootstrap_requires_identical_truth_and_cannot_emit_interaction(self):
        import numpy as np
        import summarize_history_velocity as summary
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            hist=np.broadcast_to(np.eye(18,dtype=np.int64),(150,18,18)).copy()
            for arm in summary.ARMS:
                conf=base/arm;conf.mkdir()
                for horizon in ('0.0s','1.0s','2.0s','3.0s'):
                    np.savez(conf/(horizon+'.npz'),histograms=hist,
                             scene_names=np.array(['scene%03d'%i for i in range(150)]),sample_indices=np.arange(5119))
                row=dict(arm=arm,git_revision=c.REVISION,final_audit=dict(epoch=10,iterations=29920),
                         final_confusions=str(conf),best_confusions=str(conf))
                (base/(arm+'_result.json')).write_text(json.dumps(row))
            result=summary.summarize(base)
            self.assertEqual(result['final']['future_mean_delta_pp'],0.)
            self.assertEqual(result['final']['future_delta_scene_bootstrap_ci95'],[0.,0.])
            self.assertTrue(result['final']['noninferiority_ci_pass'])
            self.assertIsNone(result['geometry_interaction'])
            wrong=hist.copy();wrong[0,0,0]+=1
            np.savez(base/'h2-velocity/0.0s.npz',histograms=wrong,
                     scene_names=np.array(['scene%03d'%i for i in range(150)]),sample_indices=np.arange(5119))
            with self.assertRaisesRegex(AssertionError,'identical scene truth'):summary.summarize(base)


if __name__=='__main__':
    unittest.main()
