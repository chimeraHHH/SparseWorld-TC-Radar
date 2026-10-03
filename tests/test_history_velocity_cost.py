"""Protect the scope change and preserve the original cost measurement body."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import json
import unittest
import sys
import copy
import collections
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

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
        for name,value in left.items():
            if name!='FrameCache':self.assertEqual(right[name], value, name)

    def test_cache_failures_cannot_leave_a_partial_paired_cache_claim(self):
        proof=[dict(voxels_exact=True,warm_voxels_exact=True,warm_reuse_without_extraction=True)]*48
        row=dict(replays=[dict(mode='chronological_feature_cache',replay=i) for i in range(3)]+[dict(mode='independent_anchor',replay=0)],cache_parity=proof)
        combined=dict(arms={arm:copy.deepcopy(row) for arm in c.ARMS})
        self.assertTrue(c.exclude_unpaired_cache(copy.deepcopy(combined))['feature_cache_claim_eligible'])
        combined['arms']['h2-velocity']['cache_rejections']=[dict(replay=1,reason='exact voxels failed')]
        result=c.exclude_unpaired_cache(combined)
        self.assertFalse(result['feature_cache_claim_eligible'])
        for arm in c.ARMS:
            self.assertEqual([r['mode'] for r in result['arms'][arm]['replays']],['independent_anchor'])
            self.assertEqual(len(result['arms'][arm]['excluded_cache_replays']),3)

    def test_cache_preserves_whole_batch_misses_and_fresh_projections(self):
        import numpy as np
        class Tensor:
            def __init__(self,data):self.data=np.asarray(data,dtype=np.float32)
            @property
            def shape(self):return self.data.shape
            @property
            def dtype(self):return self.data.dtype
            @property
            def ndim(self):return self.data.ndim
            def __getitem__(self,key):return Tensor(self.data[key])
            def detach(self):return self
            def clone(self):return Tensor(self.data.copy())
            def repeat(self,*dims):return Tensor(np.tile(self.data,dims))
            def numel(self):return self.data.size
            def element_size(self):return self.data.itemsize
        torch=SimpleNamespace(cat=lambda values,dim:Tensor(np.concatenate([x.data for x in values],axis=dim)))
        tree=ast.parse((ROOT/'tools/velocity_feature_cache.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='FrameCache')
        namespace=dict(copy=copy,collections=collections,torch=torch)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'<cache class>','exec'),namespace)
        replicate=next(x for x in ast.parse((ROOT/'models/sparse_world.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='replicate_two_visual_features')
        meta_keys=('filename','img_timestamp','lidar2img','img_shape','ori_shape','pad_shape')
        rn=dict(copy=copy,torch=torch,np=np,_VISUAL_VIEW_META_KEYS=meta_keys)
        exec(compile(ast.Module(body=[replicate],type_ignores=[]),'<scientific H2 replication>','exec'),rn)
        module=ModuleType('models.sparse_world');module.replicate_two_visual_features=rn['replicate_two_visual_features']
        for frames in (8,2):
            class Net:
                training=False
                calls=[]
                def extract_feat(self,img,metas):
                    self.calls.append(img.shape[1]);n=img.shape[1]
                    # Batch- and slot-dependent output catches the old six-view
                    # split and accidental duplicate-filename slot reuse.
                    feature=Tensor(img.data+n+np.arange(n).reshape(1,n,1,1,1))
                    for key in ('img_shape','ori_shape','pad_shape'):metas[0][key]=[(2,2,3)]*n
                    metas[0]['input_shape']=(2,2)
                    return rn['replicate_two_visual_features']([feature],metas) if frames==2 else [feature]
            net=Net();net.calls=[];cache=namespace['FrameCache'](net,frames)
            image=Tensor(np.zeros((1,frames*6,3,2,2)))
            def meta(projection=1):return dict(filename=['same-image']*(frames*6),img_timestamp=list(range(frames*6)),lidar2img=[projection]*(frames*6))
            cold_meta=meta();cold=cache.extract(net,image,[cold_meta])
            self.assertEqual(net.calls,[frames*6])
            warm_meta=meta(7)
            with patch.dict(sys.modules,{'models.sparse_world':module}):warm=cache.extract(net,image,[warm_meta])
            np.testing.assert_array_equal(cold[0].data,warm[0].data)
            self.assertEqual(net.calls,[frames*6]);self.assertEqual(cache.reuse_only_calls,1)
            self.assertEqual(len(warm_meta['lidar2img']),48);self.assertEqual(set(warm_meta['lidar2img']),{7})
            self.assertEqual(warm_meta['img_shape'],cold_meta['img_shape'])
            changed=meta();changed['filename'][0]='new-image'
            cache.extract(net,image,[changed])
            self.assertEqual(net.calls,[frames*6,frames*6]);self.assertEqual(cache.full_batch_recomputations,2)
            self.assertGreater(cache.bytes,0)

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
