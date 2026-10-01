"""Causal sources, precision, actual H2 gradient and corrected T/G sampling."""
import ast
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import importlib.util
_spec=importlib.util.spec_from_file_location('visual_budget_test',Path(__file__).with_name('test_visual_history_budget.py'))
_mod=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_mod)
model,metadata=_mod.model,_mod.metadata


def planner():
    path=Path(__file__).parents[1]/'loaders/pipelines/history_budget.py'
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.Assign))]
    import os
    scope=dict(np=np,os=os);exec(compile(ast.Module(nodes,type_ignores=[]),str(path),'exec'),scope);return scope


def fixture(count=50):
    cameras=planner()['CAMERAS'];t=1_500_000_000_000_000
    return dict(img_timestamp_us=[t+j for j in range(6)],reference_timestamp_us=t+20,
        filename=[f'current{j}' for j in range(6)],cam_sweeps=dict(prev=[
            {c:dict(timestamp=t-(i+1)*80_000+j,data_path=f'past{i}-{j}') for j,c in enumerate(cameras)} for i in range(count)]))


def test_h2_is_subset_of_actual_h8_candidates_and_async_current_is_not_history():
    p=planner();r=fixture()
    eight=p['plan_history'](r,8,True);two=p['plan_history'](r,2,True)
    assert eight==[5,11,17,23,29,35,41] and two==[5]
    assert all(t<r['reference_timestamp_us'] for t in r['img_timestamp_us'])
    assert p['plan_history'](fixture(0),2,True)==[None]
    for interval in range(4,8):
        assert p['plan_history'](r,2,False,interval=interval)[0] in p['plan_history'](r,8,False,interval=interval)


def test_incomplete_history_cannot_wrap_to_last_or_use_future_camera():
    p=planner();r=fixture(2);cams=p['CAMERAS'];r['cam_sweeps']['prev'][0].pop(cams[0])
    r['cam_sweeps']['prev'][1][cams[0]]['timestamp']=r['img_timestamp_us'][0]+1
    assert p['plan_history'](r,2,True)==[None]


def test_h2_encodes12_and_replicates_history_without_gradient_to_other_inputs():
    net=model(2);x=torch.randn(2,12,3,3,4,requires_grad=True);m=metadata(2,12);original=copy.deepcopy(m)
    out=net.extract_feat(x,m)[0];assert net.img_backbone.batch_sizes==[24]
    assert torch.equal(out[:,6:12],out[:,42:48]);assert m[0]['filename'][42:48]==original[0]['filename'][6:12]
    out.sum().backward();torch.testing.assert_close(x.grad[:,:6],torch.full_like(x.grad[:,:6],.5))
    torch.testing.assert_close(x.grad[:,6:],torch.full_like(x.grad[:,6:],3.5))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sampling_time_group_scale_rows_and_position_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    import models.sparse_world_sampling as sampler
    from models.csrc.wrapper import msmv_sampling,msmv_sampling_pytorch
    B,T,G,Q,P,N,L=2,8,4,2,2,6,4
    points=torch.zeros(B,Q,T,G,P,3,device=device);points[...,0]=.37;points[...,1]=.41;points[...,2]=1;points.requires_grad_()
    matrix=torch.eye(4,device=device).repeat(B,T*N,1,1);matrix[...,0,0]=10;matrix[...,1,1]=10
    weights=torch.randn(B,Q,G,T,P,L,device=device).softmax(-1).requires_grad_()
    # Distinct time/group/level spatial ramps expose all leading-row permutations.
    feats=[]
    for level in range(L):
        grid=torch.arange(5,device=device).float();ramp=grid[:,None]+grid[None,:]*2
        values=torch.arange(B*T*G,device=device).view(-1,1,1,1,1)*10+level*100+ramp.view(1,1,1,5,5)
        feats.append(values.expand(B*T*G,1,N,5,5).contiguous())
    calls=[]
    def reference(values,locations,actual_weights):
        expected=torch.stack([weights[b,:,g,t] for b in range(B) for t in range(T) for g in range(G)])
        torch.testing.assert_close(actual_weights,expected,atol=0,rtol=0);calls.append(True)
        return msmv_sampling_pytorch(values,locations,actual_weights)
    old=sampler.msmv_sampling;sampler.msmv_sampling=reference
    try:expected=sampler.sampling_4d(points,feats,weights,matrix,[torch.eye(4,device=device).repeat(B,1,1)],10,10)
    finally:sampler.msmv_sampling=old
    if device=='cuda':
        assert __import__('models.csrc.wrapper',fromlist=['MSMV_CUDA']).MSMV_CUDA
        cuda_feats=[f.permute(0,2,3,4,1).contiguous() for f in feats]
        actual=sampler.sampling_4d(points,cuda_feats,weights,matrix,[torch.eye(4,device=device).repeat(B,1,1)],10,10)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-4)
        ag=torch.autograd.grad(actual.sum(),(points,weights),retain_graph=True)
        eg=torch.autograd.grad(expected.sum(),(points,weights),retain_graph=True)
        for a,e in zip(ag,eg):torch.testing.assert_close(a,e,rtol=2e-4,atol=2e-3)
    gradients=torch.autograd.grad(expected.sum(),(points,weights));assert calls
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients)
