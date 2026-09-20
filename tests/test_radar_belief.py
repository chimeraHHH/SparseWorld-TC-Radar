"""Mechanistic tests independent of the full detection framework."""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('radar_belief', Path(__file__).parents[1]/'models/radar_belief.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_radial_information_and_empty_identity():
    mean = torch.zeros(1, 2)
    cov = torch.eye(2)[None] * 25
    n = torch.tensor([[[1., 0.]]])
    mu, p, info = m.information_update(mean, cov, n, torch.tensor([[5.]]), torch.ones(1,1), torch.ones(1,1))
    assert mu[0,0] > 4 and mu[0,1] == 0
    assert p[0,0,0] < 1
    torch.testing.assert_close(p[0,1,1], cov[0,1,1], rtol=1e-6, atol=1e-6)
    assert torch.linalg.matrix_rank(info).item() == 1
    empty = m.information_update(mean,cov,n,torch.zeros(1,1),torch.ones(1,1),torch.zeros(1,1))
    assert torch.equal(empty[0],mean) and torch.equal(empty[1],cov)


def test_rotation_and_finite_covariance_gradient():
    mu = torch.tensor([[1., 2.]], requires_grad=True)
    cov = torch.tensor([[[4., 1.],[1.,9.]]], requires_grad=True)
    n = torch.tensor([[[1.,0.],[.999,.001]]])
    d,w,v = torch.tensor([[5.,5.]]),torch.ones(1,2)/2,torch.ones(1,2)
    a,p,_ = m.information_update(mu,cov,n,d,v,w)
    r = torch.tensor([[0.,-1.],[1.,0.]])
    ar,pr,_ = m.information_update(mu@r.T,r@cov@r.T,n@r.T,d,v,w)
    torch.testing.assert_close(ar,a@r.T); torch.testing.assert_close(pr,r@p@r.T)
    assert (torch.linalg.eigvalsh(p)>0).all()
    (a.square().sum()+p.sum()).backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(cov.grad).all()


def fixture():
    torch.manual_seed(42)
    net=m.SharedRadarBelief(embed_dims=16,neighbors=8)
    q=torch.randn(1,6,16); centers=torch.randn(1,6,3)
    points=torch.zeros(1,16,10)
    points[0,:,:2]=torch.randn(16,2)*2
    points[0,:,6]=torch.arange(16)%4*.1
    points[0,:,7]=torch.linspace(-2,4,16)
    points[0,:,8]=1
    return net,q,centers,points,torch.ones(1,16,dtype=torch.bool)


def test_duplicate_permutation_and_no_radar_identity():
    net,q,x,p,valid=fixture();net.eval()
    a,_=net(q,x,p,valid)
    pp=torch.cat([p,p.flip(1)],1)
    b,_=net(q,x,pp,torch.ones(1,32,dtype=torch.bool))
    for k in ('mean','covariance','features','strength'):torch.testing.assert_close(a[k],b[k])
    readout=m.BeliefReadout(embed_dims=16)
    torch.nn.init.normal_(readout.output.weight)
    empty,_=net(q,x,p,valid&False)
    assert torch.equal(readout(q,x,empty,3.),q)


def test_heldout_values_do_not_enter_partial_posterior():
    net,q,x,p,valid=fixture();p=p[0];mu,cov=net.prior_state(q)
    held=m.holdout_groups(p);assert held.any() and (~held).any()
    v=torch.ones(len(p));f=torch.zeros(len(p),16)
    a=net.assimilate(x[0],mu[0],cov[0],p,v,f,~held)
    changed=p.clone();changed[held,7]+=100
    b=net.assimilate(x[0],mu[0],cov[0],changed,v,f,~held)
    assert torch.equal(m.holdout_groups(p),m.holdout_groups(changed))
    torch.testing.assert_close(a['mean'],b['mean']);torch.testing.assert_close(a['covariance'],b['covariance'])


def test_auxiliary_trains_prior_noise_and_shared_readout():
    net,q,x,p,valid=fixture();state,aux=net(q,x,p,valid)
    assert state['heldout_count']>0 and torch.isfinite(aux)
    aux.backward(retain_graph=True)
    for module in (net.prior,net.noise):
        assert sum(float(v.grad.square().sum()) for v in module.parameters() if v.grad is not None)>0
    readout=m.BeliefReadout(embed_dims=16)
    assert torch.equal(readout(q,x,state,3.),q)
    torch.nn.init.normal_(readout.output.weight,std=.01)
    out=sum(readout(q,x,state,h).square().mean() for h in (0.,1.,2.,3.))
    out.backward()
    assert readout.output.weight.grad.abs().sum()>0
    assert all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
    assert len(net.state_dict())==20


def test_independent_los_and_uncertainty_propagation():
    mean=torch.zeros(1,2);cov=torch.eye(2)[None]*25
    _,posterior,_=m.information_update(mean,cov,torch.eye(2)[None],torch.ones(1,2),torch.ones(1,2),torch.ones(1,2)/2)
    assert (posterior.diagonal(dim1=-2,dim2=-1)<2).all()
    x=torch.tensor([[[1.,2.,3.]]]);v=torch.tensor([[[2.,-1.]]]);p=cov[:,None]
    future,spread=m.propagated_position(x,v,p,3.)
    torch.testing.assert_close(future,torch.tensor([[[7.,-1.]]]))
    assert (spread.diagonal(dim1=-2,dim2=-1)>p.diagonal(dim1=-2,dim2=-1)).all()
    inv,det=m.inverse_2x2(spread)
    torch.testing.assert_close(inv,torch.linalg.inv(spread))
    torch.testing.assert_close(det,torch.linalg.det(spread))
