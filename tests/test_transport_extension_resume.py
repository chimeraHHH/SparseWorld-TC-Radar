"""The continuation must keep saved group LRs and full optimizer/AMP state."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]


def definitions():
    tree = ast.parse((ROOT/'tools/transport_extension_hooks.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in ('CheckpointConstantLrUpdaterHook','assert_equal_state')]
    for n in nodes:
        if isinstance(n, ast.ClassDef): n.decorator_list=[]
    ns = dict(torch=torch, math=__import__('math'), LrUpdaterHook=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'extension', 'exec'), ns)
    return ns


def test_resume_does_not_restart_cosine_or_restore_initial_lr():
    hook = definitions()['CheckpointConstantLrUpdaterHook']()
    rates = [2.4404913533436187e-7,2.4404913533436184e-6,2.4404913533436186e-5]
    runner = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[
        dict(lr=x, initial_lr=x*8.2) for x in rates]))
    hook.before_run(runner)
    assert hook.base_lr == rates
    for epoch in (10,14,19):
        runner.epoch = epoch
        assert [hook.get_lr(runner,x) for x in hook.base_lr] == rates


def test_optimizer_state_and_scaler_must_remain_exact():
    compare = definitions()['assert_equal_state']
    state = dict(state={0:dict(step=torch.tensor(29915.),exp_avg=torch.tensor([1.,2.]))},
        param_groups=[dict(lr=1e-6,params=[0])])
    compare(state,state)
    import copy
    changed = copy.deepcopy(state);changed['state'][0]['step'] += 1
    import pytest
    with pytest.raises(AssertionError): compare(changed,state)
    with pytest.raises(AssertionError): compare(dict(scale=512.),dict(scale=65536.))
