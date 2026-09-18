"""Check real-data loss/gradient parity for batching and checkpoint removal."""
import copy,json,argparse,gc
from pathlib import Path
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel,collate
from mmcv.runner import wrap_fp16_model
from mmdet3d.models import build_model
import models
p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--weights',required=True);p.add_argument('--out',required=True);p.add_argument('--fp32',action='store_true');a=p.parse_args()
torch.set_num_threads(4)
cfg=Config.fromfile('configs/sw-radar-m0.py');cfg.model.samplewise_loss=True;cfg.model.data_aug.img_color_aug=False
samples=torch.load(a.cache,map_location='cpu')[:2]
model=build_model(cfg.model);model.load_state_dict(torch.load(a.weights,map_location='cpu')['state_dict'],strict=True);model.cuda().train()
if not a.fp32:wrap_fp16_model(model)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
# Turn off stochastic masks only for this equivalence test. Formal training keeps dropout.
for module in model.modules():
    if isinstance(module,torch.nn.Dropout):module.p=0.0
    if isinstance(module,torch.nn.MultiheadAttention):module.dropout=0.0
net=MMDataParallel(model,[0])
def run(batch,checkpointing):
    for module in model.img_backbone.modules():
        if hasattr(module,'with_cp'):module.with_cp=checkpointing
    model.zero_grad(set_to_none=True);total=0
    for start in range(0,2,batch):
        out=net.train_step(collate(samples[start:start+batch],samples_per_gpu=batch),None)
        loss=out['loss']*(batch/2);total+=float(loss.detach());(loss*512).backward()
    grad={n:v.grad.detach().float().cpu()/512 for n,v in model.named_parameters() if v.grad is not None}
    return total,grad
loss1,g1=run(1,True);results=[]
for batch,cp in [(2,True),(2,False)]:
    loss,g=run(batch,cp);assert g.keys()==g1.keys()
    delta=sum(float((g[k]-g1[k]).double().square().sum()) for k in g)
    base=sum(float(g1[k].double().square().sum()) for k in g1)
    relative=(delta/base)**.5
    assert abs(loss-loss1)/abs(loss1)<.005,(loss,loss1)
    assert all(torch.isfinite(v).all() for v in g.values())
    results.append(dict(batch=batch,checkpointing=cp,reference_loss=loss1,loss=loss,relative_gradient_l2_error=relative,gradient_tensors=len(g),precision='fp32' if a.fp32 else 'fp16'))
    Path(a.out).write_text(json.dumps(results,indent=2))
    assert relative < .001,relative
Path(a.out).write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
