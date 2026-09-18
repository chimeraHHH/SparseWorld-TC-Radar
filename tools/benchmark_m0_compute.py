"""Bounded real-sample compute benchmark; outputs never replace training."""
import argparse,copy,json,os,random,time,statistics,gc
from pathlib import Path
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate,MMDataParallel
from mmcv.runner import build_optimizer,wrap_fp16_model
import models,loaders
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--cache',required=True);p.add_argument('--weights',required=True);a=p.parse_args()
cfg=Config.fromfile('configs/sw-radar-m0.py')
torch.set_num_threads(4);torch.manual_seed(333);np.random.seed(333);random.seed(333)
cache=Path(a.cache)
if cache.exists(): samples=torch.load(cache,map_location='cpu')
else:
    ds=build_dataset(cfg.data.train)
    samples=[]
    for index in [500,1500,3000,6000,9000,12000,17000,22000]:
        random.seed(index);np.random.seed(index);torch.manual_seed(index)
        samples.append(ds[index]);print('CACHED',index,flush=True)
    cache.parent.mkdir(parents=True,exist_ok=True);torch.save(samples,cache)
    del ds;gc.collect()
weights=torch.load(a.weights,map_location='cpu')['state_dict']
results=[]
for batch,with_cp in [(1,True),(2,True),(4,True),(4,False)]:
    torch.manual_seed(333);np.random.seed(333);random.seed(333)
    config=copy.deepcopy(cfg.model);config.samplewise_loss=True;config.img_backbone.with_cp=with_cp
    model=build_model(config);model.load_state_dict(weights,strict=True);model.cuda().train();wrap_fp16_model(model)
    optimizer=build_optimizer(model,cfg.optimizer);net=MMDataParallel(model,[0]);scaler=torch.cuda.amp.GradScaler(init_scale=512)
    batches=[collate(samples[i:i+batch],samples_per_gpu=batch) for i in range(0,8,batch)]
    timings=[];losses=[];gradients=[]
    torch.cuda.reset_peak_memory_stats()
    try:
        for update in range(10):
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            for data in batches:
                output=net.train_step(data,optimizer);loss=output['loss']/len(batches)
                assert torch.isfinite(loss),loss
                scaler.scale(loss).backward();losses.append(float(loss.detach())*len(batches))
            scaler.unscale_(optimizer)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),35)
            assert torch.isfinite(grad),grad
            radar_grad=sum(float(v.grad.float().square().sum()) for n,v in model.named_parameters() if 'radar_fusion' in n and v.grad is not None)**.5
            assert radar_grad>0
            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            timings.append(time.perf_counter()-start);gradients.append(radar_grad)
        result=dict(batch_per_gpu=batch,backbone_checkpointing=with_cp,warmup_updates=2,measured_samples=64,seconds=timings[2:],samples_per_second=64/sum(timings[2:]),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,loss_range=[min(losses),max(losses)],radar_grad_range=[min(gradients),max(gradients)],scale=scaler.get_scale())
    except torch.cuda.OutOfMemoryError as error:
        result=dict(batch_per_gpu=batch,backbone_checkpointing=with_cp,error=str(error))
    results.append(result);Path(a.out).write_text(json.dumps(results,indent=2));print('RESULT',json.dumps(result),flush=True)
    del output,loss,optimizer,net,model,batches,scaler;gc.collect();torch.cuda.empty_cache()
