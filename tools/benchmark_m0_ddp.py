"""Matched-cohort ABBA throughput benchmark on two GPUs; no formal outputs."""
import argparse,copy,gc,json,os,random,time,statistics,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from functools import partial
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import build_optimizer,wrap_fp16_model
import models,loaders
from loaders.builder import pinnable_collate
from loaders.pipelines.loading import get_nusc
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
class Cohort(Dataset):
    def __init__(self,base,indices):self.base=base;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        index=int(self.indices[i]);seed=100000+index
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        return self.base[index]
p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--out',required=True);p.add_argument('--updates',type=int,default=66);a=p.parse_args()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(4);dist.init_process_group('nccl')
cfg=Config.fromfile('configs/sw-radar-m0.py');base=build_dataset(cfg.data.train);get_nusc(cfg.dataset_root)
indices=np.random.RandomState(9801).choice(len(base),(a.updates+64)*8,replace=False);dataset=Cohort(base,indices)
weights=torch.load(a.weights,map_location='cpu')['state_dict'];results=[]
for label,batch,workers,cp in [('A1',1,4,True),('B1',4,12,False),('B2',4,12,False),('A2',1,4,True)]:
    random.seed(773);np.random.seed(773);torch.manual_seed(773)
    config=copy.deepcopy(cfg.model);config.samplewise_loss=True;config.img_backbone.with_cp=cp
    model=build_model(config);model.load_state_dict(weights,strict=True);model.cuda().train();wrap_fp16_model(model)
    optimizer=build_optimizer(model,cfg.optimizer);net=MMDistributedDataParallel(model,[rank],broadcast_buffers=False,find_unused_parameters=False)
    scaler=torch.cuda.amp.GradScaler(init_scale=512)
    loader=DataLoader(dataset,batch_size=batch,sampler=DistributedSampler(dataset,num_replicas=2,rank=rank,shuffle=False),num_workers=workers,collate_fn=partial(pinnable_collate,samples_per_gpu=batch),pin_memory=True,prefetch_factor=2,persistent_workers=True)
    iterator=iter(loader);timings=[];waits=[];losses=[];radar_grad=[];accum=4//batch
    dist.barrier();torch.cuda.reset_peak_memory_stats()
    for update in range(a.updates):
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();data_s=0
        for micro in range(accum):
            begin=time.perf_counter();data=next(iterator);data_s+=time.perf_counter()-begin
            assert data['img'].data[0].is_pinned()
            output=net.train_step(data,optimizer);loss=output['loss']/accum
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite benchmark loss')
            scaler.scale(loss).backward();losses.append(float(loss.detach())*accum)
        scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),35)
        if not torch.isfinite(grad):raise FloatingPointError('Nonfinite benchmark gradient')
        rg=sum(float(v.grad.float().square().sum()) for n,v in model.named_parameters() if 'radar_fusion' in n and v.grad is not None)**.5
        radar_grad.append(rg);scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        elapsed=torch.tensor(time.perf_counter()-start,device=rank);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        timings.append(float(elapsed));waits.append(data_s)
        if rank==0 and (update+1)%5==0:print('PROGRESS',label,update+1,round(timings[-1],3),flush=True)
    result=dict(case=label,rank=rank,batch_per_gpu=batch,workers_per_gpu=workers,checkpointing=cp,global_effective_batch=8,measured_samples=(a.updates-26)*8,samples_per_second=(a.updates-26)*8/sum(timings[26:]),seconds_per_update=timings[26:],data_wait_per_update=waits[26:],peak_GiB=torch.cuda.max_memory_allocated()/2**30,loss_range=[min(losses),max(losses)],radar_grad_range=[min(radar_grad),max(radar_grad)],loss_scale=scaler.get_scale(),cohort_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
    gathered=[None,None];dist.all_gather_object(gathered,result)
    if rank==0:
        results.append(gathered);Path(a.out).write_text(json.dumps(results,indent=2));print('RESULT',json.dumps(gathered),flush=True)
    iterator._shutdown_workers();del iterator,loader,output,loss,optimizer,net,model,scaler;gc.collect();torch.cuda.empty_cache();dist.barrier()
dist.destroy_process_group()
