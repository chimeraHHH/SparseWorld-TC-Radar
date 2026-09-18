"""Profile unchanged real input transforms; no training state is modified."""
import argparse,json,time,statistics,random
from pathlib import Path
import numpy as np
import torch
from mmcv import Config
import loaders
from mmdet3d.datasets import build_dataset
from loaders.pipelines.loading import get_nusc
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=16);a=p.parse_args()
cfg=Config.fromfile('configs/sw-radar-m0.py')
t=time.perf_counter();dataset=build_dataset(cfg.data.train);build_s=time.perf_counter()-t
t=time.perf_counter();get_nusc(cfg.dataset_root);nusc_s=time.perf_counter()-t
rows=[]
for i,index in enumerate(np.random.RandomState(701).choice(len(dataset),a.count,replace=False)):
    random.seed(int(index));np.random.seed(int(index));torch.manual_seed(int(index))
    t=time.perf_counter();result=dataset.get_data_info(int(index));dataset.pre_pipeline(result)
    row={'index':int(index),'metadata':time.perf_counter()-t}
    for stage in dataset.pipeline.transforms:
        t=time.perf_counter();result=stage(result);row[type(stage).__name__]=time.perf_counter()-t
    rows.append(row)
    print(json.dumps(row),flush=True)
out={'dataset_build_seconds':build_s,'nusc_load_seconds':nusc_s,'samples':rows,'mean_seconds':{k:statistics.mean(r[k] for r in rows) for k in rows[0] if k!='index'}}
Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps(out['mean_seconds']),flush=True)
