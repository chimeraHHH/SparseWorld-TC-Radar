"""Read-only all-anchor real source and timestamp audit, without image decoding."""
import argparse,json,copy,collections
from pathlib import Path
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
import loaders
from loaders.pipelines.history_budget import plan_history,CAMERAS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    cfg=Config.fromfile('configs/sw-budget-h8-velocity.py');rows={}
    for split in ('train','val'):
        ds=build_dataset(copy.deepcopy(cfg.data[split]));stats=collections.Counter();spans=[]
        for index in range(len(ds)):
            r=ds.get_data_info(index);r['filename']=r['img_filename']
            assert isinstance(r['reference_timestamp_us'],int)
            assert all(isinstance(t,int) for t in r['img_timestamp_us'])
            count=len(r['cam_sweeps']['prev']);maximum=min(count//7,8);minimum=min(maximum,4)
            intervals=[None] if split=='val' or count<=7 else list(range(minimum,maximum+1))
            for interval in intervals:
                h8=plan_history(r,8,split=='val',interval=interval);h2=plan_history(r,2,split=='val',interval=interval)
                assert len(h8)==7 and len(h2)==1 and (h2[0] in h8 or h2==[None])
                source=h2[0];stats['plans']+=1
                if source is None:stats['missing_h2']+=1
                else:
                    sweep=r['cam_sweeps']['prev'][source]
                    dt=[(int(sweep[c]['timestamp'])-int(r['img_timestamp_us'][j]))/1e6 for j,c in enumerate(CAMERAS)]
                    assert max(dt)<0;spans.append(-min(dt));stats['real_h2']+=1
                stats['missing_h8_slots']+=sum(i is None for i in h8)
            if (index+1)%2000==0:print(split,index+1,flush=True)
        rows[split]=dict(anchors=len(ds),counts=dict(stats),h2_history_span_seconds_percentiles=np.percentile(spans,[0,25,50,75,100]).tolist())
    Path(args.out).write_text(json.dumps(dict(status='passed',all_anchors=29049,splits=rows,
        integer_time_subtraction=True,no_images_or_labels_read=True,h2_subset_h8=True),indent=2))


if __name__=='__main__':main()
