"""Two fixed-checkpoint velocity arms; geometry interaction remains deferred."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_forecast_results import compare, load_confusions

ARMS = ('h8-velocity', 'h2-velocity')
REVISION = '0f33492d1b639da716897d8faa4a1df293354c49'


def safe(value):
    if isinstance(value, dict):return {key:safe(item) for key,item in value.items()}
    if isinstance(value, list):return [safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):return None
    return value


def summarize(root):
    receipts={arm:json.loads((root/(arm+'_result.json')).read_text()) for arm in ARMS}
    for arm,row in receipts.items():
        assert row['arm']==arm and row['git_revision']==REVISION
        assert row['final_audit']['epoch']==10 and row['final_audit']['iterations']==29920
    output=dict(science_revision=REVISION, primary_scope='final', secondary_scope='best',
                geometry_interaction=None, geometry_scope='deferred by user; no velocity attribution claim')
    for scope in ('final','best'):
        left,ln,li=load_confusions(receipts[ARMS[0]][scope+'_confusions'])
        right,rn,ri=load_confusions(receipts[ARMS[1]][scope+'_confusions'])
        assert np.array_equal(ln,rn) and np.array_equal(li,ri) and len(li)==5119 and len(ln)==150
        assert left.shape==right.shape==(4,150,18,18)
        assert np.issubdtype(left.dtype,np.integer) and np.issubdtype(right.dtype,np.integer)
        assert np.all(left>=0) and np.all(right>=0)
        assert np.array_equal(left.sum(-1),right.sum(-1)), 'Require identical scene truth'
        row=compare(left,right,samples=2000,seed=20260927)
        row.pop('engineering_threshold_pass')
        row.update(anchors=5119,contrast='H2 velocity minus H8 velocity',
                   noninferiority_margin_pp=-.3,
                   noninferiority_ci_pass=row['future_delta_scene_bootstrap_ci95'][0]>-.3)
        output[scope]=row
    return safe(output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();target=Path(args.out)
    with target.open('x') as stream:json.dump(summarize(Path(args.campaign)),stream,indent=2,allow_nan=False)


if __name__=='__main__':main()
