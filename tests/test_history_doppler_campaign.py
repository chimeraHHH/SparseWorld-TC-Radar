import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

TOOLS=Path(__file__).parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from run_history_doppler_experiment import ARMS, claim_arm, unclaimed, audit_smoke_log
from summarize_history_doppler import factorial


def test_each_arm_claimed_exactly_once_across_two_workers(tmp_path):
    got=[claim_arm(tmp_path,i%2,'revision') for i in range(5)]
    assert got==list(ARMS)+[None]
    assert unclaimed(tmp_path)==[]


def test_incomplete_learning_evidence_fails():
    with pytest.raises(ValueError):
        audit_smoke_log('')


def test_paired_factorial_zero_when_velocity_effect_same_across_history():
    base=np.ones((4,3,18,18),dtype=np.int64)
    improved=base.copy()
    for i in range(18): improved[:,:,i,i]+=3
    values={a:(improved if a.endswith('velocity') else base) for a in ARMS}
    result=factorial(values,resamples=30)
    assert result['velocity_gain_short']['delta_pp']>0
    assert result['interaction_short_minus_long']['delta_pp']==0
    assert result['interaction_short_minus_long']['ci95']==[0,0]
    assert result['short_velocity_minus_long_velocity']['ci95']==[0,0]
