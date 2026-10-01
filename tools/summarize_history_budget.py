"""Paired four-way scene bootstrap; final epoch is the primary endpoint."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from compare_forecast_results import load_confusions, semantic_iou
from run_history_budget_campaign import ARMS, write_json


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def factorial(arrays, resamples=2000, seed=20260927):
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or set(arrays) != set(ARMS):
        raise ValueError('Require four matched confusion arrays')
    n = arrays[ARMS[0]].shape[1]
    counts = np.random.RandomState(seed).multinomial(n, np.full(n, 1/n), size=resamples)
    point, scores, class_iou = {}, {}, {}
    for arm, array in arrays.items():
        # Use identical scene resampling for all four arms; mIoU is nonlinear.
        ious = semantic_iou(np.einsum('sn,hnij->shij', counts, array, optimize=True))
        scores[arm] = np.nanmean(np.nanmean(ious[:, 1:], axis=-1), axis=-1)
        class_iou[arm] = semantic_iou(array.sum(1))
        point[arm] = float(np.nanmean(np.nanmean(class_iou[arm][1:], axis=-1)))
    def contrast(weights):
        # Sum each side before subtraction; identical factorial effects should
        # not become a spurious 1e-16 signed interaction through cancellation.
        values = (sum(weights[a]*scores[a] for a in weights if weights[a] > 0)
                  - sum(-weights[a]*scores[a] for a in weights if weights[a] < 0))
        return dict(delta_pp=math.fsum(weights[a]*point[a] for a in weights),
                    ci95=np.percentile(values, [2.5,97.5]).tolist())
    short = contrast({'h2-velocity':1,'h2-geometry':-1})
    long = contrast({'h8-velocity':1,'h8-geometry':-1})
    interaction = contrast({'h2-velocity':1,'h2-geometry':-1,'h8-velocity':-1,'h8-geometry':1})
    replacement = contrast({'h2-velocity':1,'h8-velocity':-1})
    return dict(scenes=n, bootstrap_samples=resamples, seed=seed,
        future_mean_miou=point,
        miou_by_horizon={a:np.nanmean(v,axis=-1).tolist() for a,v in class_iou.items()},
        class_iou_by_horizon={a:v.tolist() for a,v in class_iou.items()},
        velocity_gain_short=short, velocity_gain_long=long,
        interaction_short_minus_long=interaction,
        short_velocity_minus_long_velocity=replacement,
        positive_interaction_ci=bool(interaction['ci95'][0]>0),
        short_noninferiority_margin_pp=.3,
        short_noninferiority_ci=bool(replacement['ci95'][0]>-.3),
        interpretation='One seed; final epoch primary, selected best secondary. Conditional on fixed checkpoints and reused validation scenes. Cost and change-domain checks still required.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',required=True)
    args=parser.parse_args()
    root=Path(args.campaign)
    receipts={a:json.loads((root/(a+'_result.json')).read_text()) for a in ARMS}
    output=dict(primary_scope='final', secondary_scope='best', arms=receipts)
    for scope in ('final','best'):
        arrays={}
        names=indices=None
        for arm,receipt in receipts.items():
            array,scene,anchor=load_confusions(receipt[scope+'_confusions'])
            if names is None:
                names,indices=scene,anchor
            if not np.array_equal(names,scene) or not np.array_equal(indices,anchor) or len(anchor)!=5119:
                raise ValueError('Scenes/anchors differ between factorial arms')
            arrays[arm]=array
        output[scope]=factorial(arrays)
    write_json(root/'comparison.json',safe(output))
    lines=['# 修正采样版本 H8/H2 × SDK速度：四组匹配实验', '',
        '主比较使用固定第10轮，最佳轮结果为辅助；均为单种子、复用验证集，不是盲测或多种子证据。', '',
        '| 输入 | 未来平均 mIoU (%) |', '|---|---:|']
    lines += ['| '+a+' | %.6f |'%output['final']['future_mean_miou'][a] for a in ARMS]
    for name in ('velocity_gain_short','velocity_gain_long','interaction_short_minus_long','short_velocity_minus_long_velocity'):
        item=output['final'][name]
        lines += ['', '%s：%.6f pp，场景配对95%%区间 [%.6f, %.6f]。'%(name,item['delta_pp'],*item['ci95'])]
    lines += ['', '必须结合当前帧、变化区域、雷达覆盖、离线实际成本解读。当前模型使用SDK补偿速度与单sweep，不等同原始独立scalar Doppler；H2通过一帧真实历史特征复制保持8槽官方参数。']
    lines += ['', '仅比较本次修正版本；H2将一帧真实历史复制到七历史槽，八槽开销保留。判定仅限测试的2与8帧，不声称全局最少历史数。实际成本见cost_benchmark.json。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(safe(output['final']),indent=2))


if __name__=='__main__':
    main()
