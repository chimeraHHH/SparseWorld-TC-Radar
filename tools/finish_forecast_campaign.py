"""Consolidate both completed arms; never infer completion from a launch receipt."""
import argparse
import fcntl
import json
from pathlib import Path

from compare_forecast_results import compare, load_confusions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    root = Path(args.campaign)
    lock = open(root/'comparison.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    for arm in ('transport', 'balanced'):
        path = root/(arm + '_status.json')
        if not path.exists() or json.loads(path.read_text()).get('state') != 'complete':
            print('Comparison waits for both independent arms to finish.', flush=True)
            return
    arms = {arm: json.loads((root/(arm + '_result.json')).read_text())
            for arm in ('transport', 'balanced')}
    references = {name: root/name/'confusions_normal'
                  for name in ('reference_official', 'reference_m0_final')}
    comparisons = {}
    def paired(reference, candidate):
        import numpy as np
        left, names, indices = load_confusions(reference)
        right, other_names, other_indices = load_confusions(candidate)
        assert np.array_equal(names, other_names) and np.array_equal(indices, other_indices)
        result = compare(left, right)
        result.update(reference=str(reference), candidate=str(candidate), anchors=len(indices))
        return result
    for arm, result in arms.items():
        for reference, directory in references.items():
            comparisons[arm + '_vs_' + reference] = paired(directory, result['best_confusions'])
    comparisons['transport_vs_balanced'] = paired(arms['balanced']['best_confusions'],
                                                   arms['transport']['best_confusions'])
    tables = []
    for reference in references:
        result = json.loads((root/reference/'normal.json').read_text())
        tables.append(dict(name=reference, future_mean_miou=result['future_mean_miou'],
                           metrics=result['metrics']))
    old_best = root.parent/'radar_forecast_20260920/m0_best_full/normal.json'
    if old_best.exists():
        result = json.loads(old_best.read_text())
        tables.append(dict(name='previous_m0_best_epoch1', future_mean_miou=result['future_mean_miou'],
                           metrics=result['metrics']))
    for arm, entry in arms.items():
        for label, filename in [('best_epoch' + str(entry['best_epoch']), entry['best_json']),
                                ('final_epoch10', entry['final_json'])]:
            result = json.loads(Path(filename).read_text())
            tables.append(dict(name=arm + '_' + label, future_mean_miou=result['future_mean_miou'],
                               metrics=result['metrics']))
    prior_best_score = max(row['future_mean_miou'] for row in tables
                           if not row['name'].startswith(('transport_', 'balanced_')))
    decisions = {}
    for arm in arms:
        comp = comparisons[arm + '_vs_reference_m0_final']
        score = next(row['future_mean_miou'] for row in tables if row['name'].startswith(arm+'_best'))
        decisions[arm] = dict(
            future_gain_over_best_available_reference=score-prior_best_score,
            passes_predeclared_final_reference_threshold=comp['engineering_threshold_pass'],
            paired_ci_vs_m0_final_positive=comp['future_delta_scene_bootstrap_ci95'][0] > 0,
            improves_over_all_available_reference_point_estimates=score > prior_best_score)
    report = dict(status='complete', arms=arms, results=tables, comparisons=comparisons,
                  decisions=decisions, limitations=[
                      'One training seed; bootstrap resamples scenes, not seeds.',
                      'Best checkpoint selected using a fixed validation subset.',
                      'Inference radar removal is not a matched camera-only training control.',
                      'Movable categories include stationary objects.'])
    tmp = root/'campaign_results.tmp'
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
    tmp.replace(root/'campaign_results.json')
    lines = ['# 两种雷达未来预测改进：训练与评估完成', '',
             '两项实验均为单卡 H200、BS8、同一官方初始化和10轮预算。', '',
             '| 模型 | 当前帧 | 1秒 | 2秒 | 3秒 | 未来平均 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in tables:
        values = [row['metrics'][h]['Semantic mIoU'] for h in ('0.0s','1.0s','2.0s','3.0s')]
        lines.append('| '+row['name']+' | '+' | '.join('%.4f'%v for v in values+[row['future_mean_miou']])+' |')
    lines += ['', '## 与原 M0 最终模型的配对比较', '']
    for arm in arms:
        comp = comparisons[arm+'_vs_reference_m0_final']
        lo, hi = comp['future_delta_scene_bootstrap_ci95']
        lines.append('- %s：未来平均变化 %+.4f 个百分点，场景重采样95%%区间 [%+.4f, %+.4f]；当前帧变化 %+.4f 个百分点。' %
                     (arm,comp['future_mean_delta_pp'],lo,hi,comp['current_delta_pp']))
    lines += ['', '详细类别差异、官方模型对照、A/B配对比较、权重路径及判定条件见 campaign_results.json。', '',
              '结果只覆盖一个训练种子；按验证子集选择最佳轮次。场景bootstrap不能替代多种子复现，推理时去雷达不能替代同预算纯相机训练对照。', '']
    (root/'CAMPAIGN_RESULTS.md').write_text('\n'.join(lines))
    print('FORECAST_CAMPAIGN_COMPLETE', root/'campaign_results.json', flush=True)


if __name__ == '__main__':
    main()
