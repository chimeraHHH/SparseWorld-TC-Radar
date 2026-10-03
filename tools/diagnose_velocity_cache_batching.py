"""Zero-update, fixed-checkpoint comparison of legacy and repaired H8 caching.

The controller owns the original GPU and evaluation locks and resource gate.
This child never submits training or changes a scientific source file.
"""
import argparse
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path
from types import MethodType
import time

import numpy as np
import torch
import mmcv
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
import models, loaders
from tools.check_censored_path_contracts import compare_voxels
from tools.check_history_doppler_contracts import tensor_parity
from velocity_feature_cache import FrameCache

LEGACY_SHA = '1e16cb8162434a081401e14f7e355db419e3626d4f53a0c0a28f3b95fe2dd4b2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(raw)
    return h.hexdigest()


def compare(left, right):
    rows = []
    assert len(left) == len(right)
    for index, (a, b) in enumerate(zip(left, right)):
        error = None
        try:
            tensor_parity(a, b, 'cache diagnostic:'+str(index))
        except (AssertionError, ValueError, FloatingPointError) as exception:
            error = repr(exception)
        rows.append(dict(shape=list(a.shape), dtype=str(a.dtype),
                         schema_equal=a.shape == b.shape and a.dtype == b.dtype,
                         finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                         exact=bool(torch.equal(a, b)),
                         max_abs=float((a.float()-b.float()).abs().max()),
                         original_tolerance_pass=error is None, error=error))
    return rows


def main():
    p = argparse.ArgumentParser()
    for name in ('campaign', 'legacy-source', 'out'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    out, legacy_source = Path(args.out), Path(args.legacy_source)
    assert not out.exists() and not out.with_suffix('.pt').exists()
    assert sha(legacy_source) == LEGACY_SHA, 'Require the preserved original cost source'
    tree = ast.parse(legacy_source.read_text())
    legacy_class = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'FrameCache')
    namespace = dict(copy=copy, collections=collections, torch=torch, np=np)
    exec(compile(ast.Module(body=[legacy_class], type_ignores=[]), str(legacy_source), 'exec'), namespace)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    cfg = mmcv.Config.fromfile('configs/sw-budget-h8-velocity.py')
    assert next(x for x in cfg.data.val.pipeline if x['type'] == 'RandomTransformImage')['training'] is False
    ds = build_dataset(copy.deepcopy(cfg.data.val))
    scenes = collections.defaultdict(list)
    for i, info in enumerate(ds.data_infos):
        scenes[info['scene_name']].append(i)
    index = sorted(scenes[sorted(scenes)[0]], key=lambda i: ds.data_infos[i]['timestamp'])[0]
    receipt = json.loads((Path(args.campaign)/'h8-velocity_result.json').read_text())
    checkpoint = Path(receipt['final_audit']['path'])
    assert receipt['git_revision'] == '0f33492d1b639da716897d8faa4a1df293354c49'
    assert receipt['final_audit']['epoch'] == 10 and receipt['final_audit']['iterations'] == 29920
    net = build_model(copy.deepcopy(cfg.model))
    net.init_weights()
    state = torch.load(checkpoint, map_location='cpu')['state_dict']
    net.load_state_dict(state, strict=True)
    del state
    net.cuda().eval()
    wrap_fp16_model(net)
    net.simple_test = net.simple_test_offline
    wrapper = MMDataParallel(net, [0])
    original = net.extract_feat
    legacy, repaired = namespace['FrameCache'](net, 8), FrameCache(net, 8)
    batch = collate([ds[index]], samples_per_gpu=1)
    started = time.monotonic()
    frozen, batches, raw_outputs = {}, [], []
    def neck_hook(module, inputs, outputs):
        batches.append(dict(views=inputs[0][0].shape[0], dtype=str(inputs[0][0].dtype)))
    def head_hook(module, inputs, output):
        raw_outputs.append([v.detach().cpu().clone() for v in
                            [output['init_points']]+output['all_cls_scores']+output['all_refine_pts']])
    with torch.no_grad():
        for _ in range(30):
            wrapper(return_loss=False, rescale=True, **copy.deepcopy(batch))
        neck = net.img_neck.register_forward_hook(neck_hook)
        head = net.pts_bbox_head.register_forward_hook(head_hook)
        try:
            for name, extractor in (
                ('reference', original),
                ('legacy_six_view', lambda img, metas: legacy.extract(net, img, metas)),
                ('repaired_cold', lambda img, metas: repaired.extract(net, img, metas)),
                ('repaired_warm', lambda img, metas: repaired.extract(net, img, metas))):
                batches.clear()
                raw_outputs.clear()
                features = []
                def capture_features(this, img, metas):
                    result = extractor(img, metas)
                    features.extend(v.detach().cpu().clone() for v in result)
                    return result
                net.extract_feat = MethodType(capture_features, net)
                prediction = wrapper(return_loss=False, rescale=True, **copy.deepcopy(batch))
                assert len(raw_outputs) == 1 and len(raw_outputs[0]) == 13
                frozen[name] = dict(features=features, raw=raw_outputs[0], voxels=prediction,
                                    extractor_batches=copy.deepcopy(batches))
        finally:
            net.extract_feat = original
            neck.remove()
            head.remove()
    frozen_path = out.with_suffix('.pt')
    torch.save(dict(index=index, token=ds.data_infos[index]['token'], variants=frozen), frozen_path)
    comparisons = {}
    for name in ('legacy_six_view', 'repaired_cold', 'repaired_warm'):
        row = dict(features=compare(frozen['reference']['features'], frozen[name]['features']),
                   raw=compare(frozen['reference']['raw'], frozen[name]['raw']))
        try:
            row['voxel_counts'] = compare_voxels(frozen['reference']['voxels'], frozen[name]['voxels'])
            row['voxels_exact'] = True
        except AssertionError as error:
            row.update(voxels_exact=False, voxel_error=repr(error))
        comparisons[name] = row
    repaired_pass = all(all(r['original_tolerance_pass'] for r in comparisons[name]['raw']) and
                        comparisons[name]['voxels_exact'] for name in ('repaired_cold', 'repaired_warm'))
    evidence = dict(status='repaired_parity_passed' if repaired_pass else 'repaired_parity_failed',
                    index=index, token=ds.data_infos[index]['token'], optimizer_updates=0,
                    checkpoint_sha256=sha(checkpoint), legacy_source_sha256=LEGACY_SHA,
                    extractor_batches={name: row['extractor_batches'] for name, row in frozen.items()},
                    comparisons=comparisons, elapsed_seconds=time.monotonic()-started,
                    frozen=dict(path=str(frozen_path), bytes=frozen_path.stat().st_size, sha256=sha(frozen_path)),
                    scope='One fixed H8 input; diagnostic only, not accuracy or cost benefit; includes CPU capture timing')
    out.write_text(json.dumps(evidence, indent=2, allow_nan=False))
    print(json.dumps({k: evidence[k] for k in ('status', 'index', 'optimizer_updates', 'extractor_batches')}))
    assert repaired_pass, 'Preserve frozen evidence; do not start measurements after failed repair'
    assert frozen['reference']['extractor_batches'][0]['views'] == 48
    assert frozen['repaired_cold']['extractor_batches'][0]['views'] == 48
    assert not frozen['repaired_warm']['extractor_batches']


if __name__ == '__main__':
    main()
