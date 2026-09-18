"""Strict model-only initialization from the published camera checkpoint."""
import hashlib
from pathlib import Path

import torch


OFFICIAL_SHA256 = '871c4da344fbb71f9e0a8067c7f10f6ba55edc6cedd4955878bd9076c6920a6a'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_camera_state(model, checkpoint, zero_radar=True):
    """Require every non-radar tensor, reject all extra or mismatched tensors."""
    source = checkpoint.get('state_dict', checkpoint)
    source = {k.removeprefix('module.'): v for k, v in source.items()}
    target = model.state_dict()
    radar = {k for k in target if '.radar_fusion.' in k}
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    mismatched = [k for k in source.keys() & target.keys()
                  if source[k].shape != target[k].shape]
    if set(missing) != radar or unexpected or mismatched:
        raise ValueError(dict(missing=missing, unexpected=unexpected,
                              shape_mismatch=mismatched, expected_new=sorted(radar)))
    if not all(torch.isfinite(v).all() for v in source.values()):
        raise ValueError('Official model contains nonfinite tensors')
    result = model.load_state_dict(source, strict=False)
    assert set(result.missing_keys) == radar and not result.unexpected_keys
    outputs = []
    if zero_radar:
        for name, module in model.named_modules():
            if name.endswith('.radar_fusion.output'):
                torch.nn.init.zeros_(module.weight)
                outputs.append(name)
        if radar and not outputs:
            raise ValueError('No radar residual outputs found')
    loaded = model.state_dict()
    assert all(torch.equal(loaded[k].cpu(), v.cpu()) for k, v in source.items())
    return dict(loaded_tensors=len(source), loaded_numel=sum(v.numel() for v in source.values()),
                new_radar_tensors=len(radar), zero_residual_outputs=outputs,
                all_camera_tensors_exact=True)


def initialize_official(model, path):
    if sha256_file(path) != OFFICIAL_SHA256:
        raise ValueError('Published checkpoint SHA256 does not match')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    report = load_camera_state(model, checkpoint)
    report.update(path=str(Path(path).resolve()), sha256=OFFICIAL_SHA256,
                  source_epoch=checkpoint.get('meta', {}).get('epoch'),
                  source_iter=checkpoint.get('meta', {}).get('iter'),
                  optimizer_restored=False, epoch_reset_to=0)
    return report
