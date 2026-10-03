"""Exact-shape feature reuse for the fixed, deterministic test augmentation.

A miss executes the original whole-anchor extractor. Splitting that extractor
into six-view forwards changes its FP16 batch shape and is not equivalent.
Cached features are served only when every slot is present in the same complete
image batch. Slot and batch identity are part of the key; fresh anchor
projections are never read from the cache.
"""
import collections
import copy
import torch


class FrameCache:
    def __init__(self, net, frames, max_groups=16):
        assert frames in (2, 8)
        self.net, self.frames, self.maximum = net, frames, max_groups
        self.entries = collections.OrderedDict()
        self.original = net.extract_feat
        self.hits = self.misses = 0
        self.full_batch_recomputations = self.reuse_only_calls = 0

    def clear(self):
        self.entries.clear()

    @property
    def bytes(self):
        return sum(f.numel()*f.element_size() for features, _ in self.entries.values() for f in features)

    def extract(self, net, img, img_metas):
        assert net is self.net and not net.training
        assert img.shape[0] == 1 and img.shape[1] == self.frames*6
        meta = img_metas[0]
        filenames = meta['filename']
        assert len(filenames) == self.frames*6
        keys = [(slot, tuple(filenames), tuple(img.shape[-2:]), str(img.dtype))
                for slot in range(self.frames)]
        missing = sum(key not in self.entries for key in keys)
        if missing:
            self.misses += missing
            self.full_batch_recomputations += 1
            # Preserve the scientific extractor, precision, history selection,
            # batch dimensions, padding and metadata updates without alteration.
            features = self.original(img, img_metas)
            derived = {k: copy.deepcopy(meta[k]) for k in
                       ('img_shape', 'ori_shape', 'pad_shape', 'input_shape') if k in meta}
            for slot, key in enumerate(keys):
                self.entries.pop(key, None)
                self.entries[key] = ([f[:, slot*6:(slot+1)*6].detach().clone() for f in features], derived)
                while len(self.entries) > self.maximum:
                    self.entries.popitem(last=False)
            return features

        self.hits += self.frames
        self.reuse_only_calls += 1
        groups = []
        for key in keys:
            features, derived = self.entries.pop(key)
            self.entries[key] = (features, derived)
            groups.append(features)
        for key, value in derived.items():
            # These are image-extractor shapes only. Anchor-specific filename,
            # camera pose, radar data and projections remain in the fresh input.
            meta[key] = copy.deepcopy(value)
        features = [torch.cat([group[level] for group in groups], dim=1)
                    for level in range(len(groups[0]))]
        if self.frames == 2:
            from models.sparse_world import replicate_two_visual_features
            features = replicate_two_visual_features(features, img_metas)
        return features
