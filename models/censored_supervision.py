"""Observed-support losses. Unknown voxels are never converted to free space."""
import torch


def observed_prediction_targets(points, semantics, observed, pc_range, voxel_size, classes):
    """Discrete supervision lookup; out-of-ROI predictions have no target."""
    indices = torch.floor((points.detach() - pc_range[:3]) / voxel_size).long()
    shape = indices.new_tensor(semantics.shape)
    inside = ((indices >= 0) & (indices < shape)).all(-1)
    clipped = torch.minimum(torch.maximum(indices, indices.new_zeros(3)), shape - 1)
    labels = semantics[clipped[:, 0], clipped[:, 1], clipped[:, 2]].long()
    known = observed[clipped[:, 0], clipped[:, 1], clipped[:, 2]].bool() & inside
    known &= (labels >= 0) & (labels <= classes)
    return labels.clamp(0, classes), known


def censored_point_loss(head, cls_scores, normalized_points, semantics, masks, observed_gt=None):
    """One scene, all horizons: visible occupied Chamfer and occupied/free focal.

    Pred->GT attraction applies only inside observed occupied voxels. Observed
    free space receives a background focal target. All other predictions are
    ignored in this direction. GT->pred still rewards explaining known evidence.
    """
    from mmcv.ops import knn
    from .bbox.utils import decode_points
    points = decode_points(normalized_points.float(), head.pc_range)
    cls_scores = cls_scores.float()
    zero = points.sum() * 0 + cls_scores.sum() * 0
    cls_terms, point_terms = [], []
    if observed_gt is None:
        observed_gt = head.get_sparse_voxels(semantics, masks, observed_only=True)
    gt_points, _, gt_labels = observed_gt
    for t in range(len(semantics)):
        pred = points[t].reshape(-1, 3)
        score = cls_scores[t].reshape(-1, head.num_classes)
        labels, known = observed_prediction_targets(
            pred, semantics[t], masks[t], head.pc_range, head.voxel_size, head.num_classes)
        known_count = int(known.sum())
        class_weights = score.new_tensor(head.train_cfg.get('cls_weights', [1.] * head.num_classes))
        weights = known[:, None].to(score.dtype) * class_weights[None]
        weights = weights * head.get_dis_weight(pred.detach())[:, None]
        cls_terms.append(head.loss_cls(score, labels, weight=weights,
                                       avg_factor=max(known_count, 1)))
        gt = gt_points[t]
        loss = zero
        if gt.numel():
            to_pred = knn(1, pred[None].contiguous(), gt[None].contiguous()).reshape(-1).long()
            closest = pred[to_pred]
            weight = gt.new_ones(len(gt))
            distant = (gt - closest.detach()).norm(dim=-1) > head.train_cfg.get('empty_dist_thr', .2)
            weight[distant] = head.train_cfg.get('empty_weights', 5.)
            for category in head.train_cfg.get('rare_classes', [0, 2, 5, 8]):
                rare = gt_labels[t] == category
                weight[rare] = weight[rare].clamp(min=head.train_cfg.get('rare_weights', 10.))
            loss = loss + head.loss_pts(gt, closest, weight=weight[:, None], avg_factor=len(gt))
            occupied = known & (labels < head.num_classes)
            if occupied.any():
                supported = pred[occupied]
                to_gt = knn(1, gt[None].contiguous(), supported[None].contiguous()).reshape(-1).long()
                loss = loss + head.loss_pts(supported, gt[to_gt], avg_factor=len(supported))
        point_terms.append(loss)
    return torch.stack(cls_terms).mean(), torch.stack(point_terms).mean()
