"""Normalized weighting helpers for the future-focused training objective."""
import torch


MOVABLE_CLASSES = (2, 3, 4, 5, 6, 7, 9, 10)


def normalized_movable_weights(labels, factor, classes=MOVABLE_CLASSES):
    if factor <= 0:
        raise ValueError('Movable-category weight must be positive')
    movable = torch.zeros_like(labels, dtype=torch.bool)
    for category in classes:
        movable |= labels == category
    weights = torch.where(movable, float(factor), 1.).float()
    return weights / weights.mean().clamp_min(1e-8) if weights.numel() else weights


def weighted_horizon_mean(losses, weights):
    if len(losses) != len(weights) or not losses or any(w <= 0 for w in weights):
        raise ValueError('Each horizon needs a positive weight')
    return sum(loss * weight for loss, weight in zip(losses, weights)) / sum(weights)
