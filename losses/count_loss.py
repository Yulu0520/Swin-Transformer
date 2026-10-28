"""Auxiliary count supervision using the existing fractional crop targets."""

import torch
import torch.nn.functional as F


def count_l1_loss(pre_density, target_list):
    """Mean per-crop count error, preserving fractional boundary targets."""
    pred_counts = pre_density.flatten(start_dim=1).sum(dim=1)
    gt_counts = torch.stack([target.sum() for target in target_list])
    return F.l1_loss(pred_counts, gt_counts, reduction='mean')
