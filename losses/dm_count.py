"""CCTrans's DM-Count objective with configurable count, OT and TV weights."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .ot_loss import OT_Loss


LOSS_TYPE = 'dm_count_v1'
TARGET_POLICY = 'center_in_crop_round_clamp_v1'


def loss_config(args):
    return dict(wot=args.wot, wtv=args.wtv, reg=args.reg,
                num_of_iter_in_ot=args.num_of_iter_in_ot,
                norm_cood=bool(args.norm_cood), crop_size=args.crop_size,
                output_stride=args.downsample_ratio,
                count_weight=getattr(args, 'count_loss_weight', 1.0),
                ot_reduction='sum', batch_size=args.batch_size, target_policy=TARGET_POLICY)


def validate_resume_loss(checkpoint, args):
    if checkpoint.get('loss_type') != LOSS_TYPE:
        raise ValueError(
            'Resume requires a DM-Count checkpoint. To switch from Bayesian '
            'training, use --init-checkpoint to load weights with a fresh optimizer.')
    if checkpoint.get('loss_config') != loss_config(args):
        raise ValueError(
            'Resume loss settings/crop size do not match the checkpoint. '
            'Use matching CLI settings, or --init-checkpoint for a new experiment.')


class DMCountLoss(nn.Module):
    """Keep CCTrans's batch reductions: mean count/TV, summed OT.

    The OT scalar is a gradient surrogate, not the Wasserstein distance;
    both the distance and OT dual objective are returned for monitoring.
    """

    def __init__(self, crop_size, stride, device, wot=0.1, wtv=0.01,
                 reg=10.0, num_of_iter_in_ot=100, norm_cood=False,
                 count_loss_weight=1.0):
        super().__init__()
        if crop_size <= 0 or stride <= 0 or crop_size % stride:
            raise ValueError('crop_size must be positive and divisible by stride')
        if any(not math.isfinite(value) or value < 0 for value in (count_loss_weight, wot, wtv)):
            raise ValueError('Count/OT/TV weights must be finite and nonnegative')
        if not math.isfinite(reg) or reg <= 0 or num_of_iter_in_ot <= 0:
            raise ValueError('Sinkhorn reg and iteration count must be positive')
        self.size = crop_size // stride
        self.wot, self.wtv = wot, wtv
        self.count_loss_weight = count_loss_weight
        self.ot = OT_Loss(crop_size, stride, norm_cood, device,
                          num_of_iter_in_ot, reg)

    def forward(self, density, points, gt_discrete):
        if (density.ndim != 4 or density.shape[0] == 0 or density.shape[1:] != (1, self.size, self.size)
                or density.shape != gt_discrete.shape or len(points) != len(density)):
            raise ValueError('Density, discrete target and point batch shapes do not match')
        # Do loss arithmetic in FP32; keep the prediction in the autograd graph.
        density = density.float()
        gt_discrete = gt_discrete.to(device=density.device, dtype=density.dtype)
        counts = density.new_tensor([len(p) for p in points])
        predicted_counts = density.sum(dim=(1, 2, 3), keepdim=True)
        normalized = density / (predicted_counts + 1e-6)
        count = F.l1_loss(predicted_counts.flatten(), counts, reduction='mean')
        target_normalized = gt_discrete / (counts[:, None, None, None] + 1e-6)
        tv = ((normalized - target_normalized).abs().sum(dim=(1, 2, 3)) * counts).mean()
        if self.wot:
            ot, wd, objective = self.ot(normalized, density, points)
            ot, objective = ot.squeeze(), objective.squeeze()
        else:
            ot, wd, objective = density.new_zeros(()), 0.0, density.new_zeros(())
        weighted_ot, weighted_tv = self.wot * ot, self.wtv * tv
        weighted_count = self.count_loss_weight * count
        return dict(total=weighted_count + weighted_ot + weighted_tv, count=count, ot=ot, tv=tv,
                    weighted_count=weighted_count,
                    weighted_ot=weighted_ot, weighted_tv=weighted_tv,
                    wasserstein=density.new_tensor(wd), ot_objective=objective)
