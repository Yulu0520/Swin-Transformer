"""Spatial CCST fusion for MAN (no scalar count regression).

Adapted from Boom5426/CCST-Crowd-Counting-with-Swin-Transformer,
CCST.py (MIT). Copyright (c) 2022 Boli-trainee.
The reference aligns all four levels to stage 1 before AFF and FPN.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _attention(channels, reduction, global_pool=False):
    hidden = max(1, channels // reduction)
    layers = [nn.AdaptiveAvgPool2d(1)] if global_pool else []
    layers.extend([
        nn.Conv2d(channels, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(),
        nn.Conv2d(hidden, channels, 1), nn.BatchNorm2d(channels),
    ])
    return nn.Sequential(*layers)


class AFF(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.local_att = _attention(channels, reduction)
        self.global_att = _attention(channels, reduction, global_pool=True)

    def forward(self, x, residual):
        combined = x + residual
        weight = torch.sigmoid(
            self.local_att(combined) + self.global_att(combined))
        return 2 * x * weight + 2 * residual * (1 - weight)


class FPN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, bias=False) for _ in range(4)])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            for _ in range(4)])

    def forward(self, features):
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, features)]
        for i in range(3, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')
        return [conv(x) for conv, x in zip(self.output_convs, laterals)]


class FAFFusion(nn.Module):
    """FAFHead's 12/34 fusion, preserving the full spatial feature map."""

    def __init__(self, channels):
        super().__init__()
        self.low = AFF(channels, reduction=4)
        self.high = AFF(channels, reduction=4)
        self.local_att = _attention(channels, 4)
        self.global_att = _attention(channels, 4, global_pool=True)

    def forward(self, features):
        low = self.low(features[0], features[1])
        high = self.high(features[2], features[3])
        weight = torch.sigmoid(self.local_att(low) + self.global_att(high))
        return 2 * low * weight + 2 * high * (1 - weight)


class CCSTFusion(nn.Module):
    def __init__(self, stage_channels, channels=512):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(c, channels, 1) for c in stage_channels])
        self.aff = nn.ModuleList([AFF(channels) for _ in range(3)])
        self.fpn = FPN(channels)
        self.faf = FAFFusion(channels)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError('CCST fusion requires four Swin stage features')
        if self.training and features[0].shape[0] == 1:
            raise ValueError('CCST BatchNorm requires batch size >= 2 per device during training')
        projected = [conv(x) for conv, x in zip(self.projections, features)]
        first = projected[0]
        aligned = [first] + [aff(F.interpolate(x, size=first.shape[-2:],
                                              mode='nearest'), first)
                             for aff, x in zip(self.aff, projected[1:])]
        fused = self.faf(self.fpn(aligned))
        # Keep MAN's token count equal to the original stride-32 backbone.
        return F.adaptive_avg_pool2d(fused, features[-1].shape[-2:])
