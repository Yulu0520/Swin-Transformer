"""Stage-2/3 PFA + MDC and a residual update to the shared MAN features.

The CBR/dilated structure follows CCTrans Regression, adapted from
Yulu0520/Swin-Transformer, finetune_regression commit 278f87a753e633ae575611999d8d8f5fc39b63ea.
Stage 4 belongs exclusively to the MAN path; this module has no density head.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def cbr(in_channels, out_channels, kernel_size=3, dilation=1):
    padding = dilation if kernel_size == 3 else 0
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class NightPfaMdc(nn.Module):
    """Fuse only stages 2/3, then produce 384-channel multiscale features."""

    def __init__(self, stage_channels):
        super().__init__()
        self.v2 = cbr(stage_channels[0], 256)
        self.v3 = cbr(stage_channels[1], 256)
        self.dilated = nn.ModuleList([cbr(256, 128, dilation=d) for d in (1, 2, 3)])
        self.skip = cbr(256, 384, kernel_size=1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, stage2, stage3):
        stage3 = F.interpolate(stage3, size=stage2.shape[-2:],
                               mode='bilinear', align_corners=True)
        fused = self.v2(stage2) + self.v3(stage3)
        return torch.cat([layer(fused) for layer in self.dilated], dim=1) + self.skip(fused)


class ResidualNightFusion(nn.Module):
    """C + alpha * H(concat(C, project(resize(M)))); output stays stride 16."""

    def __init__(self, alpha=0.1):
        super().__init__()
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError('Fusion alpha must be finite and nonnegative')
        self.project = nn.Conv2d(384, 512, kernel_size=1)
        self.update = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, crowd_features, multiscale_features):
        multiscale_features = F.interpolate(
            multiscale_features, size=crowd_features.shape[-2:],
            mode='bilinear', align_corners=False)
        multiscale_features = self.project(multiscale_features)
        update = self.update(torch.cat((crowd_features, multiscale_features), dim=1))
        return crowd_features + self.alpha * update
