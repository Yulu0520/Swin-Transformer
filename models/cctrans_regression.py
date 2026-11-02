"""CCTrans regression head adapted to Swin stage-2/3/4 channels.

Reference: Yulu0520/CCTrans-Crowd-Counting, Networks/ALTGVT.py,
commit 71210285a36eb76cb22b3789dd10ee04235a17fa, class Regression.
"""

from torch import cat, nn
from torch.nn import functional as F


def cbr(in_channels, out_channels, kernel_size=3, dilation=1):
    padding = dilation if kernel_size == 3 else 0
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CCTransRegression(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.v1 = cbr(in_channels[0], 256)
        self.v2 = cbr(in_channels[1], 256)
        self.v3 = cbr(in_channels[2], 256)
        self.stage1 = cbr(256, 128, dilation=1)
        self.stage2 = cbr(256, 128, dilation=2)
        self.stage3 = cbr(256, 128, dilation=3)
        self.stage4 = cbr(256, 384, kernel_size=1)
        self.res = nn.Sequential(
            *cbr(384, 64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
        )
        # Match Regression.init_param in the reference implementation.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, stage2, stage3, stage4):
        size = stage2.shape[-2:]
        stage3 = F.interpolate(stage3, size=size, mode='bilinear', align_corners=True)
        stage4 = F.interpolate(stage4, size=size, mode='bilinear', align_corners=True)
        fused = self.v1(stage2) + self.v2(stage3) + self.v3(stage4)
        features = cat((self.stage1(fused), self.stage2(fused),
                        self.stage3(fused)), dim=1) + self.stage4(fused)
        return fused, features

    def forward(self, stage2, stage3, stage4):
        _, features = self.forward_features(stage2, stage3, stage4)
        return self.res(features)
