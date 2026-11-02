import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import Swin_T_Weights, swin_t
from torchvision.models.swin_transformer import SwinTransformer
import os
import re

from .cctrans_regression import CCTransRegression
from .counting_checkpoint import ARCHITECTURE, OUTPUT_STRIDE
from .swin_dora import add_swin_qv_dora
from .swin_dual_dora import (
    DualQVDoRAAdapter,
    add_swin_qv_dual_dora,
    set_swin_adapter_route,
)


__all__ = ["swin_t_trans", "swin_l_trans"]


def _load_official_swin_large(backbone, path):
    """Map the official Swin-L 22K state dict onto torchvision's Swin-L."""
    if not os.path.isfile(path):
        raise FileNotFoundError('Swin-L pretrained checkpoint not found: {}'.format(path))
    try:
        checkpoint = torch.load(path, map_location='cpu', mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    source = checkpoint.get('model', checkpoint)
    expected = backbone.state_dict()
    mapped = {}
    for key, value in source.items():
        target = None
        if key.startswith('patch_embed.proj.'):
            target = key.replace('patch_embed.proj.', 'features.0.0.', 1)
        elif key.startswith('patch_embed.norm.'):
            target = key.replace('patch_embed.norm.', 'features.0.2.', 1)
        elif key.startswith('norm.'):
            target = key
        else:
            block = re.fullmatch(r'layers\.(\d+)\.blocks\.(\d+)\.(.*)', key)
            downsample = re.fullmatch(r'layers\.(\d+)\.downsample\.(.*)', key)
            if block:
                stage, index, suffix = block.groups()
                suffix = suffix.replace('mlp.fc1.', 'mlp.0.')
                suffix = suffix.replace('mlp.fc2.', 'mlp.3.')
                target = 'features.{}.{}.{}'.format(2 * int(stage) + 1, index, suffix)
            elif downsample:
                stage, suffix = downsample.groups()
                target = 'features.{}.{}'.format(2 * int(stage) + 2, suffix)
        # torchvision generates the same relative-position indices itself;
        # the source also stores resolution-specific attention masks.
        if target is None or target.endswith(('relative_position_index', 'attn_mask')):
            continue
        if target not in expected:
            raise ValueError('Unrecognized Swin-L checkpoint parameter: {}'.format(key))
        if value.shape != expected[target].shape:
            raise ValueError('Swin-L checkpoint shape mismatch: {}'.format(key))
        mapped[target] = value
    missing = [key for key in expected if key.startswith(('features.', 'norm.'))
               and key not in mapped and not key.endswith('relative_position_index')]
    if missing:
        raise ValueError('Incomplete Swin-L checkpoint; first missing key: {}'.format(missing[0]))
    backbone.load_state_dict(mapped, strict=False)


class SwinTransMultiBatch(nn.Module):
    """Swin stages 2/3/4 followed by a CCTrans density regression head."""

    architecture = ARCHITECTURE
    output_stride = OUTPUT_STRIDE

    def __init__(self, pretrained=True, variant='large', pretrained_path=None):
        super().__init__()
        if variant == 'large':
            backbone = SwinTransformer(
                patch_size=[4, 4], embed_dim=192, depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48], window_size=[12, 12])
            if pretrained:
                path = (pretrained_path or
                        'pre_models/swin_large_patch4_window12_384_22k.pth')
                _load_official_swin_large(backbone, path)
            stage_channels = (384, 768, 1536)
        elif variant == 'tiny':
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = swin_t(weights=weights)
            stage_channels = (192, 384, 768)
        else:
            raise ValueError('Unknown Swin variant: {}'.format(variant))
        self.variant = variant
        self.backbone_features = backbone.features
        self.backbone_norm = backbone.norm

        self.regression = CCTransRegression(stage_channels)

    def forward_backbone(self, x):
        # Preserve edge pixels for arbitrary validation image sizes. Training
        # crops are divisible by 8, so this is a no-op during training.
        height, width = x.shape[-2:]
        x = F.pad(x, (0, -width % self.output_stride,
                      0, -height % self.output_stride))
        stages = []
        # torchvision: patch embed, stage1, merge, stage2, merge, stage3,
        # merge, stage4. Keep the original container/keys for DoRA routing.
        for index, layer in enumerate(self.backbone_features):
            x = layer(x)
            if index in (3, 5, 7):
                feature = self.backbone_norm(x) if index == 7 else x
                stages.append(feature.permute(0, 3, 1, 2).contiguous())
        return stages

    def forward(self, x):
        """Return raw nonnegative density [B, 1, ceil(H/8), ceil(W/8)]."""
        return self.regression(*self.forward_backbone(x))

    def enable_dora(self, rank=4, alpha=4.0):
        """Freeze the clean baseline and adapt all backbone attention Q/V."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return add_swin_qv_dora(self.backbone_features, rank, alpha)

    def enable_dual_dora(
            self, dora_rank=4, dora_alpha=4.0,
            dora1_rank=4, dora1_alpha=4.0,
            active_route='dora1', trainable_route=None):
        """Attach mutually exclusive DoRA/DoRA1 branches to backbone Q/V."""
        count = add_swin_qv_dual_dora(
            self.backbone_features,
            dora_rank, dora_alpha, dora1_rank, dora1_alpha,
            active_route=active_route,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if trainable_route is not None:
            if (trainable_route != active_route
                    or trainable_route not in ('dora', 'dora1')):
                raise ValueError(
                    'trainable route must match active DoRA/DoRA1 route')
            for module in self.backbone_features.modules():
                if isinstance(module, DualQVDoRAAdapter):
                    branch = getattr(module, trainable_route)
                    for parameter in branch.parameters():
                        parameter.requires_grad_(True)
        return count

    def set_adapter_route(self, route):
        """Select baseline, DoRA, or DoRA1 for every Swin Q/V projection."""
        return set_swin_adapter_route(self.backbone_features, route)


def swin_t_trans(pretrained=True, pretrained_path=None):
    return SwinTransMultiBatch(pretrained=pretrained, variant='tiny')


def swin_l_trans(pretrained=True, pretrained_path=None):
    return SwinTransMultiBatch(
        pretrained=pretrained, variant='large', pretrained_path=pretrained_path)
