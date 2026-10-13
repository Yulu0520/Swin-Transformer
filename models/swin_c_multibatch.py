import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import Swin_T_Weights, swin_t
from torchvision.models.swin_transformer import SwinTransformer
import os
import re

from .transformer_cosine_multibatch import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .swin_dora import add_swin_qv_dora
from .swin_dual_dora import (
    DualQVDoRAAdapter,
    add_swin_qv_dual_dora,
    set_swin_adapter_route,
)


__all__ = ["swin_t_trans", "swin_l_trans"]


def normalize_fusion_stages(stages):
    stages = tuple(stages)
    if not stages or any(stage not in (1, 2, 3) for stage in stages):
        raise ValueError('fusion stages must be a nonempty selection of 1, 2, 3')
    if len(set(stages)) != len(stages):
        raise ValueError('fusion stages must not contain duplicates')
    return tuple(sorted(stages))


def validate_fusion_checkpoint(checkpoint, stages):
    """Check metadata and projection keys, including plain/legacy state dicts."""
    requested = normalize_fusion_stages(stages)
    state = checkpoint.get('model_state_dict', checkpoint)
    stored = tuple(sorted(int(key.split('.')[1]) + 1 for key in state
                          if re.fullmatch(r'stage_adapters\.\d+\.weight', key)))
    metadata = checkpoint.get('fusion_stages')
    if stored != requested or (metadata is not None and tuple(metadata) != requested):
        raise ValueError('Checkpoint fusion stages {} do not match requested {}; '
                         'use matching --fusion-stages or train a new baseline'.format(
                             stored, requested))
    weight = state.get('fusion_conv.weight')
    if weight is None or weight.shape[1] != 512 * (len(requested) + 1):
        raise ValueError('Checkpoint lacks the matching concat-convolution fusion')


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
    """Stage-4 MAN encoder with concat-convolution fusion before regression."""

    def __init__(self, pretrained=True, variant='large', pretrained_path=None,
                 fusion_stages=(3,)):
        super().__init__()
        self.fusion_stages = normalize_fusion_stages(fusion_stages)
        if variant == 'large':
            backbone = SwinTransformer(
                patch_size=[4, 4], embed_dim=192, depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48], window_size=[12, 12])
            if pretrained:
                path = (pretrained_path or
                        'pre_models/swin_large_patch4_window12_384_22k.pth')
                _load_official_swin_large(backbone, path)
            output_channels = 1536
        elif variant == 'tiny':
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = swin_t(weights=weights)
            output_channels = 768
        else:
            raise ValueError('Unknown Swin variant: {}'.format(variant))
        self.variant = variant
        self.backbone_features = backbone.features
        self.backbone_norm = backbone.norm

        # Both backbones end at stride 32; the existing MAN encoder uses 512 channels.
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(output_channels, 512, kernel_size=1),
            nn.GELU(),
        )
        # Stable stage indices preserve keys of former four-branch checkpoints.
        self.stage_adapters = nn.ModuleDict({
            str(stage - 1): nn.Conv2d(
                output_channels // (2 ** (4 - stage)), 512, kernel_size=1)
            for stage in self.fusion_stages
        })
        # Normalize each branch over channels without learned branch scaling.
        self.fusion_norms = nn.ModuleList([
            nn.LayerNorm(512, elementwise_affine=False)
            for _ in range(len(self.fusion_stages) + 1)
        ])
        self.fusion_conv = nn.Conv2d(
            (len(self.fusion_stages) + 1) * 512, 512, kernel_size=1)

        d_model = 512
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2, norm=None)

        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        input_h, input_w = x.shape[-2:]
        density_size = (input_h // 16, input_w // 16)

        # torchvision Swin features and norm use channels-last tensors.
        stage_features = {}
        for index, layer in enumerate(self.backbone_features):
            x = layer(x)
            # Capture outputs after each stage's blocks, before patch merging.
            if index in tuple(2 * stage - 1 for stage in self.fusion_stages):
                stage_features[(index + 1) // 2] = x.permute(0, 3, 1, 2).contiguous()
        x = self.backbone_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.feature_adapter(x)

        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).permute(2, 0, 1)
        tokens, consistency_features = self.encoder(tokens, (height, width))
        x = tokens.permute(1, 2, 0).reshape(
            batch_size, channels, height, width
        )

        x = F.interpolate(
            x, size=density_size, mode="bilinear", align_corners=False
        )
        x = self.fusion_norms[-1](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        branches = []
        for stage, norm in zip(self.fusion_stages, self.fusion_norms):
            feature = self.stage_adapters[str(stage - 1)](stage_features[stage])
            # Area resizing averages shallow features onto the density grid.
            if feature.shape[-2:] != density_size:
                feature = F.interpolate(feature, size=density_size, mode="area")
            feature = norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            branches.append(feature)
        # Selected stages in ascending order, followed by MAN output.
        branches.append(x)
        x = self.fusion_conv(torch.cat(branches, dim=1))
        density = self.reg_layer_0(x)
        return torch.relu(density), consistency_features

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


def swin_t_trans(pretrained=True, pretrained_path=None, fusion_stages=(3,)):
    return SwinTransMultiBatch(pretrained=pretrained, variant='tiny',
                              fusion_stages=fusion_stages)


def swin_l_trans(pretrained=True, pretrained_path=None, fusion_stages=(3,)):
    return SwinTransMultiBatch(
        pretrained=pretrained, variant='large', pretrained_path=pretrained_path,
        fusion_stages=fusion_stages)
