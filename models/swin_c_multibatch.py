import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import Swin_T_Weights, swin_t
from torchvision.models.swin_transformer import SwinTransformerBlock

from .dora import DoRALinear

from .transformer_cosine_multibatch import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


__all__ = ["swin_t_trans", "swin_t_trans_from_state_dict"]


class SwinTransMultiBatch(nn.Module):
    """Swin-T backbone followed by the original multibatch MAN head."""

    def __init__(self, pretrained=True, use_dora=True, dora_rank=8, dora_alpha=8.0):
        super().__init__()

        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = swin_t(weights=weights)
        self.backbone_features = backbone.features
        self.backbone_norm = backbone.norm

        # Swin-T stage 4 is stride 32 with 768 channels.  MAN expects 512.
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1),
            nn.GELU(),
        )

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
        self.use_dora = False
        if use_dora:
            self.enable_dora(rank=dora_rank, alpha=dora_alpha)

    def enable_dora(self, rank=8, alpha=8.0):
        """Adapt qkv and output projection in every W-MSA/SW-MSA block."""
        if self.use_dora:
            raise RuntimeError('DoRA is already enabled')
        for parameter in self.backbone_features.parameters():
            parameter.requires_grad_(False)
        for parameter in self.backbone_norm.parameters():
            parameter.requires_grad_(False)

        blocks = [
            module for module in self.backbone_features.modules()
            if isinstance(module, SwinTransformerBlock)
        ]
        if len(blocks) != 12:
            raise RuntimeError('expected 12 Swin-T blocks, found {}'.format(len(blocks)))
        for block in blocks:
            block.attn.qkv = DoRALinear(block.attn.qkv, rank, alpha)
            block.attn.proj = DoRALinear(block.attn.proj, rank, alpha)
        self.dora_rank = rank
        self.dora_alpha = alpha
        self.use_dora = True

    def train_dora_only(self):
        """Freeze the entire model except the DoRA direction and magnitude."""
        if not self.use_dora:
            raise RuntimeError('DoRA must be enabled before DoRA-only training')
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in self.modules():
            if isinstance(module, DoRALinear):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
                module.magnitude.requires_grad_(True)

    def forward(self, x):
        input_h, input_w = x.shape[-2:]
        density_size = (input_h // 16, input_w // 16)

        # torchvision Swin features and norm use channels-last tensors.
        x = self.backbone_features(x)
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
        density = self.reg_layer_0(x)
        return torch.relu(density), consistency_features


def swin_t_trans(pretrained=True, use_dora=True, dora_rank=8, dora_alpha=8.0):
    return SwinTransMultiBatch(pretrained, use_dora, dora_rank, dora_alpha)


def swin_t_trans_from_state_dict(state_dict):
    """Build a matching model for either a plain or a DoRA checkpoint."""
    rank_keys = [key for key in state_dict if key.endswith('.attn.qkv.lora_A')]
    if rank_keys:
        if len(rank_keys) != 12:
            raise ValueError('incomplete DoRA checkpoint: expected 12 qkv layers')
        first_key = rank_keys[0]
        rank = state_dict[first_key].shape[0]
        alpha_key = first_key.removesuffix('lora_A') + 'alpha'
        alpha = state_dict[alpha_key].item()
        model = swin_t_trans(
            pretrained=False, use_dora=True, dora_rank=rank, dora_alpha=alpha
        )
    else:
        model = swin_t_trans(pretrained=False, use_dora=False)
    model.load_state_dict(state_dict)
    return model
