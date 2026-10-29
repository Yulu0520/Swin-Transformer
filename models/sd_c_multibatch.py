"""Text-free SD 1.5 backbone with the unchanged MAN counting modules."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .swin_c_multibatch import SwinTransMultiBatch
from .transformer_cosine_multibatch import TransformerEncoder, TransformerEncoderLayer
from .sd_dora import DualLinearDoRA, add_dual_dora, set_route


SD15_SOURCE = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
# SD 1.5 architecture. These allow checkpoint reloads without network access.
UNET_CONFIG = dict(
    sample_size=64, in_channels=4, out_channels=4,
    down_block_types=('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D',
                      'CrossAttnDownBlock2D', 'DownBlock2D'),
    up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D',
                    'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'),
    block_out_channels=(320, 640, 1280, 1280), layers_per_block=2,
    cross_attention_dim=768, attention_head_dim=8,
    norm_num_groups=32, norm_eps=1e-5, downsample_padding=1,
)
VAE_CONFIG = dict(
    in_channels=3, out_channels=3,
    down_block_types=('DownEncoderBlock2D',) * 4,
    up_block_types=('UpDecoderBlock2D',) * 4,
    block_out_channels=(128, 256, 512, 512), layers_per_block=2,
    act_fn='silu', latent_channels=4, norm_num_groups=32,
    sample_size=512, scaling_factor=0.18215,
)


def prepare_image_backbone(vae, unet):
    from diffusers.models.attention import BasicTransformerBlock

    if len(vae.config.block_out_channels) != 4 or len(unet.config.block_out_channels) != 4:
        raise ValueError('Expected a four-level SD 1.5 VAE/U-Net')
    # Remove unused parameters rather than leaving baseline parameters without gradients.
    vae.decoder = None
    vae.post_quant_conv = None
    count = 0
    for block in unet.modules():
        if isinstance(block, BasicTransformerBlock):
            if block.only_cross_attention or not isinstance(block.norm1, nn.LayerNorm):
                raise ValueError('Expected SD 1.5 self-attention + layer-norm blocks')
            block.attn2 = None
            block.norm2 = None
            count += 1
    if not count:
        raise ValueError('No SD Transformer blocks found')
    # Return final decoder image features, retaining norm/activation. The noise
    # prediction head is unused for counting and is removed from the model.
    unet.conv_out = nn.Identity()


class SDTransMultiBatch(SwinTransMultiBatch):
    def __init__(self, pretrained=True, pretrained_path=None, vae=None, unet=None):
        nn.Module.__init__(self)
        from diffusers import AutoencoderKL, UNet2DConditionModel

        if (vae is None) != (unet is None):
            raise ValueError('Supply both VAE and U-Net or neither')
        if vae is None:
            if pretrained:
                source = pretrained_path or SD15_SOURCE
                # Never instantiate a Pipeline, tokenizer or text encoder.
                vae = AutoencoderKL.from_pretrained(source, subfolder='vae')
                unet = UNet2DConditionModel.from_pretrained(source, subfolder='unet')
                # Checkpoint reloads use the fixed SD 1.5 architecture below.
                for key in ('block_out_channels', 'down_block_types', 'up_block_types',
                            'layers_per_block', 'cross_attention_dim', 'attention_head_dim'):
                    actual, expected = getattr(unet.config, key), UNET_CONFIG[key]
                    if isinstance(expected, tuple):
                        actual = tuple(actual)
                    if actual != expected:
                        raise ValueError('Expected SD 1.5 U-Net configuration: ' + key)
                for key in ('block_out_channels', 'layers_per_block', 'latent_channels'):
                    actual, expected = getattr(vae.config, key), VAE_CONFIG[key]
                    if isinstance(expected, tuple):
                        actual = tuple(actual)
                    if actual != expected:
                        raise ValueError('Expected SD 1.5 VAE configuration: ' + key)
            else:
                vae = AutoencoderKL(**VAE_CONFIG)
                unet = UNet2DConditionModel(**UNET_CONFIG)
        prepare_image_backbone(vae, unet)
        self.vae = vae
        self.unet = unet
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(unet.config.block_out_channels[0], 512, 1), nn.GELU())
        # Same modules, dimensions and operations as SwinTransMultiBatch.
        layer = TransformerEncoderLayer(
            d_model=512, nhead=2, dim_feedforward=2048, dropout=0.1,
            activation='relu', normalize_before=False)
        self.encoder = TransformerEncoder(layer, num_layers=2, norm=None)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1))

    def forward(self, x):
        height, width = x.shape[-2:]
        if min(height, width) < 16:
            raise ValueError('SD counting inputs must be at least 16 pixels per side')
        # Pad to complete U-Net down/up sampling grids, without resizing people.
        x = F.pad(x, (0, (-width) % 64, 0, (-height) % 64), mode='replicate')
        context = nullcontext() if any(p.requires_grad for p in self.vae.parameters()) else torch.no_grad()
        with context:
            # Match T2ICount: posterior mode, no latent scaling/no added noise.
            latent = self.vae.encode(x).latent_dist.mode()
        timestep = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        feature = self.unet(latent, timestep, encoder_hidden_states=None).sample
        # Final decoder features include every U-Net stage in the gradient path.
        feature = feature[:, :, : (height + 7) // 8, : (width + 7) // 8]
        feature = F.adaptive_avg_pool2d(feature, (max(1, height // 32), max(1, width // 32)))
        feature = self.feature_adapter(feature)
        batch, channels, fh, fw = feature.shape
        tokens, consistency = self.encoder(feature.flatten(2).permute(2, 0, 1), (fh, fw))
        feature = tokens.permute(1, 2, 0).reshape(batch, channels, fh, fw)
        feature = F.interpolate(feature, (height // 16, width // 16), mode='bilinear', align_corners=False)
        return torch.relu(self.reg_layer_0(feature)), consistency

    def enable_dual_dora(self, dora_rank=4, dora_alpha=4.0,
                         dora1_rank=4, dora1_alpha=4.0,
                         active_route='dora1', trainable_route=None):
        if trainable_route is not None and (
                trainable_route != active_route or trainable_route not in ('dora', 'dora1')):
            raise ValueError('Trainable route must match the active adapter')
        count = add_dual_dora(self.unet, dora_rank, dora_alpha,
                              dora1_rank, dora1_alpha, active_route)
        self.requires_grad_(False)
        if trainable_route is not None:
            for module in self.unet.modules():
                if isinstance(module, DualLinearDoRA):
                    getattr(module, trainable_route).requires_grad_(True)
        return count

    def set_adapter_route(self, route):
        return set_route(self.unet, route)

    def enable_dora(self, rank=4, alpha=4.0):
        raise ValueError('SD uses parallel DoRA/DoRA1 checkpoints; Swin adapters are incompatible')


def sd15_man(pretrained=True, pretrained_path=None):
    return SDTransMultiBatch(pretrained=pretrained, pretrained_path=pretrained_path)
