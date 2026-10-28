"""Fixed daytime prototypes queried by nighttime Swin features."""

import math

import torch
from torch import nn


class MemoryCrossAttention(nn.Module):
    def __init__(self, codebook, heads=8, alpha_init=1.0, beta_init=0.1,
                 train_fusion=True):
        super().__init__()
        if (codebook.ndim != 2 or min(codebook.shape) <= 0
                or not codebook.is_floating_point()
                or not torch.isfinite(codebook).all()):
            raise ValueError('Memory bank must be a finite floating [K, C] tensor')
        dim = codebook.shape[1]
        if heads <= 0 or dim % heads:
            raise ValueError('Memory channels must be divisible by attention heads')
        if not all(math.isfinite(value) for value in (alpha_init, beta_init)):
            raise ValueError('Fusion weights must be finite')
        self.heads = heads
        self.dim = dim
        self.register_buffer('codebook', codebook.detach().clone().float())
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)),
                                  requires_grad=train_fusion)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)),
                                 requires_grad=train_fusion)
        self.metrics = {}

    def forward(self, night):
        if night.ndim != 4 or night.shape[-1] != self.dim:
            raise ValueError('Memory features must have shape [B, H, W, C]')
        if not torch.isfinite(night).all():
            raise FloatingPointError('Non-finite nighttime memory features')
        batch, height, width, dim = night.shape
        head_dim = dim // self.heads
        query = self.query(night.reshape(batch, height * width, dim))
        query = query.reshape(batch, -1, self.heads, head_dim).transpose(1, 2)
        key = self.key(self.codebook).reshape(-1, self.heads, head_dim)
        value = self.value(self.codebook).reshape(-1, self.heads, head_dim)
        key = key.permute(1, 0, 2).unsqueeze(0)
        value = value.permute(1, 0, 2).unsqueeze(0)
        attention = (query @ key.transpose(-1, -2) / math.sqrt(head_dim)).softmax(-1)
        cross = (attention @ value).transpose(1, 2).reshape(batch, height, width, dim)
        cross = self.output(cross)
        fused = self.alpha * night + self.beta * cross
        if not torch.isfinite(fused).all():
            raise FloatingPointError('Non-finite memory fusion; inspect alpha/beta')
        # Detached diagnostics never retain the training graph.
        with torch.no_grad():
            norm = lambda x: torch.linalg.vector_norm(x, dim=-1).mean().item()
            self.metrics = {
                'alpha': self.alpha.item(), 'beta': self.beta.item(),
                'night_norm': norm(night), 'cross_norm': norm(cross),
                'weighted_night_norm': norm(self.alpha * night),
                'weighted_cross_norm': norm(self.beta * cross),
            }
        return fused
