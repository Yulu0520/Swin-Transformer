"""Weight-decomposed low-rank adaptation for a single Linear layer."""

import torch
from torch import nn
from torch.nn import functional as F


class DoRALinear(nn.Module):
    """Keep the pretrained direction frozen and learn its update and magnitude."""

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError('DoRALinear requires an nn.Linear base layer')
        if rank <= 0 or alpha <= 0:
            raise ValueError('DoRA rank and alpha must be positive')

        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.magnitude = nn.Parameter(
            torch.linalg.vector_norm(base_layer.weight.detach().float(), dim=1)
        )
        self.register_buffer('alpha', torch.tensor(float(alpha)))
        self.rank = rank

    @property
    def weight(self):
        # torchvision's Swin attention reads .weight directly rather than
        # calling its qkv/proj modules, so expose the effective DoRA weight.
        scale = self.alpha / self.rank
        direction = self.base_layer.weight.float() + scale * (
            self.lora_B @ self.lora_A
        )
        # DoRA treats the dynamic direction norm as constant in backward.
        norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).detach()
        return self.magnitude[:, None] * direction / norm.clamp_min(1e-12)

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)
