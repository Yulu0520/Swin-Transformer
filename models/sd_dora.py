"""Mutually exclusive DoRA branches for SD U-Net self-attention Q/V."""

import torch
from torch import nn
from torch.nn.utils import parametrize


class LinearDoRA(nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError('DoRA rank and alpha must be positive')
        self.scaling = alpha / rank
        self.direction_A = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.direction_B = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        self.magnitude = nn.Parameter(weight.detach().norm(dim=1))
        nn.init.kaiming_uniform_(self.direction_A, a=5 ** 0.5)

    def forward(self, weight):
        direction = weight + self.scaling * (self.direction_B @ self.direction_A)
        norm = direction.norm(dim=1).clamp_min(torch.finfo(direction.dtype).eps).detach()
        return direction * (self.magnitude / norm).unsqueeze(1)


class DualLinearDoRA(nn.Module):
    def __init__(self, weight, rank, alpha, rank1, alpha1, route):
        super().__init__()
        self.dora = LinearDoRA(weight, rank, alpha)
        self.dora1 = LinearDoRA(weight, rank1, alpha1)
        self.set_route(route)

    def set_route(self, route):
        if route not in ('baseline', 'dora', 'dora1'):
            raise ValueError('Unknown adapter route: {}'.format(route))
        self.active_route = route

    def forward(self, weight):
        if self.active_route == 'baseline':
            return weight
        return getattr(self, self.active_route)(weight)


def self_attention_projections(unet):
    # Do not match VAE attention, MAN attention, or attn2 (text attention).
    for name, module in unet.named_modules():
        if name.endswith('attn1'):
            for projection in ('to_q', 'to_v'):
                layer = getattr(module, projection)
                if not isinstance(layer, nn.Linear):
                    raise TypeError('Expected separate Linear Q/V projections')
                yield layer


def add_dual_dora(unet, rank, alpha, rank1, alpha1, route):
    layers = list(self_attention_projections(unet))
    if not layers:
        raise ValueError('No U-Net self-attention Q/V projections found')
    for layer in layers:
        if parametrize.is_parametrized(layer, 'weight'):
            raise ValueError('SD adapters are already attached')
        parametrize.register_parametrization(
            layer, 'weight',
            DualLinearDoRA(layer.weight, rank, alpha, rank1, alpha1, route))
    return len(layers)


def set_route(unet, route):
    routers = [module for module in unet.modules() if isinstance(module, DualLinearDoRA)]
    if not routers:
        raise ValueError('SD adapters are not attached')
    for router in routers:
        router.set_route(route)
    return len(routers)
