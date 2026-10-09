"""Selectable weight-decomposed low-rank Swin attention adaptation."""

import math

import torch
from torch import nn
from torch.nn.utils import parametrize


DORA_TARGETS = ('q', 'k', 'v', 'o', 'qk', 'qv', 'qkvo')


def validate_dora_targets(targets):
    targets = targets.lower()
    if targets not in DORA_TARGETS:
        raise ValueError('DoRA targets must be one of {}'.format(DORA_TARGETS))
    return targets


class ProjectionDoRA(nn.Module):
    """Adapt selected packed QKV slices or the separate output projection.

    Empty targets implement an identity branch for dual-adapter routing.
    Parameter names retain compatibility with the original Q/V checkpoints.
    """

    def __init__(self, weight, rank, alpha, targets, packed=True):
        super().__init__()
        if rank <= 0 or alpha <= 0 or not math.isfinite(alpha):
            raise ValueError('DoRA rank and alpha must be positive and finite')
        if weight.ndim != 2:
            raise ValueError('DoRA expects a Linear weight')
        if packed and weight.shape[0] != 3 * weight.shape[1]:
            raise ValueError('DoRA expects a packed square QKV weight')
        if any(t not in ('qkv' if packed else 'o') for t in targets):
            raise ValueError('Invalid targets for this projection')
        self.targets = targets
        self.packed = packed
        self.rank = rank
        self.scaling = alpha / rank
        weights = dict(zip('qkv', weight.detach().chunk(3, dim=0))) if packed else {'o': weight.detach()}
        for target in targets:
            base = weights[target]
            a = nn.Parameter(weight.new_empty(rank, base.shape[1]))
            b = nn.Parameter(weight.new_zeros(base.shape[0], rank))
            nn.init.kaiming_uniform_(a, a=5 ** 0.5)
            self.register_parameter('dora_A_' + target, a)
            self.register_parameter('dora_B_' + target, b)
            self.register_parameter('magnitude_' + target, nn.Parameter(
                torch.linalg.vector_norm(base, dim=1)))

    @staticmethod
    def _compose(weight, update, magnitude):
        direction = weight + update
        eps = torch.finfo(direction.dtype).eps
        direction_norm = torch.linalg.vector_norm(
            direction, dim=1).clamp_min(eps).detach()
        return direction * (magnitude / direction_norm).unsqueeze(1)

    def _adapt(self, weight, target):
        if target not in self.targets:
            return weight
        update = self.scaling * (getattr(self, 'dora_B_' + target)
                                 @ getattr(self, 'dora_A_' + target))
        return self._compose(weight, update, getattr(self, 'magnitude_' + target))

    def forward(self, weight):
        if not self.targets:
            return weight
        if self.packed:
            return torch.cat([self._adapt(w, t) for t, w in
                              zip('qkv', weight.chunk(3, dim=0))], dim=0)
        return self._adapt(weight, 'o')


class QVDoRA(ProjectionDoRA):
    """Backward-compatible Q/V adapter."""

    def __init__(self, weight, rank, alpha):
        super().__init__(weight, rank, alpha, 'qv')


def add_swin_dora(features, rank, alpha, targets='qv'):
    """Attach the same rank/alpha to selected projections in every block."""
    targets = validate_dora_targets(targets)
    blocks = [module for module in features.modules()
              if hasattr(module, 'attn') and hasattr(module.attn, 'qkv')]
    if not blocks:
        raise ValueError('No Swin attention blocks found')
    layers = []
    for block in blocks:
        qkv = block.attn.qkv
        if not isinstance(qkv, nn.Linear) or qkv.out_features != 3 * qkv.in_features:
            raise ValueError('Swin attention QKV must be a packed Linear projection')
        qkv_targets = ''.join(t for t in 'qkv' if t in targets)
        if qkv_targets:
            layers.append((qkv, qkv_targets, True))
        if 'o' in targets:
            layers.append((block.attn.proj, 'o', False))
    for layer, _, _ in layers:
        if not isinstance(layer, nn.Linear):
            raise ValueError('Swin attention projection must be Linear')
        if parametrize.is_parametrized(layer, 'weight'):
            raise ValueError('DoRA is already attached to a Swin attention block')
    for layer, selected, packed in layers:
        parametrize.register_parametrization(layer, 'weight',
            ProjectionDoRA(layer.weight, rank, alpha, selected, packed))
    return len(blocks)


def add_swin_qv_dora(features, rank, alpha):
    """Backward-compatible registration entry point."""
    return add_swin_dora(features, rank, alpha, 'qv')
