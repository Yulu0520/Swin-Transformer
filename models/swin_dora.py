"""Weight-decomposed low-rank Q/V adaptation for packed Swin QKV."""

import torch
from torch import nn
from torch.nn.utils import parametrize


class QVDoRA(nn.Module):
    """Apply DoRA to Q and V while leaving K exactly unchanged."""

    def __init__(self, weight, rank, alpha):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError('DoRA rank and alpha must be positive')
        if weight.ndim != 2 or weight.shape[0] != 3 * weight.shape[1]:
            raise ValueError('DoRA expects a packed square QKV weight')

        dim = weight.shape[1]
        self.scaling = alpha / rank
        self.dora_A_q = nn.Parameter(weight.new_empty(rank, dim))
        self.dora_B_q = nn.Parameter(weight.new_zeros(dim, rank))
        self.dora_A_v = nn.Parameter(weight.new_empty(rank, dim))
        self.dora_B_v = nn.Parameter(weight.new_zeros(dim, rank))
        nn.init.kaiming_uniform_(self.dora_A_q, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.dora_A_v, a=5 ** 0.5)

        q_weight, _, v_weight = weight.detach().chunk(3, dim=0)
        self.magnitude_q = nn.Parameter(torch.linalg.vector_norm(q_weight, dim=1))
        self.magnitude_v = nn.Parameter(torch.linalg.vector_norm(v_weight, dim=1))

    @staticmethod
    def _compose(weight, update, magnitude):
        direction = weight + update
        # DoRA's simple formulation lets the norm track the current direction
        # without backpropagating through the denominator.
        eps = torch.finfo(direction.dtype).eps
        direction_norm = torch.linalg.vector_norm(
            direction, dim=1).clamp_min(eps).detach()
        return direction * (magnitude / direction_norm).unsqueeze(1)

    def forward(self, weight):
        q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
        q_update = self.scaling * (self.dora_B_q @ self.dora_A_q)
        v_update = self.scaling * (self.dora_B_v @ self.dora_A_v)
        q_weight = self._compose(q_weight, q_update, self.magnitude_q)
        v_weight = self._compose(v_weight, v_update, self.magnitude_v)
        return torch.cat((q_weight, k_weight, v_weight), dim=0)


def add_swin_qv_dora(features, rank, alpha):
    """Attach DoRA to every W-MSA/SW-MSA block found in Swin features."""
    blocks = [module for module in features.modules()
              if hasattr(module, 'attn') and hasattr(module.attn, 'qkv')]
    if not blocks:
        raise ValueError('No Swin attention blocks found')
    for block in blocks:
        qkv = block.attn.qkv
        if not isinstance(qkv, nn.Linear) or qkv.out_features != 3 * qkv.in_features:
            raise ValueError('Swin attention QKV must be a packed Linear projection')
        if parametrize.is_parametrized(qkv, 'weight'):
            raise ValueError('DoRA is already attached to a Swin attention block')
        parametrize.register_parametrization(
            qkv, 'weight', QVDoRA(qkv.weight, rank, alpha))
    return len(blocks)


class LinearDoRA(nn.Module):
    """DoRA for a rectangular Linear weight; the original bias is untouched."""

    def __init__(self, weight, rank, alpha):
        super().__init__()
        if rank <= 0 or alpha <= 0 or weight.ndim != 2:
            raise ValueError('Linear DoRA needs a matrix and positive rank/alpha')
        out_dim, in_dim = weight.shape
        self.scaling = alpha / rank
        self.dora_A = nn.Parameter(weight.new_empty(rank, in_dim))
        self.dora_B = nn.Parameter(weight.new_zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.dora_A, a=5 ** 0.5)
        self.magnitude = nn.Parameter(
            torch.linalg.vector_norm(weight.detach(), dim=1))

    def forward(self, weight):
        return QVDoRA._compose(
            weight, self.scaling * (self.dora_B @ self.dora_A), self.magnitude)


def add_swin_mlp_dora(features, rank=4, alpha=4.0):
    """Attach to the two MLP Linear layers inside each Swin block only."""
    blocks = [module for module in features.modules()
              if hasattr(module, 'attn') and hasattr(module.attn, 'qkv')]
    if not blocks:
        raise ValueError('No Swin attention blocks found')
    layers = []
    for block in blocks:
        if not hasattr(block, 'mlp'):
            raise ValueError('Swin block is missing its MLP')
        linear = [module for module in block.mlp.modules()
                  if isinstance(module, nn.Linear)]
        dim = block.attn.qkv.in_features
        if (len(linear) != 2 or linear[0].in_features != dim
                or linear[0].out_features != 4 * dim
                or linear[1].in_features != 4 * dim
                or linear[1].out_features != dim):
            raise ValueError('Expected Swin MLP dimensions C -> 4C -> C')
        layers.extend(linear)
    for layer in layers:
        if parametrize.is_parametrized(layer, 'weight'):
            raise ValueError('DoRA is already attached to a Swin MLP')
    for layer in layers:
        parametrize.register_parametrization(
            layer, 'weight', LinearDoRA(layer.weight, rank, alpha))
    return len(layers)
