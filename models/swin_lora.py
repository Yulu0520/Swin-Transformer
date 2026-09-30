"""Mutually exclusive DoRA/LoRA routing for packed Swin QKV weights."""

import math

import torch
from torch import nn
from torch.nn.utils import parametrize

from .swin_dora import QVDoRA


class QVLoRA(nn.Module):
    """Apply additive LoRA to Q and V while leaving K unchanged."""

    def __init__(self, weight, rank, alpha):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError('LoRA rank and alpha must be positive')
        if weight.ndim != 2 or weight.shape[0] != 3 * weight.shape[1]:
            raise ValueError('LoRA expects a packed square QKV weight')

        dim = weight.shape[1]
        self.scaling = alpha / rank
        self.lora_A_q = nn.Parameter(weight.new_empty(rank, dim))
        self.lora_B_q = nn.Parameter(weight.new_zeros(dim, rank))
        self.lora_A_v = nn.Parameter(weight.new_empty(rank, dim))
        self.lora_B_v = nn.Parameter(weight.new_zeros(dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_q, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.lora_A_v, a=5 ** 0.5)

    def forward(self, weight):
        q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
        q_weight = q_weight + self.scaling * (self.lora_B_q @ self.lora_A_q)
        v_weight = v_weight + self.scaling * (self.lora_B_v @ self.lora_A_v)
        return torch.cat((q_weight, k_weight, v_weight), dim=0)


class ParallelQVAdapter(nn.Module):
    """Route a baseline QKV weight through exactly one adapter branch."""

    ROUTES = ('baseline', 'dora', 'lora')

    def __init__(self, dora, lora, active_route='lora'):
        super().__init__()
        if not isinstance(dora, QVDoRA):
            raise TypeError('dora must be a QVDoRA module')
        if not isinstance(lora, QVLoRA):
            raise TypeError('lora must be a QVLoRA module')
        self.dora = dora
        self.lora = lora
        self.set_route(active_route)

    def set_route(self, route):
        if route not in self.ROUTES:
            raise ValueError(
                'adapter route must be one of {}'.format(self.ROUTES))
        self.active_route = route

    def forward(self, weight):
        # Only the selected branch is called. The inactive adapter is absent
        # from both the forward computation and the autograd graph.
        if self.active_route == 'dora':
            return self.dora(weight)
        if self.active_route == 'lora':
            return self.lora(weight)
        return weight


def _swin_qkv_layers(features):
    blocks = [module for module in features.modules()
              if hasattr(module, 'attn') and hasattr(module.attn, 'qkv')]
    if not blocks:
        raise ValueError('No Swin attention blocks found')
    qkv_layers = []
    for block in blocks:
        qkv = block.attn.qkv
        if not isinstance(qkv, nn.Linear) or qkv.out_features != 3 * qkv.in_features:
            raise ValueError('Swin attention QKV must be a packed Linear projection')
        qkv_layers.append(qkv)
    return qkv_layers


def _validate_existing_dora(dora, rank, alpha):
    actual_rank = dora.dora_A_q.shape[0]
    if actual_rank != rank or not math.isclose(
            dora.scaling, alpha / rank, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError('Existing DoRA settings do not match requested rank/alpha')


def add_swin_qv_parallel_adapters(
        features, dora_rank, dora_alpha, lora_rank, lora_alpha,
        active_route='lora'):
    """Attach parallel DoRA/LoRA branches to every Swin attention block.

    An already attached DoRA branch is preserved and moved into the router.
    Otherwise a new DoRA branch is initialized from the baseline weight.
    """
    qkv_layers = _swin_qkv_layers(features)
    for qkv in qkv_layers:
        if parametrize.is_parametrized(qkv, 'weight'):
            parametrizations = qkv.parametrizations.weight
            if (len(parametrizations) != 1
                    or not isinstance(parametrizations[0], QVDoRA)):
                raise ValueError(
                    'Expected exactly one DoRA parametrization before routing')
            dora = parametrizations[0]
            _validate_existing_dora(dora, dora_rank, dora_alpha)
            # Restore the raw baseline weight, not the current DoRA result.
            parametrize.remove_parametrizations(
                qkv, 'weight', leave_parametrized=False)
        else:
            dora = QVDoRA(qkv.weight, dora_rank, dora_alpha)

        lora = QVLoRA(qkv.weight, lora_rank, lora_alpha)
        router = ParallelQVAdapter(dora, lora, active_route)
        parametrize.register_parametrization(qkv, 'weight', router)
    return len(qkv_layers)


def set_swin_adapter_route(features, route):
    """Select one branch without evaluating the inactive adapter."""
    routers = []
    for qkv in _swin_qkv_layers(features):
        if not parametrize.is_parametrized(qkv, 'weight'):
            raise ValueError('Parallel adapters are not attached')
        parametrizations = qkv.parametrizations.weight
        if (len(parametrizations) != 1
                or not isinstance(parametrizations[0], ParallelQVAdapter)):
            raise ValueError('Expected exactly one ParallelQVAdapter')
        routers.append(parametrizations[0])
    for router in routers:
        router.set_route(route)
    return len(routers)
