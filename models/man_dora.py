"""Dual DoRA on MAN's four query projections and shared value projection."""

import math

import torch
from torch import nn
from torch.nn.utils import parametrize

from .swin_dora import QVDoRA


PROJECTION_INDICES = {
    'global_q': 0, 'global_v': 2, 'local_left_q': 3,
    'local_right_q': 5, 'local_q': 7,
}


class ProjectionDoRA(nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError('DoRA rank and alpha must be positive')
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        out_dim, in_dim = weight.shape
        self.A = nn.Parameter(weight.new_empty(rank, in_dim))
        self.B = nn.Parameter(weight.new_zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.magnitude = nn.Parameter(
            torch.linalg.vector_norm(weight.detach(), dim=1))

    def forward(self, weight):
        return QVDoRA._compose(
            weight, self.scaling * (self.B @ self.A), self.magnitude)


class MANDoRA(nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        if weight.ndim != 2 or weight.shape[0] != 9 * weight.shape[1]:
            raise ValueError('MAN DoRA expects a packed 9D x D projection')
        self.rank, self.alpha = rank, alpha
        chunks = weight.chunk(9, dim=0)
        self.projections = nn.ModuleDict({
            name: ProjectionDoRA(chunks[index], rank, alpha)
            for name, index in PROJECTION_INDICES.items()
        })

    def forward(self, weight):
        chunks = list(weight.chunk(9, dim=0))
        for name, index in PROJECTION_INDICES.items():
            chunks[index] = self.projections[name](chunks[index])
        return torch.cat(chunks, dim=0)


class DualMANDoRAAdapter(nn.Module):
    ROUTES = ('baseline', 'dora', 'dora1')

    def __init__(self, weight, rank, alpha, rank1, alpha1,
                 active_route='dora1'):
        super().__init__()
        self.dora = MANDoRA(weight, rank, alpha)
        self.dora1 = MANDoRA(weight, rank1, alpha1)
        self.set_route(active_route)

    def set_route(self, route):
        if route not in self.ROUTES:
            raise ValueError('adapter route must be one of {}'.format(self.ROUTES))
        self.active_route = route

    def forward(self, weight):
        if self.active_route == 'baseline':
            return weight
        return getattr(self, self.active_route)(weight)


def add_man_dual_dora(encoder, rank, alpha, rank1, alpha1, active_route):
    for layer in encoder.layers:
        attn = layer.self_attn
        if parametrize.is_parametrized(attn, 'in_proj_weight'):
            raise ValueError('MAN DoRA is already attached')
        parametrize.register_parametrization(
            attn, 'in_proj_weight', DualMANDoRAAdapter(
                attn.in_proj_weight, rank, alpha, rank1, alpha1, active_route))
    return len(encoder.layers)


def upgrade_man_state_dict(model, state):
    """Initialize absent MAN adapters from old checkpoint weights, strictly.

    Only entirely absent MAN parametrizations are migrated. Partial adapter
    checkpoints remain errors in load_state_dict rather than being filled in.
    """
    if any('.parametrizations.in_proj_weight.' in key for key in state):
        return state
    upgraded = state.copy()
    if hasattr(state, '_metadata'):
        upgraded._metadata = state._metadata
    for name, module in model.named_modules():
        if not parametrize.is_parametrized(module, 'in_proj_weight'):
            continue
        router = module.parametrizations.in_proj_weight[0]
        if not isinstance(router, DualMANDoRAAdapter):
            continue
        old_key = name + '.in_proj_weight'
        if old_key not in upgraded:
            continue  # Strict loading reports the missing baseline weight.
        weight = upgraded.pop(old_key)
        prefix = name + '.parametrizations.in_proj_weight.'
        upgraded[prefix + 'original'] = weight
        fresh = DualMANDoRAAdapter(
            weight, router.dora.rank, router.dora.alpha,
            router.dora1.rank, router.dora1.alpha, router.active_route)
        for key, value in fresh.state_dict().items():
            upgraded[prefix + '0.' + key] = value
    return upgraded
