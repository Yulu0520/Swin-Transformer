"""Mutually exclusive dual-DoRA routing for packed Swin QKV weights."""

import math

from torch import nn
from torch.nn.utils import parametrize

from .swin_dora import QVDoRA


LEGACY_DORA_PREFIX = '.parametrizations.weight.0.dora.'
LEGACY_LORA_PREFIX = '.parametrizations.weight.0.lora.'


def has_legacy_dora_lora_adapters(state):
    """Return whether a state dict uses the former DoRA/LoRA container."""
    return (any(LEGACY_DORA_PREFIX in key for key in state)
            and any(LEGACY_LORA_PREFIX in key for key in state))


def extract_dora_from_legacy_state(state):
    """Convert a former DoRA/LoRA state dict into standalone DoRA state.

    The trained DoRA and raw baseline are retained. The LoRA branch is dropped;
    it cannot be interpreted as DoRA because it has no magnitude parameters.
    """
    if not has_legacy_dora_lora_adapters(state):
        raise ValueError('State dict does not contain legacy DoRA/LoRA adapters')
    converted = state.__class__()
    for key, value in state.items():
        if LEGACY_LORA_PREFIX in key:
            continue
        converted[key.replace(LEGACY_DORA_PREFIX,
                              '.parametrizations.weight.0.')] = value
    if hasattr(state, '_metadata'):
        converted._metadata = state._metadata
    return converted


class DualQVDoRAAdapter(nn.Module):
    """Route one baseline QKV weight through one of two DoRA branches."""

    ROUTES = ('baseline', 'dora', 'dora1')

    def __init__(self, dora, dora1, active_route='dora1'):
        super().__init__()
        if not isinstance(dora, QVDoRA):
            raise TypeError('dora must be a QVDoRA module')
        if not isinstance(dora1, QVDoRA):
            raise TypeError('dora1 must be a QVDoRA module')
        self.dora = dora
        self.dora1 = dora1
        self.set_route(active_route)

    def set_route(self, route):
        if route not in self.ROUTES:
            raise ValueError(
                'adapter route must be one of {}'.format(self.ROUTES))
        self.active_route = route

    def forward(self, weight):
        # Only the selected branch participates in forward and autograd.
        if self.active_route == 'dora':
            return self.dora(weight)
        if self.active_route == 'dora1':
            return self.dora1(weight)
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


def add_swin_qv_dual_dora(
        features, dora_rank, dora_alpha, dora1_rank, dora1_alpha,
        active_route='dora1'):
    """Attach parallel DoRA and DoRA1 branches to every Swin Q/V block.

    If a standalone DoRA is already attached, it is preserved. Both branches
    always receive the same raw baseline weight rather than another adapter's
    output.
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
            parametrize.remove_parametrizations(
                qkv, 'weight', leave_parametrized=False)
        else:
            dora = QVDoRA(qkv.weight, dora_rank, dora_alpha)

        dora1 = QVDoRA(qkv.weight, dora1_rank, dora1_alpha)
        router = DualQVDoRAAdapter(dora, dora1, active_route)
        parametrize.register_parametrization(qkv, 'weight', router)
    return len(qkv_layers)


def set_swin_adapter_route(features, route):
    """Select one branch without evaluating the inactive adapter."""
    routers = []
    for qkv in _swin_qkv_layers(features):
        if not parametrize.is_parametrized(qkv, 'weight'):
            raise ValueError('Dual DoRA adapters are not attached')
        parametrizations = qkv.parametrizations.weight
        if (len(parametrizations) != 1
                or not isinstance(parametrizations[0], DualQVDoRAAdapter)):
            raise ValueError('Expected exactly one DualQVDoRAAdapter')
        routers.append(parametrizations[0])
    for router in routers:
        router.set_route(route)
    return len(routers)
