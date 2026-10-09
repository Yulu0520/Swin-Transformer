"""Mutually exclusive dual-DoRA routing for Swin attention projections."""

import math

from torch import nn
from torch.nn.utils import parametrize

from .swin_dora import ProjectionDoRA, validate_dora_targets


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


def extract_dora_from_dual_state(state):
    """Keep trained DoRA and baseline, dropping the inactive DoRA1 branch.

    A projection used only by DoRA1 is restored to an ordinary Linear weight.
    This permits starting a fresh DoRA1 with different targets and rank/alpha.
    """
    prefix = '.parametrizations.weight.0.dora.'
    adapted_layers = {key.split(prefix)[0] for key in state
                      if prefix + 'dora_A_' in key}
    converted = state.__class__()
    original_suffix = '.parametrizations.weight.original'
    for key, value in state.items():
        if '.parametrizations.weight.0.dora1.' in key:
            continue
        if key.endswith(original_suffix):
            layer = key[:-len(original_suffix)]
            if layer not in adapted_layers:
                key = layer + '.weight'
        converted[key.replace(prefix, '.parametrizations.weight.0.')] = value
    return converted


class DualQVDoRAAdapter(nn.Module):
    """Route one baseline projection through one of two DoRA branches."""

    ROUTES = ('baseline', 'dora', 'dora1')

    def __init__(self, dora, dora1, active_route='dora1'):
        super().__init__()
        if not isinstance(dora, ProjectionDoRA):
            raise TypeError('dora must be a ProjectionDoRA module')
        if not isinstance(dora1, ProjectionDoRA):
            raise TypeError('dora1 must be a ProjectionDoRA module')
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


def _validate_existing_dora(dora, rank, alpha, targets):
    if dora.targets != targets or dora.rank != rank or not math.isclose(
            dora.scaling, alpha / rank, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError('Existing DoRA settings do not match requested targets/rank/alpha')


def add_swin_qv_dual_dora(
        features, dora_rank, dora_alpha, dora1_rank, dora1_alpha,
        active_route='dora1', targets='qv', dora1_targets='qv'):
    """Attach parallel branches to selected projections in every Swin block.

    If a standalone DoRA is already attached, it is preserved. Both branches
    always receive the same raw baseline weight rather than another adapter's
    output.
    """
    targets = validate_dora_targets(targets)
    dora1_targets = validate_dora_targets(dora1_targets)
    qkv_layers = _swin_qkv_layers(features)
    layers = []
    for qkv in qkv_layers:
        selected = ''.join(t for t in 'qkv' if t in targets)
        selected1 = ''.join(t for t in 'qkv' if t in dora1_targets)
        if selected or selected1:
            layers.append((qkv, selected, selected1, True))
    if 'o' in targets or 'o' in dora1_targets:
        for module in features.modules():
            if hasattr(module, 'attn') and hasattr(module.attn, 'qkv'):
                layers.append((module.attn.proj,
                               'o' if 'o' in targets else '',
                               'o' if 'o' in dora1_targets else '', False))
    for layer, selected, selected1, packed in layers:
        if parametrize.is_parametrized(layer, 'weight'):
            parametrizations = layer.parametrizations.weight
            if (len(parametrizations) != 1
                    or not isinstance(parametrizations[0], ProjectionDoRA)):
                raise ValueError(
                    'Expected exactly one DoRA parametrization before routing')
            dora = parametrizations[0]
            _validate_existing_dora(dora, dora_rank, dora_alpha, selected)
            parametrize.remove_parametrizations(
                layer, 'weight', leave_parametrized=False)
        else:
            dora = ProjectionDoRA(layer.weight, dora_rank, dora_alpha, selected, packed)
        dora1 = ProjectionDoRA(layer.weight, dora1_rank, dora1_alpha, selected1, packed)
        router = DualQVDoRAAdapter(dora, dora1, active_route)
        parametrize.register_parametrization(layer, 'weight', router)
    return len(qkv_layers)


def set_swin_adapter_route(features, route):
    """Select one branch without evaluating the inactive adapter."""
    block_count = len(_swin_qkv_layers(features))
    routers = [module for module in features.modules()
               if isinstance(module, DualQVDoRAAdapter)]
    if not routers:
        raise ValueError('Dual DoRA adapters are not attached')
    for router in routers:
        router.set_route(route)
    return block_count
