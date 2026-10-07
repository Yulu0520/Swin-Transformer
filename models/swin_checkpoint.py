"""Strict loading and explicit migration to a single Swin DoRA branch."""

import torch

from .swin_dora import add_swin_mlp_dora


def read_checkpoint(path):
    """Read project checkpoints, including old optimizer/NumPy metadata."""
    try:
        return torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    except TypeError:
        # Compatibility with the minimum supported PyTorch version.
        return torch.load(path, map_location='cpu')


def single_dora_state(state):
    """Keep raw baseline and DoRA; discard the parallel DoRA1/legacy LoRA."""
    prefix = '.parametrizations.weight.0.'
    parallel = any(prefix + 'dora.' in key for key in state)
    if not parallel:
        return state
    converted = state.__class__()
    for key, value in state.items():
        if prefix + 'dora1.' in key or prefix + 'lora.' in key:
            continue
        converted[key.replace(prefix + 'dora.', prefix)] = value
    if hasattr(state, '_metadata'):
        converted._metadata = state._metadata
    return converted


def single_dora_config(checkpoint, rank=4, alpha=4.0,
                       mlp_rank=4, mlp_alpha=4.0):
    state = single_dora_state(checkpoint.get('model_state_dict', checkpoint))
    has_mlp = any('.mlp.' in key and '.parametrizations.weight.' in key
                  for key in state)
    config = {
        'adapter_targets': 'qv_mlp' if has_mlp else 'qv',
        'dora_rank': rank, 'dora_alpha': alpha,
        'mlp_dora_rank': mlp_rank, 'mlp_dora_alpha': mlp_alpha,
    }
    if 'model_state_dict' in checkpoint:
        if checkpoint.get('adapter_type') not in (
                None, 'dora', 'single_dora', 'parallel_dora_dora1',
                'parallel_dora_lora'):
            raise ValueError('Unsupported DoRA checkpoint type')
        for key in config:
            if key in checkpoint and (not key.startswith('mlp_') or has_mlp):
                config[key] = checkpoint[key]
        if config['adapter_targets'] != ('qv_mlp' if has_mlp else 'qv'):
            raise ValueError('Checkpoint targets do not match its weights')
    if any(config[key] <= 0 for key in (
            'dora_rank', 'dora_alpha', 'mlp_dora_rank', 'mlp_dora_alpha')):
        raise ValueError('DoRA rank and alpha must be positive')
    if not any(key.endswith('.dora_A_q') for key in state):
        raise ValueError('Checkpoint does not contain Q/V DoRA')
    return state, config


def load_single_dora(model, checkpoint, rank=4, alpha=4.0,
                     mlp_rank=4, mlp_alpha=4.0,
                     targets=None, train_qv=True):
    """Load existing weights before adding identity-initialized MLP adapters.

    ``targets=None`` restores the saved topology exactly. ``qv_mlp`` allows
    a Q/V-only warm start; removing existing MLP adapters is never implicit.
    """
    state, config = single_dora_config(
        checkpoint, rank, alpha, mlp_rank, mlp_alpha)
    if targets not in (None, 'qv', 'qv_mlp'):
        raise ValueError('DoRA targets must be qv or qv_mlp')
    if targets == 'qv' and config['adapter_targets'] == 'qv_mlp':
        raise ValueError('Cannot drop trained MLP adapters')
    model.enable_dora(
        config['dora_rank'], config['dora_alpha'],
        targets=config['adapter_targets'],
        mlp_rank=config['mlp_dora_rank'], mlp_alpha=config['mlp_dora_alpha'],
        train_qv=train_qv)
    model.load_state_dict(state, strict=True)
    if targets == 'qv_mlp' and config['adapter_targets'] == 'qv':
        add_swin_mlp_dora(model.backbone_features, mlp_rank, mlp_alpha)
        config.update(adapter_targets='qv_mlp',
                      mlp_dora_rank=mlp_rank, mlp_dora_alpha=mlp_alpha)
    model.configure_single_dora_training(train_qv)
    return config


def restore_swin_weights(model, checkpoint, rank=4, alpha=4.0,
                         mlp_rank=4, mlp_alpha=4.0):
    """Restore a baseline or a single DoRA model for inference/visualization."""
    expected_name = 'swin_t_trans' if model.variant == 'tiny' else 'swin_l_trans'
    if checkpoint.get('model_name', expected_name) != expected_name:
        raise ValueError('Checkpoint model name does not match')
    state = checkpoint.get('model_state_dict', checkpoint)
    if any('.parametrizations.weight.' in key for key in state):
        return load_single_dora(model, checkpoint, rank, alpha, mlp_rank, mlp_alpha)
    model.load_state_dict(state, strict=True)
    return None
