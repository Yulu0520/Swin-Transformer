"""Checkpoint loading and inference builders for Swin + CCTrans Regression."""

import glob
import os

import torch

from models import swin_c_multibatch as models
from models.counting_checkpoint import validate_counting_checkpoint
from models.swin_dual_dora import (
    extract_dora_from_legacy_state,
    has_legacy_dora_lora_adapters,
)


def read_checkpoint(path):
    """Read trusted local training checkpoints, including optimizer/NumPy state."""
    try:
        return torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_checkpoint(path):
    return validate_counting_checkpoint(read_checkpoint(path))


def load_backbone_weights(model, path):
    """Explicitly reuse only a clean Swin backbone, including old MAN baselines."""
    checkpoint = read_checkpoint(path)
    state = checkpoint.get('model_state_dict', checkpoint)
    if any('.parametrizations.weight.' in key for key in state):
        raise ValueError('--backbone-checkpoint requires a clean baseline without adapters')
    # Load strictly, so a wrong Swin variant or incomplete backbone cannot be
    # silently accepted. The regression head remains freshly initialized.
    for prefix, module in [('backbone_features.', model.backbone_features),
                           ('backbone_norm.', model.backbone_norm)]:
        weights = {key[len(prefix):]: value for key, value in state.items()
                   if key.startswith(prefix)}
        if not weights:
            raise ValueError('Backbone checkpoint is missing {}'.format(prefix))
        module.load_state_dict(weights, strict=True)


def resolve_checkpoint(explicit_path, checkpoint_root, model_name, stage):
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError('Checkpoint not found: {}'.format(explicit_path))
        return explicit_path

    pattern = os.path.join(
        checkpoint_root, model_name, stage, '*', 'best_model*.pth')
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            'No default {} checkpoint matched {}; provide --{}-checkpoint'.format(
                stage, pattern, stage))
    return max(candidates, key=os.path.getmtime)


def build_model(args, checkpoint_path, expect_dora):
    checkpoint = load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        if checkpoint.get('model_name', args.model_name) != args.model_name:
            raise ValueError(
                '{} model name does not match --model-name'.format(checkpoint_path))
        expected_stage = 'dora' if expect_dora else 'baseline'
        if checkpoint.get('stage', expected_stage) != expected_stage:
            raise ValueError(
                '{} checkpoint stage does not match {}'.format(
                    checkpoint_path, expected_stage))
        if (expect_dora and checkpoint.get('adapter_type', 'dora') not in
                ('dora', 'parallel_dora_dora1', 'parallel_dora_lora')):
            raise ValueError('{} is not a DoRA checkpoint'.format(checkpoint_path))
        state = checkpoint['model_state_dict']
        rank = checkpoint.get('dora_rank', 0)
        alpha = checkpoint.get('dora_alpha', args.dora_alpha)
        dora1_rank = checkpoint.get('dora1_rank', args.dora1_rank)
        dora1_alpha = checkpoint.get('dora1_alpha', args.dora1_alpha)
    else:
        state = checkpoint
        rank = args.dora_rank if expect_dora else 0
        alpha = args.dora_alpha
        dora1_rank, dora1_alpha = args.dora1_rank, args.dora1_alpha

    dora_markers = (
        'dora_A_q', 'dora_B_q', 'dora_A_v', 'dora_B_v',
        'magnitude_q', 'magnitude_v',
    )
    contains_parametrization = any(
        '.parametrizations.weight.' in key for key in state)
    contains_dora = all(
        any(marker in key for key in state) for marker in dora_markers)
    contains_parallel = (
        any('.parametrizations.weight.0.dora.' in key for key in state)
        and any('.parametrizations.weight.0.dora1.' in key for key in state)
    )
    contains_legacy_parallel = has_legacy_dora_lora_adapters(state)
    if expect_dora:
        if not contains_dora:
            raise ValueError(
                '{} does not contain a complete DoRA adapter'.format(
                    checkpoint_path))
        if not rank:
            raise ValueError(
                'DoRA .pth checkpoint needs --dora-rank and --dora-alpha')
    elif contains_parametrization or rank:
        raise ValueError(
            '{} contains an adapter and is not a baseline checkpoint'.format(
                checkpoint_path))

    model = getattr(models, args.model_name)(pretrained=False)
    if expect_dora:
        if contains_parallel:
            model.enable_dual_dora(
                rank, alpha, dora1_rank, dora1_alpha, active_route='dora')
        else:
            model.enable_dora(rank, alpha)
            if contains_legacy_parallel:
                state = extract_dora_from_legacy_state(state)
    model.load_state_dict(state)
    if expect_dora and contains_parallel:
        model.set_adapter_route('dora')
    return model


def build_dual_dora_model(args, checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        if checkpoint.get('model_name', args.model_name) != args.model_name:
            raise ValueError(
                '{} model name does not match --model-name'.format(checkpoint_path))
        if checkpoint.get('stage') != 'dora1':
            raise ValueError('{} is not a DoRA1-stage checkpoint'.format(
                checkpoint_path))
        if checkpoint.get('adapter_type') != 'parallel_dora_dora1':
            raise ValueError('{} is not a DoRA+DoRA1 checkpoint'.format(
                checkpoint_path))
        state = checkpoint['model_state_dict']
        dora_rank = checkpoint.get('dora_rank')
        dora_alpha = checkpoint.get('dora_alpha')
        dora1_rank = checkpoint.get('dora1_rank')
        dora1_alpha = checkpoint.get('dora1_alpha')
    else:
        state = checkpoint
        dora_rank, dora_alpha = args.dora_rank, args.dora_alpha
        dora1_rank, dora1_alpha = args.dora1_rank, args.dora1_alpha

    dora_markers = (
        'dora_A_q', 'dora_B_q', 'dora_A_v', 'dora_B_v',
        'magnitude_q', 'magnitude_v',
    )
    has_dora = all(any('.dora.{}'.format(marker) in key for key in state)
                   for marker in dora_markers)
    has_dora1 = all(any('.dora1.{}'.format(marker) in key for key in state)
                    for marker in dora_markers)
    if not has_dora or not has_dora1:
        raise ValueError('{} does not contain complete DoRA+DoRA1 adapters'.format(
            checkpoint_path))
    if not all((dora_rank, dora_alpha, dora1_rank, dora1_alpha)):
        raise ValueError('DoRA and DoRA1 rank/alpha must be positive')

    model = getattr(models, args.model_name)(pretrained=False)
    model.enable_dual_dora(
        dora_rank, dora_alpha, dora1_rank, dora1_alpha,
        active_route=args.dual_route)
    model.load_state_dict(state)
    model.set_adapter_route(args.dual_route)
    return model


