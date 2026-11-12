"""Checkpoint loading and inference builders for Swin + CCTrans Regression."""

import glob
import os

import torch
from utils.topk_checkpoint import best_ranked_path

from models import swin_c_multibatch as models
from models.counting_checkpoint import (validate_counting_checkpoint, checkpoint_mdc_mode,
                                        validate_model_mdc)
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
    if os.path.isdir(path):
        path = best_ranked_path(path)
    return validate_counting_checkpoint(read_checkpoint(path))


def load_initial_weights(model, args):
    """Load a same-stage model for a new objective, without training state."""
    checkpoint = load_checkpoint(args.init_checkpoint)
    validate_model_mdc(model, checkpoint)
    state = checkpoint.get('model_state_dict', checkpoint)
    if 'model_state_dict' in checkpoint:
        if checkpoint.get('model_name') != args.model_name or checkpoint.get('stage') != args.stage:
            raise ValueError('--init-checkpoint must match the selected model and stage')
        if args.stage in ('dora', 'dora1'):
            for name in ('dora_rank', 'dora_alpha', 'dora1_rank', 'dora1_alpha'):
                if checkpoint.get(name) != getattr(args, name):
                    raise ValueError('--init-checkpoint adapter settings do not match')
    if args.stage in ('dora', 'dora1'):
        if not any('.parametrizations.weight.0.dora1.' in key for key in state):
            raise ValueError('Adapter --init-checkpoint requires complete dual DoRA weights; '
                             'use the parent checkpoint option to start a new adapter')
        model.enable_dual_dora(
            args.dora_rank, args.dora_alpha, args.dora1_rank, args.dora1_alpha,
            active_route=args.stage, trainable_route=args.stage)
    model.load_state_dict(state)


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
        if os.path.isdir(explicit_path):
            return best_ranked_path(explicit_path)
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError('Checkpoint not found: {}'.format(explicit_path))
        return explicit_path

    root = os.path.join(checkpoint_root, model_name, stage)
    # Timestamp run names sort chronologically. Select rank 1 within the most
    # recent run by its manifest, never by weight-file modification time/rounded MAE.
    for run in sorted(glob.glob(os.path.join(root, '*')), reverse=True):
        if not os.path.isdir(run):
            continue
        if os.path.isfile(os.path.join(run, 'rankings.json')):
            return best_ranked_path(run)
        legacy = glob.glob(os.path.join(run, 'best_model*.pth'))
        if len(legacy) == 1:
            return legacy[0]
        if len(legacy) > 1:
            raise ValueError('Multiple legacy weights without ranking metadata in {}; '
                             'specify an exact checkpoint'.format(run))
    raise FileNotFoundError('No ranked {} weights under {}; provide --{}-checkpoint'.format(stage, root, stage))


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

    model = getattr(models, args.model_name)(pretrained=False, mdc_mode=checkpoint_mdc_mode(checkpoint))
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
        if checkpoint.get('stage') not in ('dora1', 'finetune'):
            raise ValueError('{} is not a DoRA1/finetune-stage checkpoint'.format(
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

    model = getattr(models, args.model_name)(pretrained=False, mdc_mode=checkpoint_mdc_mode(checkpoint))
    model.enable_dual_dora(
        dora_rank, dora_alpha, dora1_rank, dora1_alpha,
        active_route=args.dual_route)
    model.load_state_dict(state)
    model.set_adapter_route(args.dual_route)
    return model


def load_finetune_weights(model, args, path, expected_stage='dora1'):
    """Attach both frozen adapters and load a complete DoRA1 (or resume) model."""
    checkpoint = load_checkpoint(path)
    validate_model_mdc(model, checkpoint)
    state = checkpoint.get('model_state_dict', checkpoint)
    if 'model_state_dict' in checkpoint:
        if checkpoint.get('stage') != expected_stage or checkpoint.get('model_name') != args.model_name:
            raise ValueError('Fine-tune checkpoint model/stage does not match')
        for name in ('dora_rank', 'dora_alpha', 'dora1_rank', 'dora1_alpha'):
            if checkpoint.get(name) != getattr(args, name):
                raise ValueError('Fine-tune checkpoint adapter settings do not match')
    markers = ('dora_A_q', 'dora_B_q', 'dora_A_v', 'dora_B_v', 'magnitude_q', 'magnitude_v')
    if not all(any('.{}.{}'.format(branch, marker) in key for key in state)
               for branch in ('dora', 'dora1') for marker in markers):
        raise ValueError('Fine-tuning requires complete DoRA + DoRA1 weights')
    model.enable_dual_dora(args.dora_rank, args.dora_alpha,
                           args.dora1_rank, args.dora1_alpha, active_route='dora1')
    model.load_state_dict(state)
    return checkpoint


