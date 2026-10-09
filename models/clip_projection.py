"""A separately calibrated, training-only bridge from Swin to CLIP."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .swin_dual_dora import DualQVDoRAAdapter
from losses.clip_guidance import TEXT_ALIGNMENT_TARGET


class CLIPProjection(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, features):
        pooled = features.float().mean(dim=(-2, -1))
        return F.normalize(self.linear(pooled), dim=-1, eps=1e-6)


def backbone_signature(model):
    """Fingerprint raw Swin weights, ignoring both adapter branches."""
    digest = hashlib.sha256()
    entries = []
    for name, tensor in model.state_dict().items():
        if not name.startswith(('backbone_features.', 'backbone_norm.')):
            continue
        if '.parametrizations.weight.0.' in name:
            continue
        name = name.replace('.parametrizations.weight.original', '.weight')
        entries.append((name, tensor))
    for name, tensor in sorted(entries):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def branch_signature(model, route):
    """Fingerprint one adapter, independent of the active runtime route."""
    if route not in ('dora', 'dora1'):
        raise ValueError('Expected a DoRA branch')
    digest = hashlib.sha256()
    entries = [(name, tensor) for name, tensor in model.state_dict().items()
               if '.parametrizations.weight.0.{}.'.format(route) in name]
    if not entries:
        raise ValueError('No {} branch found'.format(route))
    for name, tensor in sorted(entries):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    for name, module in model.named_modules():
        if isinstance(module, DualQVDoRAAdapter):
            digest.update('{}:{}'.format(name, getattr(module, route).scaling).encode())
    return digest.hexdigest()


def restore_projection(checkpoint, model_name, clip_model, input_dim, embedding_dim,
                       signature, day_signature=None, night_signature=None,
                       text_prompts=None):
    expected = dict(model_name=model_name, clip_model=clip_model,
                    input_dim=input_dim, embedding_dim=embedding_dim,
                    backbone_signature=signature)
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError('Projection checkpoint {} mismatch'.format(key))
    if checkpoint.get('image_normalization') != 'imagenet':
        raise ValueError('Projection requires ImageNet-normalized model input')
    if day_signature is not None or text_prompts is not None:
        if checkpoint.get('alignment_target') != TEXT_ALIGNMENT_TARGET:
            raise ValueError('Projection uses old image-teacher alignment; recalibrate with day/night text centres')
        centres = checkpoint.get('text_centres')
        if not torch.is_tensor(centres) or centres.shape != (2, embedding_dim):
            raise ValueError('Projection checkpoint has invalid text centres')
    if text_prompts is not None:
        for name, expected_prompt in text_prompts.items():
            if checkpoint.get(name) != expected_prompt:
                raise ValueError('Projection checkpoint {} mismatch'.format(name))
    if day_signature is not None:
        if checkpoint.get('calibration_mode') != 'dual_dora':
            raise ValueError('Night-to-day guidance requires projection calibration on both DoRA branches')
        if checkpoint.get('day_branch_signature') != day_signature:
            raise ValueError('Projection checkpoint day branch mismatch')
    if night_signature is not None:
        if checkpoint.get('night_branch_signature') != night_signature:
            raise ValueError('Projection checkpoint initial night branch mismatch')
    projection = CLIPProjection(input_dim, embedding_dim)
    projection.load_state_dict(checkpoint['projection_state_dict'])
    projection.requires_grad_(False).eval()
    return projection, checkpoint
