"""Lightweight image-content router for day/night classification."""

import os

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ROUTER_ARCHITECTURE = 'mobilenet_v3_small'
CLASS_NAMES = ('day', 'night')


def build_day_night_router(pretrained=True, freeze_features=True):
    """Build a two-class MobileNetV3-Small router."""
    weights = (MobileNet_V3_Small_Weights.IMAGENET1K_V1
               if pretrained else None)
    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, len(CLASS_NAMES))

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad_(False)
    return model


def load_day_night_router(path):
    """Load a trained router checkpoint on CPU."""
    if not os.path.isfile(path):
        raise FileNotFoundError('Router checkpoint not found: {}'.format(path))
    try:
        checkpoint = torch.load(path, map_location='cpu', mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')

    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError('{} is not a router checkpoint'.format(path))
    if checkpoint.get('architecture') != ROUTER_ARCHITECTURE:
        raise ValueError(
            '{} has unsupported router architecture: {}'.format(
                path, checkpoint.get('architecture')))
    if tuple(checkpoint.get('class_names', ())) != CLASS_NAMES:
        raise ValueError('{} has incompatible router classes'.format(path))

    model = build_day_night_router(pretrained=False, freeze_features=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def prediction_to_route(logits):
    """Convert one router prediction into domain, adapter route, confidence."""
    if logits.ndim != 2 or logits.shape != (1, len(CLASS_NAMES)):
        raise ValueError('Router logits must have shape [1, 2]')
    probabilities = torch.softmax(logits, dim=1)
    class_index = int(probabilities.argmax(dim=1).item())
    domain = CLASS_NAMES[class_index]
    route = 'dora' if domain == 'night' else 'dora1'
    confidence = float(probabilities[0, class_index].item())
    return domain, route, confidence
