"""Load checkpoint weights and optional metadata for best-model .pth files."""

import json
import os

import torch


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location='cpu', mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    metadata_path = str(path) + '.json'
    if (isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint
            and os.path.isfile(metadata_path)):
        with open(metadata_path, encoding='utf-8') as source:
            metadata = json.load(source)
        return dict(metadata, model_state_dict=checkpoint)
    return checkpoint
