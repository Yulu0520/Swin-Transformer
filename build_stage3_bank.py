"""Build a fixed daytime Stage 3 codebook from the best frozen baseline."""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.crowd import Crowd
from utils.memory_workflow import load_research_model
from utils.stage3_bank import (TokenReservoir, cluster_tokens, file_sha256,
                               load_file, validate_bank)


def build_bank(args):
    if (args.memory_size <= 0 or args.max_tokens < args.memory_size
            or args.tokens_per_image <= 0 or args.batch_size <= 0):
        raise ValueError('Require positive K, tokens/image, batch size and max_tokens >= K')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = load_file(args.baseline_checkpoint)
    model = load_research_model(checkpoint, args.model_name)
    if model.research_config['stage'] != 'baseline':
        raise ValueError('Build the bank from the best daytime baseline only')
    config = model.research_config
    crop_size = config.get('crop_size', 384) if args.crop_size is None else args.crop_size
    is_gray = config.get('is_gray', False) if args.is_gray is None else args.is_gray
    if ('crop_size' in config and crop_size != config['crop_size']
            or 'is_gray' in config and is_gray != config['is_gray']):
        raise ValueError('Bank preprocessing must match baseline training')
    if crop_size <= 0 or crop_size % 16:
        raise ValueError('Crop size must be positive and divisible by 16')
    # Always select the training split, never val/test. Reuse baseline training
    # normalization, resizing, cropping, grayscale augmentation, and flipping.
    train_dir = os.path.join(args.data_dir, 'train')
    dataset = Crowd(train_dir, crop_size, 16, is_gray, method='train')
    if not len(dataset):
        raise ValueError('No daytime training images in ' + train_dir)
    # Single worker and sorted dataset make the augmentation order reproducible.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model.to(device).eval().requires_grad_(False)
    reservoir = TokenReservoir(args.max_tokens, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    with torch.inference_mode():
        for images, _, _, _ in tqdm(loader, desc='Daytime Stage 3 tokens'):
            features = model.extract_stage3(images.to(device))
            tokens = features.reshape(-1, model.stage3_channels).cpu()
            indices = torch.randperm(len(tokens), generator=generator)[
                :args.tokens_per_image]
            reservoir.add(tokens[indices].numpy())
    codebook = cluster_tokens(reservoir.tokens, args.memory_size, args.seed,
                               args.batch_size)
    bank = {
        'format_version': 1, 'codebook': codebook,
        'model_name': config['model_name'], 'stage': 3,
        'feature_dim': model.stage3_channels, 'memory_size': args.memory_size,
        'baseline_weights_sha256': config['baseline_weights_sha256'],
        'baseline_checkpoint': os.path.abspath(args.baseline_checkpoint),
        'baseline_checkpoint_sha256': file_sha256(args.baseline_checkpoint),
        'build_parameters': {
            'algorithm': 'MiniBatchKMeans', 'random_state': args.seed,
            'batch_size': args.batch_size, 'n_init': 3,
            'max_tokens': args.max_tokens, 'tokens_per_image': args.tokens_per_image,
            'sampled_tokens': len(reservoir.tokens), 'crop_size': crop_size,
            'is_gray': is_gray, 'preprocessing': 'Crowd.train_transform',
            'train_directory': os.path.abspath(train_dir),
        },
    }
    validate_bank(bank, config['model_name'])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(bank, args.output)
    print('Saved Stage 3 bank {}: {}'.format(tuple(codebook.shape), args.output))
    return bank


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-checkpoint', default='model/swin_t_trans/baseline/1026-154626/best_model.pth')
    parser.add_argument('--data-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/day', help='daytime root containing train/')
    parser.add_argument('--model-name', choices=['swin_t_trans', 'swin_l_trans'],
                        default='swin_t_trans', help='required for legacy raw state dictionaries')
    parser.add_argument('--output', default='stage3_memory_bank.pt')
    parser.add_argument('--memory-size', type=int, default=256)
    parser.add_argument('--max-tokens', type=int, default=65536)
    parser.add_argument('--tokens-per-image', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='MiniBatchKMeans batch size, not image batch size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--is-gray', action='store_true', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == '__main__':
    build_bank(parse_args())
