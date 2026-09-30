"""Compare baseline, DoRA, and optional parallel-adapter checkpoints."""

import argparse
import csv
import glob
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.crowd import Crowd
from models import swin_c_multibatch as models


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline, DoRA, and hybrid checkpoints')
    parser.add_argument('--model-name', choices=['swin_l_trans', 'swin_t_trans'],
                        default='swin_t_trans')
    parser.add_argument('--baseline-checkpoint', default='model/swin_t_trans/baseline/0922-200006/best_model.pth',
                        help='baseline checkpoint; defaults to the latest best_model*.pth')
    parser.add_argument('--dora-checkpoint', default='model/swin_t_trans/dora/0923-103544/best_model.pth',
                        help='DoRA checkpoint; defaults to the latest best_model*.pth')
    parser.add_argument('--lora-checkpoint', default='',
                        help='optional parallel DoRA/LoRA checkpoint')
    parser.add_argument('--hybrid-route', choices=['dora', 'lora'], default='lora',
                        help='active branch when evaluating the hybrid checkpoint')
    parser.add_argument('--checkpoint-root', default='model',
                        help='root used to find default checkpoints')
    parser.add_argument('--data-root', default='SHA',
                        help='root containing clean/, hazy/, and mix/')
    parser.add_argument('--domains', nargs='+', choices=['clean', 'hazy', 'mix'],
                        default=['clean', 'hazy', 'mix'])
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--output-csv', default='test_results.csv')
    parser.add_argument('--device', default='0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--downsample-ratio', type=int, default=16)
    parser.add_argument('--dora-rank', type=int, default=4,
                        help='required when the DoRA checkpoint is a .pth file')
    parser.add_argument('--dora-alpha', type=float, default=4.0,
                        help='must match training for a DoRA .pth checkpoint')
    parser.add_argument('--lora-rank', type=int, default=4,
                        help='required when the hybrid checkpoint is a .pth file')
    parser.add_argument('--lora-alpha', type=float, default=4.0,
                        help='must match training for a hybrid .pth checkpoint')
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', mmap=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


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
                ('dora', 'parallel_dora_lora')):
            raise ValueError('{} is not a DoRA checkpoint'.format(checkpoint_path))
        state = checkpoint['model_state_dict']
        rank = checkpoint.get('dora_rank', 0)
        alpha = checkpoint.get('dora_alpha', args.dora_alpha)
        lora_rank = checkpoint.get('lora_rank', args.lora_rank)
        lora_alpha = checkpoint.get('lora_alpha', args.lora_alpha)
    else:
        state = checkpoint
        rank = args.dora_rank if expect_dora else 0
        alpha = args.dora_alpha
        lora_rank, lora_alpha = args.lora_rank, args.lora_alpha

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
        and any('.parametrizations.weight.0.lora.' in key for key in state)
    )
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
            model.enable_parallel_adapters(
                rank, alpha, lora_rank, lora_alpha, active_route='dora')
        else:
            model.enable_dora(rank, alpha)
    model.load_state_dict(state)
    if expect_dora and contains_parallel:
        model.set_adapter_route('dora')
    return model


def build_hybrid_model(args, checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        if checkpoint.get('model_name', args.model_name) != args.model_name:
            raise ValueError(
                '{} model name does not match --model-name'.format(checkpoint_path))
        if checkpoint.get('stage') != 'lora':
            raise ValueError('{} is not a LoRA-stage checkpoint'.format(
                checkpoint_path))
        if checkpoint.get('adapter_type') != 'parallel_dora_lora':
            raise ValueError('{} is not a DoRA+LoRA checkpoint'.format(
                checkpoint_path))
        state = checkpoint['model_state_dict']
        dora_rank = checkpoint.get('dora_rank')
        dora_alpha = checkpoint.get('dora_alpha')
        lora_rank = checkpoint.get('lora_rank')
        lora_alpha = checkpoint.get('lora_alpha')
    else:
        state = checkpoint
        dora_rank, dora_alpha = args.dora_rank, args.dora_alpha
        lora_rank, lora_alpha = args.lora_rank, args.lora_alpha

    required_markers = (
        'dora_A_q', 'dora_B_q', 'dora_A_v', 'dora_B_v',
        'magnitude_q', 'magnitude_v',
        'lora_A_q', 'lora_B_q', 'lora_A_v', 'lora_B_v',
    )
    if not all(any(marker in key for key in state)
               for marker in required_markers):
        raise ValueError('{} does not contain complete DoRA+LoRA adapters'.format(
            checkpoint_path))
    if not all((dora_rank, dora_alpha, lora_rank, lora_alpha)):
        raise ValueError('DoRA and LoRA rank/alpha must be positive')

    model = getattr(models, args.model_name)(pretrained=False)
    model.enable_parallel_adapters(
        dora_rank, dora_alpha, lora_rank, lora_alpha,
        active_route=args.hybrid_route)
    model.load_state_dict(state)
    model.set_adapter_route(args.hybrid_route)
    return model


def split_large_image(inputs, maximum_size=3584):
    _, _, height, width = inputs.shape
    if height < maximum_size and width < maximum_size:
        return [inputs]

    h_splits = math.ceil(height / maximum_size)
    w_splits = math.ceil(width / maximum_size)
    h_step = height // h_splits
    w_step = width // w_splits
    return [
        inputs[
            :, :,
            i * h_step:(i + 1) * h_step if i + 1 < h_splits else height,
            j * w_step:(j + 1) * w_step if j + 1 < w_splits else width,
        ]
        for i in range(h_splits)
        for j in range(w_splits)
    ]


def evaluate(model, data_dir, args, device, description):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError('Dataset directory not found: {}'.format(data_dir))
    dataset = Crowd(data_dir, args.crop_size, args.downsample_ratio,
                    is_gray=False, method='val')
    if not dataset.im_list:
        raise ValueError('No .jpg images in {}'.format(data_dir))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    errors = []
    with torch.inference_mode():
        for inputs, count, _ in tqdm(
                loader, desc=description, dynamic_ncols=True):
            inputs = inputs.to(device, non_blocking=device.type == 'cuda')
            predicted = sum(
                model(patch)[0].sum().item()
                for patch in split_large_image(inputs)
            )
            errors.append(count.item() - predicted)

    errors = np.asarray(errors, dtype=np.float64)
    return {
        'images': len(errors),
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(np.square(errors)))),
    }


def print_results(rows):
    print('\n{:<10} {:<8} {:>8} {:>12} {:>12}'.format(
        'Model', 'Domain', 'Images', 'MAE', 'RMSE'))
    print('-' * 54)
    for row in rows:
        print('{model:<10} {domain:<8} {images:>8d} '
              '{mae:>12.3f} {rmse:>12.3f}'.format(**row))


def write_results(rows, output_path):
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    fields = ['model', 'domain', 'split', 'images', 'mae', 'rmse', 'checkpoint']
    with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []

    baseline_checkpoint = resolve_checkpoint(
        args.baseline_checkpoint, args.checkpoint_root,
        args.model_name, 'baseline')
    dora_checkpoint = resolve_checkpoint(
        args.dora_checkpoint, args.checkpoint_root,
        args.model_name, 'dora')
    print('Baseline checkpoint: {}'.format(baseline_checkpoint))
    print('DoRA checkpoint: {}'.format(dora_checkpoint))

    checkpoints = [
        ('baseline', baseline_checkpoint, False),
        ('dora', dora_checkpoint, True),
    ]
    if args.lora_checkpoint:
        if not os.path.isfile(args.lora_checkpoint):
            raise FileNotFoundError(
                'Checkpoint not found: {}'.format(args.lora_checkpoint))
        print('LoRA checkpoint: {}'.format(args.lora_checkpoint))
        checkpoints.append(('hybrid_' + args.hybrid_route,
                            args.lora_checkpoint, None))

    for model_label, checkpoint_path, expect_dora in checkpoints:
        if model_label.startswith('hybrid_'):
            model = build_hybrid_model(args, checkpoint_path)
        else:
            model = build_model(args, checkpoint_path, expect_dora)
        model.to(device).eval()
        for domain in args.domains:
            data_dir = os.path.join(args.data_root, domain, args.split)
            metrics = evaluate(
                model, data_dir, args, device,
                '{} {}'.format(model_label.capitalize(), domain))
            rows.append({
                'model': model_label,
                'domain': domain,
                'split': args.split,
                'images': metrics['images'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'checkpoint': checkpoint_path,
            })
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print_results(rows)
    write_results(rows, args.output_csv)
    print('\nCSV saved to {}'.format(os.path.abspath(args.output_csv)))


if __name__ == '__main__':
    main()
