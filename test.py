"""Compare baseline, DoRA, and optional dual-DoRA checkpoints."""

import argparse
import csv
import glob
import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.crowd import Crowd
from datasets.day_night import build_router_transform
from models import swin_c_multibatch as models
from models.day_night_router import load_day_night_router, prediction_to_route
from models.swin_dual_dora import (
    extract_dora_from_legacy_state,
    has_legacy_dora_lora_adapters,
)
from utils.stage3_bank import load_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline, DoRA, and dual-DoRA checkpoints')
    parser.add_argument('--workflow', choices=['legacy', 'memory'], default='memory',
                        help='memory enables the independent single-DoRA research phases')
    parser.add_argument('--experiment', choices=['A', 'B', 'C'], default='C',
                        help='memory workflow: baseline, single DoRA, or Stage 3 fusion')
    parser.add_argument(
        '--checkpoint',
        default='model/swin_t_trans/cross_attention/1026-160522/best_model.pth',
        help='memory workflow model checkpoint; includes the Phase 3 memory bank')
    parser.add_argument('--disable-cross-attention', action='store_true')
    parser.add_argument('--memory-bank-path', default='',
                        help='optional comparison bank .pt for experiment C; '
                             'leave empty to use the bank embedded in --checkpoint')
    parser.add_argument('--model-name', choices=['swin_l_trans', 'swin_t_trans'],
                        default='swin_t_trans')
    parser.add_argument('--baseline-checkpoint', default='model/swin_t_trans/baseline/1026-154626/best_model.pth',
                        help='baseline checkpoint; defaults to the latest best_model.pth')
    parser.add_argument('--dora-checkpoint', default='model/swin_t_trans/dora/1026-155746/best_model.pth',
                        help='DoRA checkpoint; defaults to the latest best_model.pth')
    parser.add_argument('--dora1-checkpoint', default='',
                        help='optional parallel DoRA/DoRA1 checkpoint')
    parser.add_argument('--dual-route', choices=['dora', 'dora1'], default='dora1',
                        help='active branch when evaluating the dual checkpoint')
    parser.add_argument(
        '--router-mode', choices=['fixed', 'filename', 'classifier'],
        default='classifier',
        help=('fixed evaluates all requested images with --dual-route; '
              'filename uses names; classifier uses image content'))
    parser.add_argument(
        '--router-checkpoint', default='',
        help='trained day/night router required by --router-mode classifier')
    parser.add_argument('--checkpoint-root', default='model',
                        help='root used to find default checkpoints')
    parser.add_argument('--data-root', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA',
                        help='root containing day/, night/, and mix/')
    parser.add_argument('--domains', nargs='+', choices=['day', 'night', 'mix'],
                        default=['day', 'night', 'mix'])
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--output-csv', default='test_results.csv')
    parser.add_argument(
        '--routing-csv', default='routing_results.csv',
        help='per-image results written when a router is used')
    parser.add_argument('--device', default='0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--downsample-ratio', type=int, default=16)
    parser.add_argument('--dora-rank', type=int, default=4,
                        help='required when the DoRA checkpoint is a .pth file')
    parser.add_argument('--dora-alpha', type=float, default=4.0,
                        help='must match training for a DoRA .pth checkpoint')
    parser.add_argument('--dora1-rank', type=int, default=4,
                        help='required when the dual checkpoint is a .pth file')
    parser.add_argument('--dora1-alpha', type=float, default=4.0,
                        help='must match training for a dual .pth checkpoint')
    return parser.parse_args()


def load_checkpoint(path):
    return load_file(path)


def resolve_checkpoint(explicit_path, checkpoint_root, model_name, stage):
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError('Checkpoint not found: {}'.format(explicit_path))
        return explicit_path

    pattern = os.path.join(
        checkpoint_root, model_name, stage, '*', 'best_model.pth')
    candidates = glob.glob(pattern)
    if not candidates:
        # Older --save-all runs only have numbered historical best weights.
        legacy_pattern = os.path.join(
            checkpoint_root, model_name, stage, '*', 'best_model_*.pth')
        candidates = [path for path in glob.glob(legacy_pattern)
                      if os.path.basename(path)[11:-4].isdigit()]
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


def route_from_filename(name):
    """Map the mixed-dataset filename convention to an adapter route."""
    basename = os.path.basename(name)
    if basename.lower().startswith('night_'):
        return 'night', 'dora'
    return 'day', 'dora1'


def evaluate(model, data_dir, args, device, description,
             router_mode='fixed', image_router=None, domain=None):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError('Dataset directory not found: {}'.format(data_dir))
    dataset = Crowd(data_dir, args.crop_size, args.downsample_ratio,
                    is_gray=getattr(model, 'research_config', {}).get('is_gray', False),
                    method='val')
    if not dataset.im_list:
        raise ValueError('No .jpg images in {}'.format(data_dir))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    errors = []
    routing_rows = []
    route_counts = {'dora1': 0, 'dora': 0}
    router_correct = 0
    router_transform = (build_router_transform(training=False)
                        if router_mode == 'classifier' else None)
    with torch.inference_mode():
        for inputs, count, names in tqdm(
                loader, desc=description, dynamic_ncols=True):
            image_name = names[0]
            detected_domain = ''
            route = ''
            confidence = ''
            expected_domain, _ = route_from_filename(image_name)
            inputs = inputs.to(device, non_blocking=device.type == 'cuda')
            if getattr(args, 'workflow', 'legacy') == 'memory':
                inference_domain = domain if domain in ('day', 'night') else expected_domain
                model.set_inference_domain(
                    inference_domain, not args.disable_cross_attention)
            if router_mode == 'filename':
                detected_domain, route = route_from_filename(image_name)
                confidence = 1.0
            elif router_mode == 'classifier':
                image_path = os.path.join(data_dir, image_name + '.jpg')
                with Image.open(image_path) as router_image:
                    router_input = router_transform(
                        router_image.convert('RGB')).unsqueeze(0).to(device)
                detected_domain, route, confidence = prediction_to_route(
                    image_router(router_input))
            if router_mode != 'fixed':
                model.set_adapter_route(route)
                route_counts[route] += 1
                router_correct += int(detected_domain == expected_domain)
            predicted = sum(
                model(patch)[0].sum().item()
                for patch in split_large_image(inputs)
            )
            error = count.item() - predicted
            errors.append(error)
            if router_mode != 'fixed':
                routing_rows.append({
                    'domain': domain or '',
                    'split': args.split,
                    'image': image_name,
                    'expected_domain': expected_domain,
                    'detected_domain': detected_domain,
                    'route': route,
                    'confidence': confidence,
                    'router_correct': detected_domain == expected_domain,
                    'ground_truth_count': int(count.item()),
                    'predicted_count': predicted,
                    'error': error,
                    'absolute_error': abs(error),
                })

    errors = np.asarray(errors, dtype=np.float64)
    return {
        'images': len(errors),
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(np.square(errors)))),
        'route_counts': route_counts,
        'router_accuracy': (router_correct / len(errors)
                            if router_mode != 'fixed' else None),
        'routing_rows': routing_rows,
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


def write_routing_results(rows, output_path):
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    fields = [
        'domain', 'split', 'image', 'expected_domain', 'detected_domain',
        'route', 'confidence', 'router_correct', 'ground_truth_count',
        'predicted_count', 'error', 'absolute_error',
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    device = torch.device('cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    rows = []
    routing_rows = []

    if args.workflow == 'memory':
        from utils.memory_workflow import load_research_model
        from utils.stage3_bank import validate_bank
        stage = {'A': 'baseline', 'B': 'dora', 'C': 'cross_attention'}[args.experiment]
        checkpoint_path = resolve_checkpoint(
            args.checkpoint, args.checkpoint_root, args.model_name, stage)
        model = load_research_model(load_checkpoint(checkpoint_path), args.model_name,
                                    args.dora_rank, args.dora_alpha)
        if model.research_config['stage'] != stage:
            raise ValueError('Checkpoint does not match experiment ' + args.experiment)
        if args.memory_bank_path:
            if stage != 'cross_attention':
                raise ValueError('Comparison banks require experiment C')
            bank = validate_bank(load_checkpoint(args.memory_bank_path), args.model_name,
                                 model.research_config['memory_size'])
            if bank['baseline_weights_sha256'] != model.research_config['baseline_weights_sha256']:
                raise ValueError('Comparison bank belongs to a different baseline')
            model.stage3_cross_attention.codebook.copy_(bank['codebook'])
        model.to(device).eval()
        for domain in args.domains:
            metrics = evaluate(
                model, os.path.join(args.data_root, domain, args.split), args,
                device, 'Experiment {} {}'.format(args.experiment, domain),
                domain=domain)
            rows.append({
                'model': 'experiment_' + args.experiment, 'domain': domain,
                'split': args.split, 'images': metrics['images'],
                'mae': metrics['mae'], 'rmse': metrics['rmse'],
                'checkpoint': checkpoint_path})
        print_results(rows)
        write_results(rows, args.output_csv)
        return

    if args.router_mode in ('filename', 'classifier'):
        if not args.dora1_checkpoint:
            raise ValueError(
                'routing requires --dora1-checkpoint containing '
                'both the DoRA1 and DoRA branches')
        if not os.path.isfile(args.dora1_checkpoint):
            raise FileNotFoundError(
                'Checkpoint not found: {}'.format(args.dora1_checkpoint))

        image_router = None
        if args.router_mode == 'classifier':
            if not args.router_checkpoint:
                raise ValueError(
                    '--router-mode classifier requires --router-checkpoint')
            image_router = load_day_night_router(args.router_checkpoint)
            image_router.to(device).eval()
            print('Image router checkpoint: {}'.format(
                args.router_checkpoint))

        print('Router mode: {} (day -> DoRA1, night -> DoRA)'.format(
            args.router_mode))
        print('Dual DoRA checkpoint: {}'.format(args.dora1_checkpoint))
        model = build_dual_dora_model(args, args.dora1_checkpoint)
        model.to(device).eval()
        for domain in args.domains:
            data_dir = os.path.join(args.data_root, domain, args.split)
            metrics = evaluate(
                model, data_dir, args, device,
                '{} router {}'.format(args.router_mode.capitalize(), domain),
                router_mode=args.router_mode, image_router=image_router,
                domain=domain)
            route_counts = metrics.pop('route_counts')
            router_accuracy = metrics.pop('router_accuracy')
            routing_rows.extend(metrics.pop('routing_rows'))
            print('{}: day/DoRA1={}, night/DoRA={}, router accuracy={:.2%}'.format(
                domain, route_counts['dora1'], route_counts['dora'],
                router_accuracy))
            rows.append({
                'model': args.router_mode + '_router',
                'domain': domain,
                'split': args.split,
                'images': metrics['images'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'checkpoint': args.dora1_checkpoint,
            })

        print_results(rows)
        write_results(rows, args.output_csv)
        write_routing_results(routing_rows, args.routing_csv)
        print('\nCSV saved to {}'.format(os.path.abspath(args.output_csv)))
        print('Routing CSV saved to {}'.format(
            os.path.abspath(args.routing_csv)))
        return

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
    if args.dora1_checkpoint:
        if not os.path.isfile(args.dora1_checkpoint):
            raise FileNotFoundError(
                'Checkpoint not found: {}'.format(args.dora1_checkpoint))
        print('DoRA1 checkpoint: {}'.format(args.dora1_checkpoint))
        checkpoints.append(('dual_' + args.dual_route,
                            args.dora1_checkpoint, None))

    for model_label, checkpoint_path, expect_dora in checkpoints:
        if model_label.startswith('dual_'):
            model = build_dual_dora_model(args, checkpoint_path)
        else:
            model = build_model(args, checkpoint_path, expect_dora)
        model.to(device).eval()
        for domain in args.domains:
            data_dir = os.path.join(args.data_root, domain, args.split)
            metrics = evaluate(
                model, data_dir, args, device,
                '{} {}'.format(model_label.capitalize(), domain))
            metrics.pop('route_counts')
            metrics.pop('router_accuracy')
            metrics.pop('routing_rows')
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
