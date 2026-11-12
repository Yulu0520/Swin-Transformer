"""Evaluate baseline, dual DoRA and head-finetuned crowd-counting models."""

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.crowd import Crowd, RoutedCrowd
from models.day_night_router import load_day_night_router
from utils.counting_model import build_model, build_dual_dora_model, resolve_checkpoint
from utils.routing import route_from_filename, split_large_image, select_routes, count_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Swin + CCTrans counting checkpoints')
    parser.add_argument('--model-name', choices=['swin_l_trans', 'swin_t_trans'], default='swin_t_trans')
    parser.add_argument('--baseline-checkpoint', default='', help='weight file or ranked run directory')
    parser.add_argument('--dora-checkpoint', default='', help='weight file or ranked run directory')
    parser.add_argument('--dora1-checkpoint', default='', help='dual DoRA weight file or ranked run directory')
    parser.add_argument('--finetune-checkpoint', default='', help='head-finetuned weight file or ranked run directory')
    parser.add_argument('--checkpoint-stage', choices=['dora1', 'finetune'], default='dora1',
                        help='stage to resolve automatically when routed checkpoint paths are omitted')
    parser.add_argument('--dual-route', choices=['dora', 'dora1'], default='dora1')
    parser.add_argument('--router-mode', choices=['fixed', 'filename', 'classifier'], default='classifier')
    parser.add_argument('--router-checkpoint', default='model/day_night_router/1001-144417/best_model.pth')
    parser.add_argument('--checkpoint-root', default='model')
    parser.add_argument('--data-root', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA')
    parser.add_argument('--domains', nargs='+', choices=['day', 'night', 'mix'], default=['day', 'night', 'mix'])
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--output-csv', default='test_results.csv')
    parser.add_argument('--routing-csv', default='routing_results.csv')
    parser.add_argument('--device', default='0', help='GPU index or cpu')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--downsample-ratio', type=int, choices=[8], default=8)
    parser.add_argument('--dora-rank', type=int, default=4)
    parser.add_argument('--dora-alpha', type=float, default=4.)
    parser.add_argument('--dora1-rank', type=int, default=4)
    parser.add_argument('--dora1-alpha', type=float, default=4.)
    return parser.parse_args()


def evaluate(model, data_dir, args, device, description,
             router_mode='fixed', image_router=None, domain=None, model_label=''):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError('Dataset directory not found: {}'.format(data_dir))
    routed = router_mode != 'fixed'
    dataset_class = RoutedCrowd if routed else Crowd
    options = dict(router_mode=router_mode) if routed else {}
    dataset = dataset_class(data_dir, args.crop_size, args.downsample_ratio,
                            is_gray=False, method='val', **options)
    if not dataset.im_list:
        raise ValueError('No .jpg images in {}'.format(data_dir))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    errors, routing_rows = [], []
    groups = {'day': [], 'night': []}
    route_counts, router_correct = {'dora1': 0, 'dora': 0}, 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, dynamic_ncols=True):
            inputs, count, names = batch[:3]
            image_name = names[0]
            expected_domain = domain if domain in ('day', 'night') else route_from_filename(image_name)[0]
            route, detected_domain, confidence = '', '', ''
            if routed:
                domains, routes, confidences = select_routes(router_mode, names, batch[3], image_router, device)
                detected_domain, route, confidence = domains[0], routes[0], confidences[0]
                model.set_adapter_route(route)
                route_counts[route] += 1
                router_correct += int(detected_domain == expected_domain)
            inputs = inputs.to(device, non_blocking=device.type == 'cuda')
            predicted = sum(model(patch).sum().item() for patch in split_large_image(inputs))
            error = count.item() - predicted
            errors.append(error)
            groups[expected_domain].append(error)
            if routed:
                routing_rows.append(dict(model=model_label, domain=domain or '', split=args.split,
                    image=image_name, expected_domain=expected_domain, detected_domain=detected_domain,
                    route=route, confidence=confidence, router_correct=detected_domain == expected_domain,
                    ground_truth_count=int(count.item()), predicted_count=predicted,
                    error=error, absolute_error=abs(error)))
    return dict(**count_metrics(errors), route_counts=route_counts,
                router_accuracy=router_correct / len(errors) if routed else None,
                group_metrics={name: count_metrics(values) for name, values in groups.items()},
                routing_rows=routing_rows)


def print_results(rows):
    print('\n{:<24} {:<8} {:<8} {:>8} {:>12} {:>12}'.format(
        'Model', 'Domain', 'Subset', 'Images', 'MAE', 'RMSE'))
    for row in rows:
        print('{model:<24} {domain:<8} {subset:<8} {images:>8d} {mae:>12.3f} {rmse:>12.3f}'.format(**row))


def write_results(rows, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fields = ['model', 'domain', 'subset', 'split', 'images', 'mae', 'rmse', 'checkpoint']
    with open(output_path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_routing_results(rows, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fields = ['model', 'domain', 'split', 'image', 'expected_domain', 'detected_domain',
              'route', 'confidence', 'router_correct', 'ground_truth_count',
              'predicted_count', 'error', 'absolute_error']
    with open(output_path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_jobs(args):
    """Return (label, resolved path, kind), including before/after comparisons."""
    jobs = []
    for stage in ('dora1', 'finetune'):
        path = getattr(args, stage + '_checkpoint')
        if path:
            jobs.append((stage, resolve_checkpoint(path, args.checkpoint_root, args.model_name, stage), 'dual'))
    if args.router_mode != 'fixed':
        if not jobs:
            stage = args.checkpoint_stage
            jobs.append((stage, resolve_checkpoint('', args.checkpoint_root, args.model_name, stage), 'dual'))
        return jobs
    # A fine-tune-only evaluation does not require unrelated baseline/DoRA files.
    only_requested = bool(args.finetune_checkpoint or args.checkpoint_stage == 'finetune')
    for stage, kind in [('baseline', 'baseline'), ('dora', 'dora')]:
        path = getattr(args, stage + '_checkpoint')
        if path or not only_requested:
            jobs.insert(0 if stage == 'baseline' else min(1, len(jobs)),
                (stage, resolve_checkpoint(path, args.checkpoint_root, args.model_name, stage), kind))
    if args.checkpoint_stage == 'finetune' and not args.finetune_checkpoint:
        jobs.append(('finetune', resolve_checkpoint('', args.checkpoint_root, args.model_name, 'finetune'), 'dual'))
    return jobs


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else args.device.strip()
    device = torch.device('cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    rows, routing_rows = [], []
    image_router = None
    if args.router_mode == 'classifier':
        image_router = load_day_night_router(args.router_checkpoint).to(device).eval()
        image_router.requires_grad_(False)
    for label, checkpoint, kind in checkpoint_jobs(args):
        model = (build_dual_dora_model(args, checkpoint) if kind == 'dual'
                 else build_model(args, checkpoint, expect_dora=kind == 'dora'))
        model.to(device).eval()
        mode = args.router_mode if kind == 'dual' else 'fixed'
        model_label = label + ('_' + mode if mode != 'fixed' else '_' + args.dual_route if kind == 'dual' else '')
        print('{} checkpoint: {}'.format(model_label, checkpoint))
        for domain in args.domains:
            metrics = evaluate(model, os.path.join(args.data_root, domain, args.split), args,
                device, '{} {}'.format(model_label, domain), router_mode=mode,
                image_router=image_router, domain=domain, model_label=model_label)
            routing_rows.extend(metrics['routing_rows'])
            base = dict(model=model_label, domain=domain, split=args.split, checkpoint=checkpoint)
            rows.append(dict(base, subset='overall', **{key: metrics[key] for key in ('images', 'mae', 'rmse')}))
            if domain == 'mix':
                for subset, values in metrics['group_metrics'].items():
                    if values['images']:
                        rows.append(dict(base, subset=subset, **values))
            if metrics['router_accuracy'] is not None:
                print('{} routes={} router accuracy={:.2%}'.format(domain,
                    metrics['route_counts'], metrics['router_accuracy']))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print_results(rows)
    write_results(rows, args.output_csv)
    write_routing_results(routing_rows, args.routing_csv)
    print('Saved {} and {}'.format(args.output_csv, args.routing_csv))


if __name__ == '__main__':
    main()
