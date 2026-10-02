"""Train the standalone image-content day/night router."""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.day_night import DayNightDataset
from models.day_night_router import (
    CLASS_NAMES,
    ROUTER_ARCHITECTURE,
    build_day_night_router,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a MobileNetV3 day/night router')
    parser.add_argument(
        '--data-root',
        default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/mix',
        help='directory containing train/ and val/')
    parser.add_argument('--save-dir', default='model/day_night_router')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument(
        '--unfreeze-backbone', action='store_true',
        help='fine-tune all MobileNet parameters instead of only classifier')
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error('epochs, batch size, and learning rate must be positive')
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def confusion_matrix(predictions, labels):
    matrix = torch.zeros(2, 2, dtype=torch.long)
    for truth, prediction in zip(labels.view(-1), predictions.view(-1)):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    matrix = torch.zeros(2, 2, dtype=torch.long)

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels, _ in tqdm(
                loader, desc='train' if training else 'val',
                leave=False, dynamic_ncols=True):
            images = images.to(device, non_blocking=device.type == 'cuda')
            labels = labels.to(device, non_blocking=device.type == 'cuda')
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            matrix += confusion_matrix(
                logits.argmax(dim=1).cpu(), labels.cpu())

    total = int(matrix.sum().item())
    correct = int(matrix.diag().sum().item())
    per_class = matrix.diag().float() / matrix.sum(dim=1).clamp_min(1)
    return {
        'loss': loss_sum / total,
        'accuracy': correct / total,
        'day_accuracy': float(per_class[0].item()),
        'night_accuracy': float(per_class[1].item()),
        'confusion_matrix': matrix.tolist(),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')

    train_dataset = DayNightDataset(
        os.path.join(args.data_root, 'train'), training=True)
    val_dataset = DayNightDataset(
        os.path.join(args.data_root, 'val'), training=False)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    model = build_day_night_router(
        pretrained=not args.no_pretrained,
        freeze_features=not args.unfreeze_backbone)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    run_dir = os.path.join(
        args.save_dir, datetime.now().strftime('%m%d-%H%M%S'))
    os.makedirs(run_dir, exist_ok=False)
    best_path = os.path.join(run_dir, 'best_model.pth')
    best_accuracy = -1.0

    print('Device: {}'.format(device))
    print('Train images: {}; val images: {}'.format(
        len(train_dataset), len(val_dataset)))
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(
            model, val_loader, criterion, device)
        print(
            'Epoch {}/{} | train loss {:.4f} acc {:.2%} | '
            'val loss {:.4f} acc {:.2%} day {:.2%} night {:.2%}'.format(
                epoch + 1, args.epochs,
                train_metrics['loss'], train_metrics['accuracy'],
                val_metrics['loss'], val_metrics['accuracy'],
                val_metrics['day_accuracy'], val_metrics['night_accuracy']))

        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            torch.save({
                'architecture': ROUTER_ARCHITECTURE,
                'class_names': CLASS_NAMES,
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_metrics': val_metrics,
                'pretrained': not args.no_pretrained,
                'features_frozen': not args.unfreeze_backbone,
            }, best_path)
            print('Saved best router: {}'.format(best_path))

    print('Best validation accuracy: {:.2%}'.format(best_accuracy))
    print('Best router checkpoint: {}'.format(os.path.abspath(best_path)))


if __name__ == '__main__':
    main()
