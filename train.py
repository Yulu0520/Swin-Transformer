"""Train a baseline and mutually exclusive DoRA/DoRA1 adapter branches."""

import argparse
import copy
import gc
import math
import os
import random

import numpy as np
import torch

from utils.regression_trainer_cosine_multibatch import RegTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Swin + MAN')
    parser.add_argument('--model-name', choices=['swin_l_trans', 'swin_t_trans'],
                        default='swin_t_trans',
                        help='Swin-L has 24 attention blocks; Swin-T has 12')
    parser.add_argument(
        '--stage', choices=['all', 'both', 'baseline', 'dora', 'dora1'],
        default='dora1',
        help='all: baseline->DoRA->DoRA1; both: baseline->DoRA')
    
    parser.add_argument('--data-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/day',
                        help='clean baseline root containing train/ and val/')
    
    parser.add_argument('--dora-train-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/night/train')
    parser.add_argument('--dora-val-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/night/val')

    parser.add_argument('--dora1-train-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/night/train',
                        help='DoRA1-stage training directory')
    parser.add_argument('--dora1-val-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/day/val',
                        help='DoRA1-stage validation directory')
    
    parser.add_argument('--baseline-checkpoint', default='',
                        help='trained clean baseline .pth or .tar for DoRA-only training')
    parser.add_argument('--dora-checkpoint', default='model/swin_t_trans/dora/1013-133009/best_model.pth',
                        help='trained DoRA .pth or .tar for DoRA1-only training')
    parser.add_argument('--pretrained-path',
                        default='pre_models/swin_large_patch4_window12_384_22k.pth',
                        help='official ImageNet-22K Swin-L checkpoint; also works with 256 crops')
    parser.add_argument('--no-pretrained-backbone', dest='pretrained_backbone',
                        action='store_false')
    
    parser.set_defaults(pretrained_backbone=True)
    parser.add_argument('--dora-rank', type=int, default=4)
    parser.add_argument('--dora-alpha', type=float, default=4.0)
    parser.add_argument('--dora1-rank', type=int, default=4)
    parser.add_argument('--dora1-alpha', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', default='model')
    parser.add_argument('--save-all', action='store_true')
    parser.add_argument('--lr', type=float, default=5*1e-6)
    parser.add_argument('--dora-lr', type=float, default=2e-4)
    parser.add_argument('--dora1-lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='baseline weight decay')
    parser.add_argument('--dora-weight-decay', type=float, default=1e-5,
                        help='DoRA-stage weight decay')
    parser.add_argument('--dora1-weight-decay', type=float, default=1e-5,
                        help='DoRA1-stage weight decay')
    parser.add_argument('--lr-scheduler', choices=['plateau', 'none'], default='plateau',
                        help='reduce learning rate when validation RMSE plateaus')
    parser.add_argument('--scheduler-factor', type=float, default=0.5)
    parser.add_argument('--scheduler-patience', type=int, default=3,
                        help='number of tolerated non-improving validation calls')
    parser.add_argument('--scheduler-min-lr', type=float, default=1e-7)
    parser.add_argument('--scheduler-threshold', type=float, default=1e-4,
                        help='relative threshold for RMSE improvement')
    parser.add_argument('--scheduler-cooldown', type=int, default=0)
    parser.add_argument('--reset-scheduler', action='store_true',
                        help='restart scheduler and stage learning rate when resuming')
    parser.add_argument('--resume', default='',
                        help='continue the selected stage from its .tar checkpoint')
    
    parser.add_argument('--max-model-num', type=int, default=1)
    parser.add_argument('--pretrain-epochs', type=int, default=600,
                        help='number of clean baseline training epochs')
    parser.add_argument('--pretrain-val-epoch', type=int, default=5,
                        help='validate the baseline every N pretrain epochs')
    parser.add_argument('--pretrain-val-start', type=int, default=200,
                        help='first zero-based pretrain epoch eligible for validation')
    parser.add_argument('--dora-epochs', '--max-epoch', dest='dora_epochs',
                        type=int, default=1200,
                        help='number of DoRA training epochs')
    parser.add_argument('--dora-val-epoch', '--val-epoch', dest='dora_val_epoch',
                        type=int, default=5,
                        help='validate DoRA every N DoRA training epochs')
    parser.add_argument('--dora-val-start', '--val-start', dest='dora_val_start',
                        type=int, default=200,
                        help='first zero-based DoRA epoch eligible for validation')
    parser.add_argument('--dora1-epochs', type=int, default=1200,
                        help='number of parallel DoRA1-branch training epochs')
    parser.add_argument('--dora1-val-epoch', type=int, default=5,
                        help='validate DoRA1 every N epochs')
    parser.add_argument('--dora1-val-start', type=int, default=0,
                        help='first zero-based DoRA1 epoch eligible for validation')
    
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--tensorboard-log-interval', type=int, default=10,
                        help='batch interval for TensorBoard and train.log loss')
    parser.add_argument('--is-gray', action='store_true')
    parser.add_argument('--crop-size', type=int, default=384,
                        help='shared crop-size fallback for all training stages')
    parser.add_argument('--baseline-crop-size', type=int, default=256,
                        help='baseline training crop size (default: 256)')
    parser.add_argument('--dora-crop-size', type=int, default=384,
                        help='DoRA training crop size (default: 384)')
    parser.add_argument('--dora1-crop-size', type=int, default=384,
                        help='DoRA1 training crop size (default: 384)')
    parser.add_argument('--downsample-ratio', type=int, default=16)
    parser.add_argument('--use-background', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=8.0)
    parser.add_argument('--background-ratio', type=float, default=0.15)
    
    args = parser.parse_args()
    for name in ('lr', 'dora_lr', 'dora1_lr'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error('--{} must be finite and positive'.format(name.replace('_', '-')))
    for name in ('weight_decay', 'dora_weight_decay', 'dora1_weight_decay'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error('--{} must be finite and nonnegative'.format(name.replace('_', '-')))
    if not math.isfinite(args.scheduler_factor) or not 0 < args.scheduler_factor < 1:
        parser.error('--scheduler-factor must be between 0 and 1')
    if args.scheduler_patience < 0 or args.scheduler_cooldown < 0:
        parser.error('scheduler patience and cooldown must be nonnegative')
    for name in ('scheduler_min_lr', 'scheduler_threshold'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error('--{} must be finite and nonnegative'.format(name.replace('_', '-')))
    if args.scheduler_threshold >= 1:
        parser.error('--scheduler-threshold must be less than 1')
    active_stages = {'all': ('baseline', 'dora', 'dora1'),
                     'both': ('baseline', 'dora')}.get(args.stage, (args.stage,))
    stage_lrs = {'baseline': args.lr, 'dora': args.dora_lr, 'dora1': args.dora1_lr}
    if args.lr_scheduler == 'plateau' and any(
            args.scheduler_min_lr > stage_lrs[stage] for stage in active_stages):
        parser.error('--scheduler-min-lr must not exceed an active stage learning rate')
    if args.stage == 'dora' and not (args.baseline_checkpoint or args.resume):
        parser.error('--stage dora needs --baseline-checkpoint or --resume')
    if args.stage == 'dora1' and not (args.dora_checkpoint or args.resume):
        parser.error('--stage dora1 needs --dora-checkpoint or --resume')
    if args.stage in ('both', 'all') and args.dora_checkpoint:
        parser.error('pipeline stages use their newly trained DoRA checkpoint; '
                     'do not pass --dora-checkpoint')
    if args.stage in ('dora1', 'all') and not (
            args.dora1_train_dir and args.dora1_val_dir):
        parser.error('--dora1-train-dir and --dora1-val-dir are required for '
                     'the DoRA1 stage')
    for name, epochs, val_epoch, val_start in (
        ('pretrain', args.pretrain_epochs, args.pretrain_val_epoch,
         args.pretrain_val_start),
        ('dora', args.dora_epochs, args.dora_val_epoch, args.dora_val_start),
        ('dora1', args.dora1_epochs, args.dora1_val_epoch,
         args.dora1_val_start),
    ):
        if epochs <= 0 or val_epoch <= 0 or val_start < 0:
            parser.error('{} epochs and validation interval must be positive; '
                         'validation start must be nonnegative'.format(name))
        if (((args.stage in ('both', 'all') and name in ('pretrain', 'dora')) or
             (args.stage == 'all' and name == 'dora1') or
             (args.stage == 'baseline' and name == 'pretrain') or
             (args.stage == 'dora' and name == 'dora') or
             (args.stage == 'dora1' and name == 'dora1')) and
                val_start >= epochs):
            parser.error('--{}-val-start must be less than --{}-epochs '
                         'to select a best checkpoint'.format(name, name))
    if args.downsample_ratio <= 0:
        parser.error('--downsample-ratio must be positive')
    if args.baseline_crop_size is None:
        args.baseline_crop_size = args.crop_size
    if args.dora_crop_size is None:
        args.dora_crop_size = args.crop_size
    if args.dora1_crop_size is None:
        args.dora1_crop_size = args.crop_size
    for name in ('crop_size', 'baseline_crop_size', 'dora_crop_size',
                 'dora1_crop_size'):
        size = getattr(args, name)
        if size <= 0 or size % args.downsample_ratio:
            parser.error('--{} must be positive and divisible by --downsample-ratio'.format(
                name.replace('_', '-')))
    if args.dora_rank <= 0 or args.dora_alpha <= 0:
        parser.error('DoRA rank and alpha must be positive')
    if args.dora1_rank <= 0 or args.dora1_alpha <= 0:
        parser.error('DoRA1 rank and alpha must be positive')
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_stage(args, stage, parent_checkpoint=''):
    stage_args = copy.copy(args)
    stage_args.stage = stage
    stage_args.crop_size = {
        'baseline': args.baseline_crop_size,
        'dora': args.dora_crop_size,
        'dora1': args.dora1_crop_size,
    }[stage]
    if stage == 'baseline':
        stage_args.max_epoch = args.pretrain_epochs
        stage_args.val_epoch = args.pretrain_val_epoch
        stage_args.val_start = args.pretrain_val_start
    elif stage == 'dora':
        stage_args.max_epoch = args.dora_epochs
        stage_args.val_epoch = args.dora_val_epoch
        stage_args.val_start = args.dora_val_start
    else:
        stage_args.max_epoch = args.dora1_epochs
        stage_args.val_epoch = args.dora1_val_epoch
        stage_args.val_start = args.dora1_val_start
    stage_args.save_dir = os.path.join(args.save_dir, args.model_name, stage)
    stage_args.lr = {
        'baseline': args.lr, 'dora': args.dora_lr, 'dora1': args.dora1_lr,
    }[stage]
    stage_args.weight_decay = {
        'baseline': args.weight_decay,
        'dora': args.dora_weight_decay,
        'dora1': args.dora1_weight_decay,
    }[stage]
    if parent_checkpoint:
        if stage == 'dora':
            stage_args.baseline_checkpoint = parent_checkpoint
        elif stage == 'dora1':
            stage_args.dora_checkpoint = parent_checkpoint
        stage_args.resume = ''
    seed_everything(stage_args.seed)
    trainer = RegTrainer(stage_args)
    trainer.setup()
    trainer.train()
    return trainer.best_model_path


def run_pipeline(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    if args.stage in ('both', 'all'):
        best_baseline = run_stage(args, 'baseline')
        if not best_baseline:
            raise RuntimeError('Baseline produced no best checkpoint')
        gc.collect()
        torch.cuda.empty_cache()
        best_dora = run_stage(args, 'dora', parent_checkpoint=best_baseline)
        if args.stage == 'both':
            return best_baseline, best_dora
        gc.collect()
        torch.cuda.empty_cache()
        best_dora1 = run_stage(args, 'dora1', parent_checkpoint=best_dora)
        return best_baseline, best_dora, best_dora1
    return run_stage(args, args.stage)


if __name__ == '__main__':
    run_pipeline(parse_args())
