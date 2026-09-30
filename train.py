"""Train a baseline and mutually exclusive DoRA/LoRA adapter branches."""

import argparse
import copy
import gc
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
        '--stage', choices=['all', 'both', 'baseline', 'dora', 'lora'],
        default='lora',
        help='all: baseline->DoRA->LoRA; both: baseline->DoRA')
    parser.add_argument('--data-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/day',
                        help='clean baseline root containing train/ and val/')
    parser.add_argument('--dora-train-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/night/train')
    parser.add_argument('--dora-val-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/mix/val')
    parser.add_argument('--lora-train-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/night/train',
                        help='LoRA-stage training directory; set this explicitly')
    parser.add_argument('--lora-val-dir', default='/home/mmslab-1080/Desktop/img2img-turbo/SHA/mix/val',
                        help='LoRA-stage validation directory; set this explicitly')
    parser.add_argument('--baseline-checkpoint', default='',
                        help='trained clean baseline .pth or .tar for DoRA-only training')
    parser.add_argument('--dora-checkpoint', default='/home/mmslab-1080/Desktop/Dora_in_swin/model/swin_t_trans/dora/0929-012559/best_model.pth',
                        help='trained DoRA .pth or .tar for LoRA-only training')
    parser.add_argument('--pretrained-path',
                        default='pre_models/swin_large_patch4_window12_384_22k.pth',
                        help='official ImageNet-22K Swin-L checkpoint; also works with 256 crops')
    parser.add_argument('--no-pretrained-backbone', dest='pretrained_backbone',
                        action='store_false')
    
    parser.set_defaults(pretrained_backbone=True)
    parser.add_argument('--dora-rank', type=int, default=4)
    parser.add_argument('--dora-alpha', type=float, default=4.0)
    parser.add_argument('--lora-rank', type=int, default=4)
    parser.add_argument('--lora-alpha', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', default='model')
    parser.add_argument('--save-all', action='store_true')
    parser.add_argument('--lr', type=float, default=5*1e-6)
    parser.add_argument('--dora-lr', type=float, default=2e-4)
    parser.add_argument('--lora-lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--resume', default='',
                        help='continue the selected stage from its .tar checkpoint')
    
    parser.add_argument('--max-model-num', type=int, default=1)
    parser.add_argument('--pretrain-epochs', type=int, default=10,
                        help='number of clean baseline training epochs')
    parser.add_argument('--pretrain-val-epoch', type=int, default=5,
                        help='validate the baseline every N pretrain epochs')
    parser.add_argument('--pretrain-val-start', type=int, default=0,
                        help='first zero-based pretrain epoch eligible for validation')
    parser.add_argument('--dora-epochs', '--max-epoch', dest='dora_epochs',
                        type=int, default=10,
                        help='number of DoRA training epochs')
    parser.add_argument('--dora-val-epoch', '--val-epoch', dest='dora_val_epoch',
                        type=int, default=5,
                        help='validate DoRA every N DoRA training epochs')
    parser.add_argument('--dora-val-start', '--val-start', dest='dora_val_start',
                        type=int, default=0,
                        help='first zero-based DoRA epoch eligible for validation')
    parser.add_argument('--lora-epochs', type=int, default=10,
                        help='number of parallel LoRA-branch training epochs')
    parser.add_argument('--lora-val-epoch', type=int, default=5,
                        help='validate LoRA every N epochs')
    parser.add_argument('--lora-val-start', type=int, default=0,
                        help='first zero-based LoRA epoch eligible for validation')
    
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--tensorboard-log-interval', type=int, default=10,
                        help='batch interval for TensorBoard and train.log loss')
    parser.add_argument('--is-gray', action='store_true')
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--downsample-ratio', type=int, default=16)
    parser.add_argument('--use-background', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=8.0)
    parser.add_argument('--background-ratio', type=float, default=0.15)
    
    args = parser.parse_args()
    if args.stage == 'dora' and not (args.baseline_checkpoint or args.resume):
        parser.error('--stage dora needs --baseline-checkpoint or --resume')
    if args.stage == 'lora' and not (args.dora_checkpoint or args.resume):
        parser.error('--stage lora needs --dora-checkpoint or --resume')
    if args.stage in ('both', 'all') and (
            args.baseline_checkpoint or args.dora_checkpoint):
        parser.error('pipeline stages use newly trained checkpoints; do not pass '
                     '--baseline-checkpoint or --dora-checkpoint')
    if args.stage in ('lora', 'all') and not (
            args.lora_train_dir and args.lora_val_dir):
        parser.error('--lora-train-dir and --lora-val-dir are required for '
                     'the LoRA stage')
    for name, epochs, val_epoch, val_start in (
        ('pretrain', args.pretrain_epochs, args.pretrain_val_epoch,
         args.pretrain_val_start),
        ('dora', args.dora_epochs, args.dora_val_epoch, args.dora_val_start),
        ('lora', args.lora_epochs, args.lora_val_epoch, args.lora_val_start),
    ):
        if epochs <= 0 or val_epoch <= 0 or val_start < 0:
            parser.error('{} epochs and validation interval must be positive; '
                         'validation start must be nonnegative'.format(name))
        if (((args.stage in ('both', 'all') and name in ('pretrain', 'dora')) or
             (args.stage == 'all' and name == 'lora') or
             (args.stage == 'baseline' and name == 'pretrain') or
             (args.stage == 'dora' and name == 'dora') or
             (args.stage == 'lora' and name == 'lora')) and
                val_start >= epochs):
            parser.error('--{}-val-start must be less than --{}-epochs '
                         'to select a best checkpoint'.format(name, name))
    if args.crop_size <= 0 or args.crop_size % args.downsample_ratio:
        parser.error('--crop-size must be positive and divisible by --downsample-ratio')
    if args.dora_rank <= 0 or args.dora_alpha <= 0:
        parser.error('DoRA rank and alpha must be positive')
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        parser.error('LoRA rank and alpha must be positive')
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
    if stage == 'baseline':
        stage_args.max_epoch = args.pretrain_epochs
        stage_args.val_epoch = args.pretrain_val_epoch
        stage_args.val_start = args.pretrain_val_start
    elif stage == 'dora':
        stage_args.max_epoch = args.dora_epochs
        stage_args.val_epoch = args.dora_val_epoch
        stage_args.val_start = args.dora_val_start
    else:
        stage_args.max_epoch = args.lora_epochs
        stage_args.val_epoch = args.lora_val_epoch
        stage_args.val_start = args.lora_val_start
    stage_args.save_dir = os.path.join(args.save_dir, args.model_name, stage)
    stage_args.lr = {
        'baseline': args.lr, 'dora': args.dora_lr, 'lora': args.lora_lr,
    }[stage]
    if parent_checkpoint:
        if stage == 'dora':
            stage_args.baseline_checkpoint = parent_checkpoint
        elif stage == 'lora':
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
        best_lora = run_stage(args, 'lora', parent_checkpoint=best_dora)
        return best_baseline, best_dora, best_lora
    return run_stage(args, args.stage)


if __name__ == '__main__':
    run_pipeline(parse_args())
