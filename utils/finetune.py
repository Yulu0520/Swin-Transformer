"""Head-only fine-tuning policy and reproducible conservative LR scheduling."""

import hashlib
import math
from pathlib import Path

from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def freeze_for_finetune(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.regression.parameters():
        parameter.requires_grad_(True)
    set_finetune_mode(model)


def set_finetune_mode(model):
    model.eval()
    model.regression.train()
    for module in model.regression.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()  # affine parameters still receive gradients


def make_finetune_scheduler(optimizer, args):
    def factor(index):
        if args.finetune_warmup_epochs and index < args.finetune_warmup_epochs:
            lr = args.finetune_start_lr + (args.finetune_lr - args.finetune_start_lr) * index / args.finetune_warmup_epochs
        else:
            progress = min(max((index - args.finetune_warmup_epochs) /
                (args.finetune_epochs - args.finetune_warmup_epochs), 0.0), 1.0)
            lr = args.finetune_min_lr + 0.5 * (args.finetune_lr - args.finetune_min_lr) * (1 + math.cos(math.pi * progress))
        return lr / args.finetune_lr
    return LambdaLR(optimizer, factor)


def router_fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finetune_config(args):
    return dict(router_mode=args.finetune_router_mode,
        router_sha256=(router_fingerprint(args.router_checkpoint)
                       if args.finetune_router_mode == 'classifier' else None),
        train_dir=str(Path(args.finetune_train_dir).resolve()),
        val_dir=str(Path(args.finetune_val_dir).resolve()),
        epochs=args.finetune_epochs, warmup_epochs=args.finetune_warmup_epochs,
        start_lr=args.finetune_start_lr, peak_lr=args.finetune_lr,
        min_lr=args.finetune_min_lr, freeze_bn_statistics=True,
        trainable='regression.*', route_input='original_image_224_before_augmentation')
