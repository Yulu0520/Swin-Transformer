"""Head-only fine-tuning policy with constant Adam LR and shared weight decay."""

import hashlib
from pathlib import Path

from torch import nn


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
        epochs=args.finetune_epochs, lr_schedule='constant',
        lr=args.finetune_lr, optimizer='Adam', weight_decay=args.weight_decay,
        freeze_bn_statistics=True,
        trainable='regression.*', route_input='original_image_224_before_augmentation')
