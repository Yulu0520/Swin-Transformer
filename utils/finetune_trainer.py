"""Shared-head fine-tuning with frozen, per-image dual-DoRA routing."""

import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.crowd import Crowd
from datasets.day_night import build_router_transform
from losses.bay_loss import Bay_Loss
from losses.count_loss import count_l1_loss
from losses.post_prob import Post_Prob
from models.day_night_router import load_day_night_router, prediction_to_route
from utils.regression_trainer_cosine_multibatch import seed_worker, train_collate
from utils.trainer import Trainer
from utils.logger import setlogger
from test import build_dual_dora_model, evaluate, route_from_filename


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(dataset):
    """Hash images and annotations, detecting content changes as well as renames."""
    return [(os.path.basename(path), file_digest(path),
             file_digest(str(Path(path).with_suffix('.npy'))))
            for path in dataset.im_list]


def validate_resume_config(saved, current):
    differences = [key for key in sorted(set(saved) | set(current))
                   if saved.get(key) != current.get(key)]
    if differences:
        raise ValueError('Fine-tune resume settings differ: ' + ', '.join(differences))


def ranked_candidates(ranking, epoch, mae, rmse):
    score = mae + 2 * rmse
    if not math.isfinite(score):
        raise ValueError('Non-finite fine-tune validation score')
    entry = dict(epoch=epoch, mae=mae, rmse=rmse, score=score,
                 filename='best_epoch_{:03d}.pth'.format(epoch))
    return sorted(ranking + [entry], key=lambda row: (row['score'], row['epoch']))[:5]


def atomic_save(value, path):
    temporary = str(path) + '.tmp'
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_training_checkpoint(path):
    # Training state contains Python/NumPy RNG state and must be loaded in full.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


class RoutedCrowd(Dataset):
    def __init__(self, crowd, routes):
        self.crowd, self.routes = crowd, routes

    def __len__(self):
        return len(self.crowd)

    def __getitem__(self, index):
        return self.crowd[index], self.routes[index]


def routed_collate(batch):
    samples, routes = zip(*batch)
    return (*train_collate(samples), routes)


def backward_routed_batch(model, batch, device, post_prob, criterion, count_weight):
    """Accumulate sample-weighted gradients before one optimizer step."""
    inputs, points, targets, sizes, routes = batch
    if not routes or any(route not in ('dora', 'dora1') for route in routes):
        raise ValueError('Unknown fine-tune adapter route')
    inputs, sizes = inputs.to(device), sizes.to(device)
    points = [point.to(device) for point in points]
    targets = [target.to(device) for target in targets]
    totals = dict(total=0., bayesian=0., consistency=0., count=0.)
    for route in ('dora', 'dora1'):
        indices = [i for i, selected in enumerate(routes) if selected == route]
        if not indices:
            continue
        model.set_adapter_route(route)
        density, features = model(inputs[indices])
        group_targets = [targets[i] for i in indices]
        probabilities = post_prob([points[i] for i in indices], sizes[indices])
        bayesian = criterion(probabilities, group_targets, density)
        consistency = density.new_zeros(())
        for feature in features:
            consistency += (1 - F.cosine_similarity(
                feature, feature.mean(dim=1, keepdim=True), dim=2, eps=1e-5)
            ).sum(dim=1).mean()
        count = count_l1_loss(density, group_targets)
        loss = bayesian + consistency + count_weight * count
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite routed training loss')
        weight = len(indices) / len(routes)
        (loss * weight).backward()
        for key, value in zip(totals, (loss, bayesian, consistency, count)):
            totals[key] += value.detach().item() * weight
    return totals


class FineTuneTrainer(Trainer):
    @property
    def stage_label(self):
        return 'Fine-tune' if self.args.stage == 'finetune' else self.args.stage.title()

    @property
    def validation_router_mode(self):
        return self.args.router_mode

    def prepare_model(self, model_args, source, resumed):
        source_state = resumed if resumed else load_training_checkpoint(source)
        if 'model_state_dict' in source_state:
            if source_state.get('stage') != ('finetune' if resumed else 'dora1'):
                raise ValueError('Fine-tune must start from a completed DoRA1 checkpoint')
            self.check_source_settings(source_state)
        model = build_dual_dora_model(model_args, source)
        model.enable_finetune()
        return model

    def check_source_settings(self, source_state):
        for key in ('model_name', 'dora_rank', 'dora_alpha', 'dora1_rank', 'dora1_alpha'):
            if source_state.get(key) != getattr(self.args, key):
                raise ValueError('Source checkpoint differs from --' + key.replace('_', '-'))

    def configure_train_mode(self):
        self.model.finetune_train_mode()

    def checkpoint_metadata(self):
        return {}

    def extra_config(self):
        return {}

    def __init__(self, args):
        if args.resume:
            if not os.path.isfile(args.resume):
                raise FileNotFoundError(args.resume)
            self.args = args
            self.save_dir = os.path.dirname(os.path.abspath(args.resume))
            setlogger(os.path.join(self.save_dir, 'train.log'))
        else:
            super().__init__(args)

    def setup(self):
        args = self.args
        if args.downsample_ratio != 16:
            raise ValueError('Swin MAN density maps use --downsample-ratio 16')
        self.device = torch.device('cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu')
        resumed = load_training_checkpoint(args.resume) if args.resume else None
        resume_keys = ('config', 'ranking', 'epoch', 'optimizer_state_dict', 'rng', 'global_step')
        if resumed and (resumed.get('stage') != args.stage
                        or any(key not in resumed for key in resume_keys)):
            raise ValueError('Resume needs a matching-stage last_checkpoint.tar')
        if resumed:
            self.save_dir = os.path.dirname(os.path.abspath(args.resume))
        self.train_data = Crowd(args.finetune_train_dir, args.crop_size,
                                args.downsample_ratio, args.is_gray, 'train')
        self.val_data = Crowd(args.finetune_val_dir, args.crop_size,
                              args.downsample_ratio, False, 'val')
        if not len(self.train_data) or not len(self.val_data):
            raise ValueError('Fine-tune train and val must both contain .jpg images')
        config_keys = ('model_name', 'dora_rank', 'dora_alpha', 'dora1_rank',
                       'dora1_alpha', 'lr', 'weight_decay', 'count_loss_weight',
                       'crop_size', 'downsample_ratio', 'is_gray', 'sigma',
                       'background_ratio', 'use_background', 'batch_size',
                       'num_workers', 'seed', 'router_mode')
        self.config = {key: getattr(args, key) for key in config_keys}
        logging.info('Fine-tune: fingerprinting train/val images and annotations')
        self.config.update(
            train_root=os.path.realpath(args.finetune_train_dir),
            val_root=os.path.realpath(args.finetune_val_dir),
            train_manifest=dataset_manifest(self.train_data),
            val_manifest=dataset_manifest(self.val_data),
            router_digest=(file_digest(args.router_checkpoint)
                           if args.router_mode == 'classifier' else None),
            router_transform='resize224_imagenet_v1',
            loss_digests={name: file_digest(Path(__file__).resolve().parents[1] / name)
                          for name in ('losses/bay_loss.py', 'losses/count_loss.py',
                                       'losses/post_prob.py', 'utils/finetune_trainer.py')},
            schema_version=1,
        )
        self.config.update(self.extra_config())
        if resumed:
            validate_resume_config(resumed['config'], self.config)
            if args.max_epoch < resumed['epoch']:
                raise ValueError('--finetune-epochs is less than the completed epoch')
            for row in resumed['ranking']:
                if not os.path.isfile(os.path.join(self.save_dir, row['filename'])):
                    raise FileNotFoundError('Ranked weight missing: ' + row['filename'])
        logging.info('Fine-tune: hashing complete; loading frozen model/router')
        model_args = copy.copy(args)
        model_args.dual_route = 'dora1'
        source = args.resume or args.dora1_checkpoint
        self.model = self.prepare_model(model_args, source, resumed).to(self.device)
        logging.info('%s trainable parameters: %d / %d', self.stage_label,
                     sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                     sum(p.numel() for p in self.model.parameters()))
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)
        self.router = None
        if args.router_mode == 'classifier':
            self.router = load_day_night_router(args.router_checkpoint).to(self.device)
            self.router.requires_grad_(False).eval()
        # The frozen router sees each full image, before random crop/augmentation.
        transform = build_router_transform(False)
        routes = []
        with torch.inference_mode():
            for path in tqdm(self.train_data.im_list, desc='Cache full-image routes'):
                if self.validation_router_mode == 'fixed':
                    route = 'dora'
                elif self.router is None:
                    route = route_from_filename(path)[1]
                else:
                    with Image.open(path) as image:
                        pixels = transform(image.convert('RGB')).unsqueeze(0).to(self.device)
                    route = prediction_to_route(self.router(pixels))[1]
                routes.append(route)
        self.generator = torch.Generator().manual_seed(args.seed)
        self.loader = DataLoader(
            RoutedCrowd(self.train_data, routes), batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, collate_fn=routed_collate,
            generator=self.generator, worker_init_fn=seed_worker)
        self.post_prob = Post_Prob(args.sigma, args.crop_size, args.downsample_ratio,
                                   args.background_ratio, args.use_background, self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.ranking = [] if resumed is None else resumed['ranking']
        self.completed_epoch = -1 if resumed is None else resumed['epoch']
        self.global_step = 0 if resumed is None else resumed['global_step']
        self.eval_args = SimpleNamespace(
            crop_size=args.crop_size, downsample_ratio=args.downsample_ratio,
            num_workers=args.num_workers, split='val')
        self.origin = (resumed['origin'] if resumed else os.path.abspath(source))
        self.origin_digest = resumed['origin_digest'] if resumed else file_digest(source)
        if resumed:
            self.optimizer.load_state_dict(resumed['optimizer_state_dict'])
            random.setstate(resumed['rng']['python'])
            np.random.set_state(resumed['rng']['numpy'])
            torch.set_rng_state(resumed['rng']['torch'])
            if torch.cuda.is_available() and resumed['rng']['cuda'] is not None:
                torch.cuda.set_rng_state_all(resumed['rng']['cuda'])
            self.generator.set_state(resumed['rng']['loader'])
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'),
                                    purge_step=self.completed_epoch + 1 if resumed else None)
        self.best_model_path = None

    def weights(self, epoch):
        metadata = {key: getattr(self.args, key) for key in
                    ('model_name', 'dora_rank', 'dora_alpha', 'dora1_rank', 'dora1_alpha')}
        return dict(metadata, stage=self.args.stage, adapter_type='parallel_dora_dora1',
                    epoch=epoch, model_state_dict=self.model.state_dict(),
                    config=self.config, origin=self.origin, origin_digest=self.origin_digest,
                    **self.checkpoint_metadata())

    def validate_and_save(self, epoch):
        self.model.eval()
        if self.validation_router_mode == 'fixed':
            self.model.set_adapter_route('dora')
        metrics = evaluate(self.model, self.args.finetune_val_dir, self.eval_args,
                           self.device, '{} val epoch {}'.format(self.stage_label, epoch),
                           router_mode=self.validation_router_mode, image_router=self.router,
                           domain='mix')
        for domain, values in dict(mix=metrics, **metrics['domain_metrics']).items():
            for key in ('mae', 'rmse'):
                self.writer.add_scalar('val/{}/{}'.format(domain, key), values[key], epoch)
        score = metrics['mae'] + 2 * metrics['rmse']
        logging.info('%s epoch %d MAE %.4f RMSE %.4f score %.4f',
                     self.stage_label, epoch, metrics['mae'], metrics['rmse'], score)
        if hasattr(self.model, 'night_fusion'):
            self.writer.add_scalar('train/fusion_alpha', self.model.night_fusion.alpha.item(), epoch)
        ranking = ranked_candidates(self.ranking, epoch, metrics['mae'], metrics['rmse'])
        if any(row['epoch'] == epoch for row in ranking):
            atomic_save(self.weights(epoch), os.path.join(
                self.save_dir, 'best_epoch_{:03d}.pth'.format(epoch)))
        self.ranking = ranking
        checkpoint = self.weights(epoch)
        checkpoint.update(
            optimizer_state_dict=self.optimizer.state_dict(), ranking=self.ranking,
            global_step=self.global_step,
            rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                     torch=torch.get_rng_state(), loader=self.generator.get_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        # Commit resume state before pruning: interruption cannot delete the old top 5.
        atomic_save(checkpoint, os.path.join(self.save_dir, 'last_checkpoint.tar'))
        self.sync_ranking()
        self.completed_epoch = epoch
        self.best_model_path = os.path.join(self.save_dir, ranking[0]['filename'])
        self.writer.flush()

    def sync_ranking(self):
        ranking_path = os.path.join(self.save_dir, 'ranking.json')
        with open(ranking_path + '.tmp', 'w') as output:
            json.dump(self.ranking, output, indent=2)
        os.replace(ranking_path + '.tmp', ranking_path)
        retained = {row['filename'] for row in self.ranking}
        for path in Path(self.save_dir).glob('best_epoch_*.pth'):
            if path.name not in retained:
                path.unlink()

    def train(self):
        try:
            if self.ranking:
                self.sync_ranking()
            if self.completed_epoch == -1:
                self.validate_and_save(0)
            for epoch in range(self.completed_epoch + 1, self.args.max_epoch + 1):
                self.configure_train_mode()
                totals = dict(total=0., bayesian=0., consistency=0., count=0.)
                seen = 0
                for batch in tqdm(self.loader, desc='{} epoch {}'.format(self.stage_label, epoch)):
                    self.optimizer.zero_grad(set_to_none=True)
                    values = backward_routed_batch(
                        self.model, batch, self.device, self.post_prob,
                        self.criterion, self.args.count_loss_weight)
                    self.optimizer.step()
                    size = len(batch[-1])
                    seen += size
                    for key, value in values.items():
                        totals[key] += value * size
                        if self.global_step % self.args.tensorboard_log_interval == 0:
                            self.writer.add_scalar('train_step/' + key, value, self.global_step)
                    self.global_step += 1
                for key, value in totals.items():
                    self.writer.add_scalar('train/' + key, value / seen, epoch)
                logging.info('%s epoch %d train total %.6f Bayesian %.6f '
                             'consistency %.6f count L1 %.6f', self.stage_label, epoch,
                             *(totals[key] / seen for key in totals))
                self.validate_and_save(epoch)
            self.best_model_path = os.path.join(self.save_dir, self.ranking[0]['filename'])
        finally:
            self.writer.close()
