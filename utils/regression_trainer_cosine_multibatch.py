from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
# The system protobuf extension can be incompatible with the active libstdc++.
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models import swin_c_multibatch as models
from models.counting_checkpoint import OUTPUT_STRIDE, validate_model_mdc
from utils.counting_model import (load_checkpoint, load_backbone_weights,
                                  load_initial_weights, load_finetune_weights)
from utils.topk_checkpoint import TopKCheckpoints
from utils.finetune import freeze_for_finetune, set_finetune_mode, finetune_config
from utils.routing import (select_routes, forward_routed_regression,
                           route_from_filename, count_metrics)
from models.day_night_router import load_day_night_router
from models.swin_dual_dora import (
    extract_dora_from_legacy_state,
    has_legacy_dora_lora_adapters,
)
from datasets.crowd import Crowd, RoutedCrowd
from losses.dm_count import DMCountLoss, LOSS_TYPE, loss_config, validate_resume_loss
from math import ceil
import random


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def has_dual_dora_adapters(state):
    return (any('.parametrizations.weight.0.dora.' in key for key in state)
            and any('.parametrizations.weight.0.dora1.' in key for key in state))


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes


def routed_train_collate(batch):
    images, points, discrete = train_collate(batch)
    return images, points, discrete, torch.stack([item[3] for item in batch]), [item[4] for item in batch]


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if args.device == 'cpu':
            self.device, self.device_count = torch.device('cpu'), 1
            logging.info('using CPU')
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        if args.downsample_ratio != OUTPUT_STRIDE:
            raise ValueError('CCTrans Regression requires downsample_ratio=8')
        self.downsample_ratio = args.downsample_ratio
        data_roots = {
            'baseline': {
                'train': os.path.join(args.data_dir, 'train'),
                'val': os.path.join(args.data_dir, 'val'),
            },
            'dora': {
                'train': args.dora_train_dir,
                'val': args.dora_val_dir,
            },
            'dora1': {
                'train': args.dora1_train_dir,
                'val': args.dora1_val_dir,
            },
            'finetune': {'train': args.finetune_train_dir, 'val': args.finetune_val_dir},
        }
        active_roots = data_roots[args.stage]
        dataset_class = RoutedCrowd if args.stage == 'finetune' else Crowd
        dataset_options = dict(router_mode=args.finetune_router_mode) if args.stage == 'finetune' else {}
        self.datasets = {x: dataset_class(active_roots[x],
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x, **dataset_options) for x in ['train', 'val']}
        for split in ['train', 'val']:
            if not self.datasets[split].im_list:
                raise ValueError('No .jpg images in {}'.format(active_roots[split]))
        loader_generators = {
            'train': torch.Generator().manual_seed(args.seed),
            'val': torch.Generator().manual_seed(args.seed + 1),
        }
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=((routed_train_collate if args.stage == 'finetune' else train_collate)
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False),
                                          worker_init_fn=seed_worker,
                                          generator=loader_generators[x])
                            for x in ['train', 'val']}
        model_builder = getattr(models, args.model_name, None)
        if model_builder is None:
            raise ValueError('unknown model: {}'.format(args.model_name))
        # A resumed checkpoint already contains the backbone weights.
        use_pretrained = (args.stage == 'baseline' and args.pretrained_backbone
                          and not args.resume and not args.backbone_checkpoint
                          and not args.init_checkpoint)
        self.model = model_builder(pretrained=use_pretrained,
                                   pretrained_path=args.pretrained_path,
                                   mdc_mode=args.mdc_mode)
        if args.stage == 'baseline' and args.backbone_checkpoint:
            load_backbone_weights(self.model, args.backbone_checkpoint)
        if args.stage == 'finetune':
            load_finetune_weights(self.model, args, args.resume or args.finetune_checkpoint,
                                  expected_stage='finetune' if args.resume else 'dora1')
            freeze_for_finetune(self.model)
        elif args.init_checkpoint:
            load_initial_weights(self.model, args)
            logging.info('Initialized model weights from %s; optimizer/epoch/best metrics reset',
                         args.init_checkpoint)
        elif args.stage == 'dora':
            if not args.resume:
                checkpoint = load_checkpoint(args.baseline_checkpoint)
                validate_model_mdc(self.model, checkpoint)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    if checkpoint.get('stage', 'baseline') != 'baseline':
                        raise ValueError('DoRA needs a baseline checkpoint')
                    if checkpoint.get('model_name', args.model_name) != args.model_name:
                        raise ValueError('Baseline checkpoint model name does not match')
                state = checkpoint.get('model_state_dict', checkpoint)
                self.model.load_state_dict(state)
            count = self.model.enable_dual_dora(
                args.dora_rank, args.dora_alpha,
                args.dora1_rank, args.dora1_alpha,
                active_route='dora', trainable_route='dora')
            logging.info(
                'Dual DoRA added to %d Swin blocks; only DoRA is active',
                count)
        elif args.stage == 'dora1':
            if args.resume:
                count = self.model.enable_dual_dora(
                    args.dora_rank, args.dora_alpha,
                    args.dora1_rank, args.dora1_alpha,
                    active_route='dora1', trainable_route='dora1')
            else:
                checkpoint = load_checkpoint(args.dora_checkpoint)
                validate_model_mdc(self.model, checkpoint)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    if checkpoint.get('stage', 'dora') != 'dora':
                        raise ValueError('DoRA1 needs a DoRA checkpoint')
                    if checkpoint.get('model_name', args.model_name) != args.model_name:
                        raise ValueError('DoRA checkpoint model name does not match')
                    adapter_type = checkpoint.get('adapter_type', 'dora')
                    if (adapter_type not in (
                            'dora', 'parallel_dora_dora1',
                            'parallel_dora_lora') or
                            checkpoint.get('dora_rank') != args.dora_rank or
                            checkpoint.get('dora_alpha') != args.dora_alpha):
                        raise ValueError('DoRA checkpoint settings do not match')
                    if (adapter_type == 'parallel_dora_dora1' and
                            (checkpoint.get('dora1_rank') != args.dora1_rank or
                             checkpoint.get('dora1_alpha') != args.dora1_alpha)):
                        raise ValueError('Frozen DoRA1 settings do not match')
                state = checkpoint.get('model_state_dict', checkpoint)
                if has_dual_dora_adapters(state):
                    count = self.model.enable_dual_dora(
                        args.dora_rank, args.dora_alpha,
                        args.dora1_rank, args.dora1_alpha,
                        active_route='dora1', trainable_route='dora1')
                    self.model.load_state_dict(state)
                else:
                    # Migrate standalone DoRA or the former DoRA/LoRA
                    # container without changing trained DoRA or baseline.
                    if has_legacy_dora_lora_adapters(state):
                        state = extract_dora_from_legacy_state(state)
                    self.model.enable_dora(args.dora_rank, args.dora_alpha)
                    self.model.load_state_dict(state)
                    count = self.model.enable_dual_dora(
                        args.dora_rank, args.dora_alpha,
                        args.dora1_rank, args.dora1_alpha,
                        active_route='dora1', trainable_route='dora1')
            logging.info(
                'Dual DoRA active in %d Swin blocks; only DoRA1 is active',
                count)
        self.model.to(self.device)
        self.image_router = None
        self.finetune_settings = None
        if args.stage == 'finetune':
            self.finetune_settings = finetune_config(args)
            if args.finetune_router_mode == 'classifier':
                self.image_router = load_day_night_router(args.router_checkpoint).to(self.device).eval()
                self.image_router.requires_grad_(False)
        named_trainable = [(name, parameter)
                           for name, parameter in self.model.named_parameters()
                           if parameter.requires_grad]
        expected_marker = {
            'dora': '.parametrizations.weight.0.dora.',
            'dora1': '.parametrizations.weight.0.dora1.',
        }.get(args.stage)
        if expected_marker is not None:
            unexpected = [name for name, _ in named_trainable
                          if expected_marker not in name]
            if unexpected or not named_trainable:
                raise RuntimeError(
                    '{} stage has unexpected trainable parameters: {}'.format(
                        args.stage, unexpected))
        if args.stage == 'finetune' and (not named_trainable or any(
                not name.startswith('regression.') for name, _ in named_trainable)):
            raise RuntimeError('Fine-tuning must optimize only regression parameters')
        trainable = [parameter for _, parameter in named_trainable]
        logging.info('trainable parameters: %d / %d',
                     sum(parameter.numel() for parameter in trainable),
                     sum(parameter.numel() for parameter in self.model.parameters()))
        self.optimizer = optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = None  # All stages use a constant LR with Adam weight decay.

        self.start_epoch = 0
        resumed_state = {}
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = load_checkpoint(args.resume)
                validate_model_mdc(self.model, checkpoint)
                validate_resume_loss(checkpoint, args)
                if args.stage == 'finetune' and checkpoint.get('finetune_config') != self.finetune_settings:
                    raise ValueError('Fine-tune router/data/LR schedule settings or weight decay do not match '
                                     'the checkpoint. Warmup/cosine runs cannot be resumed as fixed-LR runs.')
                if args.stage == 'finetune' and checkpoint.get('scheduler_state_dict') is not None:
                    raise ValueError('Fixed-LR fine-tuning cannot resume a warmup/cosine scheduler')
                resumed_state = checkpoint
                if checkpoint.get('stage', 'baseline') != args.stage:
                    raise ValueError('Resume checkpoint stage does not match')
                if checkpoint.get('model_name', args.model_name) != args.model_name:
                    raise ValueError('Resume checkpoint model name does not match')
                if args.stage in ('dora', 'dora1', 'finetune') and (
                    checkpoint.get('adapter_type') != 'parallel_dora_dora1'
                    or checkpoint.get('dora_rank') != args.dora_rank
                    or checkpoint.get('dora_alpha') != args.dora_alpha
                    or checkpoint.get('dora1_rank') != args.dora1_rank
                    or checkpoint.get('dora1_alpha') != args.dora1_alpha
                ):
                    raise ValueError('Resume checkpoint adapter settings do not match')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # New checkpoints store both the human-facing 1-based epoch
                # and its internal 0-based index. Older checkpoints only have
                # the 0-based ``epoch`` field.
                if 'epoch_index' in checkpoint:
                    self.start_epoch = checkpoint['epoch_index'] + 1
                else:
                    self.start_epoch = checkpoint['epoch'] + 1
            else:
                raise ValueError('--resume requires a .tar; use --init-checkpoint for weights only')

        self.criterion = DMCountLoss(
            args.crop_size, args.downsample_ratio, self.device,
            count_loss_weight=getattr(args, 'count_loss_weight', 1.0),
            wot=args.wot, wtv=args.wtv, reg=args.reg,
            num_of_iter_in_ot=args.num_of_iter_in_ot, norm_cood=bool(args.norm_cood))
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = resumed_state.get('best_mae', np.inf)
        self.best_mse = resumed_state.get('best_mse', np.inf)
        self.best_model_path = resumed_state.get('best_model_path')
        self.best_count = resumed_state.get('best_count', 0)
        self.top_checkpoints = TopKCheckpoints(self.save_dir)
        if resumed_state.get('top_checkpoints'):
            self.top_checkpoints.restore(resumed_state['top_checkpoints'], os.path.dirname(args.resume))
            self.best_model_path = self.top_checkpoints.best_path
        if args.resume and not self.best_model_path:
            previous_best = os.path.join(os.path.dirname(args.resume), 'best_model.pth')
            if os.path.isfile(previous_best):
                self.best_model_path = previous_best
        if self.best_model_path and not os.path.isfile(self.best_model_path):
            local_best = os.path.join(os.path.dirname(args.resume), os.path.basename(self.best_model_path))
            if os.path.isfile(local_best):
                self.best_model_path = local_best
            else:
                raise FileNotFoundError('Saved best model is missing: {}'.format(self.best_model_path))
        if self.best_model_path and not self.top_checkpoints.entries:
            self.top_checkpoints.import_legacy_best(self.best_model_path, self.best_mae, self.best_mse)
            self.best_model_path = self.top_checkpoints.best_path
        if self.top_checkpoints.entries:
            best = self.top_checkpoints.entries[0]
            self.best_mae, self.best_mse = best['mae'], best['rmse']
        self.initial_evaluated = resumed_state.get('initial_evaluated', False)
        self.global_step = self.start_epoch * len(self.dataloaders['train'])
        if args.tensorboard_log_interval <= 0:
            raise ValueError('tensorboard_log_interval must be greater than zero')
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'))

    def train(self):
        """training process"""
        args = self.args
        try:
            if args.stage == 'finetune' and not self.initial_evaluated:
                self.epoch, self.epoch_index = 0, -1
                self.val_epoch()
                self.initial_evaluated = True
                self.save_epoch()
            for epoch_index in range(self.start_epoch, args.max_epoch):
                self.epoch_index = epoch_index
                self.epoch = epoch_index + 1
                logging.info(
                    '-'*5 + 'Epoch {}/{}'.format(self.epoch, args.max_epoch) + '-'*5
                )
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.epoch)
                self.train_eopch()
                if (
                    self.epoch_index >= args.val_start
                    and (self.epoch_index - args.val_start) % args.val_epoch == 0
                ):
                    self.val_epoch()
                self.save_epoch()
        finally:
            self.writer.close()

    def train_eopch(self):
        metric_names = ('total', 'count', 'ot', 'tv', 'weighted_count', 'weighted_ot',
                        'weighted_tv', 'wasserstein', 'ot_objective')
        meters = {name: AverageMeter() for name in metric_names}
        epoch_mae, epoch_mse = AverageMeter(), AverageMeter()
        epoch_start = time.time()
        # Freeze dropout/stochastic depth and head BN statistics in adapter stages.
        if self.args.stage == 'finetune':
            set_finetune_mode(self.model)
        else:
            self.model.train(self.args.stage == 'baseline')
        route_counts = {'dora': 0, 'dora1': 0}
        confidences, router_correct = [], 0
        train_bar = tqdm(
            enumerate(self.dataloaders['train']), total=len(self.dataloaders['train']),
            desc='{} {}/{}'.format(self.args.stage.title(), self.epoch, self.args.max_epoch),
            dynamic_ncols=True, mininterval=0.5)
        for step, batch in train_bar:
            inputs, points, gt_discrete = batch[:3]
            inputs, gt_discrete = inputs.to(self.device), gt_discrete.to(self.device)
            points = [p.to(self.device) for p in points]
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            routes = None
            if self.args.stage == 'finetune':
                router_inputs, names = batch[3:]
                domains, routes, confidence = select_routes(self.args.finetune_router_mode,
                    names, router_inputs, self.image_router, self.device)
                confidences.extend(confidence)
                for name, domain, route in zip(names, domains, routes):
                    route_counts[route] += 1
                    router_correct += int(domain == route_from_filename(name)[0])
            with torch.set_grad_enabled(True):
                outputs = (forward_routed_regression(self.model, inputs, routes)
                           if routes is not None else self.model(inputs))
                terms = self.criterion(outputs, points, gt_discrete)
                loss = terms['total']
                if not all(torch.isfinite(value).all() for value in terms.values()):
                    raise FloatingPointError('Nonfinite DM-Count loss at step {}'.format(step))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_size = inputs.size(0)
            predicted = outputs.flatten(1).sum(1).detach().cpu().numpy()
            errors = predicted - gd_count
            epoch_mae.update(np.mean(abs(errors)), batch_size)
            epoch_mse.update(np.mean(errors * errors), batch_size)
            values = {name: terms[name].item() for name in metric_names}
            for name, value in values.items():
                meters[name].update(value, batch_size)
            if self.global_step % self.args.tensorboard_log_interval == 0:
                for name, value in values.items():
                    tag = name + '_loss' if name not in ('wasserstein', 'ot_objective') else name
                    self.writer.add_scalar('train_step/' + tag, value, self.global_step)
            if step == 0 or (step + 1) % self.args.tensorboard_log_interval == 0:
                logging.info(
                    '%s Epoch %d/%d Step %d/%d Train: total_loss=%.4f, '
                    'count_loss=%.4f, ot_loss=%.4f, tv_loss=%.4f, '
                    'weighted_count_loss=%.4f, weighted_ot_loss=%.4f, weighted_tv_loss=%.4f, wasserstein=%.4f',
                    self.args.stage, self.epoch, self.args.max_epoch,
                    step + 1, len(self.dataloaders['train']), values['total'],
                    values['count'], values['ot'], values['tv'],
                    values['weighted_count'], values['weighted_ot'], values['weighted_tv'], values['wasserstein'],
                    extra={'file_only': True})
            train_bar.set_postfix(loss='{:.3f}'.format(values['total']),
                                  count='{:.3f}'.format(values['count']),
                                  weighted_count='{:.3f}'.format(values['weighted_count']),
                                  ot='{:.3f}'.format(values['weighted_ot']),
                                  tv='{:.3f}'.format(values['weighted_tv']), refresh=False)
            self.global_step += 1
        for name, meter in meters.items():
            tag = name + '_loss' if name not in ('wasserstein', 'ot_objective') else name
            self.writer.add_scalar('train/' + tag, meter.get_avg(), self.epoch)
        train_rmse = np.sqrt(epoch_mse.get_avg())
        self.writer.add_scalar('train/mae', epoch_mae.get_avg(), self.epoch)
        self.writer.add_scalar('train/crop_count_mae', meters['count'].get_avg(), self.epoch)
        self.writer.add_scalar('train/rmse', train_rmse, self.epoch)
        if confidences:
            self.writer.add_scalar('train/router_confidence', np.mean(confidences), self.epoch)
            self.writer.add_scalar('train/router_accuracy', router_correct / len(confidences), self.epoch)
            for route, count in route_counts.items():
                self.writer.add_scalar('train/route_count/' + route, count, self.epoch)
        self.writer.flush()
        logging.info(
            '%s Epoch %d Train, Loss: %.4f, Count L1: %.4f, OT: %.4f, TV: %.4f, '
            'Weighted Count: %.4f, Weighted OT: %.4f, Weighted TV: %.4f, RMSE: %.2f, MAE: %.2f, Cost %.1f sec',
            self.args.stage, self.epoch, meters['total'].get_avg(),
            meters['count'].get_avg(), meters['ot'].get_avg(), meters['tv'].get_avg(),
            meters['weighted_count'].get_avg(), meters['weighted_ot'].get_avg(), meters['weighted_tv'].get_avg(),
            train_rmse, epoch_mae.get_avg(), time.time() - epoch_start)

    def save_epoch(self):
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'epoch_index': self.epoch_index,
            'model_name': self.args.model_name,
            'architecture': self.model.architecture,
            'mdc_mode': self.model.mdc_mode,
            'output_stride': self.model.output_stride,
            'stage': self.args.stage,
            'crop_size': self.args.crop_size,
            'loss_type': LOSS_TYPE,
            'loss_config': loss_config(self.args),
            'initial_checkpoint': self.args.init_checkpoint,
            'top_checkpoints': (self.top_checkpoints.state_dict() if hasattr(self, 'top_checkpoints') else []),
            'scheduler_state_dict': self.scheduler.state_dict() if getattr(self, 'scheduler', None) else None,
            'finetune_config': getattr(self, 'finetune_settings', None),
            'initial_evaluated': getattr(self, 'initial_evaluated', False),
            'parent_finetune_checkpoint': getattr(self.args, 'finetune_checkpoint', ''),
            'router_checkpoint': getattr(self.args, 'router_checkpoint', ''),
            'adapter_type': ({
                'dora': 'parallel_dora_dora1',
                'dora1': 'parallel_dora_dora1',
                'finetune': 'parallel_dora_dora1',
            }.get(self.args.stage)),
            'dora_rank': (self.args.dora_rank
                          if self.args.stage in ('dora', 'dora1', 'finetune') else 0),
            'dora_alpha': (self.args.dora_alpha
                           if self.args.stage in ('dora', 'dora1', 'finetune') else 0),
            'dora1_rank': (self.args.dora1_rank
                           if self.args.stage in ('dora', 'dora1', 'finetune') else 0),
            'dora1_alpha': (self.args.dora1_alpha
                            if self.args.stage in ('dora', 'dora1', 'finetune') else 0),
            'parent_dora_checkpoint': (self.args.dora_checkpoint
                                       if self.args.stage == 'dora1' else ''),
            'seed': self.args.seed,
            'best_mae': self.best_mae,
            'best_mse': self.best_mse,
            'best_model_path': self.best_model_path,
            'best_count': self.best_count,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        domain_errors = {'day': [], 'night': []}
        confidences, router_correct = [], 0
        route_counts = {'dora': 0, 'dora1': 0}
        running_abs_error = 0.0
        running_squared_error = 0.0
        val_bar = tqdm(
            self.dataloaders['val'],
            total=len(self.dataloaders['val']),
            desc='{} Val {}/{}'.format(
                self.args.stage.title(), self.epoch, self.args.max_epoch),
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for batch in val_bar:
            inputs, count, name = batch[:3]
            if self.args.stage == 'finetune':
                domains, routes, confidence = select_routes(self.args.finetune_router_mode,
                    name, batch[3], self.image_router, self.device)
                self.model.set_adapter_route(routes[0])
                route_counts[routes[0]] += 1
                confidences.extend(confidence)
                router_correct += int(domains[0] == route_from_filename(name[0])[0])
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            if h >= 3584 or w >= 3584:
                h_stride = int(ceil(1.0 * h / 3584))
                w_stride = int(ceil(1.0 * w / 3584))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    predicted_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = self.model(input)
                        predicted_count += torch.sum(output).item()
            else:
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    predicted_count = torch.sum(outputs).item()

            ground_truth = count[0].item()
            res = ground_truth - predicted_count
            epoch_res.append(res)
            domain_errors[route_from_filename(name[0])[0]].append(res)
            running_abs_error += abs(res)
            running_squared_error += res * res
            processed_samples = len(epoch_res)
            val_bar.set_postfix(
                gt='{:.1f}'.format(ground_truth),
                pred='{:.1f}'.format(predicted_count),
                mae='{:.2f}'.format(running_abs_error / processed_samples),
                rmse='{:.2f}'.format(
                    np.sqrt(running_squared_error / processed_samples)
                ),
                refresh=False,
            )

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.writer.add_scalar('val/mae', mae, self.epoch)
        self.writer.add_scalar('val/rmse', mse, self.epoch)
        logging.info('{} Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.args.stage, self.epoch, mse, mae,
                             time.time()-epoch_start))

        if not hasattr(self, 'top_checkpoints'):
            self.top_checkpoints = TopKCheckpoints(self.save_dir)
        if self.top_checkpoints.consider(self.model.state_dict(), mae, mse, self.epoch):
            logging.info('Saved top-5 validation weights for epoch %d', self.epoch)
        best = self.top_checkpoints.entries[0]
        self.best_mae, self.best_mse = best['mae'], best['rmse']
        self.best_model_path = self.top_checkpoints.best_path

        groups = {domain: count_metrics(errors) for domain, errors in domain_errors.items()}
        if self.args.stage == 'finetune':
            for domain, metrics in groups.items():
                if metrics['images']:
                    for metric in ('mae', 'rmse'):
                        self.writer.add_scalar('val/{}/{}'.format(domain, metric), metrics[metric], self.epoch)
                    logging.info('Fine-tune Val %s: images=%d MAE=%.2f RMSE=%.2f',
                        domain, metrics['images'], metrics['mae'], metrics['rmse'])
            if confidences:
                self.writer.add_scalar('val/router_confidence', np.mean(confidences), self.epoch)
                self.writer.add_scalar('val/router_accuracy', router_correct / len(confidences), self.epoch)
                for route, count in route_counts.items():
                    self.writer.add_scalar('val/route_count/' + route, count, self.epoch)

        self.writer.add_scalar('best/mae', self.best_mae, self.epoch)
        self.writer.add_scalar('best/rmse', self.best_mse, self.epoch)
        self.writer.flush()
        return dict(mae=float(mae), rmse=float(mse), groups=groups)
