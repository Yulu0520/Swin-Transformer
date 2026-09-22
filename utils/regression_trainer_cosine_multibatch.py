from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models import swin_c_multibatch as models
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from math import ceil


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        model_builder = getattr(models, args.model_name, None)
        if model_builder is None:
            raise ValueError('unknown model: {}'.format(args.model_name))
        if args.resume and args.init_checkpoint:
            raise ValueError('--resume and --init-checkpoint cannot be used together')
        if args.max_epoch <= 0 or args.dora_epochs <= 0:
            raise ValueError('--max-epoch and --dora-epochs must be positive')
        if args.val_start <= 0 or args.val_epoch <= 0:
            raise ValueError('--val-start and --val-epoch must be positive')
        if args.dora_val_start is not None and args.dora_val_start <= 0:
            raise ValueError('--dora-val-start must be positive')
        self.start_epoch = 0
        resume_optimizer_state = None
        self.phase = 'pretrain'
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_model_path = None
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                state_dict = checkpoint['model_state_dict']
                self.model = models.swin_t_trans_from_state_dict(state_dict)
                self.phase = checkpoint.get(
                    'phase', 'dora' if self.model.use_dora else 'pretrain'
                )
                if self.phase not in ('pretrain', 'dora'):
                    raise ValueError('unknown checkpoint phase: {}'.format(self.phase))
                if self.model.use_dora != (self.phase == 'dora'):
                    raise ValueError('checkpoint model does not match training phase')
                if self.phase == 'dora' and not args.use_dora:
                    raise ValueError('cannot resume DoRA training with --no-dora')
                if self.phase == 'dora':
                    self.model.train_dora_only()
                self.best_mae = checkpoint.get('best_mae', np.inf)
                self.best_mse = checkpoint.get('best_mse', np.inf)
                best_name = '{}_best_model.pth'.format(self.phase)
                candidate = os.path.join(os.path.dirname(args.resume), best_name)
                if os.path.isfile(candidate):
                    self.best_model_path = candidate
                elif self.phase == 'pretrain':
                    legacy = os.path.join(os.path.dirname(args.resume), 'best_model.pth')
                    if os.path.isfile(legacy):
                        self.best_model_path = legacy
                if (self.phase == 'pretrain' and np.isfinite(self.best_mae)
                        and self.best_model_path is None):
                    raise FileNotFoundError(
                        'pretrain best model is missing beside the resume checkpoint'
                    )
                resume_optimizer_state = checkpoint['optimizer_state_dict']
                # New checkpoints store both the human-facing 1-based epoch
                # and its internal 0-based index. Older checkpoints only have
                # the 0-based ``epoch`` field.
                if 'epoch_index' in checkpoint:
                    self.start_epoch = checkpoint['epoch_index'] + 1
                else:
                    self.start_epoch = checkpoint['epoch'] + 1
                if self.phase == 'dora' and 'phase' not in checkpoint:
                    # Older DoRA runs also trained the head and adapter, so
                    # their optimizer groups cannot be restored for DoRA-only.
                    logging.warning(
                        'Legacy DoRA checkpoint: restarting DoRA-only epochs '
                        'with a fresh optimizer while keeping model weights'
                    )
                    self.start_epoch = 0
                    resume_optimizer_state = None
            elif suf == 'pth':
                raise ValueError('use --init-checkpoint for .pth files; '
                                 '--resume requires a full .tar checkpoint')
            else:
                raise ValueError('unsupported resume checkpoint: {}'.format(args.resume))
        elif args.init_checkpoint:
            checkpoint = torch.load(args.init_checkpoint, self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if any(key.endswith('.lora_A') for key in state_dict):
                raise ValueError('--init-checkpoint expects a model without DoRA; '
                                 'use --resume for a full DoRA checkpoint')
            self.model = model_builder(pretrained=False, use_dora=False)
            self.model.load_state_dict(state_dict)
            if args.use_dora:
                self.phase = 'dora'
                self.model.enable_dora(args.dora_rank, args.dora_alpha)
                self.model.train_dora_only()
        else:
            self.model = model_builder(
                pretrained=args.pretrained_backbone,
                use_dora=False,
            )

        self.model.to(self.device)
        self.optimizer = optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=args.lr, weight_decay=args.weight_decay,
        )
        if resume_optimizer_state is not None:
            self.optimizer.load_state_dict(resume_optimizer_state)
        logging.info('Training phase: {}'.format(self.phase))
        logging.info('DoRA enabled: {}'.format(self.model.use_dora))
        logging.info('Trainable parameters: {}'.format(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        ))
        self.set_dataloaders_for_phase()
        if self.phase == 'pretrain' and args.use_dora and args.dora_data_dir:
            # Validate the second dataset before the potentially long pretrain.
            self.build_datasets(args.dora_data_dir)

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        # self.criterion = torch.nn.MSELoss(reduction='sum')
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.save_all = args.save_all
        self.best_count = 0
        self.global_step = (
            checkpoint.get('global_step', self.start_epoch * len(self.dataloaders['train']))
            if args.resume else 0
        )
        if args.tensorboard_log_interval <= 0:
            raise ValueError('tensorboard_log_interval must be greater than zero')
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'))

    def phase_data_dir(self):
        if self.phase == 'dora' and self.args.dora_data_dir:
            return self.args.dora_data_dir
        return self.args.data_dir

    def build_datasets(self, data_dir):
        args = self.args
        datasets = {
            split: Crowd(
                os.path.join(data_dir, split), args.crop_size,
                args.downsample_ratio, args.is_gray, split,
            )
            for split in ('train', 'val')
        }
        for split, dataset in datasets.items():
            if len(dataset) == 0:
                raise ValueError(
                    'no .jpg images in {} dataset: {}'.format(
                        split, os.path.join(data_dir, split)
                    )
                )
        return datasets

    def set_dataloaders_for_phase(self):
        args = self.args
        data_dir = self.phase_data_dir()
        logging.info('{} data directory: {}'.format(self.phase, data_dir))
        self.datasets = self.build_datasets(data_dir)
        self.dataloaders = {
            split: DataLoader(
                self.datasets[split],
                collate_fn=(train_collate if split == 'train' else default_collate),
                batch_size=(args.batch_size if split == 'train' else 1),
                shuffle=(split == 'train'),
                num_workers=args.num_workers * self.device_count,
                pin_memory=(split == 'train'),
            )
            for split in ('train', 'val')
        }

    def train(self):
        """Pretrain the full model, then adapt the best model with DoRA only."""
        args = self.args
        try:
            if self.phase == 'pretrain':
                self.run_phase(args.max_epoch)
                if args.use_dora:
                    self.start_dora_phase()
            if self.phase == 'dora':
                self.run_phase(args.dora_epochs)
        finally:
            self.writer.close()

    def run_phase(self, total_epochs):
        val_start = (
            self.args.dora_val_start
            if self.phase == 'dora' and self.args.dora_val_start is not None
            else self.args.val_start
        )
        for epoch_index in range(self.start_epoch, total_epochs):
            self.epoch_index = epoch_index
            self.epoch = epoch_index + 1
            logging.info(
                '----- {} Epoch {}/{} -----'.format(
                    self.phase, self.epoch, total_epochs
                )
            )
            self.train_eopch()
            if (
                (self.epoch >= val_start
                 and (self.epoch - val_start) % self.args.val_epoch == 0)
                or self.epoch == total_epochs
            ):
                self.val_epoch()
            self.save_checkpoint()

    def start_dora_phase(self):
        if self.best_model_path is None:
            raise RuntimeError('pretrain has no validated best checkpoint')
        logging.info('Loading best pretrain model: {}'.format(self.best_model_path))
        state_dict = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.enable_dora(self.args.dora_rank, self.args.dora_alpha)
        self.model.train_dora_only()
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.args.lr, weight_decay=self.args.weight_decay,
        )
        self.phase = 'dora'
        self.set_dataloaders_for_phase()
        self.start_epoch = 0
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_model_path = None
        self.best_count = 0
        self.save_list = Save_Handle(max_num=self.args.max_model_num)
        logging.info('DoRA-only trainable parameters: {}'.format(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        ))

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_bayesian_loss = AverageMeter()
        epoch_consistency_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()
        if self.phase == 'dora':
            # Parameter freezing does not disable dropout or stochastic depth.
            # The frozen base should also behave deterministically.
            self.model.eval()

        train_bar = tqdm(
            enumerate(self.dataloaders['train']),
            total=len(self.dataloaders['train']),
            desc='{} Train {}/{}'.format(
                self.phase, self.epoch,
                self.args.max_epoch if self.phase == 'pretrain' else self.args.dora_epochs,
            ),
            dynamic_ncols=True,
        )
        for step, (inputs, points, targets, st_sizes) in train_bar:
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                outputs, features = self.model(inputs)
                prob_list = self.post_prob(points, st_sizes)
                bayesian_loss = self.criterion(prob_list, targets, outputs)
                loss_c = outputs.new_zeros(())
                for layer_features in features:
                    # [batch, tokens, channels]. Preserve the original sum over
                    # tokens, but average over the batch so its scale does not
                    # depend on batch size.
                    mean_feature = layer_features.mean(dim=1, keepdim=True)
                    cosine = 1 - F.cosine_similarity(
                        layer_features, mean_feature, dim=2, eps=1e-5
                    )
                    loss_c += cosine.sum(dim=1).mean()
                loss = bayesian_loss + loss_c

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_bayesian_loss.update(bayesian_loss.item(), N)
                epoch_consistency_loss.update(loss_c.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

                if self.global_step % self.args.tensorboard_log_interval == 0:
                    self.writer.add_scalar(
                        '{}/train_step/total_loss'.format(self.phase),
                        loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        '{}/train_step/bayesian_loss'.format(self.phase),
                        bayesian_loss.item(),
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        '{}/train_step/consistency_loss'.format(self.phase),
                        loss_c.item(),
                        self.global_step,
                    )

                train_bar.set_postfix(
                    loss='{:.3f}'.format(loss.item()),
                    bayes='{:.3f}'.format(bayesian_loss.item()),
                    consistency='{:.3f}'.format(loss_c.item()),
                    mae='{:.2f}'.format(np.mean(abs(res))),
                    lr='{:.2e}'.format(self.optimizer.param_groups[0]['lr']),
                )
                self.global_step += 1

        train_rmse = np.sqrt(epoch_mse.get_avg())
        self.writer.add_scalar(
            '{}/train/total_loss'.format(self.phase), epoch_loss.get_avg(), self.epoch
        )
        self.writer.add_scalar(
            '{}/train/bayesian_loss'.format(self.phase),
            epoch_bayesian_loss.get_avg(), self.epoch
        )
        self.writer.add_scalar(
            '{}/train/consistency_loss'.format(self.phase),
            epoch_consistency_loss.get_avg(),
            self.epoch,
        )
        self.writer.add_scalar(
            '{}/train/mae'.format(self.phase), epoch_mae.get_avg(), self.epoch
        )
        self.writer.add_scalar(
            '{}/train/rmse'.format(self.phase), train_rmse, self.epoch
        )
        self.writer.flush()

        logging.info(
            'Epoch {} Train, Loss: {:.2f}, Bayesian Loss: {:.2f}, '
            'Consistency Loss: {:.2f}, RMSE: {:.2f}, MAE: {:.2f}, Cost {:.1f} sec'
            .format(
                self.epoch,
                epoch_loss.get_avg(),
                epoch_bayesian_loss.get_avg(),
                epoch_consistency_loss.get_avg(),
                train_rmse,
                epoch_mae.get_avg(),
                time.time()-epoch_start,
            )
        )
    def save_checkpoint(self):
        save_path = os.path.join(
            self.save_dir, '{}_{}_ckpt.tar'.format(self.phase, self.epoch)
        )
        torch.save({
            'epoch': self.epoch,
            'epoch_index': self.epoch_index,
            'phase': self.phase,
            'model_name': self.args.model_name,
            'use_dora': self.model.use_dora,
            'dora_rank': self.model.dora_rank if self.model.use_dora else None,
            'dora_alpha': self.model.dora_alpha if self.model.use_dora else None,
            'best_mae': self.best_mae,
            'best_mse': self.best_mse,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.model.state_dict(),
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        running_abs_error = 0.0
        running_squared_error = 0.0
        val_bar = tqdm(
            self.dataloaders['val'],
            total=len(self.dataloaders['val']),
            desc='{} Val {}/{}'.format(
                self.phase, self.epoch,
                self.args.max_epoch if self.phase == 'pretrain' else self.args.dora_epochs,
            ),
            dynamic_ncols=True,
        )
        for inputs, count, name in val_bar:
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
                        output = self.model(input)[0]
                        predicted_count += torch.sum(output).item()
            else:
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)[0]
                    predicted_count = torch.sum(outputs).item()

            ground_truth = count[0].item()
            res = ground_truth - predicted_count
            epoch_res.append(res)
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
            )

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.writer.add_scalar(
            '{}/val/mae'.format(self.phase), mae, self.epoch
        )
        self.writer.add_scalar(
            '{}/val/rmse'.format(self.phase), mse, self.epoch
        )
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        logging.info("best mse {:.2f} mae {:.2f}".format(self.best_mse, self.best_mae))
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            self.best_model_path = os.path.join(
                self.save_dir, '{}_best_model.pth'.format(self.phase)
            )
            torch.save(model_state_dic, self.best_model_path)
            if self.save_all:
                torch.save(model_state_dic, os.path.join(
                    self.save_dir,
                    '{}_best_model_{}.pth'.format(self.phase, self.best_count),
                ))
                self.best_count += 1
            if self.phase == 'dora' or not self.args.use_dora:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))

        self.writer.add_scalar(
            '{}/best/mae'.format(self.phase), self.best_mae, self.epoch
        )
        self.writer.add_scalar(
            '{}/best/rmse'.format(self.phase), self.best_mse, self.epoch
        )
        self.writer.flush()
