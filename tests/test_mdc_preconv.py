"""Paper-style MDC branches, architecture compatibility and real stage handoff."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import train
from models.cctrans_regression import CCTransRegression
from models.counting_checkpoint import (ARCHITECTURE, LEGACY_ARCHITECTURE,
    checkpoint_mdc_mode, validate_counting_checkpoint, validate_model_mdc)
from models.swin_c_multibatch import swin_t_trans
from utils.counting_model import build_model, build_dual_dora_model, load_checkpoint
from utils.counting_visualization import load_visualization_model
from utils.topk_checkpoint import TopKCheckpoints


class MDCPreconvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_spatial_sizes_branch_gradients_and_parameter_cost(self):
        torch.manual_seed(9)
        head = CCTransRegression((4, 8, 16)).train()
        legacy = CCTransRegression((4, 8, 16), mdc_mode='legacy')
        delta = sum(p.numel() for p in head.parameters()) - sum(p.numel() for p in legacy.parameters())
        self.assertEqual(delta, 2296064)
        for index, kernel in enumerate((1, 3, 5), 1):
            convolution = getattr(head, 'pre' + str(index))[0]
            self.assertEqual(convolution.kernel_size, (kernel, kernel))
            self.assertEqual(convolution.padding, (kernel // 2, kernel // 2))
            self.assertEqual((convolution.in_channels, convolution.out_channels), (256, 256))
        with torch.no_grad():
            head.res[3].bias.fill_(1.)
        inputs = [torch.randn(2, channels, height, width, requires_grad=True)
                  for channels, height, width in [(4, 8, 10), (8, 4, 5), (16, 2, 3)]]
        seen = {}
        handles = [getattr(head, 'pre' + str(index)).register_forward_hook(
            lambda module, args, output, index=index: seen.update({index: tuple(output.shape)}))
            for index in (1, 2, 3)]
        try:
            output = head(*inputs)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(output.shape, (2, 1, 8, 10))
        self.assertEqual(seen, {index: (2, 256, 8, 10) for index in (1, 2, 3)})
        output.sum().backward()
        for index in (1, 2, 3):
            self.assertGreater(getattr(head, 'pre' + str(index))[0].weight.grad.abs().sum().item(), 0.)

    def test_old_and_new_raw_metadata_dual_and_visualization_loaders(self):
        args = SimpleNamespace(model_name='swin_t_trans', dora_rank=4, dora_alpha=4.,
                               dora1_rank=4, dora1_alpha=4., dual_route='dora1')
        inputs = torch.randn(1, 3, 64, 64)
        for mode, architecture in [('legacy', LEGACY_ARCHITECTURE), ('preconv', ARCHITECTURE)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = swin_t_trans(pretrained=False, mdc_mode=mode).eval()
                raw = root / 'baseline.pth'
                torch.save(source.state_dict(), raw)
                metadata = dict(architecture=architecture, mdc_mode=mode, output_stride=8,
                    stage='baseline', model_name=args.model_name, model_state_dict=source.state_dict())
                tar = root / 'baseline.tar'
                torch.save(metadata, tar)
                for path in (raw, tar):
                    restored = build_model(args, str(path), expect_dora=False).eval()
                    self.assertEqual(restored.mdc_mode, mode)
                    with torch.no_grad():
                        torch.testing.assert_close(restored(inputs), source(inputs))
                    del restored
                source.enable_dual_dora()
                metadata.update(stage='finetune', adapter_type='parallel_dora_dora1',
                    dora_rank=4, dora_alpha=4., dora1_rank=4, dora1_alpha=4.,
                    model_state_dict=source.state_dict())
                torch.save(metadata, tar)
                restored = build_dual_dora_model(args, str(tar)).eval()
                self.assertEqual(restored.mdc_mode, mode)
                del restored
                ranked = TopKCheckpoints(root / 'ranked')
                ranked.consider(source.state_dict(), 1., 2., 0)
                args.model_path = str(root / 'ranked')
                restored = load_visualization_model(args, torch.device('cpu'))
                self.assertEqual(restored.mdc_mode, mode)
                with torch.no_grad():
                    torch.testing.assert_close(restored(inputs), source(inputs))
                del restored
                opposite = 'legacy' if mode == 'preconv' else 'preconv'
                target = swin_t_trans(pretrained=False, mdc_mode=opposite)
                with self.assertRaisesRegex(ValueError, '--mdc-mode'):
                    validate_model_mdc(target, metadata)
                metadata['architecture'] = ARCHITECTURE if mode == 'legacy' else LEGACY_ARCHITECTURE
                with self.assertRaisesRegex(ValueError, 'MDC weight keys'):
                    validate_counting_checkpoint(metadata)

    def test_partial_preconv_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            checkpoint_mdc_mode({'regression.pre1.0.weight': torch.ones(1)})

    def test_cli_constant_lr_defaults_and_invalid_values(self):
        flags = ['train.py', '--stage', 'finetune', '--finetune-checkpoint', 'dual.pth',
                 '--finetune-router-mode', 'filename', '--finetune-epochs', '1']
        with patch('sys.argv', flags):
            args = train.parse_args()
        self.assertEqual((args.mdc_mode, args.finetune_lr, args.weight_decay), ('preconv', 1e-7, 1e-5))
        self.assertFalse(hasattr(args, 'finetune_warmup_epochs'))
        for option in [['--finetune-lr', '0'], ['--finetune-lr', 'nan'],
                       ['--weight-decay', '-1'], ['--weight-decay', 'inf'],
                       ['--finetune-epochs', '0'], ['--finetune-warmup-epochs', '5']]:
            with self.subTest(option=option), patch('sys.argv', flags + option), patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    train.parse_args()

    def test_real_four_stage_pipeline_with_preconv_head_and_fixed_lr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for domain in ('day', 'night', 'mix'):
                for split in ('train', 'val'):
                    folder = root / domain / split
                    folder.mkdir(parents=True)
                    for index in (1, 2):
                        name = ('night_' if domain == 'night' or domain == 'mix' and index == 2 else '') + 'IMG_' + str(index)
                        Image.new('RGB', (64, 64), (120, 90 + index * 10, 80)).save(folder / (name + '.jpg'))
                        np.save(folder / (name + '.npy'), [[24., 28.]])
            flags = ['train.py', '--stage', 'all', '--finetune-after-all',
                '--finetune-router-mode', 'filename', '--no-pretrained-backbone',
                '--device', 'cpu', '--crop-size', '64', '--batch-size', '2', '--num-workers', '0',
                '--pretrain-epochs', '1', '--dora-epochs', '1', '--dora1-epochs', '1',
                '--finetune-epochs', '1', '--save-dir', str(root / 'models'),
                '--data-dir', str(root / 'day'), '--dora-train-dir', str(root / 'night' / 'train'),
                '--dora-val-dir', str(root / 'night' / 'val'),
                '--dora1-train-dir', str(root / 'night' / 'train'),
                '--dora1-val-dir', str(root / 'day' / 'val'),
                '--finetune-train-dir', str(root / 'mix' / 'train'),
                '--finetune-val-dir', str(root / 'mix' / 'val')]
            with patch('sys.argv', flags):
                args = train.parse_args()
            paths = train.run_pipeline(args)
            self.assertEqual(len(paths), 4)
            for path in paths:
                self.assertEqual(checkpoint_mdc_mode(load_checkpoint(path)), 'preconv')
            snapshot = torch.load(Path(paths[-1]).parent / '1_ckpt.tar', map_location='cpu', weights_only=False)
            self.assertEqual(snapshot['architecture'], ARCHITECTURE)
            self.assertEqual(snapshot['mdc_mode'], 'preconv')
            group = snapshot['optimizer_state_dict']['param_groups'][0]
            self.assertEqual((group['lr'], group['weight_decay']), (1e-7, 1e-5))
            self.assertIsNone(snapshot['scheduler_state_dict'])
