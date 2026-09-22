import unittest
import os
import tempfile
from types import SimpleNamespace

import torch
from torch import nn
from PIL import Image

from models.dora import DoRALinear
from models.swin_c_multibatch import swin_t_trans, swin_t_trans_from_state_dict
from utils.regression_trainer_cosine_multibatch import RegTrainer


class DoRALinearTests(unittest.TestCase):
    def test_initial_output_matches_base_and_base_is_frozen(self):
        torch.manual_seed(0)
        base = nn.Linear(16, 24)
        inputs = torch.randn(3, 16)
        expected = base(inputs)
        adapted = DoRALinear(base, rank=4, alpha=4)

        self.assertTrue(torch.allclose(adapted(inputs), expected, atol=1e-6))
        adapted(inputs).sum().backward()
        self.assertIsNone(adapted.base_layer.weight.grad)
        self.assertIsNotNone(adapted.lora_B.grad)
        self.assertIsNotNone(adapted.magnitude.grad)


class SwinDoRATests(unittest.TestCase):
    def test_all_attention_blocks_and_multibatch_output(self):
        torch.manual_seed(0)
        model = swin_t_trans(pretrained=False, use_dora=True).eval()
        per_stage = [
            sum(isinstance(block.attn.qkv, DoRALinear) for block in model.backbone_features[i])
            for i in (1, 3, 5, 7)
        ]
        self.assertEqual(per_stage, [2, 2, 6, 2])
        self.assertEqual(
            sum(isinstance(module, DoRALinear) for module in model.modules()), 24
        )
        self.assertFalse(model.backbone_norm.weight.requires_grad)

        with torch.no_grad():
            density, features = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(density.shape), (2, 1, 4, 4))
        self.assertEqual([tuple(feature.shape) for feature in features],
                         [(2, 4, 512), (2, 4, 512)])

    def test_checkpoint_reconstructs_dora_and_plain_models(self):
        adapted = swin_t_trans(pretrained=False, use_dora=True, dora_rank=4,
                               dora_alpha=12)
        loaded = swin_t_trans_from_state_dict(adapted.state_dict())
        self.assertTrue(loaded.use_dora)
        self.assertEqual(loaded.backbone_features[1][0].attn.qkv.rank, 4)
        self.assertEqual(loaded.backbone_features[1][0].attn.qkv.alpha.item(), 12)

        plain = swin_t_trans(pretrained=False, use_dora=False)
        self.assertFalse(swin_t_trans_from_state_dict(plain.state_dict()).use_dora)

    def test_pretrain_to_dora_only_transition(self):
        plain = swin_t_trans(pretrained=False, use_dora=False)
        self.assertTrue(all(parameter.requires_grad for parameter in plain.parameters()))
        with tempfile.TemporaryDirectory() as tmpdir:
            best_path = os.path.join(tmpdir, 'pretrain_best_model.pth')
            best_weight = plain.backbone_features[1][0].attn.qkv.weight.detach().clone()
            torch.save(plain.state_dict(), best_path)
            with torch.no_grad():
                plain.backbone_features[1][0].attn.qkv.weight.add_(1)

            trainer = RegTrainer.__new__(RegTrainer)
            trainer.args = SimpleNamespace(
                dora_rank=4, dora_alpha=4.0, lr=1e-5, weight_decay=1e-5,
                max_model_num=1,
            )
            trainer.model = plain
            trainer.device = torch.device('cpu')
            trainer.best_model_path = best_path
            loader_phases = []
            trainer.set_dataloaders_for_phase = lambda: loader_phases.append(
                trainer.phase
            )
            trainer.start_dora_phase()

        self.assertEqual(trainer.phase, 'dora')
        self.assertEqual(loader_phases, ['dora'])
        self.assertTrue(torch.equal(
            trainer.model.backbone_features[1][0].attn.qkv.base_layer.weight,
            best_weight,
        ))
        self.assertEqual(trainer.start_epoch, 0)
        self.assertEqual(
            sum(isinstance(module, DoRALinear) for module in trainer.model.modules()),
            24,
        )
        trainable = {
            name for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(len(trainable), 72)
        self.assertTrue(all(name.endswith(
            ('.lora_A', '.lora_B', '.magnitude')
        ) for name in trainable))
        self.assertEqual(
            sum(parameter.numel() for parameter in trainer.model.parameters()
                if parameter.requires_grad),
            123648,
        )
        optimizer_ids = {
            id(parameter) for group in trainer.optimizer.param_groups
            for parameter in group['params']
        }
        self.assertEqual(
            optimizer_ids,
            {id(parameter) for parameter in trainer.model.parameters()
             if parameter.requires_grad},
        )

    def test_last_pretrain_epoch_is_always_validated(self):
        trainer = RegTrainer.__new__(RegTrainer)
        trainer.args = SimpleNamespace(val_start=5, val_epoch=5)
        trainer.phase = 'pretrain'
        trainer.start_epoch = 0
        events = []
        trainer.train_eopch = lambda: events.append(('train', trainer.epoch))
        trainer.val_epoch = lambda: events.append(('val', trainer.epoch))
        trainer.save_checkpoint = lambda: events.append(('save', trainer.epoch))

        trainer.run_phase(2)
        self.assertEqual(events, [
            ('train', 1), ('save', 1),
            ('train', 2), ('val', 2), ('save', 2),
        ])

    def test_dora_has_independent_validation_start(self):
        trainer = RegTrainer.__new__(RegTrainer)
        trainer.args = SimpleNamespace(
            val_start=5, dora_val_start=2, val_epoch=5
        )
        trainer.phase = 'dora'
        trainer.start_epoch = 0
        validated = []
        trainer.train_eopch = lambda: None
        trainer.val_epoch = lambda: validated.append(trainer.epoch)
        trainer.save_checkpoint = lambda: None

        trainer.run_phase(10)
        self.assertEqual(validated, [2, 7, 10])

    def test_each_phase_selects_its_own_data_dir(self):
        trainer = RegTrainer.__new__(RegTrainer)
        trainer.args = SimpleNamespace(
            data_dir='/data/day', dora_data_dir='/data/night'
        )
        trainer.phase = 'pretrain'
        self.assertEqual(trainer.phase_data_dir(), '/data/day')
        trainer.phase = 'dora'
        self.assertEqual(trainer.phase_data_dir(), '/data/night')
        trainer.args.dora_data_dir = None
        self.assertEqual(trainer.phase_data_dir(), '/data/day')

    def test_dora_dataloaders_use_separate_train_and_val_folders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dora_dir = os.path.join(tmpdir, 'night')
            for split in ('train', 'val'):
                split_dir = os.path.join(dora_dir, split)
                os.makedirs(split_dir)
                Image.new('RGB', (16, 16)).save(
                    os.path.join(split_dir, 'sample.jpg')
                )
            trainer = RegTrainer.__new__(RegTrainer)
            trainer.args = SimpleNamespace(
                data_dir=os.path.join(tmpdir, 'day'), dora_data_dir=dora_dir,
                crop_size=256, downsample_ratio=16, is_gray=False,
                batch_size=2, num_workers=0,
            )
            trainer.device_count = 1
            trainer.phase = 'dora'
            trainer.set_dataloaders_for_phase()
            self.assertEqual(
                trainer.datasets['train'].root_path, os.path.join(dora_dir, 'train')
            )
            self.assertEqual(
                trainer.datasets['val'].root_path, os.path.join(dora_dir, 'val')
            )
            self.assertEqual(trainer.dataloaders['train'].batch_size, 2)
            self.assertEqual(trainer.dataloaders['val'].batch_size, 1)


if __name__ == '__main__':
    unittest.main()
