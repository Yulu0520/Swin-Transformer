"""Fine-tune integration checks using small deterministic models and images."""
import copy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

import test as evaluation
from train import parse_args, run_stage, seed_everything
from models.swin_c_multibatch import swin_t_trans, SwinTransMultiBatch
from utils.finetune_trainer import (
    FineTuneTrainer, backward_routed_batch, load_training_checkpoint,
    ranked_candidates, validate_resume_config,
)


class ToyModel(nn.Module):
    enable_finetune = SwinTransMultiBatch.enable_finetune
    finetune_train_mode = SwinTransMultiBatch.finetune_train_mode

    def __init__(self):
        super().__init__()
        self.backbone_features = nn.Linear(3, 2)
        self.backbone_norm = nn.LayerNorm(2)
        self.dora = nn.Parameter(torch.tensor(.1))
        self.dora1 = nn.Parameter(torch.tensor(.2))
        self.feature_adapter = nn.Linear(2, 2)
        self.encoder = nn.Linear(2, 2)
        self.reg_layer_0 = nn.Linear(2, 1)
        self.route = 'dora1'
        self.calls = []

    def set_adapter_route(self, route):
        self.route = route

    def forward(self, inputs):
        self.calls.append((self.route, len(inputs)))
        x = inputs.mean(dim=(2, 3))
        x = self.backbone_norm(self.backbone_features(x)) + getattr(self, self.route)
        x = self.encoder(self.feature_adapter(x))
        h, w = inputs.shape[-2:]
        density = torch.nn.functional.softplus(self.reg_layer_0(x))
        return density.reshape(-1, 1, 1, 1).expand(-1, 1, h // 16, w // 16), [x[:, None, :]]


def toy_builder(args, path):
    model = ToyModel()
    checkpoint = load_training_checkpoint(path)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    return model


class FineTuneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_swin_freezes_both_adapters_and_updates_only_shared_modules(self):
        model = swin_t_trans(pretrained=False)
        model.enable_dual_dora(2, 2., 3, 3.)
        model.enable_finetune()
        model.finetune_train_mode()
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(('feature_adapter.', 'encoder.', 'reg_layer_0.'))
                            for name in trainable))
        self.assertFalse(model.backbone_features.training)
        self.assertFalse(model.backbone_norm.training)
        self.assertTrue(model.encoder.training)
        with torch.no_grad():
            model.reg_layer_0[-1].bias.fill_(1.)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-7)
        for route in ('dora', 'dora1'):
            model.set_adapter_route(route)
            density, _ = model(torch.randn(1, 3, 64, 64))
            density.sum().backward()
        optimizer.step()
        for name, p in model.named_parameters():
            if name in frozen:
                self.assertIsNone(p.grad)
                self.assertTrue(torch.equal(p, frozen[name]), name)
        self.assertIsNotNone(model.feature_adapter[0].weight.grad)
        self.assertIsNotNone(model.encoder.layers[0].self_attn.in_proj_weight.grad)
        self.assertIsNotNone(model.reg_layer_0[-1].weight.grad)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'finetune.pth'
            torch.save(dict(stage='finetune', model_name='swin_t_trans',
                            adapter_type='parallel_dora_dora1', dora_rank=2,
                            dora_alpha=2., dora1_rank=3, dora1_alpha=3.,
                            model_state_dict=model.state_dict()), path)
            args = SimpleNamespace(model_name='swin_t_trans', dual_route='dora1')
            restored = evaluation.build_dual_dora_model(args, path)
            restored.eval()
            model.eval()
            image = torch.randn(1, 3, 64, 64)
            for route in ('dora', 'dora1'):
                model.set_adapter_route(route)
                restored.set_adapter_route(route)
                with torch.inference_mode():
                    torch.testing.assert_close(model(image)[0], restored(image)[0])

    def test_grouped_gradients_equal_per_image_mean(self):
        torch.manual_seed(4)
        grouped = ToyModel()
        reference = copy.deepcopy(grouped)
        grouped.enable_finetune()
        reference.enable_finetune()
        batch = (torch.randn(3, 3, 32, 32), [torch.empty(0, 2)] * 3,
                 [torch.empty(0)] * 3, torch.tensor([32.] * 3),
                 ('dora1', 'dora', 'dora1'))
        def prob(points, sizes):
            return [None] * len(points)
        def criterion(probabilities, targets, density):
            return density.flatten(1).sum(1).mean()
        backward_routed_batch(grouped, batch, 'cpu', prob, criterion, .005)
        for i, route in enumerate(batch[-1]):
            reference.set_adapter_route(route)
            density, _ = reference(batch[0][i:i+1])
            (density.sum() * 1.005 / 3).backward()
        self.assertEqual(grouped.calls, [('dora', 1), ('dora1', 2)])
        for (name, p), (_, q) in zip(grouped.named_parameters(), reference.named_parameters()):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad, msg=name)

    def test_top_five_includes_epoch_zero_and_stable_ties(self):
        ranking = []
        for epoch in range(8):
            ranking = ranked_candidates(ranking, epoch, 1., 2.)
        self.assertEqual([r['epoch'] for r in ranking], list(range(5)))
        with self.assertRaises(ValueError):
            ranked_candidates(ranking, 8, float('nan'), 0)

    def args(self, root, name, epochs, resume=''):
        with patch('sys.argv', ['train.py', '--stage', 'finetune',
                               '--dora1-checkpoint', str(root / 'source.pth'),
                               '--router-mode', 'filename', '--crop-size', '32',
                               '--batch-size', '2', '--num-workers', '0',
                               '--finetune-train-dir', str(root / 'mix/train'),
                               '--finetune-val-dir', str(root / 'mix/val'),
                               '--save-dir', str(root / name), '--resume', resume,
                               '--finetune-epochs', str(epochs)]):
            return parse_args()

    def fixture(self, root):
        for split in ('train', 'val'):
            directory = root / 'mix' / split
            directory.mkdir(parents=True)
            for index, name in enumerate(('IMG_1', 'night_IMG_1', 'IMG_2')):
                Image.new('RGB', (40, 40), (40 + index * 60, 70, 90)).save(directory / (name + '.jpg'))
                np.save(directory / (name + '.npy'), np.array([[16., 16., 8.]], dtype=np.float32))
        seed_everything(7)
        torch.save(ToyModel().state_dict(), root / 'source.pth')

    @patch('utils.finetune_trainer.torch.cuda.is_available', return_value=False)
    @patch('utils.finetune_trainer.build_dual_dora_model', side_effect=toy_builder)
    def test_resume_matches_uninterrupted_and_comparison_csv(self, builder, cuda):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            full = run_stage(self.args(root, 'full', 6), 'finetune')
            partial = run_stage(self.args(root, 'partial', 2), 'finetune')
            last = str(Path(partial).parent / 'last_checkpoint.tar')
            # Reproduce an interruption after committing last state but before cleanup.
            (Path(last).parent / 'best_epoch_999.pth').write_bytes(b'orphan')
            (Path(last).parent / 'ranking.json').write_text('interrupted write')
            run_stage(self.args(root, 'ignored', 6, last), 'finetune')
            full_state = load_training_checkpoint(Path(full).parent / 'last_checkpoint.tar')
            resumed = load_training_checkpoint(last)
            self.assertEqual(resumed['epoch'], 6)
            self.assertEqual(resumed['ranking'], full_state['ranking'])
            self.assertEqual(len(list(Path(last).parent.glob('best_epoch_*.pth'))), 5)
            for key, value in full_state['model_state_dict'].items():
                torch.testing.assert_close(value, resumed['model_state_dict'][key], rtol=0, atol=0)
            for epoch in ('0', '1', '6'):
                self.assertIn('Fine-tune epoch ' + epoch, (Path(last).parent / 'train.log').read_text())
            config = resumed['config']
            for key in ('lr', 'weight_decay', 'count_loss_weight', 'router_mode', 'train_manifest'):
                altered = copy.deepcopy(config)
                altered[key] = 'different'
                with self.assertRaisesRegex(ValueError, key):
                    validate_resume_config(config, altered)
            args = SimpleNamespace(
                router_mode='filename', finetune_checkpoint=full,
                dora1_checkpoint=str(root / 'source.pth'), compare_before_after=True,
                data_root=str(root), split='val', crop_size=32, downsample_ratio=16,
                num_workers=0, output_csv=str(root / 'summary.csv'),
                routing_csv=str(root / 'routing.csv'))
            with patch('test.build_dual_dora_model', side_effect=toy_builder):
                evaluation.evaluate_finetune_comparison(args, torch.device('cpu'))
            with open(args.output_csv) as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 6)
            self.assertEqual({r['domain'] for r in rows}, {'mix', 'day', 'night'})
            with open(args.routing_csv) as source:
                details = list(csv.DictReader(source))
            self.assertEqual(len(details), 6)
            self.assertEqual({r['model'] for r in details}, {'before', 'after'})
            np.save(root / 'mix/train/IMG_1.npy', np.array([[1., 1., 8.]]))
            with self.assertRaisesRegex(ValueError, 'train_manifest'):
                run_stage(self.args(root, 'ignored', 6, last), 'finetune')

    @patch('utils.finetune_trainer.torch.cuda.is_available', return_value=False)
    @patch('utils.finetune_trainer.build_dual_dora_model', side_effect=toy_builder)
    def test_classifier_is_frozen_and_sees_full_images_before_crop(self, builder, cuda):
        class Router(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.))
                self.shapes = []
            def forward(self, images):
                self.shapes.append(tuple(images.shape))
                return torch.stack((images.mean((1, 2, 3)) * self.weight,
                                    -images.mean((1, 2, 3)) * self.weight), dim=1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            args = self.args(root, 'classifier', 1)
            args.router_mode = 'classifier'
            args.router_checkpoint = str(root / 'router.pth')
            Path(args.router_checkpoint).write_bytes(b'router fixture')
            args.max_epoch, args.lr = 1, args.finetune_lr
            router = Router()
            def checked_evaluate(model, *positional, **keywords):
                self.assertFalse(model.training)
                self.assertFalse(model.encoder.training)
                return evaluation.evaluate(model, *positional, **keywords)
            with patch('utils.finetune_trainer.load_day_night_router', return_value=router), \
                    patch('utils.finetune_trainer.evaluate', side_effect=checked_evaluate):
                trainer = FineTuneTrainer(args)
                trainer.setup()
                trainer.train()
            self.assertFalse(router.training)
            self.assertFalse(router.weight.requires_grad)
            self.assertIsNone(router.weight.grad)
            self.assertEqual(router.weight.item(), 1.)
            self.assertTrue(all(shape == (1, 3, 224, 224) for shape in router.shapes))
            self.assertEqual(len(router.shapes), 9)  # 3 cached train + 2 x 3 val images


if __name__ == '__main__':
    unittest.main()
