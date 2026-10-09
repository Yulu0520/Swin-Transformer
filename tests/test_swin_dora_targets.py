import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.utils import parametrize

from models.swin_dora import DORA_TARGETS, add_swin_dora
from models.swin_dual_dora import DualQVDoRAAdapter, add_swin_qv_dual_dora, set_swin_adapter_route, extract_dora_from_dual_state
from test import build_model, build_dual_dora_model


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()


class ToyModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.features = nn.Sequential(Block(), Block())

    def enable_dual_dora(self, rank, alpha, rank1, alpha1, active_route, targets, dora1_targets='qv'):
        return add_swin_qv_dual_dora(
            self.features, rank, alpha, rank1, alpha1, active_route, targets, dora1_targets)

    def set_adapter_route(self, route):
        set_swin_adapter_route(self.features, route)


class SwinTargetsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_all_configs_preserve_initial_weights_and_only_update_selected_targets(self):
        for targets in DORA_TARGETS:
            with self.subTest(targets=targets):
                features = nn.Sequential(Block(), Block())
                before = [(b.attn.qkv.weight.detach().clone(), b.attn.proj.weight.detach().clone()) for b in features]
                for p in features.parameters():
                    p.requires_grad_(False)
                self.assertEqual(add_swin_dora(features, 3, 6, targets), 2)
                trainable = [p for p in features.parameters() if p.requires_grad]
                self.assertEqual(sum(p.numel() for p in trainable), 2 * len(targets) * (2 * 8 * 3 + 8))
                optimizer = torch.optim.SGD(trainable, lr=0.1)
                loss = 0
                for b, (qkv, proj) in zip(features, before):
                    torch.testing.assert_close(b.attn.qkv.weight, qkv)
                    torch.testing.assert_close(b.attn.proj.weight, proj)
                    loss = loss + b.attn.qkv.weight.sum() + b.attn.proj.weight.sum()
                loss.backward()
                self.assertTrue(all(p.grad is not None for p in trainable))
                self.assertTrue(all(p.grad is None for p in features.parameters() if not p.requires_grad))
                optimizer.step()
                for b, (qkv, proj) in zip(features, before):
                    for t, actual, original in zip('qkv', b.attn.qkv.weight.chunk(3), qkv.chunk(3)):
                        if t in targets:
                            self.assertFalse(torch.equal(actual, original))
                        else:
                            self.assertTrue(torch.equal(actual, original))
                    if 'o' in targets:
                        self.assertFalse(torch.equal(b.attn.proj.weight, proj))
                    else:
                        self.assertTrue(torch.equal(b.attn.proj.weight, proj))

    def test_dual_routes_and_checkpoint_loading_for_all_configs(self):
        for targets, targets1 in itertools.product(DORA_TARGETS, repeat=2):
            with self.subTest(targets=targets, targets1=targets1), tempfile.TemporaryDirectory() as directory:
                source = ToyModel()
                before = [(b.attn.qkv.weight.detach().clone(), b.attn.proj.weight.detach().clone()) for b in source.features]
                source.enable_dual_dora(3, 6, 2, 2, 'dora', targets, targets1)
                for p in source.parameters():
                    p.requires_grad_(False)
                for module in source.modules():
                    if isinstance(module, DualQVDoRAAdapter):
                        for p in module.dora.parameters():
                            p.requires_grad_(True)
                        self.assertEqual(module.dora.rank, 3)
                        self.assertEqual(module.dora.scaling, 2)
                loss = sum(b.attn.qkv.weight.sum() + b.attn.proj.weight.sum() for b in source.features)
                loss.backward()
                self.assertTrue(all(p.grad is not None for p in source.parameters() if p.requires_grad))
                self.assertTrue(all(p.grad is None for p in source.parameters() if not p.requires_grad))
                with torch.no_grad():
                    for p in source.parameters():
                        if p.requires_grad:
                            p.add_(0.2)
                expected = [(b.attn.qkv.weight.detach().clone(), b.attn.proj.weight.detach().clone()) for b in source.features]
                source.set_adapter_route('baseline')
                for b, (qkv, proj) in zip(source.features, before):
                    torch.testing.assert_close(b.attn.qkv.weight, qkv)
                    torch.testing.assert_close(b.attn.proj.weight, proj)
                path = Path(directory) / 'weights.pth'
                torch.save(source.state_dict(), path)
                args = SimpleNamespace(model_name='swin_t_trans', dora_targets=targets, dora1_targets=targets1, dora_rank=3, dora_alpha=6, dora1_rank=2, dora1_alpha=2)
                with patch('test.models.swin_t_trans', ToyModel):
                    restored = build_model(args, str(path), True)
                for b, (qkv, proj) in zip(restored.features, expected):
                    torch.testing.assert_close(b.attn.qkv.weight, qkv)
                    torch.testing.assert_close(b.attn.proj.weight, proj)
                # Sidecar metadata must override evaluation defaults.
                from utils.regression_trainer_cosine_multibatch import RegTrainer
                args.stage, args.seed, args.baseline_checkpoint = 'dora', 42, 'baseline.pth'
                trainer = RegTrainer.__new__(RegTrainer)
                trainer.args, trainer.best_model_path = args, str(path)
                trainer.save_best_metadata()
                args.dora_targets, args.dora_rank, args.dora_alpha = 'qv', 4, 4
                args.dora1_targets = 'qv'
                with patch('test.models.swin_t_trans', ToyModel):
                    restored = build_model(args, str(path), True)
                for b, (qkv, proj) in zip(restored.features, expected):
                    torch.testing.assert_close(b.attn.qkv.weight, qkv)
                    torch.testing.assert_close(b.attn.proj.weight, proj)
                source.set_adapter_route('dora1')
                args.stage, args.dual_route = 'dora1', 'dora1'
                args.dora_targets, args.dora_rank, args.dora_alpha = targets, 3, 6
                args.dora1_targets = targets1
                trainer.save_best_metadata()
                with patch('test.models.swin_t_trans', ToyModel):
                    restored = build_dual_dora_model(args, str(path))
                for a, b in zip(source.features, restored.features):
                    torch.testing.assert_close(a.attn.qkv.weight, b.attn.qkv.weight)
                    torch.testing.assert_close(a.attn.proj.weight, b.attn.proj.weight)

    def test_fresh_dora1_preserves_trained_dora_and_updates_only_its_targets(self):
        for targets, targets1 in itertools.product(DORA_TARGETS, repeat=2):
            with self.subTest(targets=targets, targets1=targets1):
                source = ToyModel()
                baseline = [(b.attn.qkv.weight.detach().clone(), b.attn.proj.weight.detach().clone()) for b in source.features]
                # Simulate a DoRA-stage checkpoint with an inactive old Q/V branch.
                source.enable_dual_dora(3, 6, 2, 2, 'dora', targets, 'qv')
                with torch.no_grad():
                    for module in source.modules():
                        if isinstance(module, DualQVDoRAAdapter):
                            for p in module.dora.parameters():
                                p.add_(0.2)
                            for p in module.dora1.parameters():
                                p.add_(0.4)
                expected = [(b.attn.qkv.weight.detach().clone(), b.attn.proj.weight.detach().clone()) for b in source.features]
                restored = ToyModel()
                add_swin_dora(restored.features, 3, 6, targets)
                restored.load_state_dict(extract_dora_from_dual_state(source.state_dict()))
                # DoRA1 can now select different projections and rank/alpha.
                restored.enable_dual_dora(3, 6, 4, 12, 'dora1', targets, targets1)
                for b, (qkv, proj) in zip(restored.features, baseline):
                    torch.testing.assert_close(b.attn.qkv.weight, qkv)
                    torch.testing.assert_close(b.attn.proj.weight, proj)
                for p in restored.parameters():
                    p.requires_grad_(False)
                for module in restored.modules():
                    if isinstance(module, DualQVDoRAAdapter):
                        self.assertEqual(module.dora1.rank, 4)
                        self.assertEqual(module.dora1.scaling, 3)
                        for p in module.dora1.parameters():
                            p.requires_grad_(True)
                trainable = [p for p in restored.parameters() if p.requires_grad]
                self.assertEqual(sum(p.numel() for p in trainable), 2 * len(targets1) * (2 * 8 * 4 + 8))
                optimizer = torch.optim.SGD(trainable, lr=0.1)
                sum(b.attn.qkv.weight.sum() + b.attn.proj.weight.sum() for b in restored.features).backward()
                self.assertTrue(all(p.grad is not None for p in trainable))
                self.assertTrue(all(p.grad is None for p in restored.parameters() if not p.requires_grad))
                optimizer.step()
                for b, (qkv, proj) in zip(restored.features, baseline):
                    for t, actual, original in zip('qkv', b.attn.qkv.weight.chunk(3), qkv.chunk(3)):
                        if t in targets1:
                            self.assertFalse(torch.equal(actual, original))
                        else:
                            self.assertTrue(torch.equal(actual, original))
                    if 'o' in targets1:
                        self.assertFalse(torch.equal(b.attn.proj.weight, proj))
                    else:
                        self.assertTrue(torch.equal(b.attn.proj.weight, proj))
                restored.set_adapter_route('dora')
                for b, (qkv, proj) in zip(restored.features, expected):
                    torch.testing.assert_close(b.attn.qkv.weight, qkv)
                    torch.testing.assert_close(b.attn.proj.weight, proj)

    def test_real_swin_forward_initialization_and_output_projection_gradient(self):
        from models.swin_c_multibatch import swin_t_trans
        with torch.random.fork_rng():
            model = swin_t_trans(pretrained=False).eval()
            image = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                baseline = model(image)[0]
            self.assertEqual(model.enable_dual_dora(
                2, 4, 2, 2, active_route='dora', trainable_route='dora', targets='qkvo'), 12)
            adapted = model(image)[0]
            torch.testing.assert_close(adapted, baseline)
            adapted.sum().backward()
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIn('.dora.', name)
                    self.assertIsNotNone(p.grad, name)
                else:
                    self.assertIsNone(p.grad, name)

    def test_train_cli_and_invalid_settings(self):
        from train import parse_args
        for targets in DORA_TARGETS:
            with patch('sys.argv', ['train.py', '--stage', 'dora', '--dora-targets', 'q', '--dora1-targets', targets.upper(), '--dora-rank', '8', '--dora-alpha', '16']):
                args = parse_args()
            self.assertEqual((args.dora_targets, args.dora1_targets, args.dora_rank, args.dora_alpha), ('q', targets, 8, 16))
        for targets in ('', 'qvo', 'invalid'):
            with self.assertRaises(ValueError):
                add_swin_dora(nn.Sequential(Block()), 2, 2, targets)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
