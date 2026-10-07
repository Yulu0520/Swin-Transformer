import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
from torch import nn

from models.swin_c_multibatch import swin_t_trans
from models.swin_checkpoint import load_single_dora, read_checkpoint, restore_swin_weights
from models.swin_dora import LinearDoRA, QVDoRA
from models.swin_dual_dora import DualQVDoRAAdapter
from utils.regression_trainer_cosine_multibatch import RegTrainer
import test as evaluation
import train


class SwinMLPDoRATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_rectangular_weights_preserve_output_and_learn_magnitude(self):
        for out_dim, in_dim in ((32, 8), (8, 32)):
            weight = torch.randn(out_dim, in_dim)
            adapter = LinearDoRA(weight, 2, 4.0)
            torch.testing.assert_close(adapter(weight), weight)
            with torch.no_grad():
                adapter.dora_B.normal_()
                adapter.magnitude.add_(0.25)
            torch.testing.assert_close(
                adapter(weight).norm(dim=1), adapter.magnitude)
        zero = torch.zeros(8, 32)
        self.assertTrue(torch.isfinite(LinearDoRA(zero, 2, 2.0)(zero)).all())

    def test_real_swin_t_initial_output_counts_and_gradients(self):
        model = swin_t_trans(pretrained=False).eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            baseline = model(images)
        self.assertEqual(model.enable_dora(targets='qv_mlp'), 12)
        self.assertEqual(sum(isinstance(m, LinearDoRA) for m in model.modules()), 24)
        self.assertEqual(sum(isinstance(m, QVDoRA) for m in model.modules()), 12)
        self.assertFalse(any(isinstance(m, DualQVDoRAAdapter) for m in model.modules()))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 278208)
        outputs, features = model(images)
        torch.testing.assert_close(outputs, baseline[0], atol=1e-5, rtol=1e-5)
        for actual, expected in zip(features, baseline[1]):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        loss = outputs.sum() + sum(feature.square().mean() for feature in features)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=2e-4)
        original = {name: p.detach().clone() for name, p in model.named_parameters()
                    if not p.requires_grad}
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in original:
                torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
        self.assertTrue(any(m.dora_B.abs().sum() > 0 for m in model.modules()
                            if isinstance(m, LinearDoRA)))

    def test_mlp_only_training_freezes_qv(self):
        model = swin_t_trans(pretrained=False)
        model.enable_dora(targets='qv_mlp', train_qv=False)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 198720)
        for module in model.modules():
            if isinstance(module, QVDoRA):
                self.assertFalse(any(p.requires_grad for p in module.parameters()))
            elif isinstance(module, LinearDoRA):
                self.assertTrue(all(p.requires_grad for p in module.parameters()))

    def test_dual_qv_migration_preserves_trained_route_and_loaded_mlp(self):
        source = swin_t_trans(pretrained=False).eval()
        source.enable_dual_dora(active_route='dora')
        with torch.no_grad():
            for module in source.modules():
                if isinstance(module, DualQVDoRAAdapter):
                    module.dora.dora_B_q.normal_(std=0.01)
                    module.dora.magnitude_v.add_(0.1)
                elif isinstance(module, nn.Linear):
                    # Ensure MLP norms must come from the loaded weights.
                    if module.out_features == 4 * module.in_features:
                        module.weight.mul_(2)
        restored = swin_t_trans(pretrained=False).eval()
        config = load_single_dora(restored, source.state_dict(), targets='qv_mlp')
        self.assertEqual(config['adapter_targets'], 'qv_mlp')
        self.assertFalse(any('dora1' in key for key in restored.state_dict()))
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected, expected_features = source(images)
            actual, actual_features = restored(images)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for actual, expected in zip(actual_features, expected_features):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_new_checkpoint_round_trip_restores_nondefault_scaling(self):
        source = swin_t_trans(pretrained=False).eval()
        source.enable_dora(2, 6, targets='qv_mlp', mlp_rank=3, mlp_alpha=9)
        with torch.no_grad():
            for module in source.modules():
                if isinstance(module, LinearDoRA):
                    module.dora_B.normal_(std=0.01)
        trainer = object.__new__(RegTrainer)
        trainer.single_dora = True
        trainer.args = SimpleNamespace(
            adapter_targets='qv_mlp', dora_rank=2, dora_alpha=6,
            mlp_dora_rank=3, mlp_dora_alpha=9, freeze_qv_dora=False,
            model_name='swin_t_trans', crop_size=384)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'best_model.pth'
            torch.save(trainer.best_checkpoint(source.state_dict()), checkpoint_path)
            args = SimpleNamespace(model_name='swin_t_trans', dora_rank=4, dora_alpha=4,
                                   mlp_dora_rank=4, mlp_dora_alpha=4)
            restored = evaluation.build_model(args, str(checkpoint_path), True).eval()
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(restored(images)[0], source(images)[0])
        self.assertTrue(all(m.scaling == 3 for m in restored.modules()
                            if isinstance(m, (QVDoRA, LinearDoRA))))

    def test_incomplete_checkpoint_and_dropping_mlp_are_rejected(self):
        source = swin_t_trans(pretrained=False)
        source.enable_dora(targets='qv_mlp')
        state = source.state_dict()
        with self.assertRaisesRegex(ValueError, 'targets do not match'):
            load_single_dora(swin_t_trans(pretrained=False),
                             {'model_state_dict': state, 'adapter_targets': 'qv'})
        with self.assertRaisesRegex(ValueError, 'drop trained MLP'):
            load_single_dora(swin_t_trans(pretrained=False), state, targets='qv')
        state.pop(next(key for key in state if key.endswith('.magnitude')))
        with self.assertRaises(RuntimeError):
            load_single_dora(swin_t_trans(pretrained=False), state)

    def test_legacy_epoch_numpy_metadata_can_be_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old_ckpt.tar'
            torch.save({'best_mae': np.float64(2.5),
                        'model_state_dict': {'weight': torch.ones(2)}}, path)
            checkpoint = read_checkpoint(path)
        self.assertEqual(checkpoint['best_mae'], 2.5)

    def test_shared_visualization_loader_preserves_baseline(self):
        source = swin_t_trans(pretrained=False)
        restored = swin_t_trans(pretrained=False)
        self.assertIsNone(restore_swin_weights(restored, source.state_dict()))
        for key, value in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)
        with self.assertRaisesRegex(ValueError, 'model name'):
            restore_swin_weights(restored, {'model_name': 'swin_l_trans'})

    def test_cli_defaults_and_disabled_routes(self):
        with patch('sys.argv', ['train.py']):
            args = train.parse_args()
        self.assertEqual((args.model_name, args.stage, args.adapter_mode, args.adapter_targets),
                         ('swin_t_trans', 'dora', 'single', 'qv_mlp'))
        with patch('sys.argv', ['test.py']):
            args = evaluation.parse_args()
        self.assertEqual((args.evaluation_mode, args.router_mode), ('single', 'fixed'))
        for argv, parser in ((['train.py', '--stage', 'dora1'], train.parse_args),
                             (['test.py', '--router-mode', 'classifier'], evaluation.parse_args)):
            with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser()

    def test_single_evaluation_never_loads_router_or_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / 'summary.csv')
            argv = ['test.py', '--dora-checkpoint', 'chosen.pth',
                    '--domains', 'mix', '--output-csv', output]
            with patch('sys.argv', argv), patch.object(evaluation, 'resolve_checkpoint', return_value='chosen.pth') as resolve, \
                    patch.object(evaluation, 'build_model') as build, \
                    patch.object(evaluation, 'load_day_night_router') as router, \
                    patch.object(evaluation, 'evaluate', return_value={'images': 1, 'mae': 2.0, 'rmse': 3.0}), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.main()
            router.assert_not_called()
            self.assertEqual(resolve.call_count, 1)
            build.assert_called_once_with(unittest.mock.ANY, 'chosen.pth', expect_dora=True)
            self.assertIn('single_dora,mix', Path(output).read_text())


if __name__ == '__main__':
    unittest.main()
