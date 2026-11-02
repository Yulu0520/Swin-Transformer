"""Integration checks for multiscale regression, Bayesian training and routing."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from losses.bay_loss import Bay_Loss
from losses.count_loss import count_l1_loss
from losses.post_prob import Post_Prob
from models.counting_checkpoint import (
    ARCHITECTURE, OUTPUT_STRIDE, validate_counting_checkpoint,
)
from models.swin_c_multibatch import swin_l_trans, swin_t_trans
from models.swin_dual_dora import DualQVDoRAAdapter
from utils.regression_trainer_cosine_multibatch import RegTrainer
from utils.helper import Save_Handle
from utils.counting_model import build_dual_dora_model, build_model, load_checkpoint, load_backbone_weights


class CCTransRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_tiny_stages_and_native_stride_for_training_and_odd_images(self):
        model = swin_t_trans(pretrained=False).eval()
        self.assertFalse(hasattr(model, 'encoder'))
        self.assertFalse(hasattr(model, 'feature_adapter'))
        with torch.inference_mode():
            inputs = torch.randn(1, 3, 384, 384)
            stages = model.forward_backbone(inputs)
            self.assertEqual([tuple(x.shape) for x in stages], [
                (1, 192, 48, 48), (1, 384, 24, 24), (1, 768, 12, 12),
            ])
            output = model(inputs)
            self.assertEqual(output.shape, (1, 1, 48, 48))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue((output >= 0).all())
            for h, w in [(65, 83), (72, 88)]:
                output = model(torch.randn(1, 3, h, w))
                self.assertEqual(output.shape, (1, 1, math.ceil(h / 8), math.ceil(w / 8)))

    def test_large_backbone_and_head_channels(self):
        model = swin_l_trans(pretrained=False).eval()
        self.assertEqual([model.regression.v1[0].in_channels,
                          model.regression.v2[0].in_channels,
                          model.regression.v3[0].in_channels], [384, 768, 1536])
        with torch.inference_mode():
            output = model(torch.randn(1, 3, 65, 83))
        self.assertEqual(output.shape, (1, 1, 9, 11))
        self.assertTrue(torch.isfinite(output).all())

    def test_bayesian_and_fractional_count_loss_reach_all_stages_and_head(self):
        model = swin_t_trans(pretrained=False).train()
        with torch.no_grad():
            model.regression.res[3].bias.fill_(0.1)
        stages = model.forward_backbone(torch.randn(2, 3, 64, 64))
        for stage in stages:
            stage.retain_grad()
        output = model.regression(*stages)
        points = [torch.tensor([[24., 28.], [50., 50.]]), torch.empty(0, 2)]
        targets = [torch.tensor([1., 0.4]), torch.empty(0)]
        post = Post_Prob(8., 64, 8, 0.15, True, torch.device('cpu'))
        bayesian = Bay_Loss(True, torch.device('cpu'))(post(points, torch.tensor([64., 64.])),
                                                     targets, output)
        loss = bayesian + 0.1 * count_l1_loss(output, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for stage in stages:
            self.assertGreater(stage.grad.abs().sum().item(), 0)
        for module in [model.regression.v1, model.regression.v2, model.regression.v3,
                       model.regression.stage1, model.regression.stage2,
                       model.regression.stage3, model.regression.stage4, model.regression.res]:
            self.assertGreater(module[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone_features[0][0].weight.grad.abs().sum().item(), 0)

    def test_dual_routes_freeze_head_and_batchnorm_and_receive_gradients(self):
        model = swin_t_trans(pretrained=False).eval()
        with torch.no_grad():
            model.regression.res[3].bias.fill_(0.1)
        inputs = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            baseline = model(inputs)
        self.assertEqual(model.enable_dual_dora(active_route='dora', trainable_route='dora'), 12)
        batchnorms = [m for m in model.regression.modules() if isinstance(m, nn.BatchNorm2d)]
        means = [m.running_mean.clone() for m in batchnorms]
        with torch.inference_mode():
            for route in ['baseline', 'dora', 'dora1']:
                model.set_adapter_route(route)
                torch.testing.assert_close(model(inputs), baseline)
        for route in ['dora', 'dora1']:
            # Same policy as the trainer: eval mode, gradients only on active route.
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.set_adapter_route(route)
            for module in model.backbone_features.modules():
                if isinstance(module, DualQVDoRAAdapter):
                    for parameter in getattr(module, route).parameters():
                        parameter.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            model(inputs).sum().backward()
            trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            self.assertTrue(trainable)
            self.assertTrue(all('.{}.'.format(route) in name for name, _ in trainable))
            self.assertTrue(all(p.grad is not None for _, p in trainable))
            self.assertTrue(any(p.grad.abs().sum() > 0 for _, p in trainable))
            self.assertTrue(all(p.grad is None for p in model.regression.parameters()))
        for module, mean in zip(batchnorms, means):
            torch.testing.assert_close(module.running_mean, mean)

    def test_checkpoint_round_trip_and_inference_builders(self):
        source = swin_t_trans(pretrained=False).eval()
        args = SimpleNamespace(model_name='swin_t_trans', dora_rank=4, dora_alpha=4.,
                               dora1_rank=4, dora1_alpha=4., dual_route='dora1')
        inputs = torch.randn(1, 3, 64, 64)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'baseline.pth')
            torch.save(source.state_dict(), path)
            restored = build_model(args, path, expect_dora=False).eval()
            with torch.inference_mode():
                torch.testing.assert_close(source(inputs), restored(inputs))
        source.enable_dual_dora(active_route='dora')
        metadata = dict(architecture=ARCHITECTURE, output_stride=OUTPUT_STRIDE,
                        model_name=args.model_name, stage='dora',
                        adapter_type='parallel_dora_dora1', dora_rank=4, dora_alpha=4.,
                        dora1_rank=4, dora1_alpha=4., model_state_dict=source.state_dict())
        validate_counting_checkpoint(metadata)
        with patch('utils.counting_model.load_checkpoint', return_value=metadata):
            restored = build_model(args, 'unused.tar', expect_dora=True)
        self.assertTrue(all(m.active_route == 'dora' for m in restored.modules()
                            if isinstance(m, DualQVDoRAAdapter)))
        metadata['stage'] = 'dora1'
        with patch('utils.counting_model.load_checkpoint', return_value=metadata):
            restored = build_dual_dora_model(args, 'unused.tar')
        self.assertTrue(all(m.active_route == 'dora1' for m in restored.modules()
                            if isinstance(m, DualQVDoRAAdapter)))

    def test_explicit_old_backbone_reuse_does_not_load_old_head(self):
        source = swin_t_trans(pretrained=False)
        state = {key: value for key, value in source.state_dict().items()
                 if key.startswith(('backbone_features.', 'backbone_norm.'))}
        state['encoder.old_head'] = torch.ones(1)
        target = swin_t_trans(pretrained=False)
        head_before = target.regression.v1[0].weight.detach().clone()
        with patch('utils.counting_model.read_checkpoint', return_value=state):
            load_backbone_weights(target, 'old_baseline.pth')
        torch.testing.assert_close(target.backbone_features[0][0].weight,
                                   source.backbone_features[0][0].weight)
        torch.testing.assert_close(target.regression.v1[0].weight, head_before)
        state['backbone_features.adapter.parametrizations.weight.original'] = torch.ones(1)
        with patch('utils.counting_model.read_checkpoint', return_value=state):
            with self.assertRaisesRegex(ValueError, 'without adapters'):
                load_backbone_weights(target, 'old_adapter.pth')

    def test_training_loop_checkpoint_metadata_and_adapter_batchnorm(self):
        # Run the real training loop on CPU, bypassing only GPU/dataset setup.
        model = swin_t_trans(pretrained=False)
        with torch.no_grad():
            model.regression.res[3].bias.fill_(0.1)
        inputs = torch.randn(2, 3, 64, 64)
        points = [torch.tensor([[24., 28.]]), torch.empty(0, 2)]
        targets = [torch.tensor([0.4]), torch.empty(0)]
        trainer = object.__new__(RegTrainer)
        trainer.model = model
        trainer.device = torch.device('cpu')
        trainer.args = SimpleNamespace(stage='baseline', max_epoch=1,
            tensorboard_log_interval=1, count_loss_weight=0.1,
            model_name='swin_t_trans', crop_size=64, dora_rank=4, dora_alpha=4.,
            dora1_rank=4, dora1_alpha=4., dora_checkpoint='', seed=42,
            dual_route='dora1')
        trainer.epoch, trainer.epoch_index, trainer.global_step = 1, 0, 0
        trainer.dataloaders = {'train': [(inputs, points, targets, torch.tensor([64., 64.]))]}
        trainer.post_prob = Post_Prob(8., 64, 8, 0.15, True, trainer.device)
        trainer.criterion = Bay_Loss(True, trainer.device)
        trainer.writer = unittest.mock.Mock()
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
        trainer.train_eopch()
        scalars = {call.args[0]: call.args[1] for call in trainer.writer.add_scalar.call_args_list}
        self.assertAlmostEqual(scalars['train/total_loss'],
            scalars['train/bayesian_loss'] + scalars['train/weighted_count_loss'], places=4)
        self.assertFalse(any('consistency' in key for key in scalars))
        for route in ['dora', 'dora1']:
            if route == 'dora':
                model.enable_dual_dora(active_route=route, trainable_route=route)
            else:
                model.set_adapter_route(route)
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                for module in model.modules():
                    if isinstance(module, DualQVDoRAAdapter):
                        for parameter in module.dora1.parameters():
                            parameter.requires_grad_(True)
            trainer.args.stage = route
            trainer.optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=1e-6)
            bn = model.regression.v1[1]
            before = bn.running_mean.clone()
            trainer.train_eopch()
            self.assertFalse(model.training)
            torch.testing.assert_close(bn.running_mean, before)
        trainer.best_mae, trainer.best_mse = float('inf'), float('inf')
        trainer.best_model_path, trainer.best_count = None, 0
        trainer.save_list = Save_Handle(max_num=1)
        with tempfile.TemporaryDirectory() as directory:
            trainer.save_dir = directory
            trainer.save_epoch()
            saved = load_checkpoint(str(Path(directory) / '1_ckpt.tar'))
            self.assertEqual(saved['architecture'], ARCHITECTURE)
            self.assertEqual(saved['output_stride'], 8)
            self.assertEqual(saved['stage'], 'dora1')
            restored = build_dual_dora_model(trainer.args, str(Path(directory) / '1_ckpt.tar')).eval()
            with torch.inference_mode():
                torch.testing.assert_close(model(inputs), restored(inputs))

    def test_rejects_old_man_and_wrong_architecture_or_stride(self):
        with self.assertRaisesRegex(ValueError, 'Legacy MAN'):
            validate_counting_checkpoint({'encoder.layers.0.weight': torch.ones(1)})
        state = {'regression.v1.0.weight': torch.ones(1), 'regression.res.3.weight': torch.ones(1)}
        for changes in [dict(architecture='old'), dict(output_stride=16)]:
            checkpoint = dict(architecture=ARCHITECTURE, output_stride=8, model_state_dict=state)
            checkpoint.update(changes)
            with self.assertRaises(ValueError):
                validate_counting_checkpoint(checkpoint)


if __name__ == '__main__':
    unittest.main()
