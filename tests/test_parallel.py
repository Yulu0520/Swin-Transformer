"""Behavioral checks for day bypass, night fusion, freezing and stage recovery."""

import csv
import gc
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
import train
from models.swin_c_multibatch import swin_t_trans
from utils.finetune_trainer import backward_routed_batch, load_training_checkpoint
from utils.parallel_trainer import ParallelTrainer


def model_args():
    return SimpleNamespace(model_name='swin_t_trans', dual_route='dora1',
                           dora_rank=2, dora_alpha=2., dora1_rank=3, dora1_alpha=3.)


class ParallelModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(10)
        model = swin_t_trans(pretrained=False)
        model.enable_dual_dora(2, 2., 3, 3.)
        with torch.no_grad():
            model.reg_layer_0[-1].bias.fill_(1.)
        return model

    def test_day_is_identical_and_never_calls_night_modules(self):
        model = self.model().eval()
        image = torch.randn(2, 3, 65, 83)
        with torch.inference_mode():
            original, features = model(image)
        model.enable_night_fusion()
        model.eval()
        calls = []
        hooks = [module.register_forward_hook(lambda *args: calls.append(1))
                 for module in (model.night_branch, model.night_fusion)]
        try:
            with torch.inference_mode():
                day, new_features = model(image)
            torch.testing.assert_close(day, original, rtol=0, atol=0)
            for old, new in zip(features, new_features):
                torch.testing.assert_close(new, old, rtol=0, atol=0)
            self.assertEqual(calls, [])
            model.set_adapter_route('dora')
            stage_shapes = []
            hooks.append(model.night_branch.register_forward_pre_hook(
                lambda module, inputs: stage_shapes.extend(tuple(x.shape) for x in inputs)))
            with torch.inference_mode():
                night, _ = model(image)
            self.assertEqual(calls, [1, 1])
            self.assertEqual([shape[1] for shape in stage_shapes], [192, 384])
            self.assertEqual(night.shape, original.shape)
            self.assertTrue(torch.isfinite(night).all())
        finally:
            for hook in hooks:
                hook.remove()

    def test_zero_residual_recovers_old_night_and_checkpoint_roundtrip(self):
        model = self.model().eval()
        model.set_adapter_route('dora')
        image = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            original = model(image)[0]
        model.enable_night_fusion(alpha=0.)
        model.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(image)[0], original, rtol=0, atol=0)
        with torch.no_grad():
            model.night_fusion.alpha.fill_(.2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pth'
            torch.save(model.state_dict(), path)
            restored = evaluation.build_dual_dora_model(model_args(), path).eval()
            for route in ('dora', 'dora1'):
                model.set_adapter_route(route)
                restored.set_adapter_route(route)
                with torch.inference_mode():
                    torch.testing.assert_close(restored(image)[0], model(image)[0])
            state = model.state_dict()
            del state['night_fusion.project.weight']
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                evaluation.build_dual_dora_model(model_args(), path)

    def test_fusion_loss_updates_new_modules_through_frozen_regression(self):
        model = self.model()
        model.enable_night_fusion()
        model.enable_parallel_training('fusion')
        model.parallel_train_mode('fusion')
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        batch = (torch.randn(2, 3, 64, 64), [torch.empty(0, 2)] * 2,
                 [torch.empty(0)] * 2, torch.tensor([64., 64.]), ('dora', 'dora'))
        def probabilities(points, sizes):
            return [None] * len(points)
        def criterion(probabilities, targets, density):
            return density.flatten(1).sum(1).mean()
        values = backward_routed_batch(model, batch, 'cpu', probabilities, criterion, .005)
        self.assertTrue(np.isfinite(values['total']))
        for module in (model.night_branch.v2, model.night_branch.v3,
                       model.night_fusion.project, model.night_fusion.update):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertIsNotNone(model.night_fusion.alpha.grad)
        optimizer.step()
        for name, p in model.named_parameters():
            if name in frozen:
                self.assertIsNone(p.grad, name)
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.reg_layer_0.training)

    def test_joint_updates_only_crowd_aware_and_regression_for_both_routes(self):
        model = self.model()
        model.enable_night_fusion()
        model.enable_parallel_training('joint')
        model.parallel_train_mode('joint')
        prefixes = ('feature_adapter.', 'encoder.', 'reg_layer_0.')
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(prefixes) for name in trainable))
        frozen = {name: value.clone() for name, value in model.state_dict().items()
                  if not name.startswith(prefixes)}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        image = torch.randn(2, 3, 64, 64)
        for route in ('dora1', 'dora'):
            optimizer.zero_grad(set_to_none=True)
            model.set_adapter_route(route)
            before = model.feature_adapter[0].weight.detach().clone()
            model(image)[0].sum().backward()
            for module in (model.night_branch, model.night_fusion,
                           model.backbone_features, model.backbone_norm):
                self.assertTrue(all(p.grad is None for p in module.parameters()))
            for parameter in (model.feature_adapter[0].weight,
                              model.encoder.layers[0].self_attn.in_proj_weight,
                              model.reg_layer_0[-1].weight):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
            optimizer.step()
            self.assertFalse(torch.equal(before, model.feature_adapter[0].weight))
            for name, value in model.state_dict().items():
                if name in frozen:
                    torch.testing.assert_close(value, frozen[name], rtol=0, atol=0, msg=name)
        self.assertFalse(model.backbone_features.training)
        self.assertFalse(model.night_branch.training)
        self.assertFalse(model.night_fusion.training)
        self.assertTrue(model.encoder.training)
        self.assertTrue(model.reg_layer_0.training)
        self.assertTrue(all(not m.training for m in model.night_branch.modules()
                            if isinstance(m, nn.BatchNorm2d)))


class ParallelTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def args(self, root, stage, source='', resume=''):
        flags = ['train.py', '--stage', stage, '--device', 'cpu', '--router-mode', 'filename',
                 '--crop-size', '64', '--batch-size', '2', '--num-workers', '0',
                 '--dora-rank', '2', '--dora-alpha', '2', '--dora1-rank', '3', '--dora1-alpha', '3',
                 '--fusion-epochs', '1', '--joint-epochs', '1',
                 '--fusion-train-dir', str(root / 'night/train'), '--fusion-val-dir', str(root / 'night/val'),
                 '--joint-train-dir', str(root / 'mix/train'), '--joint-val-dir', str(root / 'mix/val'),
                 '--save-dir', str(root / 'runs'), '--resume', str(resume)]
        if source:
            flags += ['--dora1-checkpoint' if stage == 'fusion' else '--fusion-checkpoint', str(source)]
        with patch('sys.argv', flags):
            return train.parse_args()

    def test_real_two_stages_resume_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for domain in ('night', 'mix'):
                for split in ('train', 'val'):
                    folder = root / domain / split
                    folder.mkdir(parents=True)
                    # Deliberately use IMG names in the night-only folder:
                    # stage fusion must force DoRA, independent of filenames.
                    for name in ('IMG_1', 'night_IMG_2'):
                        Image.new('RGB', (64, 64), (100, 120, 80)).save(folder / (name + '.jpg'))
                        np.save(folder / (name + '.npy'), np.array([[24., 28., 8.]], dtype=np.float32))
            source = swin_t_trans(pretrained=False)
            source.enable_dual_dora(2, 2., 3, 3.)
            with torch.no_grad():
                source.reg_layer_0[-1].bias.fill_(1.)
            path = root / 'source.pth'
            torch.save(source.state_dict(), path)
            del source
            fusion_routes = []
            def checked_backward(model, batch, *args):
                fusion_routes.extend(batch[-1])
                return backward_routed_batch(model, batch, *args)
            with patch('utils.finetune_trainer.backward_routed_batch', side_effect=checked_backward):
                fusion = train.run_stage(self.args(root, 'fusion', path), 'fusion')
            self.assertEqual(fusion_routes, ['dora', 'dora'])
            checkpoint = load_training_checkpoint(fusion)
            self.assertEqual(checkpoint['stage'], 'fusion')
            self.assertEqual(checkpoint['output_stride'], 16)
            del checkpoint
            gc.collect()
            joint = train.run_stage(self.args(root, 'joint', fusion), 'joint')
            last = Path(joint).parent / 'last_checkpoint.tar'
            # Epoch-boundary resume must restore the selected stage/config and RNG.
            args = self.args(root, 'joint', resume=last)
            args.joint_epochs = 2
            resumed = train.run_stage(args, 'joint')
            checkpoint = load_training_checkpoint(last)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(checkpoint['stage'], 'joint')
            self.assertEqual(len(checkpoint['ranking']), 3)
            del checkpoint
            with patch('sys.argv', ['test.py', '--parallel-checkpoint', resumed,
                '--router-mode', 'filename', '--device', 'cpu', '--num-workers', '0',
                '--data-root', str(root), '--domains', 'mix', '--compare-before-after',
                '--output-csv', str(root / 'summary.csv'), '--routing-csv', str(root / 'routing.csv')]):
                evaluation.main()
            with (root / 'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row['subset'] for row in rows}, {'overall', 'day', 'night'})
            with (root / 'routing.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row['route'] for row in rows}, {'dora', 'dora1'})
            self.assertEqual({row['model'] for row in rows}, {'before', 'after'})
            bad = self.args(root, 'joint', resume=last)
            bad.fusion_alpha = .2
            with self.assertRaisesRegex(ValueError, 'fusion_alpha_init'):
                train.run_stage(bad, 'joint')

    def test_fusion_forces_night_and_pipeline_passes_best_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, 'fusion', 'source.pth')
            args.stage = 'parallel'
            with patch('train.run_stage', side_effect=['fusion.pth', 'joint.pth']) as run:
                self.assertEqual(train.run_pipeline(args), ('fusion.pth', 'joint.pth'))
            self.assertEqual(run.call_args_list[1].kwargs['parent_checkpoint'], 'fusion.pth')
            args.stage = 'all'
            with patch('train.run_stage', side_effect=['base', 'dora', 'dora1', 'fusion', 'joint']) as run:
                self.assertEqual(train.run_pipeline(args), ('base', 'dora', 'dora1', 'fusion', 'joint'))
            self.assertEqual([call.args[1] for call in run.call_args_list],
                             ['baseline', 'dora', 'dora1', 'fusion', 'joint'])


if __name__ == '__main__':
    unittest.main()
