import argparse
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from datasets.crowd import Crowd
from losses.clip_guidance import (
    CLIPTeacher, cosine_loss, directional_loss, warmup_factor, rgb_to_imagenet,
    text_alignment_loss, text_alignment_logits, TEXT_ALIGNMENT_TARGET,
    DEFAULT_DAY_PROMPT, DEFAULT_NIGHT_PROMPT,
)
from models.clip_projection import (
    CLIPProjection, backbone_signature, branch_signature, restore_projection,
)
from models.swin_c_multibatch import swin_t_trans
from models.swin_dual_dora import DualQVDoRAAdapter
from train_clip_projection import (
    main as calibrate_projection, split_scenes, extract_calibration_features,
)
from utils.clip_model_loading import load_trained_dual
from utils.regression_trainer_cosine_multibatch import RegTrainer, train_collate
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob


class _Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, *args):
        self.scalars[args[0]] = args[1]

    def flush(self):
        pass


class _Teacher(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding_dim = dim
        centres = torch.eye(2, dim)
        self.register_buffer('text_centres', centres)
        self.register_buffer('text_direction', torch.nn.functional.normalize(
            centres[0:1] - centres[1:2], dim=-1))


class CLIPGuidanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pairs_share_geometry_and_preserve_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            night, day = root / 'night', root / 'day'
            night.mkdir()
            day.mkdir()
            # Identical source pixels make any mismatched augmentation visible.
            grid = np.arange(96 * 96 * 3, dtype=np.uint8).reshape(96, 96, 3)
            Image.fromarray(grid).save(night / 'night_IMG_1.jpg')
            Image.fromarray(grid).save(day / 'IMG_1.jpg')
            np.save(night / 'night_IMG_1.npy', np.array([[40., 40., 8.]]))
            dataset = Crowd(str(night), 64, 16, pair_root=str(day))
            with patch('datasets.crowd.random.random', side_effect=[0.5, 0.9]), \
                    patch('datasets.crowd.random_crop', return_value=(8, 12, 64, 64)):
                inputs, points, targets, size, pair = dataset[0]
            mean = torch.tensor([.485, .456, .406])[:, None, None]
            std = torch.tensor([.229, .224, .225])[:, None, None]
            torch.testing.assert_close(inputs * std + mean, pair)
            torch.testing.assert_close(points, torch.tensor([[36., 32.]]))
            torch.testing.assert_close(targets, torch.ones(1))
            batch = train_collate([dataset[0], dataset[0]])
            self.assertEqual(len(batch), 5)
            self.assertEqual(batch[4].shape, (2, 3, 64, 64))
            (day / 'IMG_1.jpg').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Missing paired'):
                Crowd(str(night), 64, 16, pair_root=str(day))

    def test_pair_shape_mismatch_fails(self):
        dataset = Crowd('/nonexistent', 64, 16)
        with self.assertRaisesRegex(ValueError, 'same geometry'):
            dataset.train_transform(Image.new('RGB', (96, 96)),
                                    np.empty((0, 3)), Image.new('RGB', (96, 80)))

    @unittest.skipUnless(importlib.util.find_spec('clip'), 'OpenAI CLIP is optional')
    def test_official_clip_text_centres_are_frozen_without_image_encoding(self):
        from clip.model import CLIP
        model = CLIP(embed_dim=8, image_resolution=32, vision_layers=1,
                     vision_width=64, vision_patch_size=16, context_length=77,
                     vocab_size=49408, transformer_width=64,
                     transformer_heads=1, transformer_layers=1)
        with patch('clip.load', return_value=(model, None)), \
                patch.object(model, 'encode_image', side_effect=AssertionError('Image encoder called')):
            teacher = CLIPTeacher('test', 'cpu', '/tmp')
        teacher.train()
        self.assertEqual(teacher.text_centres.shape, (2, 8))
        self.assertTrue(torch.isfinite(teacher.text_centres).all())
        self.assertFalse(teacher.text_centres.requires_grad)
        self.assertFalse(teacher.training)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        torch.testing.assert_close(teacher.text_centres.norm(dim=-1), torch.ones(2))
        self.assertEqual(teacher.text_direction.shape, (1, 8))
        self.assertFalse(hasattr(teacher, 'model'))

    def test_cached_text_centres_need_no_clip_package_or_model(self):
        centres = torch.eye(2, 8, requires_grad=True)
        with patch.dict('sys.modules', {'clip': None}):
            teacher = CLIPTeacher('test', 'cpu', '/tmp', cached_centres=centres)
        torch.testing.assert_close(teacher.text_centres, centres.detach())
        self.assertFalse(teacher.text_centres.requires_grad)
        with self.assertRaisesRegex(ValueError, 'Invalid cached'):
            CLIPTeacher('test', 'cpu', '/tmp', cached_centres=torch.zeros(2, 8))
        with self.assertRaisesRegex(ValueError, 'no direction'):
            CLIPTeacher('test', 'cpu', '/tmp', cached_centres=torch.ones(2, 8))

    def test_calibration_holdout_keeps_scene_pairs_together(self):
        paths = ['/day/IMG_{}.jpg'.format(i) for i in range(10)]
        paths += ['/night/night_IMG_{}.jpg'.format(i) for i in range(10)]
        train, val = split_scenes(paths, .2, 42)
        scenes = lambda items: {Path(p).name.replace('night_', '') for p in items}
        self.assertTrue(scenes(train).isdisjoint(scenes(val)))
        self.assertEqual((len(train), len(val)), (16, 4))

    def test_dual_calibration_command_records_both_branch_signatures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            day, night = root / 'day', root / 'night'
            day.mkdir()
            night.mkdir()
            for i in range(3):
                Image.new('RGB', (80, 80), (50, 80, 100)).save(day / f'IMG_{i}.jpg')
                Image.new('RGB', (80, 80), (10, 20, 30)).save(night / f'night_IMG_{i}.jpg')
            model = swin_t_trans(pretrained=False)
            model.enable_dual_dora(2, 2, 2, 2, active_route='dora')
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, DualQVDoRAAdapter):
                        module.dora1.dora_B_v.normal_(std=.01)
            source, output = root / 'dual.pth', root / 'projection.pth'
            torch.save(model.state_dict(), source)
            argv = ['train_clip_projection.py', '--dual-checkpoint', str(source),
                    '--train-dirs', str(day), str(night), '--clip-model', 'test',
                    '--output', str(output), '--epochs', '1', '--batch-size', '2',
                    '--crop-size', '64', '--num-workers', '0', '--device', 'cpu',
                    '--dora-rank', '2', '--dora-alpha', '2',
                    '--dora1-rank', '2', '--dora1-alpha', '2']
            with patch('sys.argv', argv), \
                    patch('train_clip_projection.CLIPTeacher', return_value=_Teacher(8)):
                calibrate_projection()
            checkpoint = torch.load(output, map_location='cpu')
            self.assertEqual(checkpoint['calibration_mode'], 'dual_dora')
            self.assertEqual(checkpoint['alignment_target'], TEXT_ALIGNMENT_TARGET)
            self.assertEqual(checkpoint['day_prompt'], DEFAULT_DAY_PROMPT)
            self.assertEqual(checkpoint['night_prompt'], DEFAULT_NIGHT_PROMPT)
            self.assertTrue(np.isfinite(checkpoint['holdout_direction_cosine']))
            self.assertGreaterEqual(checkpoint['holdout_accuracy'], 0)
            self.assertLessEqual(checkpoint['holdout_accuracy'], 1)
            projection, _ = restore_projection(
                checkpoint, 'swin_t_trans', 'test', 768, 8, backbone_signature(model),
                branch_signature(model, 'dora1'), branch_signature(model, 'dora'))
            self.assertTrue(np.isfinite(checkpoint['holdout_loss']))
            self.assertFalse(projection.linear.weight.requires_grad)

    def test_initial_zero_displacement_has_finite_nonzero_gradient(self):
        baseline = torch.tensor([[1., 0., 0.]], requires_grad=True)
        adapted = baseline.detach().clone().requires_grad_()
        target = torch.tensor([[0., 1., 0.]], requires_grad=True)
        loss = directional_loss(adapted, baseline, target)
        loss.backward()
        self.assertTrue(torch.isfinite(adapted.grad).all())
        self.assertGreater(adapted.grad.abs().sum().item(), 0)
        self.assertGreater(adapted.grad[0, 1].item(), 0)
        self.assertIsNone(baseline.grad)
        self.assertIsNone(target.grad)
        self.assertEqual(warmup_factor(0, 5), 0)
        self.assertEqual(warmup_factor(2, 5), .4)
        self.assertEqual(warmup_factor(6, 5), 1)
        self.assertEqual(warmup_factor(0, 0), 1)

    def test_direction_is_day_minus_night(self):
        night = torch.tensor([[1., 0.]])
        day = torch.tensor([[1., 1.]])
        text = torch.tensor([[0., 1.]])
        torch.testing.assert_close(directional_loss(night, day, text), torch.tensor(0.))
        torch.testing.assert_close(directional_loss(night, day, -text), torch.tensor(2.))

    def test_text_alignment_prefers_correct_domain_and_keeps_text_fixed(self):
        centres = torch.eye(2, requires_grad=True)
        images = torch.eye(2, requires_grad=True)
        labels = torch.tensor([0, 1])
        aligned = text_alignment_loss(images, centres, labels)
        reversed_loss = text_alignment_loss(images, centres, 1 - labels)
        self.assertLess(aligned.item(), reversed_loss.item())
        reversed_loss.backward()
        self.assertIsNone(centres.grad)
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(images.grad.norm().item(), 0)
        with self.assertRaisesRegex(ValueError, 'temperature'):
            text_alignment_logits(images, centres, 0)

    def test_text_alignment_alone_reaches_night_dora_through_frozen_adapter(self):
        model = swin_t_trans(pretrained=False)
        model.enable_dual_dora(2, 2, 2, 2, active_route='dora', trainable_route='dora')
        model.eval()
        adapter = CLIPProjection(768, 8).requires_grad_(False)
        features = model.extract_backbone_features(torch.randn(1, 3, 64, 64))
        loss = text_alignment_loss(adapter(features), torch.eye(2, 8), torch.ones(1, dtype=torch.long))
        loss.backward()
        gradients = [p.grad for name, p in model.named_parameters()
                     if '.dora.dora_B_v' in name and p.grad is not None]
        self.assertTrue(gradients)
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))

    def test_text_direction_cli_requires_night_training_and_a_day_pair(self):
        from train import parse_args
        argv = ['train.py', '--stage', 'dora', '--dual-checkpoint', 'dual.pth',
                '--clip-direction-weight', '.001', '--clip-pair-root', '/day/train']
        with patch('sys.argv', argv):
            args = parse_args()
        self.assertEqual(args.clip_negative_prompt, DEFAULT_NIGHT_PROMPT)
        self.assertEqual(args.clip_positive_prompt, DEFAULT_DAY_PROMPT)
        for invalid in (
                ['train.py', '--stage', 'dora1', '--clip-direction-weight', '.001'],
                ['train.py', '--stage', 'dora', '--dual-checkpoint', 'dual.pth',
                 '--clip-direction-weight', '.001'],
                ['train.py', '--stage', 'dora', '--dual-checkpoint', 'dual.pth',
                 '--clip-pair-weight', '.1']):
            with patch('sys.argv', invalid), patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_frozen_projection_still_propagates_feature_gradients(self):
        projection = CLIPProjection(4, 8).requires_grad_(False)
        features = torch.randn(2, 4, 2, 2, requires_grad=True)
        target = torch.randn(2, 8, requires_grad=True)
        cosine_loss(projection(features), target).backward()
        self.assertGreater(features.grad.norm().item(), 0)
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is None for p in projection.parameters()))

    def test_night_day_epoch_checkpoint_keeps_inference_routes(self):
        from test import build_dual_dora_model
        model = swin_t_trans(pretrained=False)
        model.enable_dual_dora(2, 2, 2, 2, active_route='dora')
        checkpoint = dict(model_state_dict=model.state_dict(), model_name='swin_t_trans',
                          stage='dora', adapter_type='parallel_dora_dora1',
                          dora_rank=2, dora_alpha=2., dora1_rank=2, dora1_alpha=2.,
                          clip_config=dict(alignment_target=TEXT_ALIGNMENT_TARGET))
        args = argparse.Namespace(model_name='swin_t_trans', dual_route='dora')
        with patch('test.load_checkpoint', return_value=checkpoint):
            restored = build_dual_dora_model(args, '/unused/epoch.tar')
        self.assertEqual(branch_signature(restored, 'dora1'), branch_signature(model, 'dora1'))
        restored.set_adapter_route('dora1')
        self.assertTrue(all(m.active_route == 'dora1' for m in restored.modules()
                            if isinstance(m, DualQVDoRAAdapter)))

    def test_swin_forward_and_projection_checkpoint_compatibility(self):
        model = swin_t_trans(pretrained=False).eval()
        inputs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            density, consistency = model(inputs)
            extended = model(inputs, return_backbone_features=True)
        torch.testing.assert_close(density, extended[0])
        for left, right in zip(consistency, extended[1]):
            torch.testing.assert_close(left, right)
        self.assertEqual(extended[2].shape, (1, 768, 2, 2))
        self.assertEqual(density.shape, (1, 1, 4, 4))
        signature = backbone_signature(model)
        projection = CLIPProjection(768, 8)
        checkpoint = dict(model_name='swin_t_trans', clip_model='test',
                          input_dim=768, embedding_dim=8,
                          image_normalization='imagenet', backbone_signature=signature,
                          projection_state_dict=projection.state_dict())
        restored, _ = restore_projection(checkpoint, 'swin_t_trans', 'test', 768, 8, signature)
        torch.testing.assert_close(projection(extended[2]), restored(extended[2]))
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        model.enable_dual_dora(2, 2, 2, 2, active_route='dora1', trainable_route='dora1')
        self.assertEqual(signature, backbone_signature(model))
        with self.assertRaisesRegex(ValueError, 'backbone_signature mismatch'):
            restore_projection(checkpoint, 'swin_t_trans', 'test', 768, 8, 'other')

    def test_cpu_night_day_direction_updates_only_night_dora(self):
        model = swin_t_trans(pretrained=False)
        model.enable_dual_dora(2, 2, 2, 2, active_route='dora', trainable_route='dora')
        projection = CLIPProjection(768, 8).requires_grad_(False).eval()
        trainer = RegTrainer.__new__(RegTrainer)
        trainer.args = argparse.Namespace(stage='dora', max_epoch=1,
                                         clip_warmup_epochs=0, clip_align_weight=.1, clip_temperature=.07,
                                         clip_direction_weight=.001,
                                         tensorboard_log_interval=1)
        trainer.device = torch.device('cpu')
        trainer.model, trainer.clip_projection = model, projection
        trainer.clip_teacher, trainer.clip_config = _Teacher(8), {'enabled': True}
        trainer.epoch_index, trainer.epoch, trainer.global_step = 0, 1, 0
        trainer.writer = _Writer()
        trainer.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4)
        trainer.criterion = Bay_Loss(True, trainer.device)
        trainer.post_prob = Post_Prob(8, 64, 16, .15, True, trainer.device)
        sample = (torch.randn(3, 64, 64), torch.tensor([[24., 24.]]),
                  torch.ones(1), 64., torch.rand(3, 64, 64))
        trainer.dataloaders = {'train': [train_collate([sample])]}
        branches = [m for m in model.modules() if isinstance(m, DualQVDoRAAdapter)]
        before = branches[0].dora.dora_B_v.detach().clone()
        frozen_signature = branch_signature(model, 'dora1')
        records = []
        original_extract = model.extract_backbone_features

        def extract(inputs):
            features = original_extract(inputs)
            records.append((branches[0].active_route, inputs.detach().clone(),
                            projection(features).detach().clone(), torch.is_grad_enabled()))
            return features

        with patch.object(model, 'extract_backbone_features', side_effect=extract):
            trainer.train_eopch()
        self.assertEqual([record[0] for record in records], ['dora1', 'dora'])
        self.assertEqual([record[3] for record in records], [False, True])
        torch.testing.assert_close(records[0][1], rgb_to_imagenet(sample[4][None]))
        torch.testing.assert_close(records[1][1], sample[0][None])
        expected = directional_loss(records[1][2], records[0][2],
                                    trainer.clip_teacher.text_direction)
        self.assertAlmostEqual(trainer.writer.scalars['train_step/clip_direction_loss'], expected.item())
        alignment = text_alignment_loss(records[1][2], trainer.clip_teacher.text_centres,
                                        torch.ones(1, dtype=torch.long), .07)
        self.assertAlmostEqual(trainer.writer.scalars['train_step/clip_alignment_loss'], alignment.item())
        self.assertGreater((before - branches[0].dora.dora_B_v).abs().sum().item(), 0)
        self.assertEqual(frozen_signature, branch_signature(model, 'dora1'))
        self.assertTrue(all(m.active_route == 'dora' for m in branches))
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad, name)
            elif parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue(all(p.grad is None for p in projection.parameters()))

    def test_dual_checkpoint_loading_and_calibration_routes(self):
        source = swin_t_trans(pretrained=False)
        source.enable_dual_dora(2, 2, 2, 2, active_route='dora1')
        with torch.no_grad():
            for module in source.modules():
                if isinstance(module, DualQVDoRAAdapter):
                    module.dora1.dora_B_v.normal_(std=.01)
        restored = swin_t_trans(pretrained=False)
        load_trained_dual(restored, source.state_dict(), 'swin_t_trans', 2, 2, 2, 2,
                          trainable_route='dora')
        self.assertEqual(branch_signature(source, 'dora1'), branch_signature(restored, 'dora1'))
        self.assertTrue(all('.dora.' in name for name, p in restored.named_parameters() if p.requires_grad))
        restored.eval()
        inputs = torch.randn(2, 3, 64, 64)
        features = extract_calibration_features(restored, inputs, torch.tensor([1, 0]))
        with torch.no_grad():
            restored.set_adapter_route('dora')
            night = restored.extract_backbone_features(inputs[:1])
            restored.set_adapter_route('dora1')
            day = restored.extract_backbone_features(inputs[1:])
        torch.testing.assert_close(features, torch.cat([night, day]))
        self.assertFalse(features.requires_grad)
        # Same parameter tensors with different scaling must be incompatible.
        signature = branch_signature(restored, 'dora1')
        for module in restored.modules():
            if isinstance(module, DualQVDoRAAdapter):
                module.dora1.scaling *= 2
                break
        self.assertNotEqual(signature, branch_signature(restored, 'dora1'))

    def test_dual_projection_rejects_wrong_teacher_and_allows_student_resume(self):
        model = swin_t_trans(pretrained=False)
        model.enable_dual_dora(2, 2, 2, 2, active_route='dora')
        base_sig = backbone_signature(model)
        day_sig, night_sig = branch_signature(model, 'dora1'), branch_signature(model, 'dora')
        checkpoint = dict(model_name='swin_t_trans', clip_model='test', input_dim=768,
                          embedding_dim=8, image_normalization='imagenet',
                          backbone_signature=base_sig, calibration_mode='dual_dora',
                          alignment_target=TEXT_ALIGNMENT_TARGET, text_centres=torch.eye(2, 8),
                          day_prompt=DEFAULT_DAY_PROMPT, night_prompt=DEFAULT_NIGHT_PROMPT,
                          day_branch_signature=day_sig, night_branch_signature=night_sig,
                          projection_state_dict=CLIPProjection(768, 8).state_dict())
        restore_projection(checkpoint, 'swin_t_trans', 'test', 768, 8, base_sig,
                           day_sig, night_sig)
        with self.assertRaisesRegex(ValueError, 'day branch mismatch'):
            restore_projection(checkpoint, 'swin_t_trans', 'test', 768, 8, base_sig, 'wrong')
        with self.assertRaisesRegex(ValueError, 'initial night branch mismatch'):
            restore_projection(checkpoint, 'swin_t_trans', 'test', 768, 8, base_sig, day_sig, 'wrong')
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, DualQVDoRAAdapter):
                    module.dora.dora_B_v.add_(.1)
        # Resume validates the fixed teacher; the trained student has evolved.
        restore_projection(checkpoint, 'swin_t_trans', 'test', 768, 8,
                           backbone_signature(model), branch_signature(model, 'dora1'))
        checkpoint['calibration_mode'] = 'baseline'
        with self.assertRaisesRegex(ValueError, 'both DoRA branches'):
            restore_projection(checkpoint, 'swin_t_trans', 'test', 768, 8, base_sig, day_sig)
        checkpoint['calibration_mode'] = 'dual_dora'
        with self.assertRaisesRegex(ValueError, 'day_prompt mismatch'):
            restore_projection(checkpoint, 'swin_t_trans', 'test', 768, 8, base_sig,
                               day_sig, text_prompts={'day_prompt': 'different'})
        checkpoint.pop('alignment_target')
        with self.assertRaisesRegex(ValueError, 'old image-teacher'):
            restore_projection(checkpoint, 'swin_t_trans', 'test', 768, 8, base_sig, day_sig)


if __name__ == '__main__':
    unittest.main()
