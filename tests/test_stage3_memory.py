from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from models import swin_c_multibatch as models
from models.stage3_cross_attention import Stage3CrossAttention
from models.swin_dora import QVDoRA
from utils.memory_workflow import (attach_memory, load_research_model,
                                   research_payload)
from utils.stage3_bank import (STAGE3_DIMS, TokenReservoir, baseline_fingerprint,
                               cluster_tokens, load_file, validate_bank)


def make_bank(model, size=8):
    return {
        'codebook': torch.randn(size, model.stage3_channels),
        'model_name': 'swin_t_trans', 'stage': 3,
        'feature_dim': model.stage3_channels, 'memory_size': size,
        'baseline_weights_sha256': baseline_fingerprint(model.state_dict()),
        'build_parameters': {'algorithm': 'synthetic test fixture'},
    }


def original_forward(model, images):
    """The pre-change forward path, with unchanged adapter/MAN/density head."""
    x = model.backbone_norm(model.backbone_features(images))
    x = model.feature_adapter(x.permute(0, 3, 1, 2).contiguous())
    batch, channels, height, width = x.shape
    tokens, features = model.encoder(x.flatten(2).permute(2, 0, 1), (height, width))
    x = tokens.permute(1, 2, 0).reshape(batch, channels, height, width)
    x = F.interpolate(x, size=(images.shape[-2] // 16, images.shape[-1] // 16),
                      mode='bilinear', align_corners=False)
    return torch.relu(model.reg_layer_0(x)), features


class Stage3MemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def baseline(self):
        model = models.swin_t_trans(pretrained=False).eval()
        with torch.no_grad():
            model.reg_layer_0[-1].bias.fill_(1.0)
        return model

    def night(self):
        model = self.baseline()
        model.enable_dora(2, 2.0)
        model.research_config = {
            'stage': 'dora', 'model_name': 'swin_t_trans',
            'dora_rank': 2, 'dora_alpha': 2.0,
            'baseline_weights_sha256': baseline_fingerprint(model.state_dict()),
        }
        return model

    def test_baseline_is_identical_to_original_forward(self):
        model = self.baseline()
        images = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            expected, expected_features = original_forward(model, images)
            actual, features = model(images)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for first, second in zip(features, expected_features):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_stage3_tiny_and_large_dimensions(self):
        model = self.baseline()
        with torch.no_grad():
            features = model.extract_stage3(torch.randn(1, 3, 64, 80))
        self.assertEqual(tuple(features.shape), (1, 4, 5, 384))
        # Real torchvision Swin-L topology on meta avoids allocating its 197M
        # parameters just to check Stage 3 shape; CPU arithmetic is tested on T.
        with torch.device('meta'):
            large = models.swin_l_trans(pretrained=False).eval()
            large_features = large.extract_stage3(torch.empty(1, 3, 64, 80))
        self.assertEqual(tuple(large_features.shape), (1, 4, 5, 768))

    def test_bank_shapes_serialization_and_metadata_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, dim in STAGE3_DIMS.items():
                bank = dict(codebook=torch.randn(256, dim), model_name=name,
                            stage=3, feature_dim=dim, memory_size=256,
                            baseline_weights_sha256='synthetic')
                path = Path(temporary) / (name + '.pt')
                torch.save(bank, path)
                restored = validate_bank(load_file(path), name, 256)
                torch.testing.assert_close(restored['codebook'], bank['codebook'])
                with self.assertRaises(ValueError):
                    validate_bank(dict(bank, stage=4))
                with self.assertRaises(ValueError):
                    validate_bank(bank, name, 128)

    def test_legacy_tar_with_numpy_validation_scalars_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'legacy.tar'
            torch.save({'best_mae': np.float64(1.25), 'best_mse': np.float32(2.5),
                        'model_state_dict': {'weight': torch.ones(2)}}, path)
            payload = load_file(path)
            self.assertEqual(float(payload['best_mae']), 1.25)
            self.assertEqual(float(payload['best_mse']), 2.5)

    def test_cross_attention_shape_initial_weights_and_manual_formula(self):
        module = Stage3CrossAttention(torch.randn(7, 16), heads=8)
        self.assertEqual(module.alpha.item(), 1.0)
        self.assertAlmostEqual(module.beta.item(), 0.1)
        night = torch.randn(2, 3, 4, 16)
        output = module(night)
        self.assertEqual(output.shape, night.shape)
        q = module.query(night.flatten(1, 2)).reshape(2, 12, 8, 2).transpose(1, 2)
        k = module.key(module.codebook).reshape(7, 8, 2).permute(1, 0, 2)
        v = module.value(module.codebook).reshape(7, 8, 2).permute(1, 0, 2)
        attended = ((q @ k.transpose(-1, -2) / 2**0.5).softmax(-1) @ v)
        attended = module.output(attended.transpose(1, 2).reshape_as(night))
        torch.testing.assert_close(output, night + module.beta * attended)
        with self.assertRaises(ValueError):
            module(torch.randn(1, 3, 4, 8))
        with torch.no_grad():
            module.alpha.fill_(float('inf'))
        with self.assertRaises(FloatingPointError):
            module(night)

    def _assert_counting_gradients_and_frozen_weights(self, count_loss_only=False):
        from losses.bay_loss import Bay_Loss
        from losses.count_loss import count_l1_loss
        from losses.post_prob import Post_Prob
        model = self.night()
        bank = make_bank(model)
        attach_memory(model, bank)
        frozen = {name: value.detach().clone() for name, value in model.named_parameters()
                  if not value.requires_grad}
        trainable = {name: value for name, value in model.named_parameters()
                     if value.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith('stage3_cross_attention.')
                            for name in trainable))
        optimizer = torch.optim.Adam(trainable.values(), lr=1e-3)
        points = [torch.tensor([[16., 16.], [40., 40.]])]
        probabilities = Post_Prob(8, 64, 16, 0.15, True, 'cpu')(
            points, torch.tensor([64.]))
        density, features = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(density.shape), (1, 1, 4, 4))
        loss = (count_l1_loss(density, [torch.tensor([1.0, 0.4])])
                if count_loss_only else
                Bay_Loss(True, 'cpu')(probabilities, [torch.ones(2)], density))
        # Verify the COUNT loss alone, without relying on consistency gradients.
        loss.backward()
        for name, parameter in trainable.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        before_alpha = model.stage3_cross_attention.alpha.item()
        before_beta = model.stage3_cross_attention.beta.item()
        optimizer.step()
        self.assertNotEqual(model.stage3_cross_attention.alpha.item(), before_alpha)
        self.assertNotEqual(model.stage3_cross_attention.beta.item(), before_beta)
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        self.assertFalse(model.stage3_cross_attention.codebook.requires_grad)
        torch.testing.assert_close(model.stage3_cross_attention.codebook, bank['codebook'],
                                   rtol=0, atol=0)

    def test_counting_gradients_and_frozen_weights(self):
        self._assert_counting_gradients_and_frozen_weights()

    def test_count_l1_reaches_cross_attention_and_keeps_baseline_frozen(self):
        self._assert_counting_gradients_and_frozen_weights(count_loss_only=True)

    def test_day_bypasses_memory_and_adaptation_and_preserves_baseline(self):
        model = self.baseline()
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = model(images)[0]
        model.enable_dora(2, 2.0)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, QVDoRA):
                    module.dora_B_q.normal_()
                    module.magnitude_v.mul_(1.5)
        model.enable_cross_attention(torch.randn(8, 384))
        model.eval().set_inference_domain('day')
        with patch.object(model.stage3_cross_attention, 'forward',
                          side_effect=AssertionError('day must bypass the bank')):
            with torch.no_grad():
                actual = model(images)[0]
        # Parametrized/frozen Linear dispatch can differ at float32 rounding.
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        model.set_inference_domain('night')
        with patch.object(model.stage3_cross_attention, 'forward',
                          wraps=model.stage3_cross_attention.forward) as forward:
            with torch.no_grad():
                model(images)
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(tuple(forward.call_args.args[0].shape), (1, 4, 4, 384))

    def test_legacy_baseline_single_and_dual_dora_checkpoint_loading(self):
        model = self.baseline()
        images = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            expected = model(images)[0]
        restored = load_research_model(model.state_dict(), 'swin_t_trans')
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(images)[0], expected)
        del restored
        model.enable_dora(2, 2.0)
        with torch.no_grad():
            expected = model(images)[0]
        model.enable_dual_dora(2, 2.0, 3, 3.0, active_route='dora')
        restored = load_research_model(
            model.state_dict(), 'swin_t_trans', dora_rank=2, dora_alpha=2.0).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(images)[0], expected)
        self.assertFalse(any('dora1' in name for name in restored.state_dict()))

    def test_self_contained_checkpoint_roundtrip_and_disabled_cross_ablation(self):
        model = self.night()
        attach_memory(model, make_bank(model), train_fusion=False)
        model.eval()
        self.assertFalse(model.stage3_cross_attention.alpha.requires_grad)
        self.assertFalse(model.stage3_cross_attention.beta.requires_grad)
        images = torch.randn(1, 3, 48, 64)
        with torch.no_grad():
            expected = model(images)[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'phase3.pth'
            torch.save(research_payload(model), path)
            payload = load_file(path)
            self.assertNotIn('codebook', payload['research_config']['memory_metadata'])
            restored = load_research_model(payload).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(images)[0], expected, rtol=0, atol=0)
        restored.set_inference_domain('night', use_cross_attention=False)
        with patch.object(restored.stage3_cross_attention, 'forward',
                          side_effect=AssertionError('ablation must bypass bank')):
            with torch.no_grad():
                restored(images)

    def test_memory_provenance_mismatch_rejected(self):
        model = self.night()
        bank = make_bank(model)
        bank['baseline_weights_sha256'] = 'different-baseline'
        with self.assertRaisesRegex(ValueError, 'different baselines'):
            attach_memory(model, bank)

    def test_sampling_is_bounded_and_kmeans_reproducible(self):
        data = np.random.default_rng(42).normal(size=(400, 16)).astype(np.float32)
        first, second = TokenReservoir(80, 42), TokenReservoir(80, 42)
        for chunk in np.array_split(data, 8):
            first.add(chunk)
            second.add(chunk)
            self.assertLessEqual(len(first.tokens), 80)
        np.testing.assert_array_equal(first.tokens, second.tokens)
        torch.testing.assert_close(cluster_tokens(first.tokens, 8),
                                   cluster_tokens(second.tokens, 8), rtol=0, atol=0)

    def _assert_three_training_phases_bank_build_evaluation_and_resume(self, memory_stage):
        import sys
        if memory_stage == 3:
            from build_stage3_bank import build_bank
        else:
            from build_memory_bank import build_bank
        from train import parse_args, run_stage
        from test import evaluate

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8)
            for domain in ('day', 'night'):
                for split in ('train', 'val'):
                    directory = root / domain / split
                    directory.mkdir(parents=True)
                    Image.fromarray(image).save(directory / 'IMG_1.jpg')
                    np.save(directory / 'IMG_1.npy', np.array(
                        [[16., 16., 8.], [40., 40., 8.]], dtype=np.float32))
            common = [
                '--workflow', 'memory', '--device', 'cpu', '--no-pretrained-backbone',
                '--crop-size', '64', '--num-workers', '0', '--batch-size', '1',
                '--data-dir', str(root / 'day'), '--save-dir', str(root / 'models'),
                '--dora-train-dir', str(root / 'night' / 'train'),
                '--dora-val-dir', str(root / 'night' / 'val'),
                '--pretrain-epochs', '1', '--dora-epochs', '1',
                '--dora-val-start', '0', '--cross-attention-epochs', '1',
                '--memory-size', '2', '--dora-rank', '2', '--dora-alpha', '2',
                '--memory-stage', str(memory_stage),
            ]

            def run(stage, extra=()):
                with patch.object(sys, 'argv', ['train.py', '--stage', stage] + common + list(extra)):
                    args = parse_args()
                return run_stage(args, stage)

            baseline = run('baseline')
            self.assertTrue((Path(baseline).parent / 'baseline_best.pth').is_symlink())
            self.assertFalse(any('stage3_cross_attention.' in key
                                 for key in load_file(baseline)['model_state_dict']))
            bank_path = root / 'stage{}_memory_bank.pt'.format(memory_stage)
            bank_args = SimpleNamespace(
                baseline_checkpoint=baseline, data_dir=str(root / 'day'),
                model_name='swin_t_trans', output=str(bank_path), memory_size=2,
                max_tokens=32, tokens_per_image=16, batch_size=16, seed=42,
                crop_size=None, is_gray=None, device='cpu', memory_stage=memory_stage)
            bank = build_bank(bank_args)
            self.assertEqual(tuple(bank['codebook'].shape), (2, 384 if memory_stage == 3 else 768))
            self.assertEqual(bank['stage'], memory_stage)
            self.assertEqual(bank['build_parameters']['train_directory'],
                             str(root / 'day' / 'train'))
            night = run('dora', ['--baseline-checkpoint', baseline])
            self.assertTrue((Path(night).parent / 'night_dora_best.pth').is_symlink())
            self.assertFalse(any('dora1' in key or 'stage3_cross_attention.' in key
                                 for key in load_file(night)['model_state_dict']))
            cross = run('cross_attention', ['--dora-checkpoint', night,
                                            '--memory-bank-path', str(bank_path)])
            self.assertTrue((Path(cross).parent / 'night_cross_attention_best.pth').is_symlink())
            cross_payload = load_file(cross)
            self.assertEqual(cross_payload['research_config']['memory_stage'], memory_stage)
            self.assertEqual(Path(cross).parent.parent.name, 'stage{}'.format(memory_stage))
            self.assertEqual(cross_payload['count_loss_weight'], 0.1)
            self.assertEqual(cross_payload['research_config']['count_loss_weight'], 0.1)
            checkpoint = Path(cross).parent / '1_ckpt.tar'
            resumed = run('cross_attention', ['--resume', str(checkpoint),
                                              '--cross-attention-epochs', '2',
                                              '--memory-stage', str(7 - memory_stage),
                                              '--count-loss-weight', '0.2'])
            self.assertNotEqual(Path(resumed).parent, Path(cross).parent)
            self.assertTrue(checkpoint.is_file())
            payload = load_file(Path(resumed).parent / '2_ckpt.tar')
            self.assertEqual(payload['epoch_index'], 1)
            self.assertEqual(payload['research_config']['stage'], 'cross_attention')
            self.assertEqual(payload['research_config']['memory_stage'], memory_stage)
            self.assertEqual(payload['count_loss_weight'], 0.2)
            self.assertEqual(payload['research_config']['count_loss_weight'], 0.2)
            model = load_research_model(load_file(resumed)).eval()
            args = SimpleNamespace(crop_size=64, downsample_ratio=16, num_workers=0,
                                   workflow='memory', disable_cross_attention=False)
            for domain in ('day', 'night'):
                metrics = evaluate(model, str(root / domain / 'val'), args,
                                   torch.device('cpu'), 'synthetic ' + domain,
                                   domain=domain)
                self.assertEqual(metrics['images'], 1)
                self.assertTrue(np.isfinite(metrics['mae']))

    def test_three_training_phases_bank_build_evaluation_and_resume(self):
        self._assert_three_training_phases_bank_build_evaluation_and_resume(3)

    def test_stage4_training_bank_build_evaluation_and_resume(self):
        self._assert_three_training_phases_bank_build_evaluation_and_resume(4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_backward(self):
        model = self.night()
        attach_memory(model, make_bank(model))
        model.cuda().eval()
        density, _ = model(torch.randn(1, 3, 64, 64, device='cuda'))
        density.sum().backward()
        self.assertGreater(model.stage3_cross_attention.query.weight.grad.abs().sum(), 0)


if __name__ == '__main__':
    unittest.main()
