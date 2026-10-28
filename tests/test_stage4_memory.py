import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from losses.count_loss import count_l1_loss
from models import swin_c_multibatch as models
from models.swin_dora import QVDoRA
from utils.memory_workflow import attach_memory, load_research_model, research_payload
from utils.stage3_bank import (FEATURE_POSITIONS, STAGE4_DIMS, baseline_fingerprint,
                               load_file, validate_bank)


class Stage4MemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def night(self):
        model = models.swin_t_trans(pretrained=False).eval()
        with torch.no_grad():
            model.reg_layer_0[-1].bias.fill_(1.0)
        model.enable_dora(2, 2.0)
        model.research_config = {
            'stage': 'dora', 'model_name': 'swin_t_trans',
            'dora_rank': 2, 'dora_alpha': 2.0,
            'baseline_weights_sha256': baseline_fingerprint(model.state_dict()),
        }
        return model

    def bank(self, model, memory_stage=4):
        return {
            'codebook': torch.randn(8, getattr(model, 'stage{}_channels'.format(memory_stage))),
            'model_name': 'swin_t_trans', 'stage': memory_stage,
            'feature_dim': getattr(model, 'stage{}_channels'.format(memory_stage)),
            'feature_position': FEATURE_POSITIONS[memory_stage], 'memory_size': 8,
            'baseline_weights_sha256': model.research_config['baseline_weights_sha256'],
        }

    def test_stage4_extraction_tiny_and_large_before_norm(self):
        model = self.night()
        with patch.object(model.backbone_norm, 'forward', side_effect=AssertionError('no norm')):
            with torch.no_grad():
                features = model.extract_stage4(torch.randn(1, 3, 64, 80))
        self.assertEqual(tuple(features.shape), (1, 2, 3, 768))
        with torch.device('meta'):
            large = models.swin_l_trans(pretrained=False).eval()
            features = large.extract_stage4(torch.empty(1, 3, 64, 80))
        self.assertEqual(tuple(features.shape), (1, 2, 3, 1536))

    def test_bank_dimensions_serialization_and_stage_position_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, dim in STAGE4_DIMS.items():
                bank = dict(codebook=torch.randn(256, dim), model_name=name, stage=4,
                            feature_dim=dim, memory_size=256,
                            feature_position=FEATURE_POSITIONS[4], baseline_weights_sha256='test')
                path = Path(temporary) / (name + '.pt')
                torch.save(bank, path)
                restored = validate_bank(load_file(path), name, 256, 4)
                torch.testing.assert_close(restored['codebook'], bank['codebook'])
                with self.assertRaisesRegex(ValueError, 'does not match --memory-stage'):
                    validate_bank(bank, name, memory_stage=3)
                for position in (None, 'after_backbone_norm'):
                    with self.assertRaisesRegex(ValueError, 'feature position'):
                        validate_bank(dict(bank, feature_position=position))

    def test_fusion_runs_once_after_stage4_and_before_norm(self):
        model = self.night()
        attach_memory(model, self.bank(model), memory_stage=4)
        model.eval()
        events, values, handles = [], {}, []

        def after_stage4(module, inputs, output):
            events.append('stage4')
            values['stage4'] = output.detach().clone()

        def before_cross(module, inputs):
            events.append('cross')
            torch.testing.assert_close(inputs[0], values['stage4'], rtol=0, atol=0)

        def after_cross(module, inputs, output):
            values['fused'] = output.detach().clone()

        def before_norm(module, inputs):
            events.append('norm')
            torch.testing.assert_close(inputs[0], values['fused'], rtol=0, atol=0)

        handles.append(model.backbone_features[6].register_forward_hook(
            lambda *args: events.append('merge')))
        handles.append(model.backbone_features[7].register_forward_hook(after_stage4))
        handles.append(model.active_cross_attention.register_forward_pre_hook(before_cross))
        handles.append(model.active_cross_attention.register_forward_hook(after_cross))
        handles.append(model.backbone_norm.register_forward_pre_hook(before_norm))
        try:
            with torch.no_grad():
                density, _ = model(torch.randn(1, 3, 64, 64))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(events, ['merge', 'stage4', 'cross', 'norm'])
        self.assertEqual(tuple(values['stage4'].shape), (1, 2, 2, 768))
        self.assertEqual(tuple(density.shape), (1, 1, 4, 4))
        self.assertIsNone(model.stage3_cross_attention)
        with self.assertRaisesRegex(ValueError, 'already attached'):
            model.enable_cross_attention(torch.randn(8, 384), memory_stage=3)

    def test_count_l1_gradients_and_all_frozen_parameters_unchanged(self):
        model = self.night()
        bank = self.bank(model)
        attach_memory(model, bank, memory_stage=4)
        model.eval()
        frozen = {name: param.detach().clone() for name, param in model.named_parameters()
                  if not param.requires_grad}
        trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
        self.assertTrue(all(name.startswith('stage4_cross_attention.') for name in trainable))
        optimizer = torch.optim.Adam(trainable.values(), lr=1e-3)
        density, _ = model(torch.randn(1, 3, 64, 64))
        count_l1_loss(density, [torch.tensor([1.0, 0.4])]).backward()
        for suffix in ('query.weight', 'key.weight', 'value.weight', 'output.weight',
                       'alpha', 'beta'):
            parameter = trainable['stage4_cross_attention.' + suffix]
            self.assertIsNotNone(parameter.grad, suffix)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, suffix)
        before_alpha = model.active_cross_attention.alpha.item()
        before_beta = model.active_cross_attention.beta.item()
        optimizer.step()
        self.assertNotEqual(model.active_cross_attention.alpha.item(), before_alpha)
        self.assertNotEqual(model.active_cross_attention.beta.item(), before_beta)
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        torch.testing.assert_close(model.active_cross_attention.codebook, bank['codebook'], rtol=0, atol=0)
        self.assertEqual(baseline_fingerprint(model.state_dict()), bank['baseline_weights_sha256'])

    def test_daytime_and_ablation_bypass_cross_attention(self):
        model = self.night()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, QVDoRA):
                    module.dora_B_q.normal_()
                    module.magnitude_v.mul_(1.5)
        images = torch.randn(1, 3, 64, 64)
        model.set_inference_domain('day')
        with torch.no_grad():
            baseline = model(images)[0]
        attach_memory(model, self.bank(model), memory_stage=4)
        model.eval().set_inference_domain('day')
        with patch.object(model.active_cross_attention, 'forward',
                          side_effect=AssertionError('must bypass bank')):
            with torch.no_grad():
                torch.testing.assert_close(model(images)[0], baseline, rtol=1e-6, atol=1e-6)
            model.set_inference_domain('night', use_cross_attention=False)
            with torch.no_grad():
                model(images)

    def test_self_contained_stage4_roundtrip_and_metadata_mismatch(self):
        model = self.night()
        attach_memory(model, self.bank(model), train_fusion=False, memory_stage=4)
        model.eval()
        with torch.no_grad():
            model.active_cross_attention.alpha.fill_(1.2)
            model.active_cross_attention.beta.fill_(0.3)
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = model(images)[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'phase3_stage4.pth'
            torch.save(research_payload(model), path)
            payload = load_file(path)
        self.assertEqual(payload['format_version'], 2)
        self.assertNotIn('codebook', payload['research_config']['memory_metadata'])
        restored = load_research_model(payload).eval()
        self.assertEqual(restored.memory_stage, 4)
        self.assertFalse(restored.active_cross_attention.alpha.requires_grad)
        self.assertFalse(restored.active_cross_attention.beta.requires_grad)
        with torch.no_grad():
            torch.testing.assert_close(restored(images)[0], expected, rtol=0, atol=0)
        for key, value in [('memory_stage', 3), ('feature_position', 'after_backbone_norm'),
                           ('feature_dim', 384)]:
            corrupted = dict(payload, research_config=dict(payload['research_config'], **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_research_model(corrupted)

    def test_original_stage3_checkpoint_without_new_fields_remains_loadable(self):
        model = self.night()
        attach_memory(model, self.bank(model, memory_stage=3))
        model.eval()
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = model(images)[0]
        payload = research_payload(model)
        payload['research_config'] = copy.deepcopy(payload['research_config'])
        for key in ('memory_stage', 'feature_dim', 'feature_position'):
            payload['research_config'].pop(key)
        payload['research_config']['memory_metadata'].pop('feature_position')
        restored = load_research_model(payload).eval()
        self.assertEqual(restored.memory_stage, 3)
        with torch.no_grad():
            torch.testing.assert_close(restored(images)[0], expected, rtol=0, atol=0)

    def test_cli_defaults_and_legacy_builder_are_unambiguous(self):
        from build_memory_bank import parse_args as parse_bank
        from build_stage3_bank import parse_args as parse_old_bank
        from train import parse_args as parse_train
        with patch('sys.argv', ['train.py', '--stage', 'baseline']):
            args = parse_train()
        self.assertEqual(args.memory_stage, 4)
        self.assertEqual(args.memory_bank_path, 'stage4_memory_bank.pt')
        with patch('sys.argv', ['build_memory_bank.py']):
            args = parse_bank()
        self.assertEqual(args.memory_stage, 4)
        self.assertEqual(args.output, 'stage4_memory_bank.pt')
        with patch('sys.argv', ['build_stage3_bank.py']):
            args = parse_old_bank()
        self.assertEqual(args.memory_stage, 3)
        self.assertEqual(args.output, 'stage3_memory_bank.pt')

    def test_checkpoint_discovery_separates_stages_and_supports_legacy_layout(self):
        from test import resolve_checkpoint
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'swin_t_trans' / 'cross_attention'
            paths = [root / 'old_run' / 'best_model.pth',
                     root / 'stage3' / 'run' / 'best_model.pth',
                     root / 'stage4' / 'run' / 'best_model.pth']
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True)
                path.touch()
                os.utime(path, (300 - index * 100, 300 - index * 100))
            self.assertEqual(resolve_checkpoint('', temporary, 'swin_t_trans', 'cross_attention', 4),
                             str(paths[2]))
            self.assertEqual(resolve_checkpoint('', temporary, 'swin_t_trans', 'cross_attention', 3),
                             str(paths[0]))
            paths[0].unlink()
            self.assertEqual(resolve_checkpoint('', temporary, 'swin_t_trans', 'cross_attention', 3),
                             str(paths[1]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_stage4_count_loss_backward(self):
        model = self.night()
        attach_memory(model, self.bank(model), memory_stage=4)
        model.cuda().eval()
        density, _ = model(torch.randn(1, 3, 64, 64, device='cuda'))
        count_l1_loss(density, [torch.ones(2, device='cuda')]).backward()
        self.assertGreater(model.active_cross_attention.query.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
