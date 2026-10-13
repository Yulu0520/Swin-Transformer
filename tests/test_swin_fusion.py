import unittest

import torch
from torch.nn import functional as F

from models.swin_c_multibatch import swin_t_trans


class SwinFusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.model = swin_t_trans(pretrained=False).eval()

    def test_stage4_only_encoder_and_equal_sum_before_regression(self):
        captured = {}
        handles = []

        def capture(name):
            def hook(module, inputs, output):
                captured[name] = output
            return hook

        for index in (1, 3, 5):
            handles.append(self.model.backbone_features[index].register_forward_hook(
                capture('stage_{}'.format(index))))
        handles.append(self.model.feature_adapter.register_forward_hook(capture('stage4')))
        handles.append(self.model.encoder.register_forward_hook(capture('encoder')))

        def capture_encoder_input(module, inputs):
            captured['encoder_input'] = inputs

        def capture_regression_input(module, inputs):
            captured['regression_input'] = inputs[0]

        handles.append(self.model.encoder.register_forward_pre_hook(capture_encoder_input))
        handles.append(self.model.reg_layer_0.register_forward_pre_hook(capture_regression_input))
        try:
            # Odd, rectangular dimensions exercise Swin patch padding and resizing.
            with torch.no_grad():
                density, consistency = self.model(torch.randn(2, 3, 65, 81))
                stage4 = captured['stage4']
                tokens, shape = captured['encoder_input']
                torch.testing.assert_close(tokens, stage4.flatten(2).permute(2, 0, 1))
                self.assertEqual(shape, stage4.shape[-2:])
                self.assertEqual(len(consistency), 2)
                self.assertIs(consistency, captured['encoder'][1])
                self.assertEqual(density.shape, (2, 1, 4, 5))
                self.assertTrue(torch.isfinite(density).all())
                self.assertTrue((density >= 0).all())

                encoder = captured['encoder'][0].permute(1, 2, 0).reshape_as(stage4)
                encoder = F.interpolate(encoder, size=(4, 5), mode='bilinear',
                                        align_corners=False)
                branches = [encoder]
                for index, adapter in zip((1, 3, 5), self.model.stage_adapters):
                    feature = adapter(captured['stage_{}'.format(index)].permute(0, 3, 1, 2))
                    branches.append(F.interpolate(feature, size=(4, 5), mode='area'))
                expected = sum(
                    F.layer_norm(feature.permute(0, 2, 3, 1), (512,)).permute(0, 3, 1, 2)
                    for feature in branches
                )
                torch.testing.assert_close(captured['regression_input'], expected)
        finally:
            for handle in handles:
                handle.remove()

    def test_baseline_gradients_reach_all_branches(self):
        # Keep the final ReLU active so this checks the complete density path.
        with torch.no_grad():
            self.model.reg_layer_0[-1].bias.fill_(10.)
        density, _ = self.model(torch.randn(1, 3, 64, 64))
        density.square().mean().backward()
        modules = list(self.model.stage_adapters) + [
            self.model.feature_adapter[0], self.model.encoder.layers[-1].linear2,
            self.model.backbone_features[1][0].attn.qkv,
            self.model.reg_layer_0[-1],
        ]
        for module in modules:
            gradient = module.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_dora_freezes_fusion_and_preserves_initial_prediction(self):
        inputs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected, _ = self.model(inputs)
        self.model.enable_dual_dora(
            dora_rank=2, dora_alpha=2., dora1_rank=2, dora1_alpha=2.,
            active_route='dora', trainable_route='dora')
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in self.model.stage_adapters.parameters()))
        for route in ('dora', 'dora1', 'baseline'):
            self.model.set_adapter_route(route)
            with torch.no_grad():
                actual, _ = self.model(inputs)
            torch.testing.assert_close(actual, expected)
        names = [name for name, parameter in self.model.named_parameters()
                 if parameter.requires_grad]
        self.assertTrue(names)
        self.assertTrue(all('.parametrizations.weight.0.dora.' in name for name in names))


if __name__ == '__main__':
    unittest.main()
