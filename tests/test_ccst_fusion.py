import unittest

import torch

from models.ccst_fusion import CCSTFusion
from models.swin_c_multibatch import swin_t_ccst_trans


class CCSTFusionTest(unittest.TestCase):
    def test_all_four_stages_receive_gradients(self):
        fusion = CCSTFusion([4, 8, 16, 32], channels=16).train()
        features = [torch.randn(2, c, h, w, requires_grad=True)
                    for c, h, w in [(4, 16, 24), (8, 8, 12),
                                    (16, 4, 6), (32, 2, 3)]]
        output = fusion(features)
        self.assertEqual(output.shape, (2, 16, 2, 3))
        output.square().mean().backward()
        for feature in features:
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(feature.grad.abs().sum().item(), 0)
        for parameter in fusion.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_single_image_evaluation_and_training_guard(self):
        fusion = CCSTFusion([4, 8, 16, 32], channels=16)
        features = [torch.randn(1, c, h, w)
                    for c, h, w in [(4, 16, 24), (8, 8, 12),
                                    (16, 4, 6), (32, 2, 3)]]
        with self.assertRaisesRegex(ValueError, 'batch size'):
            fusion(features)
        fusion.eval()
        with torch.no_grad():
            self.assertEqual(fusion(features).shape, (1, 16, 2, 3))

    def test_man_density_contract_and_adapter_freezing(self):
        model = swin_t_ccst_trans(pretrained=False).train()
        inputs = torch.randn(2, 3, 64, 96)
        density, features = model(inputs)
        self.assertEqual(density.shape, (2, 1, 4, 6))
        self.assertTrue(torch.isfinite(density).all())
        self.assertTrue((density >= 0).all())
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0].shape, (2, 6, 512))
        loss = density.mean() + sum(feature.square().mean() for feature in features)
        loss.backward()
        self.assertGreater(model.ccst_fusion.projections[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone_features[1][0].attn.qkv.weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            reference = model(inputs[:1])[0]
        self.assertEqual(model.enable_dual_dora(active_route='dora1',
                                              trainable_route='dora1'), 12)
        self.assertTrue(all(not p.requires_grad for p in model.ccst_fusion.parameters()))
        with torch.no_grad():
            torch.testing.assert_close(model(inputs[:1])[0], reference)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
