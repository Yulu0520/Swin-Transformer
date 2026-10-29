"""Exercise real Diffusers VAE/U-Net blocks at reduced widths, on CPU."""

import copy
import io
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from diffusers import AutoencoderKL, UNet2DConditionModel

from models.sd_c_multibatch import SDTransMultiBatch, UNET_CONFIG, VAE_CONFIG
from models.sd_dora import DualLinearDoRA, self_attention_projections
from datasets.crowd import Crowd
from utils.image_tiles import split_image


def tiny_model():
    vae_config = dict(VAE_CONFIG, block_out_channels=(8, 8, 16, 16),
                      norm_num_groups=4, layers_per_block=1)
    unet_config = dict(UNET_CONFIG, block_out_channels=(8, 16, 32, 32),
                       norm_num_groups=4, layers_per_block=1,
                       cross_attention_dim=12, attention_head_dim=2)
    model = SDTransMultiBatch(vae=AutoencoderKL(**vae_config),
                              unet=UNet2DConditionModel(**unet_config))
    # Ensure regression ReLU is active for gradient/update assertions.
    with torch.no_grad():
        model.reg_layer_0[-1].bias.fill_(1)
    return model


class SDBackboneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.model = tiny_model()
        self.image = torch.rand(2, 3, 64, 64) * 2 - 1

    def objective(self, output):
        density, features = output
        consistency = sum((1 - torch.nn.functional.cosine_similarity(
            feature, feature.mean(1, keepdim=True), dim=2, eps=1e-5)).sum(1).mean()
                          for feature in features)
        return density.sum() + consistency

    def test_no_text_and_all_baseline_parameters_receive_gradients(self):
        model = self.model
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertIsNone(model.vae.decoder)
        self.assertIsNone(model.vae.post_quant_conv)
        self.assertIsInstance(model.unet.conv_out, nn.Identity)
        self.assertFalse(any('attn2' in name or 'text_encoder' in name
                             or '.transformer_blocks.' in name and '.norm2.' in name
                             for name, _ in model.named_parameters()))
        calls = []
        handles = [module.register_forward_hook(lambda *args: calls.append(True))
                   for name, module in model.unet.named_modules() if name.endswith('attn1')]
        density, features = model(self.image)
        for handle in handles:
            handle.remove()
        self.assertEqual(density.shape, (2, 1, 4, 4))
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0].shape, (2, 4, 512))
        self.assertEqual(len(calls), len(handles))
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        self.objective((density, features)).backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.vae.encoder.conv_in.weight.grad.abs().sum().item(), 0)
        torch.optim.SGD(model.parameters(), lr=1e-3).step()
        for prefix in ('vae.', 'unet.', 'feature_adapter.', 'encoder.', 'reg_layer_0.'):
            self.assertTrue(any(not torch.equal(before[name], p)
                                for name, p in model.named_parameters() if name.startswith(prefix)), prefix)

    def test_only_active_adapter_updates_and_other_branch_is_not_called(self):
        for route in ('dora', 'dora1'):
            with self.subTest(route=route):
                model = copy.deepcopy(self.model).eval()
                original = model(self.image)[0].detach()
                model.enable_dual_dora(2, 2, 3, 3, active_route=route, trainable_route=route)
                torch.testing.assert_close(model(self.image)[0], original)
                other = 'dora1' if route == 'dora' else 'dora'
                calls = []
                handles = [getattr(m, other).register_forward_hook(lambda *args: calls.append(True))
                           for m in model.modules() if isinstance(m, DualLinearDoRA)]
                marker = '.parametrizations.weight.0.{}.'.format(route)
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                self.objective(model(self.image)).backward()
                optimizer.step()
                for handle in handles:
                    handle.remove()
                self.assertEqual(calls, [])
                changed = []
                for name, parameter in model.named_parameters():
                    self.assertEqual(parameter.requires_grad, marker in name, name)
                    if marker in name:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                    if not torch.equal(parameter, before[name]):
                        changed.append(name)
                self.assertTrue(changed)
                self.assertTrue(all(marker in name for name in changed))

    def test_stage_transition_shared_baseline_and_checkpoint_roundtrip(self):
        model = self.model.eval()
        baseline_state = copy.deepcopy(model.state_dict())
        baseline = model(self.image)[0].detach()
        model.enable_dual_dora(2, 2, 3, 3, active_route='dora', trainable_route='dora')
        for projection in self_attention_projections(model.unet):
            router = projection.parametrizations.weight[0]
            with torch.no_grad():
                router.dora.direction_B.normal_(std=0.01)
        trained_dora = model(self.image)[0].detach()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        next_model = tiny_model().eval()
        next_model.enable_dual_dora(2, 2, 3, 3, active_route='dora1', trainable_route='dora1')
        next_model.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(next_model(self.image)[0], baseline)
        next_model.set_adapter_route('dora')
        torch.testing.assert_close(next_model(self.image)[0], trained_dora)
        next_model.set_adapter_route('baseline')
        torch.testing.assert_close(next_model(self.image)[0], baseline)
        # Raw baseline Q/V weights remain shared and unchanged.
        for name, value in next_model.state_dict().items():
            if name.endswith('.parametrizations.weight.original'):
                key = name.replace('.parametrizations.weight.original', '.weight')
                torch.testing.assert_close(value, baseline_state[key], rtol=0, atol=0)

    def test_normalization_padding_and_tiling(self):
        dataset = Crowd('/tmp/nonexistent-sd-test', 64, 16, sd_normalization=True)
        from PIL import Image
        normalized = dataset.trans(Image.new('RGB', (32, 32), (0, 255, 0)))
        torch.testing.assert_close(normalized[:, 0, 0], torch.tensor([-1., 1., -1.]))
        density, _ = self.model(torch.zeros(2, 3, 70, 83))
        self.assertEqual(density.shape, (2, 1, 4, 5))
        image = torch.arange(2 * 3 * 129 * 193).reshape(2, 3, 129, 193)
        tiles = split_image(image, 64)
        self.assertEqual(sum(tile.numel() for tile in tiles), image.numel())
        self.assertEqual(sum(tile.sum().item() for tile in tiles), image.sum().item())
        self.assertTrue(all(max(tile.shape[-2:]) <= 64 for tile in tiles))

    def test_evaluation_loaders_accept_sd_adapter_names(self):
        from test import build_model, build_dual_dora_model
        args = SimpleNamespace(model_name='sd15_man', dora_rank=2, dora_alpha=2,
                               dora1_rank=3, dora1_alpha=3, dual_route='dora1')
        model = self.model.eval()
        expected = model(self.image)[0].detach()
        with tempfile.TemporaryDirectory() as directory, patch(
                'test.models.sd15_man', side_effect=lambda **kwargs: tiny_model()):
            path = Path(directory) / 'baseline.pth'
            torch.save(model.state_dict(), path)
            restored = build_model(args, str(path), expect_dora=False).eval()
            torch.testing.assert_close(restored(self.image)[0], expected)
            model.enable_dual_dora(2, 2, 3, 3, active_route='dora')
            path = Path(directory) / 'dora.pth'
            torch.save(model.state_dict(), path)
            restored = build_model(args, str(path), expect_dora=True).eval()
            torch.testing.assert_close(restored(self.image)[0], expected)
            path = Path(directory) / 'dora1.tar'
            torch.save(dict(model_state_dict=model.state_dict(), model_name='sd15_man',
                            stage='dora1', adapter_type='parallel_dora_dora1',
                            dora_rank=2, dora_alpha=2, dora1_rank=3, dora1_alpha=3), path)
            restored = build_dual_dora_model(args, str(path)).eval()
            torch.testing.assert_close(restored(self.image)[0], expected)


if __name__ == '__main__':
    unittest.main()
