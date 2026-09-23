import unittest

import torch
from torch import nn
from torch.nn.utils import parametrize

from models.swin_dora import QVDoRA, add_swin_qv_dora


class _Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _Attention(dim)


class SwinDoRATest(unittest.TestCase):
    def test_initialization_preserves_qkv_weight(self):
        torch.manual_seed(1)
        weight = torch.randn(18, 6)
        adapter = QVDoRA(weight, rank=2, alpha=2.0)

        torch.testing.assert_close(adapter(weight), weight)

    def test_k_is_unchanged_and_qv_norms_follow_magnitude(self):
        torch.manual_seed(2)
        weight = torch.randn(18, 6)
        adapter = QVDoRA(weight, rank=2, alpha=4.0)
        with torch.no_grad():
            adapter.dora_B_q.normal_()
            adapter.dora_B_v.normal_()
            adapter.magnitude_q.add_(0.5)
            adapter.magnitude_v.add_(0.25)

        q_weight, k_weight, v_weight = adapter(weight).chunk(3, dim=0)
        torch.testing.assert_close(k_weight, weight.chunk(3, dim=0)[1])
        torch.testing.assert_close(
            torch.linalg.vector_norm(q_weight, dim=1),
            adapter.magnitude_q,
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(v_weight, dim=1),
            adapter.magnitude_v,
        )

    def test_registration_adds_only_dora_parameters(self):
        features = nn.Sequential(_Block(8), _Block(8))
        original_weights = [
            block.attn.qkv.weight.detach().clone() for block in features
        ]
        for parameter in features.parameters():
            parameter.requires_grad_(False)

        count = add_swin_qv_dora(features, rank=2, alpha=2.0)

        self.assertEqual(count, 2)
        for block, original_weight in zip(features, original_weights):
            self.assertTrue(parametrize.is_parametrized(
                block.attn.qkv, 'weight'))
            torch.testing.assert_close(block.attn.qkv.weight, original_weight)
            trainable_names = {
                name for name, parameter in block.attn.qkv.named_parameters()
                if parameter.requires_grad
            }
            self.assertEqual(
                trainable_names,
                {
                    'parametrizations.weight.0.dora_A_q',
                    'parametrizations.weight.0.dora_B_q',
                    'parametrizations.weight.0.dora_A_v',
                    'parametrizations.weight.0.dora_B_v',
                    'parametrizations.weight.0.magnitude_q',
                    'parametrizations.weight.0.magnitude_v',
                },
            )

            block.attn.qkv.weight.sum().backward()
            adapter = block.attn.qkv.parametrizations.weight[0]
            for parameter in adapter.parameters():
                self.assertIsNotNone(parameter.grad)
            self.assertIsNone(
                block.attn.qkv.parametrizations.weight.original.grad)

    def test_rejects_duplicate_registration(self):
        features = nn.Sequential(_Block(8))
        add_swin_qv_dora(features, rank=2, alpha=2.0)
        with self.assertRaisesRegex(ValueError, 'already attached'):
            add_swin_qv_dora(features, rank=2, alpha=2.0)


if __name__ == '__main__':
    unittest.main()
