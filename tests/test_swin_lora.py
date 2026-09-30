import unittest

import torch
from torch import nn
from torch.nn.utils import parametrize

from models.swin_dora import add_swin_qv_dora
from models.swin_lora import (
    ParallelQVAdapter,
    QVLoRA,
    add_swin_qv_parallel_adapters,
    set_swin_adapter_route,
)


class _Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _Attention(dim)


class SwinParallelAdapterTest(unittest.TestCase):
    def test_lora_zero_initialization_preserves_baseline_weight(self):
        torch.manual_seed(1)
        weight = torch.randn(18, 6)
        adapter = QVLoRA(weight, rank=2, alpha=2.0)
        torch.testing.assert_close(adapter(weight), weight)

    def test_lora_updates_qv_and_leaves_k_unchanged(self):
        torch.manual_seed(2)
        baseline_weight = torch.randn(18, 6)
        adapter = QVLoRA(baseline_weight, rank=2, alpha=4.0)
        with torch.no_grad():
            adapter.lora_B_q.normal_()
            adapter.lora_B_v.normal_()

        q_weight, k_weight, v_weight = adapter(baseline_weight).chunk(3, dim=0)
        base_q, base_k, base_v = baseline_weight.chunk(3, dim=0)
        torch.testing.assert_close(k_weight, base_k)
        self.assertFalse(torch.equal(q_weight, base_q))
        self.assertFalse(torch.equal(v_weight, base_v))

    def test_trained_dora_is_preserved_as_parallel_branch(self):
        features = nn.Sequential(_Block(8), _Block(8))
        baseline_weights = [
            block.attn.qkv.weight.detach().clone() for block in features
        ]
        for parameter in features.parameters():
            parameter.requires_grad_(False)
        add_swin_qv_dora(features, rank=2, alpha=2.0)
        with torch.no_grad():
            for block in features:
                dora = block.attn.qkv.parametrizations.weight[0]
                dora.dora_B_q.normal_()
                dora.dora_B_v.normal_()
        dora_weights = [
            block.attn.qkv.weight.detach().clone() for block in features
        ]

        count = add_swin_qv_parallel_adapters(
            features, 2, 2.0, 3, 3.0, active_route='lora')

        self.assertEqual(count, 2)
        for block, baseline_weight, dora_weight in zip(
                features, baseline_weights, dora_weights):
            parametrizations = block.attn.qkv.parametrizations.weight
            self.assertEqual(len(parametrizations), 1)
            self.assertIsInstance(parametrizations[0], ParallelQVAdapter)

            # Zero-initialized LoRA reads the raw baseline, not DoRA output.
            torch.testing.assert_close(block.attn.qkv.weight, baseline_weight)
            parametrizations[0].set_route('dora')
            torch.testing.assert_close(block.attn.qkv.weight, dora_weight)

    def test_only_selected_branch_is_called(self):
        features = nn.Sequential(_Block(8))
        add_swin_qv_parallel_adapters(
            features, 2, 2.0, 2, 2.0, active_route='dora')
        qkv = features[0].attn.qkv
        router = qkv.parametrizations.weight[0]
        calls = {'dora': 0, 'lora': 0}

        dora_hook = router.dora.register_forward_hook(
            lambda *unused: calls.__setitem__('dora', calls['dora'] + 1))
        lora_hook = router.lora.register_forward_hook(
            lambda *unused: calls.__setitem__('lora', calls['lora'] + 1))
        try:
            _ = qkv.weight
            self.assertEqual(calls, {'dora': 1, 'lora': 0})

            calls.update(dora=0, lora=0)
            set_swin_adapter_route(features, 'lora')
            _ = qkv.weight
            self.assertEqual(calls, {'dora': 0, 'lora': 1})

            calls.update(dora=0, lora=0)
            set_swin_adapter_route(features, 'baseline')
            _ = qkv.weight
            self.assertEqual(calls, {'dora': 0, 'lora': 0})
        finally:
            dora_hook.remove()
            lora_hook.remove()

    def test_dora_training_freezes_baseline_and_lora(self):
        features = nn.Sequential(_Block(8), _Block(8))
        add_swin_qv_parallel_adapters(
            features, 2, 2.0, 3, 3.0, active_route='dora')
        for parameter in features.parameters():
            parameter.requires_grad_(False)
        for module in features.modules():
            if isinstance(module, ParallelQVAdapter):
                for parameter in module.dora.parameters():
                    parameter.requires_grad_(True)

        for block in features:
            qkv = block.attn.qkv
            router = qkv.parametrizations.weight[0]
            trainable_names = {
                name for name, parameter in qkv.named_parameters()
                if parameter.requires_grad
            }
            self.assertEqual(
                trainable_names,
                {
                    'parametrizations.weight.0.dora.dora_A_q',
                    'parametrizations.weight.0.dora.dora_B_q',
                    'parametrizations.weight.0.dora.dora_A_v',
                    'parametrizations.weight.0.dora.dora_B_v',
                    'parametrizations.weight.0.dora.magnitude_q',
                    'parametrizations.weight.0.dora.magnitude_v',
                },
            )

            qkv.weight.sum().backward()
            self.assertTrue(all(parameter.grad is not None
                                for parameter in router.dora.parameters()))
            self.assertTrue(all(parameter.grad is None
                                for parameter in router.lora.parameters()))
            self.assertIsNone(qkv.parametrizations.weight.original.grad)

    def test_lora_training_freezes_baseline_and_dora(self):
        features = nn.Sequential(_Block(8), _Block(8))
        add_swin_qv_parallel_adapters(
            features, 2, 2.0, 3, 3.0, active_route='lora')
        for parameter in features.parameters():
            parameter.requires_grad_(False)
        for module in features.modules():
            if isinstance(module, ParallelQVAdapter):
                for parameter in module.lora.parameters():
                    parameter.requires_grad_(True)

        for block in features:
            qkv = block.attn.qkv
            router = qkv.parametrizations.weight[0]
            trainable_names = {
                name for name, parameter in qkv.named_parameters()
                if parameter.requires_grad
            }
            self.assertEqual(
                trainable_names,
                {
                    'parametrizations.weight.0.lora.lora_A_q',
                    'parametrizations.weight.0.lora.lora_B_q',
                    'parametrizations.weight.0.lora.lora_A_v',
                    'parametrizations.weight.0.lora.lora_B_v',
                },
            )

            qkv.weight.sum().backward()
            self.assertTrue(all(parameter.grad is None
                                for parameter in router.dora.parameters()))
            self.assertTrue(all(parameter.grad is not None
                                for parameter in router.lora.parameters()))
            self.assertIsNone(qkv.parametrizations.weight.original.grad)

    def test_route_validation(self):
        features = nn.Sequential(_Block(8))
        add_swin_qv_parallel_adapters(
            features, 2, 2.0, 2, 2.0, active_route='lora')
        with self.assertRaisesRegex(ValueError, 'adapter route'):
            set_swin_adapter_route(features, 'both')


if __name__ == '__main__':
    unittest.main()
