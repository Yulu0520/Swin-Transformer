import unittest

import torch
from torch import nn

from models.swin_dora import add_swin_qv_dora
from models.swin_dual_dora import (
    DualQVDoRAAdapter,
    add_swin_qv_dual_dora,
    extract_dora_from_legacy_state,
    has_legacy_dora_lora_adapters,
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


class SwinDualDoRATest(unittest.TestCase):
    def test_both_branches_start_from_the_same_baseline(self):
        features = nn.Sequential(_Block(8), _Block(8))
        baseline_weights = [
            block.attn.qkv.weight.detach().clone() for block in features
        ]

        count = add_swin_qv_dual_dora(
            features, 2, 2.0, 3, 3.0, active_route='dora')

        self.assertEqual(count, 2)
        for block, baseline_weight in zip(features, baseline_weights):
            torch.testing.assert_close(block.attn.qkv.weight, baseline_weight)
            block.attn.qkv.parametrizations.weight[0].set_route('dora1')
            torch.testing.assert_close(block.attn.qkv.weight, baseline_weight)

    def test_trained_dora_is_preserved_when_dora1_is_attached(self):
        features = nn.Sequential(_Block(8), _Block(8))
        baseline_weights = [
            block.attn.qkv.weight.detach().clone() for block in features
        ]
        add_swin_qv_dora(features, rank=2, alpha=2.0)
        with torch.no_grad():
            for block in features:
                dora = block.attn.qkv.parametrizations.weight[0]
                dora.dora_B_q.normal_()
                dora.dora_B_v.normal_()
                dora.magnitude_q.add_(0.25)
        dora_weights = [
            block.attn.qkv.weight.detach().clone() for block in features
        ]

        add_swin_qv_dual_dora(
            features, 2, 2.0, 3, 3.0, active_route='dora1')

        for block, baseline_weight, dora_weight in zip(
                features, baseline_weights, dora_weights):
            router = block.attn.qkv.parametrizations.weight[0]
            self.assertIsInstance(router, DualQVDoRAAdapter)
            torch.testing.assert_close(block.attn.qkv.weight, baseline_weight)
            router.set_route('dora')
            torch.testing.assert_close(block.attn.qkv.weight, dora_weight)

    def test_only_selected_branch_is_called(self):
        features = nn.Sequential(_Block(8))
        add_swin_qv_dual_dora(
            features, 2, 2.0, 2, 2.0, active_route='dora')
        qkv = features[0].attn.qkv
        router = qkv.parametrizations.weight[0]
        calls = {'dora': 0, 'dora1': 0}
        dora_hook = router.dora.register_forward_hook(
            lambda *unused: calls.__setitem__('dora', calls['dora'] + 1))
        dora1_hook = router.dora1.register_forward_hook(
            lambda *unused: calls.__setitem__('dora1', calls['dora1'] + 1))
        try:
            _ = qkv.weight
            self.assertEqual(calls, {'dora': 1, 'dora1': 0})

            calls.update(dora=0, dora1=0)
            set_swin_adapter_route(features, 'dora1')
            _ = qkv.weight
            self.assertEqual(calls, {'dora': 0, 'dora1': 1})

            calls.update(dora=0, dora1=0)
            set_swin_adapter_route(features, 'baseline')
            _ = qkv.weight
            self.assertEqual(calls, {'dora': 0, 'dora1': 0})
        finally:
            dora_hook.remove()
            dora1_hook.remove()

    def test_dual_dora_state_dict_round_trip(self):
        source = nn.Sequential(_Block(8))
        add_swin_qv_dual_dora(
            source, 2, 2.0, 3, 3.0, active_route='dora1')
        with torch.no_grad():
            router = source[0].attn.qkv.parametrizations.weight[0]
            router.dora.dora_B_q.normal_()
            router.dora1.dora_B_v.normal_()
            router.dora1.magnitude_v.add_(0.5)
        state = source.state_dict()

        restored = nn.Sequential(_Block(8))
        add_swin_qv_dual_dora(
            restored, 2, 2.0, 3, 3.0, active_route='dora1')
        restored.load_state_dict(state)

        for route in ('baseline', 'dora', 'dora1'):
            set_swin_adapter_route(source, route)
            set_swin_adapter_route(restored, route)
            torch.testing.assert_close(
                source[0].attn.qkv.weight,
                restored[0].attn.qkv.weight,
            )

    def test_legacy_dora_lora_checkpoint_keeps_trained_dora(self):
        source = nn.Sequential(_Block(8))
        add_swin_qv_dual_dora(
            source, 2, 2.0, 3, 3.0, active_route='dora')
        with torch.no_grad():
            dora = source[0].attn.qkv.parametrizations.weight[0].dora
            dora.dora_B_q.normal_()
            dora.magnitude_v.add_(0.25)
        expected = source[0].attn.qkv.weight.detach().clone()

        # Reproduce the former container prefixes. The migration intentionally
        # drops the second branch rather than treating it as DoRA1.
        state = source.state_dict()
        legacy = state.__class__()
        for key, value in state.items():
            legacy[key.replace(
                '.parametrizations.weight.0.dora1.',
                '.parametrizations.weight.0.lora.')] = value
        self.assertTrue(has_legacy_dora_lora_adapters(legacy))

        restored = nn.Sequential(_Block(8))
        add_swin_qv_dora(restored, rank=2, alpha=2.0)
        restored.load_state_dict(extract_dora_from_legacy_state(legacy))
        torch.testing.assert_close(restored[0].attn.qkv.weight, expected)

    def _assert_only_branch_trainable(self, trainable_route):
        features = nn.Sequential(_Block(8), _Block(8))
        add_swin_qv_dual_dora(
            features, 2, 2.0, 3, 3.0, active_route=trainable_route)
        for parameter in features.parameters():
            parameter.requires_grad_(False)
        for module in features.modules():
            if isinstance(module, DualQVDoRAAdapter):
                for parameter in getattr(module, trainable_route).parameters():
                    parameter.requires_grad_(True)

        frozen_route = 'dora1' if trainable_route == 'dora' else 'dora'
        expected_suffixes = {
            'dora_A_q', 'dora_B_q', 'dora_A_v', 'dora_B_v',
            'magnitude_q', 'magnitude_v',
        }
        for block in features:
            qkv = block.attn.qkv
            router = qkv.parametrizations.weight[0]
            trainable_names = {
                name for name, parameter in qkv.named_parameters()
                if parameter.requires_grad
            }
            self.assertEqual(
                trainable_names,
                {'parametrizations.weight.0.{}.{}'.format(
                    trainable_route, suffix) for suffix in expected_suffixes},
            )

            qkv.weight.sum().backward()
            self.assertTrue(all(
                parameter.grad is not None
                for parameter in getattr(router, trainable_route).parameters()))
            self.assertTrue(all(
                parameter.grad is None
                for parameter in getattr(router, frozen_route).parameters()))
            self.assertIsNone(qkv.parametrizations.weight.original.grad)

    def test_dora_training_freezes_baseline_and_dora1(self):
        self._assert_only_branch_trainable('dora')

    def test_dora1_training_freezes_baseline_and_dora(self):
        self._assert_only_branch_trainable('dora1')

    def test_route_validation(self):
        features = nn.Sequential(_Block(8))
        add_swin_qv_dual_dora(
            features, 2, 2.0, 2, 2.0, active_route='dora1')
        with self.assertRaisesRegex(ValueError, 'adapter route'):
            set_swin_adapter_route(features, 'both')


if __name__ == '__main__':
    unittest.main()
