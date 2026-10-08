import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.utils import parametrize

from models.man_dora import DualMANDoRAAdapter, PROJECTION_INDICES
from models.swin_c_multibatch import SwinTransMultiBatch
from models.transformer_cosine_multibatch import (
    TransformerEncoder, TransformerEncoderLayer,
)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 24)


class _SmallModel(SwinTransMultiBatch):
    """Exercise production routing/freezing/loading with a small real MAN."""
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone_features = nn.Sequential(_Block(), _Block())
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(8, 2, 16, dropout=0), 2)
        self.head = nn.Linear(8, 1)


class MANDoRATest(unittest.TestCase):
    def test_initial_output_and_k_weights(self):
        torch.manual_seed(42)
        model = _SmallModel().eval()
        tokens = torch.randn(4, 2, 8)
        expected, expected_features = model.encoder(tokens, (2, 2))
        raw_weights = [layer.self_attn.in_proj_weight.detach().clone()
                       for layer in model.encoder.layers]
        model.enable_dual_dora(2, 2, 3, 3)
        for route in ('baseline', 'dora', 'dora1'):
            model.set_adapter_route(route)
            actual, actual_features = model.encoder(tokens, (2, 2))
            torch.testing.assert_close(actual, expected)
            for actual_feature, expected_feature in zip(
                    actual_features, expected_features):
                torch.testing.assert_close(actual_feature, expected_feature)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, DualMANDoRAAdapter):
                    for projection in module.dora.projections.values():
                        projection.B.normal_()
                        projection.magnitude.add_(0.5)
        model.set_adapter_route('dora')
        for layer, original in zip(model.encoder.layers, raw_weights):
            actual = layer.self_attn.in_proj_weight
            for index in range(9):
                part = slice(index * 8, (index + 1) * 8)
                if index in PROJECTION_INDICES.values():
                    self.assertFalse(torch.equal(actual[part], original[part]))
                else:
                    self.assertTrue(torch.equal(actual[part], original[part]))

    def test_only_selected_branch_trains_and_routes_are_synchronized(self):
        for route in ('dora', 'dora1'):
            model = _SmallModel().eval()
            model.enable_dual_dora(2, 2, 2, 2,
                                   active_route=route, trainable_route=route)
            before = {name: value.detach().clone()
                      for name, value in model.named_parameters()}
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=0.01)
            output, features = model.encoder(torch.randn(4, 2, 8), (2, 2))
            loss = output.square().mean()
            for block in model.backbone_features:
                loss = loss + block.attn.qkv(torch.randn(2, 8)).square().mean()
            # Match the production cosine regularizer so mask projections
            # also have a direct differentiable supervision path.
            for feature in features:
                loss = loss + (1 - nn.functional.cosine_similarity(
                    feature, feature.mean(dim=1, keepdim=True), dim=2)).sum()
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIn('.0.' + route + '.', name)
                    self.assertIsNotNone(parameter.grad, name)
                else:
                    self.assertIsNone(parameter.grad, name)
            optimizer.step()
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    self.assertTrue(torch.equal(parameter, before[name]), name)
            self.assertTrue(any(
                not torch.equal(parameter, before[name])
                for name, parameter in model.named_parameters()
                if name.startswith('encoder.') and parameter.requires_grad))
            for selected in ('baseline', 'dora', 'dora1'):
                model.set_adapter_route(selected)
                for layer in model.encoder.layers:
                    self.assertEqual(layer.self_attn.parametrizations
                                     .in_proj_weight[0].active_route, selected)
                for block in model.backbone_features:
                    self.assertEqual(block.attn.qkv.parametrizations
                                     .weight[0].active_route, selected)

    def test_one_man_composition_per_forward_and_inactive_branch_not_called(self):
        model = _SmallModel().eval()
        model.enable_dual_dora(2, 2, 2, 2, active_route='dora')
        adapter = model.encoder.layers[0].self_attn.parametrizations.in_proj_weight[0]
        with patch.object(adapter.dora, 'forward', wraps=adapter.dora.forward) as active:
            with patch.object(adapter.dora1, 'forward', wraps=adapter.dora1.forward) as inactive:
                model.encoder(torch.randn(4, 1, 8), (2, 2))
                self.assertEqual(active.call_count, 1)
                inactive.assert_not_called()

    def test_new_checkpoint_round_trip_and_partial_checkpoint_rejected(self):
        model = _SmallModel().eval()
        model.enable_dual_dora(2, 2, 3, 3)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, DualMANDoRAAdapter):
                    for branch in (module.dora, module.dora1):
                        for projection in branch.projections.values():
                            projection.B.normal_()
        state = model.state_dict()
        restored = _SmallModel().eval()
        restored.enable_dual_dora(2, 2, 3, 3)
        restored.load_compatible_state_dict(state)
        tokens = torch.randn(4, 1, 8)
        for route in ('baseline', 'dora', 'dora1'):
            model.set_adapter_route(route)
            restored.set_adapter_route(route)
            torch.testing.assert_close(model.encoder(tokens, (2, 2))[0],
                                       restored.encoder(tokens, (2, 2))[0])
        partial = state.copy()
        del partial['encoder.layers.0.self_attn.parametrizations.in_proj_weight'
                    '.0.dora.projections.global_q.magnitude']
        with self.assertRaises(RuntimeError):
            restored.load_compatible_state_dict(partial)

    def test_old_swin_only_checkpoint_uses_checkpoint_magnitudes(self):
        source = _SmallModel().eval()
        source.enable_dual_dora(2, 2, 3, 3)
        # Produce the old format, with no MAN adapters and trained Swin DoRA.
        with torch.no_grad():
            source.backbone_features[0].attn.qkv.parametrizations.weight[0].dora.dora_B_q.normal_()
        for layer in source.encoder.layers:
            parametrize.remove_parametrizations(
                layer.self_attn, 'in_proj_weight', leave_parametrized=False)
        old = source.state_dict()
        restored = _SmallModel().eval()
        restored.enable_dual_dora(2, 2, 3, 3,
                                  active_route='dora1', trainable_route='dora1')
        restored.load_compatible_state_dict(old)
        tokens = torch.randn(4, 1, 8)
        expected = source.encoder(tokens, (2, 2))[0]
        for route in ('baseline', 'dora', 'dora1'):
            source.set_adapter_route(route)
            restored.set_adapter_route(route)
            torch.testing.assert_close(restored.encoder(tokens, (2, 2))[0], expected)
            torch.testing.assert_close(
                source.backbone_features[0].attn.qkv.weight,
                restored.backbone_features[0].attn.qkv.weight)


if __name__ == '__main__':
    unittest.main()
