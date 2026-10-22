import unittest

import torch

from losses.count_loss import count_l1_loss


class CountL1LossTest(unittest.TestCase):
    def test_uses_fractional_targets_and_means_over_images(self):
        density = torch.tensor([[[[1.0, 2.0]]], [[[2.0, 3.0]]]])
        targets = [torch.tensor([1.0, 0.4]), torch.tensor([1.0, 1.0, 0.3])]
        # (|3 - 1.4| + |5 - 2.3|) / 2, not len(targets[i]).
        self.assertAlmostEqual(count_l1_loss(density, targets).item(), 2.15, places=5)

    def test_opposing_errors_do_not_cancel_and_gradients_reach_density(self):
        density = torch.tensor([[[[6.0, 6.0]]], [[[3.0, 4.0]]]],
                               requires_grad=True)
        targets = [torch.ones(10), torch.ones(9)]
        loss = count_l1_loss(density, targets)
        self.assertEqual(loss.item(), 2.0)
        loss.backward()
        torch.testing.assert_close(
            density.grad, torch.tensor([[[[0.5, 0.5]]], [[[-0.5, -0.5]]]])
        )

    def test_empty_crop_has_zero_target_and_penalizes_false_positives(self):
        density = torch.tensor([[[[0.0, 2.0]]], [[[0.0, 0.0]]]],
                               requires_grad=True)
        loss = count_l1_loss(density, [torch.empty(0), torch.empty(0)])
        self.assertEqual(loss.item(), 1.0)
        loss.backward()
        torch.testing.assert_close(
            density.grad, torch.tensor([[[[0.5, 0.5]]], [[[0.0, 0.0]]]])
        )


if __name__ == '__main__':
    unittest.main()
