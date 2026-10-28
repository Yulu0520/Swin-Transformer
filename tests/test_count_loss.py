import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from losses.count_loss import count_l1_loss


class CountL1LossTest(unittest.TestCase):
    def test_fractional_targets_and_batch_average(self):
        density = torch.tensor([[[[1.0, 2.0]]], [[[2.0, 3.0]]]])
        targets = [torch.tensor([1.0, 0.4]), torch.tensor([1.0, 1.0, 0.3])]
        self.assertAlmostEqual(count_l1_loss(density, targets).item(), 2.15, places=5)

    def test_opposing_errors_do_not_cancel_and_gradients_reach_density(self):
        density = torch.tensor([[[[6.0, 6.0]]], [[[3.0, 4.0]]]], requires_grad=True)
        loss = count_l1_loss(density, [torch.ones(10), torch.ones(9)])
        self.assertEqual(loss.item(), 2.0)
        loss.backward()
        torch.testing.assert_close(
            density.grad, torch.tensor([[[[0.5, 0.5]]], [[[-0.5, -0.5]]]]))

    def test_empty_crops_penalize_false_positives(self):
        density = torch.tensor([[[[0.0, 2.0]]], [[[0.0, 0.0]]]], requires_grad=True)
        loss = count_l1_loss(density, [torch.empty(0), torch.empty(0)])
        self.assertEqual(loss.item(), 1.0)
        loss.backward()
        torch.testing.assert_close(
            density.grad, torch.tensor([[[[0.5, 0.5]]], [[[0.0, 0.0]]]]))

    def test_training_loop_weighting_logging_and_zero_weight(self):
        from utils.regression_trainer_cosine_multibatch import RegTrainer

        class DensityModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.count = torch.nn.Parameter(torch.tensor(3.0))

            def forward(self, images):
                return self.count.reshape(1, 1, 1, 1), []

        for weight in (0.0, 0.1):
            with self.subTest(weight=weight):
                trainer = object.__new__(RegTrainer)
                trainer.args = SimpleNamespace(stage='baseline', max_epoch=1,
                                               tensorboard_log_interval=1,
                                               workflow='legacy', count_loss_weight=weight)
                trainer.device = torch.device('cpu')
                trainer.model = DensityModel()
                trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
                trainer.writer = Mock()
                trainer.epoch, trainer.global_step = 1, 0
                trainer.post_prob = lambda *args: []
                trainer.criterion = lambda probs, targets, outputs: outputs.sum() * 0 + 0.5
                trainer.dataloaders = {'train': [(
                    torch.zeros(1, 3, 16, 16), [torch.zeros(2, 2)],
                    [torch.tensor([1.0, 0.4])], torch.tensor([16.]))]}
                trainer.train_eopch()
                scalars = {call.args[0]: call.args[1]
                           for call in trainer.writer.add_scalar.call_args_list}
                self.assertAlmostEqual(scalars['train/count_loss'], 1.6, places=5)
                self.assertAlmostEqual(scalars['train/crop_count_mae'], 1.6, places=5)
                self.assertAlmostEqual(scalars['train/weighted_count_loss'], weight * 1.6,
                                       places=5)
                self.assertAlmostEqual(scalars['train/total_loss'], 0.5 + weight * 1.6,
                                       places=5)
                self.assertAlmostEqual(trainer.model.count.item(), 3.0 - 0.1 * weight,
                                       places=5)

    def test_cli_rejects_negative_or_nonfinite_weight(self):
        from train import parse_args
        for value in ('-0.1', 'nan', 'inf'):
            with self.subTest(value=value), patch('sys.argv', [
                    'train.py', '--stage', 'baseline', '--count-loss-weight', value]):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parse_args()
                self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
