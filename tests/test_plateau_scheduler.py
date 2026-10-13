import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from utils.helper import Save_Handle
from utils.regression_trainer_cosine_multibatch import RegTrainer, load_checkpoint


class PlateauSchedulerTest(unittest.TestCase):
    def make_trainer(self):
        t = RegTrainer.__new__(RegTrainer)
        t.args = SimpleNamespace(
            lr_scheduler='plateau', scheduler_factor=.5, scheduler_patience=1,
            scheduler_min_lr=2.5e-5, scheduler_threshold=1e-4,
            scheduler_cooldown=0, reset_scheduler=False, stage='baseline',
            model_name='swin_t_trans', crop_size=256, seed=42, fusion_stages=(3,),
            dora_rank=4, dora_alpha=4., dora1_rank=4, dora1_alpha=4.)
        t.model = nn.Linear(2, 1)
        t.optimizer = torch.optim.Adam(t.model.parameters(), lr=1e-4,
                                       weight_decay=1e-5)
        t.writer = Mock()
        t.epoch, t.epoch_index = 1, 0
        t.configure_scheduler({})
        return t

    def test_plateau_reduces_lr_and_respects_floor_without_changing_decay(self):
        t = self.make_trainer()
        rates = []
        for metric in (100., 101., 102., 103., 104., 105., 106.):
            t.step_scheduler(metric)
            rates.append(t.optimizer.param_groups[0]['lr'])
        self.assertEqual(rates, [1e-4, 1e-4, 5e-5, 5e-5, 2.5e-5, 2.5e-5, 2.5e-5])
        self.assertEqual(t.optimizer.param_groups[0]['weight_decay'], 1e-5)

    def test_saved_epoch_continues_plateau_patience(self):
        t = self.make_trainer()
        t.step_scheduler(100.)
        t.step_scheduler(101.)
        with tempfile.TemporaryDirectory() as directory:
            t.save_dir = directory
            t.save_list = Save_Handle(1)
            t.best_mae, t.best_mse = 60., 100.
            t.best_model_path, t.best_count, t.metric_bests = None, 0, {}
            t.save_epoch()
            checkpoint = load_checkpoint(str(Path(directory) / '1_ckpt.tar'))
        resumed = self.make_trainer()
        resumed.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resumed.configure_scheduler(checkpoint)
        resumed.step_scheduler(102.)
        t.step_scheduler(102.)
        self.assertEqual(resumed.optimizer.param_groups[0]['lr'], 5e-5)
        self.assertEqual(resumed.scheduler.state_dict(), t.scheduler.state_dict())

    def test_reset_disabled_and_nonfinite_metrics(self):
        t = self.make_trainer()
        t.step_scheduler(100.)
        state = t.scheduler.state_dict()
        t.step_scheduler(float('nan'))
        self.assertEqual(state, t.scheduler.state_dict())
        t.args.reset_scheduler = True
        t.configure_scheduler({'scheduler_state_dict': state})
        self.assertEqual(t.scheduler.num_bad_epochs, 0)
        self.assertEqual(t.scheduler.best, float('inf'))
        t.args.lr_scheduler = 'none'
        t.configure_scheduler({})
        t.step_scheduler(101.)
        self.assertIsNone(t.scheduler)


if __name__ == '__main__':
    unittest.main()
