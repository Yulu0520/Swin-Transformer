import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from utils.helper import Save_Handle
from utils.regression_trainer_cosine_multibatch import RegTrainer, load_checkpoint


class MetricCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.trainer = RegTrainer.__new__(RegTrainer)
        t = self.trainer
        t.save_dir = self.temp.name
        t.model = nn.ModuleDict({name: nn.Linear(2, 1)
                                for name in ('baseline', 'dora', 'dora1')})
        t.args = SimpleNamespace(
            model_name='swin_t_trans', stage='dora1', crop_size=384, fusion_stages=(3,),
            dora_rank=4, dora_alpha=4., dora1_rank=4, dora1_alpha=4.,
            seed=42, dora_checkpoint='parent.pth', lr=2e-4, weight_decay=1e-5)
        t.datasets = {'val': SimpleNamespace(root_path='/data/day/val')}
        t.metric_bests = {}
        t.optimizer = torch.optim.Adam(t.model.parameters(), lr=t.args.lr,
                                       weight_decay=t.args.weight_decay)
        t.scheduler = None
        t.epoch, t.epoch_index = 1, 0

    def weights(self, metric):
        return load_checkpoint(str(Path(self.temp.name) / ('best_' + metric + '.pth')))

    def test_independent_selection_and_complete_loadable_weights(self):
        t = self.trainer
        t.save_metric_bests(10., 20., 182)
        first = self.weights('rmse')
        nn.ModuleDict({name: nn.Linear(2, 1) for name in
                       ('baseline', 'dora', 'dora1')}).load_state_dict(first)
        with torch.no_grad():
            for p in t.model.parameters():
                p.add_(1.)
        t.epoch, t.epoch_index = 2, 1
        t.save_metric_bests(9., 21., 182)
        self.assertEqual(t.metric_bests['mae']['epoch'], 2)
        self.assertEqual(t.metric_bests['rmse']['epoch'], 1)
        for key, value in first.items():
            torch.testing.assert_close(self.weights('rmse')[key], value)
            torch.testing.assert_close(self.weights('mae')[key], value + 1.)
        t.epoch, t.epoch_index = 3, 2
        t.save_metric_bests(11., 19., 182)
        self.assertEqual(t.metric_bests['mae']['epoch'], 2)
        self.assertEqual(t.metric_bests['rmse']['epoch'], 3)
        metadata = json.loads((Path(t.save_dir) / 'best_rmse.json').read_text())
        self.assertEqual(metadata['mae'], 11.)
        self.assertEqual(metadata['mse'], 361.)
        self.assertEqual(metadata['active_route'], 'dora1')
        self.assertEqual(metadata['validation_images'], 182)

    def test_ties_and_nonfinite_metrics_do_not_overwrite(self):
        t = self.trainer
        t.save_metric_bests(10., 20., 182)
        t.epoch, t.epoch_index = 2, 1
        t.save_metric_bests(10., 20., 182)
        t.save_metric_bests(float('nan'), float('inf'), 182)
        self.assertEqual(t.metric_bests['mae']['epoch'], 1)
        self.assertEqual(t.metric_bests['rmse']['epoch'], 1)

    def test_epoch_checkpoint_preserves_independent_minima(self):
        t = self.trainer
        t.save_metric_bests(10., 20., 182)
        t.optimizer = torch.optim.Adam(t.model.parameters())
        t.save_list = Save_Handle(max_num=1)
        t.best_mae, t.best_mse = 12., 18.
        t.best_model_path, t.best_count = 'best_model.pth', 0
        t.save_epoch()
        checkpoint = load_checkpoint(str(Path(t.save_dir) / '1_ckpt.tar'))
        t.metric_bests = checkpoint['metric_bests']
        t.epoch, t.epoch_index = 2, 1
        t.save_metric_bests(11., 21., 182)
        self.assertEqual(t.metric_bests['mae']['epoch'], 1)
        self.assertEqual(t.metric_bests['rmse']['epoch'], 1)
        t.save_epoch()
        self.assertFalse((Path(t.save_dir) / '1_ckpt.tar').exists())
        self.assertTrue((Path(t.save_dir) / 'best_mae.pth').exists())
        self.assertTrue((Path(t.save_dir) / 'best_rmse.pth').exists())


if __name__ == '__main__':
    unittest.main()
