import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from utils.topk_checkpoint import TopKCheckpoints, best_ranked_path
from utils.counting_model import resolve_checkpoint


class TopKCheckpointTest(unittest.TestCase):
    def test_keeps_five_best_validation_results_not_only_record_breakers(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = TopKCheckpoints(directory)
            # Epochs 2-5 are not record breakers but still belong in the top five.
            values = [(10., 10.), (11., 10.), (12., 10.), (13., 10.),
                      (14., 10.), (15., 10.), (9., 10.), (8., 10.)]
            for epoch, (mae, rmse) in enumerate(values, 1):
                manager.consider({'weight': torch.tensor(epoch)}, mae, rmse, epoch)
            self.assertEqual([entry['epoch'] for entry in manager.entries], [8, 7, 1, 2, 3])
            files = list(Path(directory).glob('*.pth'))
            self.assertEqual(len(files), 5)
            self.assertEqual(Path(manager.best_path).name, 'MAE8.00_epoch8.pth')
            self.assertFalse((Path(directory) / 'MAE13.00_epoch4.pth').exists())
            data = json.loads((Path(directory) / 'rankings.json').read_text())
            self.assertEqual(data['entries'], manager.state_dict())

    def test_ranking_uses_full_precision_score_and_survives_resume(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'source'
            manager = TopKCheckpoints(source)
            manager.consider({}, 9.001, 20., 1)
            manager.consider({}, 9.002, 10., 2)
            manager.consider({}, 9.003, 11., 3)
            # Rounded MAE and modification time must not determine rank 1.
            os.utime(source / manager.entries[-1]['file'], (9999999999, 9999999999))
            self.assertEqual(Path(best_ranked_path(source)).name, 'MAE9.00_epoch2.pth')
            restored = TopKCheckpoints(Path(root) / 'resumed')
            restored.restore(manager.state_dict(), source)
            restored.consider({}, 8., 10., 4)
            self.assertEqual(restored.entries[0]['epoch'], 4)
            self.assertEqual(len(restored.entries), 4)
            self.assertEqual(Path(resolve_checkpoint(str(source), root, 'tiny', 'baseline')).name,
                             'MAE9.00_epoch2.pth')

    def test_auto_resolution_uses_run_index_not_file_timestamp(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / 'tiny' / 'finetune' / '1112-120000'
            manager = TopKCheckpoints(run)
            manager.consider({}, 8., 10., 1)
            manager.consider({}, 7., 20., 2)
            self.assertEqual(resolve_checkpoint('', root, 'tiny', 'finetune'), manager.best_path)

    def test_ties_epoch_zero_invalid_metrics_and_missing_ranked_file(self):
        with tempfile.TemporaryDirectory() as root:
            manager = TopKCheckpoints(root)
            manager.consider({}, 10., 10., 0)
            manager.consider({}, 10., 10., 1)
            self.assertEqual(manager.entries[0]['epoch'], 0)
            with self.assertRaises(ValueError):
                manager.consider({}, float('nan'), 1., 2)
            with self.assertRaises(ValueError):
                manager.consider({}, 10., 10., 1)
            Path(manager.best_path).unlink()
            with self.assertRaises(FileNotFoundError):
                best_ranked_path(root)
