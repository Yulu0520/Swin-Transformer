import ast
import glob
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from utils.top_checkpoints import TopCheckpoints
from utils.helper import Save_Handle


class TopCheckpointsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manager = TopCheckpoints(self.root / 'original', 5)

    def submit(self, epoch, score, manager=None):
        manager = manager or self.manager
        return manager.consider(
            epoch, score / 3, score / 3,
            lambda path: Path(path).write_text(str(epoch)))

    def assert_weights(self, manager, epochs):
        self.assertEqual([record['epoch'] for record in manager.records], epochs)
        self.assertEqual(
            [Path(manager.path(rank)).read_text()
             for rank in range(1, len(epochs) + 1)],
            [str(epoch) for epoch in epochs])
        self.assertEqual(len(list(Path(manager.directory).glob('*.pth'))),
                         len(epochs))
        with open(Path(manager.directory) / 'top_models.json') as stream:
            self.assertEqual(json.load(stream), manager.records)

    def test_global_top_five_includes_results_that_do_not_beat_best(self):
        for epoch, score in enumerate([3, 30, 15, 24, 9, 12, 6, 99], 1):
            self.submit(epoch, score)
        self.assert_weights(self.manager, [1, 7, 5, 6, 3])

    def test_new_best_shifts_all_ranks_and_evicts_fifth(self):
        for epoch in range(1, 6):
            self.submit(epoch, epoch * 3)
        self.assertEqual(self.submit(6, 0), 1)
        self.assert_weights(self.manager, [6, 1, 2, 3, 4])

    def test_ties_keep_earlier_results(self):
        for epoch in range(1, 6):
            self.submit(epoch, 3)
        self.assertIsNone(self.submit(6, 3))
        self.assert_weights(self.manager, [1, 2, 3, 4, 5])

    def test_invalid_metrics_never_write(self):
        def unexpected_save(path):
            self.fail('invalid metrics must not save weights')
        for mae, rmse in [(float('nan'), 1), (1, float('inf')),
                          (float('-inf'), 1), (1e308, 1e308)]:
            self.assertIsNone(self.manager.consider(1, mae, rmse, unexpected_save))
        self.assertEqual(list(Path(self.manager.directory).iterdir()), [])

    def test_failed_save_preserves_retained_weights(self):
        self.submit(1, 9)

        def fail(path):
            Path(path).write_text('partial')
            raise OSError('disk full')

        with self.assertRaisesRegex(OSError, 'disk full'):
            self.manager.consider(2, 1, 1, fail)
        self.assert_weights(self.manager, [1])
        self.assertFalse(list(Path(self.manager.directory).glob('checkpoint-*.tmp')))

    def test_resume_copies_then_continues_without_modifying_source(self):
        for epoch in range(1, 6):
            self.submit(epoch, epoch * 3)
        resumed = TopCheckpoints(self.root / 'resumed', 5)
        resumed.restore(self.manager.records, self.manager.directory)
        self.submit(6, 0, resumed)
        self.assert_weights(resumed, [6, 1, 2, 3, 4])
        self.assert_weights(self.manager, [1, 2, 3, 4, 5])

    def test_resume_with_smaller_limit(self):
        for epoch in range(1, 6):
            self.submit(epoch, epoch * 3)
        resumed = TopCheckpoints(self.root / 'resumed', 2)
        resumed.restore(self.manager.records, self.manager.directory)
        self.assert_weights(resumed, [1, 2])
        self.submit(6, 0, resumed)
        self.assert_weights(resumed, [6, 1])

    def test_resume_rejects_missing_or_mismatched_weights(self):
        self.submit(1, 3)
        resumed = TopCheckpoints(self.root / 'resumed', 5)
        stale = [dict(self.manager.records[0], epoch=99)]
        with self.assertRaisesRegex(ValueError, 'ranking differs'):
            resumed.restore(stale, self.manager.directory)
        Path(self.manager.path()).unlink()
        with self.assertRaises(FileNotFoundError):
            resumed.restore(self.manager.records, self.manager.directory)

    def test_single_best_mode_rejects_non_improvements(self):
        manager = TopCheckpoints(self.root / 'single', 1)
        self.submit(1, 9, manager)
        self.assertIsNone(self.submit(2, 12, manager))
        self.submit(3, 3, manager)
        self.assert_weights(manager, [3])

    def test_checkpoint_resolution_ignores_newer_lower_ranks(self):
        # Load the actual resolver alone so this filesystem test needs no torch.
        source = Path(__file__).resolve().parents[1] / 'test.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == 'resolve_checkpoint')
        namespace = {'os': os, 'glob': glob}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(source), 'exec'), namespace)
        run = self.root / 'swin' / 'baseline' / 'run'
        run.mkdir(parents=True)
        best = run / 'best_model.pth'
        best.touch()
        lower = run / 'best_model_rank2.pth'
        lower.touch()
        os.utime(best, (100, 100))
        os.utime(lower, (200, 200))
        resolve = namespace['resolve_checkpoint']
        self.assertEqual(resolve('', str(self.root), 'swin', 'baseline'), str(best))
        self.assertEqual(resolve(str(lower), '', '', ''), str(lower))
        best.unlink()
        with self.assertRaises(FileNotFoundError):
            resolve('', str(self.root), 'swin', 'baseline')
        legacy = run / 'best_model_12.pth'
        legacy.touch()
        self.assertEqual(resolve('', str(self.root), 'swin', 'baseline'), str(legacy))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
    def test_trainer_validation_and_resume_payload_with_real_torch_weights(self):
        import torch
        from utils.regression_trainer_cosine_multibatch import RegTrainer

        class DensityModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.density = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, inputs):
                return (self.density.reshape(1, 1, 1, 1),)

        for save_all in (False, True):
            with self.subTest(save_all=save_all):
                trainer = object.__new__(RegTrainer)
                trainer.save_dir = str(self.root / str(save_all))
                trainer.top_models = TopCheckpoints(
                    trainer.save_dir, 1 if save_all else 5)
                trainer.args = SimpleNamespace(
                    stage='baseline', max_epoch=7, model_name='swin_t_trans',
                    crop_size=384, dora_rank=12, dora_alpha=12,
                    dora1_rank=8, dora1_alpha=8, seed=42)
                trainer.device = torch.device('cpu')
                trainer.model = DensityModel()
                trainer.optimizer = torch.optim.Adam(trainer.model.parameters())
                trainer.writer = Mock()
                trainer.best_mae = trainer.best_mse = float('inf')
                trainer.best_model_path = None
                trainer.best_count = 0
                trainer.save_all = save_all
                trainer.save_list = Save_Handle(1)
                trainer.dataloaders = {'val': [(
                    torch.ones(1, 1, 1, 1), torch.tensor([0.0]), ['image'])]}
                for epoch, error in enumerate([3, 10, 5, 8, 2, 4, 1], 1):
                    trainer.epoch = epoch
                    trainer.epoch_index = epoch - 1
                    with torch.no_grad():
                        trainer.model.density.fill_(error)
                    trainer.val_epoch()
                    trainer.save_epoch()
                expected = [7] if save_all else [7, 5, 1, 6, 3]
                self.assertEqual([item['epoch'] for item in
                                  trainer.top_models.records], expected)
                for rank, epoch in enumerate(expected, 1):
                    weights = torch.load(trainer.top_models.path(rank),
                                         map_location='cpu', weights_only=True)
                    self.assertEqual(weights['density'].item(),
                                     [3, 10, 5, 8, 2, 4, 1][epoch - 1])
                payload = torch.load(Path(trainer.save_dir) / '7_ckpt.tar',
                                     map_location='cpu', weights_only=True)
                self.assertEqual(payload['top_models'], trainer.top_models.records)
                self.assertEqual(payload['best_model_path'], trainer.best_model_path)
                self.assertEqual(len(list(Path(trainer.save_dir).glob('*.tar'))), 1)
                if save_all:
                    self.assertEqual(trainer.best_count, 3)
                    self.assertEqual(len(list(Path(trainer.save_dir).glob(
                        'best_model_[0-9]*.pth'))), 3)


if __name__ == '__main__':
    unittest.main()
