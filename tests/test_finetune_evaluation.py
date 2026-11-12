import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

import test as evaluation
import train


class RoutedCounter(nn.Module):
    def __init__(self):
        super().__init__()
        self.route = 'dora1'
        self.calls = []

    def set_adapter_route(self, route):
        self.route = route

    def forward(self, images):
        self.calls.append(self.route)
        return images.new_tensor([1. if self.route == 'dora1' else 3.])


class NightRouter(nn.Module):
    def forward(self, images):
        return images.new_tensor([[0., 3.]]).expand(len(images), -1)


class FineTuneEvaluationTest(unittest.TestCase):
    def test_real_dataset_routes_metrics_and_comparison_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, count in [('IMG_1', 2), ('night_IMG_1', 4)]:
                Image.new('RGB', (64, 64), (100, 120, 80)).save(root / (name + '.jpg'))
                np.save(root / (name + '.npy'), np.full((count, 2), 20.))
            args = SimpleNamespace(crop_size=64, downsample_ratio=8, num_workers=0, split='val')
            model = RoutedCounter()
            classifier = evaluation.evaluate(model, str(root), args, torch.device('cpu'),
                'classifier', router_mode='classifier', image_router=NightRouter(),
                domain='mix', model_label='finetune_classifier')
            self.assertEqual(model.calls, ['dora', 'dora'])
            self.assertEqual(classifier['route_counts'], {'dora': 2, 'dora1': 0})
            self.assertEqual(classifier['router_accuracy'], .5)
            self.assertEqual(classifier['mae'], 1.)
            for group in ('day', 'night'):
                self.assertEqual(classifier['group_metrics'][group]['images'], 1)
                self.assertEqual(classifier['group_metrics'][group]['mae'], 1.)
            model.calls.clear()
            filename = evaluation.evaluate(model, str(root), args, torch.device('cpu'),
                'filename', router_mode='filename', domain='mix', model_label='dora1_filename')
            self.assertCountEqual(model.calls, ['dora1', 'dora'])
            self.assertEqual(filename['router_accuracy'], 1.)
            output = root / 'routing.csv'
            evaluation.write_routing_results(classifier['routing_rows'] + filename['routing_rows'], output)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row['model'] for row in rows}, {'finetune_classifier', 'dora1_filename'})
            model.calls.clear()
            model.set_adapter_route('dora1')
            fixed = evaluation.evaluate(model, str(root), args, torch.device('cpu'), 'fixed', domain='mix')
            self.assertEqual(model.calls, ['dora1', 'dora1'])
            self.assertEqual(fixed['mae'], 2.)
            self.assertEqual(fixed['routing_rows'], [])

    def test_finetune_only_jobs_and_before_after_comparison(self):
        args = SimpleNamespace(finetune_checkpoint='fine', dora1_checkpoint='',
            baseline_checkpoint='', dora_checkpoint='', router_mode='fixed',
            checkpoint_stage='dora1', checkpoint_root='model', model_name='swin_t_trans')
        with patch.object(evaluation, 'resolve_checkpoint', side_effect=lambda path, *rest: path or rest[-1]):
            self.assertEqual(evaluation.checkpoint_jobs(args), [('finetune', 'fine', 'dual')])
            args.router_mode = 'classifier'
            args.dora1_checkpoint = 'before'
            self.assertEqual(evaluation.checkpoint_jobs(args),
                [('dora1', 'before', 'dual'), ('finetune', 'fine', 'dual')])
            args.finetune_checkpoint = args.dora1_checkpoint = ''
            args.checkpoint_stage = 'finetune'
            self.assertEqual(evaluation.checkpoint_jobs(args), [('finetune', 'finetune', 'dual')])

    def test_pipeline_passes_each_best_checkpoint_to_next_stage(self):
        with patch('sys.argv', ['train.py', '--stage', 'all', '--finetune-after-all',
                                '--finetune-router-mode', 'filename']):
            args = train.parse_args()
        with patch.object(train, 'run_stage', side_effect=['base', 'night', 'dual', 'fine']) as run:
            self.assertEqual(train.run_pipeline(args), ('base', 'night', 'dual', 'fine'))
        self.assertEqual([call.args[1] for call in run.call_args_list], ['baseline', 'dora', 'dora1', 'finetune'])
        self.assertEqual([call.kwargs.get('parent_checkpoint') for call in run.call_args_list],
                         [None, 'base', 'night', 'dual'])
