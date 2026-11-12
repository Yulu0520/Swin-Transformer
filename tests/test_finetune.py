import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from datasets.crowd import RoutedCrowd, gen_discrete_map
from datasets.day_night import build_router_transform
from losses.dm_count import DMCountLoss
from models.swin_c_multibatch import swin_t_trans
from utils.counting_model import build_dual_dora_model
from utils.finetune import freeze_for_finetune, make_finetune_scheduler, finetune_config
from utils.regression_trainer_cosine_multibatch import RegTrainer
from utils.routing import forward_routed_regression, select_routes
from utils.topk_checkpoint import best_ranked_path


class FakeRouter(nn.Module):
    def forward(self, x):
        # Explicit hard decisions from image pixels, irrespective of filenames.
        return torch.stack((x[:, 0, 0, 0], -x[:, 0, 0, 0]), dim=1)


def parse_finetune(source, data, **options):
    import train
    flags = ['train.py', '--stage', 'finetune', '--device', 'cpu',
        '--finetune-checkpoint', str(source), '--finetune-router-mode', 'filename',
        '--finetune-train-dir', str(data / 'train'), '--finetune-val-dir', str(data / 'val'),
        '--crop-size', '64', '--batch-size', '2', '--num-workers', '0',
        '--finetune-epochs', '3', '--finetune-warmup-epochs', '1']
    for name, value in options.items():
        flags += ['--' + name.replace('_', '-'), str(value)]
    with patch('sys.argv', flags):
        args = train.parse_args()
    args.max_epoch, args.val_epoch, args.val_start = args.finetune_epochs, 1, 0
    args.lr = args.finetune_lr
    return args


class FineTuneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_classifier_is_pixel_based_and_filename_mode_is_optional(self):
        router = FakeRouter()
        x = torch.tensor([[[[-2.]]], [[[2.]]]])
        domains, routes, confidence = select_routes('classifier', ['IMG_1', 'night_IMG_2'], x, router, 'cpu')
        self.assertEqual(routes, ['dora', 'dora1'])
        self.assertEqual(domains, ['night', 'day'])
        self.assertTrue(all(value > .9 for value in confidence))
        self.assertFalse(router.training)
        self.assertEqual(select_routes('filename', ['IMG_1', 'night_IMG_2'])[1], ['dora1', 'dora'])

    def test_router_input_uses_original_pixels_before_crop_and_grayscale(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            image = Image.new('RGB', (80, 80), (200, 20, 70))
            image.save(root / 'night_IMG_1.jpg')
            np.save(root / 'night_IMG_1.npy', [[20., 20.]])
            dataset = RoutedCrowd(str(root), 64, 8, method='train')
            with patch('datasets.crowd.random.random', side_effect=[.95, .5, .1]):
                crop, points, target, router_input, name = dataset[0]
            with Image.open(root / 'night_IMG_1.jpg') as saved:
                expected = build_router_transform(False)(saved.convert('RGB'))
            torch.testing.assert_close(router_input, expected)
            self.assertEqual(crop.shape, (3, 64, 64))
            self.assertEqual(name, 'night_IMG_1')

    def test_grouped_frozen_features_preserve_order_and_only_head_updates(self):
        torch.manual_seed(4)
        model = swin_t_trans(pretrained=False)
        model.enable_dual_dora()
        with torch.no_grad():
            model.regression.res[3].bias.fill_(.1)
            for module in model.backbone_features.modules():
                if hasattr(module, 'dora') and hasattr(module, 'dora1'):
                    module.dora.dora_B_q.fill_(.05)
        freeze_for_finetune(model)
        inputs = torch.randn(3, 3, 64, 64)
        routes = ['dora', 'dora1', 'dora']
        with torch.no_grad():
            expected = []
            for index, route in enumerate(routes):
                model.set_adapter_route(route)
                expected.append(model(inputs[index:index + 1]))
            expected = torch.cat(expected)
        output = forward_routed_regression(model, inputs, routes)
        torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-5)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-6)
        output.sum().backward()
        optimizer.step()
        updated = model.state_dict()
        for name, tensor in before.items():
            if not name.startswith('regression.') or name.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                torch.testing.assert_close(updated[name], tensor, rtol=0, atol=0)
        self.assertFalse(model.backbone_features.training)
        self.assertTrue(model.regression.training)
        self.assertFalse(model.regression.v1[1].training)
        self.assertTrue(model.regression.v1[1].weight.requires_grad)
        self.assertTrue(any(not torch.equal(updated[name], tensor)
                            for name, tensor in before.items() if name.startswith('regression.')))
        self.assertTrue(all(p.grad is None for p in model.backbone_features.parameters()))

    def test_scheduler_starts_small_warms_up_decays_and_resumes_exactly(self):
        from types import SimpleNamespace
        args = SimpleNamespace(finetune_start_lr=1e-7, finetune_lr=1e-6,
            finetune_min_lr=1e-8, finetune_warmup_epochs=5, finetune_epochs=50)
        p = nn.Parameter(torch.ones(1))
        opt = torch.optim.Adam([p], lr=args.finetune_lr)
        scheduler = make_finetune_scheduler(opt, args)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 1e-7)
        values = []
        for epoch in range(51):
            values.append(opt.param_groups[0]['lr'])
            if epoch == 13:
                opt_state, schedule_state = copy.deepcopy(opt.state_dict()), copy.deepcopy(scheduler.state_dict())
            opt.step()
            scheduler.step()
        self.assertAlmostEqual(values[5], 1e-6)
        self.assertAlmostEqual(values[50], 1e-8)
        self.assertTrue(all(a <= b for a, b in zip(values[:5], values[1:6])))
        opt2 = torch.optim.Adam([nn.Parameter(torch.ones(1))], lr=args.finetune_lr)
        scheduler2 = make_finetune_scheduler(opt2, args)
        opt2.load_state_dict(opt_state)
        scheduler2.load_state_dict(schedule_state)
        for expected in values[13:]:
            self.assertEqual(opt2.param_groups[0]['lr'], expected)
            opt2.step()
            scheduler2.step()

    def test_real_finetune_setup_training_top5_and_resume_on_cpu(self):
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ['train', 'val']:
                folder = root / 'mix' / split
                folder.mkdir(parents=True)
                for name in ['IMG_1', 'night_IMG_1']:
                    Image.new('RGB', (64, 64), (120, 100, 80)).save(folder / (name + '.jpg'))
                    np.save(folder / (name + '.npy'), [[24., 28.]])
            source = swin_t_trans(pretrained=False)
            source.enable_dual_dora()
            with torch.no_grad():
                source.regression.res[3].bias.fill_(.1)
            source_path = root / 'source.pth'
            torch.save(source.state_dict(), source_path)
            args = parse_finetune(source_path, root / 'mix')
            args.save_dir = str(root / 'first')
            trainer = RegTrainer(args)
            trainer.setup()
            frozen_before = {name: value.clone() for name, value in trainer.model.state_dict().items()
                             if not name.startswith('regression.')}
            trainer.args.max_epoch = 1  # Interrupt after one epoch of the planned 3-epoch schedule.
            trainer.train()
            self.assertEqual([entry['epoch'] for entry in trainer.top_checkpoints.entries].count(0), 1)
            resume = Path(trainer.save_dir) / '1_ckpt.tar'
            resumed_args = parse_finetune(source_path, root / 'mix', resume=resume)
            resumed_args.save_dir = str(root / 'resumed')
            resumed = RegTrainer(resumed_args)
            resumed.setup()
            self.assertEqual(resumed.start_epoch, 1)
            self.assertEqual(resumed.optimizer.param_groups[0]['lr'], trainer.optimizer.param_groups[0]['lr'])
            resumed.train()
            self.assertEqual(len(resumed.top_checkpoints.entries), 4)
            self.assertTrue(Path(best_ranked_path(resumed.save_dir)).is_file())
            for name, before in frozen_before.items():
                torch.testing.assert_close(resumed.model.state_dict()[name], before, rtol=0, atol=0)
            checkpoint = Path(resumed.save_dir) / '3_ckpt.tar'
            args.dual_route = 'dora1'
            loaded = build_dual_dora_model(args, str(checkpoint)).eval()
            resumed.model.set_adapter_route('dora1')
            resumed.model.eval()
            with torch.no_grad():
                inputs = torch.randn(1, 3, 64, 64)
                torch.testing.assert_close(loaded(inputs), resumed.model(inputs))
            # A changed schedule cannot silently resume the old LR trajectory.
            changed = parse_finetune(source_path, root / 'mix', resume=checkpoint, finetune_lr=2e-6)
            changed.save_dir = str(root / 'bad_resume')
            bad = RegTrainer(changed)
            with self.assertRaisesRegex(ValueError, 'LR schedule settings'):
                bad.setup()

    def test_classifier_finetune_and_evaluation_cli_with_real_router_checkpoint(self):
        import csv
        import test as evaluation
        from models.day_night_router import build_day_night_router, ROUTER_ARCHITECTURE, CLASS_NAMES
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ['train', 'val']:
                folder = root / 'mix' / split
                folder.mkdir(parents=True)
                for name in ['IMG_1', 'night_IMG_1']:
                    Image.new('RGB', (64, 64), (120, 100, 80)).save(folder / (name + '.jpg'))
                    np.save(folder / (name + '.npy'), [[24., 28.]])
            source = swin_t_trans(pretrained=False)
            source.enable_dual_dora()
            with torch.no_grad():
                source.regression.res[3].bias.fill_(.1)
            source_path = root / 'source.pth'
            torch.save(source.state_dict(), source_path)
            del source
            router = build_day_night_router(pretrained=False, freeze_features=False)
            with torch.no_grad():
                router.classifier[-1].weight.zero_()
                router.classifier[-1].bias.copy_(torch.tensor([0., 3.]))
            router_path = root / 'router.pth'
            torch.save(dict(architecture=ROUTER_ARCHITECTURE, class_names=CLASS_NAMES,
                            model_state_dict=router.state_dict()), router_path)
            before = {name: value.clone() for name, value in router.state_dict().items()}
            del router
            args = parse_finetune(source_path, root / 'mix',
                finetune_router_mode='classifier', router_checkpoint=router_path,
                count_loss_weight=.5)
            args.save_dir = str(root / 'classifier')
            trainer = RegTrainer(args)
            trainer.setup()
            self.assertEqual(trainer.criterion.count_loss_weight, .5)
            self.assertTrue(all(not p.requires_grad for p in trainer.image_router.parameters()))
            trainer.args.max_epoch = 1
            trainer.train()
            snapshot = torch.load(Path(trainer.save_dir) / '1_ckpt.tar',
                                  map_location='cpu', weights_only=False)
            self.assertEqual(snapshot['loss_config']['count_weight'], .5)
            del snapshot
            for name, value in trainer.image_router.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            with patch('sys.argv', ['test.py', '--device', 'cpu', '--num-workers', '0',
                '--router-mode', 'classifier', '--router-checkpoint', str(router_path),
                '--finetune-checkpoint', trainer.save_dir, '--data-root', str(root),
                '--domains', 'mix', '--output-csv', str(root / 'metrics.csv'),
                '--routing-csv', str(root / 'routing.csv')]):
                evaluation.main()
            with (root / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row['subset'] for row in rows}, {'overall', 'day', 'night'})
            with (root / 'routing.csv').open() as stream:
                routes = list(csv.DictReader(stream))
            self.assertEqual(len(routes), 2)
            self.assertEqual({row['route'] for row in routes}, {'dora'})
            self.assertEqual({row['model'] for row in routes}, {'finetune_classifier'})
