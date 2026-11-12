"""Reference-gradient, mass-conservation and objective migration checks."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from datasets.crowd import Crowd, gen_discrete_map, load_points
from losses.dm_count import DMCountLoss, LOSS_TYPE, loss_config, validate_resume_loss
from models.counting_checkpoint import ARCHITECTURE
from models.swin_c_multibatch import swin_t_trans
from models.swin_dual_dora import DualQVDoRAAdapter
from utils.counting_model import load_initial_weights
from utils.regression_trainer_cosine_multibatch import train_collate


def settings(**kwargs):
    args = SimpleNamespace(wot=0.1, wtv=0.01, reg=10., num_of_iter_in_ot=100,
        norm_cood=0, crop_size=16, downsample_ratio=8, batch_size=2)
    for name, value in kwargs.items():
        setattr(args, name, value)
    return args


class DMCountLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_values_and_gradients_match_original_cctrans_reference(self):
        # Golden values computed with the unmodified reference OT_Loss and
        # training formula at commit 71210285a36eb76cb22b3789dd10ee04235a17fa.
        expected = {
            False: (35.60020446777344, 22.518035888671875,
                [-5.6657981873, 2.5365152359, 1.1502206326, -1.6548950672]),
            True: (1.0074126720428467, -11.772907257080078,
                [-0.8706773520, -0.1265408099, -1.5128350258, -0.4132157862]),
        }
        for norm_cood, (distance, objective, gradients) in expected.items():
            with self.subTest(norm_cood=norm_cood):
                density = torch.tensor([[[[.2, .4], [.1, .3]]],
                                        [[[.3, .1], [.6, .2]]]], requires_grad=True)
                points = [torch.tensor([[3., 3.], [11., 11.]]), torch.empty(0, 2)]
                target = torch.tensor([[[[1., 0.], [0., 1.]]], [[[0., 0.], [0., 0.]]]])
                terms = DMCountLoss(16, 8, torch.device('cpu'), norm_cood=norm_cood)(density, points, target)
                self.assertAlmostEqual(terms['count'].item(), 1.1, places=6)
                self.assertAlmostEqual(terms['tv'].item(), 0.99999952316, places=6)
                self.assertAlmostEqual(terms['total'].item(), 1.11, places=6)
                self.assertAlmostEqual(terms['wasserstein'].item(), distance, places=4)
                self.assertAlmostEqual(terms['ot_objective'].item(), objective, places=4)
                terms['total'].backward()
                torch.testing.assert_close(density.grad.flatten(),
                    torch.tensor(gradients + [0.5] * 4), atol=2e-5, rtol=2e-5)

    def test_ot_gradient_sums_over_batch_while_count_and_tv_mean(self):
        density = torch.tensor([[[[.2, .4], [.1, .3]]]], requires_grad=True)
        points = [torch.tensor([[3., 3.], [11., 11.]])]
        target = torch.tensor([[[[1., 0.], [0., 1.]]]])
        criterion = DMCountLoss(16, 8, torch.device('cpu'))
        single = criterion(density, points, target)
        single_grad = torch.autograd.grad(single['ot'], density)[0]
        doubled_density = density.detach().repeat(2, 1, 1, 1).requires_grad_()
        doubled = criterion(doubled_density, points * 2, target.repeat(2, 1, 1, 1))
        doubled_grad = torch.autograd.grad(doubled['ot'], doubled_density)[0]
        torch.testing.assert_close(doubled_grad[0], single_grad[0])
        torch.testing.assert_close(doubled_grad[1], single_grad[0])
        torch.testing.assert_close(single['count'], doubled['count'])
        torch.testing.assert_close(single['tv'], doubled['tv'])
        torch.testing.assert_close(doubled['ot'], 2 * single['ot'])

    def test_empty_crops_only_count_penalty_and_finite_gradients(self):
        density = torch.tensor([[[[.2, .4], [.1, .3]]],
                                [[[.3, .1], [.6, .2]]]], requires_grad=True)
        terms = DMCountLoss(16, 8, torch.device('cpu'))(
            density, [torch.empty(0, 2), torch.empty(0, 2)], torch.zeros_like(density))
        self.assertEqual(terms['ot'].item(), 0.)
        self.assertEqual(terms['tv'].item(), 0.)
        self.assertAlmostEqual(terms['count'].item(), 1.1, places=6)
        terms['total'].backward()
        torch.testing.assert_close(density.grad, torch.full_like(density, 0.5))

    def test_zero_and_tiny_density_are_finite(self):
        points = [torch.tensor([[3., 3.]])]
        target = torch.tensor([[[[1., 0.], [0., 0.]]]])
        for magnitude in [0., 1e-12, 1e-5]:
            density = torch.full((1, 1, 2, 2), magnitude, requires_grad=True)
            terms = DMCountLoss(16, 8, torch.device('cpu'))(density, points, target)
            self.assertTrue(all(torch.isfinite(x).all() for x in terms.values()))
            terms['total'].backward()
            self.assertTrue(torch.isfinite(density.grad).all())

    def test_zero_weights_disable_distribution_losses(self):
        density = torch.ones(1, 1, 2, 2, requires_grad=True)
        points = [torch.tensor([[3., 3.]])]
        target = torch.tensor([[[[1., 0.], [0., 0.]]]])
        loss = DMCountLoss(16, 8, torch.device('cpu'), wot=0., wtv=0.)
        with patch.object(loss.ot, 'forward', side_effect=AssertionError('OT should be skipped')):
            terms = loss(density, points, target)
        self.assertEqual(terms['total'].item(), 3.)
        terms['total'].backward()
        torch.testing.assert_close(density.grad, torch.ones_like(density))

    def test_count_weight_scales_only_count_values_and_gradients(self):
        density = torch.tensor([[[[.2, .4], [.1, .3]]],
                                [[[.3, .1], [.6, .2]]]], requires_grad=True)
        points = [torch.tensor([[3., 3.], [11., 11.]]), torch.empty(0, 2)]
        target = torch.tensor([[[[1., 0.], [0., 1.]]], [[[0., 0.], [0., 0.]]]])
        baseline = DMCountLoss(16, 8, 'cpu')(density, points, target)
        baseline_grad = torch.autograd.grad(baseline['total'], density, retain_graph=True)[0]
        count_grad = torch.autograd.grad(baseline['count'], density)[0]
        for weight in (0., .25, 1., 2.):
            with self.subTest(weight=weight):
                prediction = density.detach().clone().requires_grad_()
                terms = DMCountLoss(16, 8, 'cpu', count_loss_weight=weight)(prediction, points, target)
                for name in ('count', 'ot', 'tv', 'weighted_ot', 'weighted_tv'):
                    torch.testing.assert_close(terms[name], baseline[name])
                torch.testing.assert_close(terms['weighted_count'], weight * baseline['count'])
                torch.testing.assert_close(terms['total'],
                    terms['weighted_count'] + terms['weighted_ot'] + terms['weighted_tv'])
                terms['total'].backward()
                torch.testing.assert_close(prediction.grad, baseline_grad + (weight - 1.) * count_grad)

    def test_zero_count_weight_removes_empty_crop_penalty(self):
        density = torch.ones(1, 1, 2, 2, requires_grad=True)
        terms = DMCountLoss(16, 8, 'cpu', count_loss_weight=0.)(
            density, [torch.empty(0, 2)], torch.zeros_like(density))
        self.assertEqual(terms['count'].item(), 4.)
        self.assertEqual(terms['weighted_count'].item(), 0.)
        self.assertEqual(terms['total'].item(), 0.)
        terms['total'].backward()
        torch.testing.assert_close(density.grad, torch.zeros_like(density))

    def test_rejects_wrong_shapes_and_invalid_hyperparameters(self):
        for kwargs in [dict(count_loss_weight=-1.), dict(count_loss_weight=float('nan')),
                       dict(count_loss_weight=float('inf')), dict(wot=-1.), dict(wtv=float('nan')), dict(reg=0.),
                       dict(num_of_iter_in_ot=0)]:
            with self.assertRaises(ValueError):
                DMCountLoss(16, 8, torch.device('cpu'), **kwargs)
        with self.assertRaises(ValueError):
            DMCountLoss(16, 8, torch.device('cpu'))(torch.ones(1, 1, 3, 3),
                [torch.empty(0, 2)], torch.zeros(1, 1, 3, 3))


class DMCountDatasetTest(unittest.TestCase):
    def dataset(self, root, crop_size=16):
        return Crowd(str(root), crop_size, 8, method='train')

    def test_duplicate_and_rounded_edge_points_preserve_count(self):
        points = np.array([[0., 0.], [7., 7.], [7., 7.], [15.9, 15.9]], dtype=np.float32)
        result = gen_discrete_map(16, 16, points)
        np.testing.assert_array_equal(result, [[[3., 0.], [0., 1.]]])

    def test_crop_filters_centers_not_fractional_boxes_and_flip_aligns_targets(self):
        dataset = self.dataset('/tmp')
        points = np.array([[-.1, 5., 128.], [16., 5., 128.], [5., 16., 128.],
                           [0., 0., 16.], [15.9, 15.9, 16.], [5., 5., 16.], [5., 5., 16.]], dtype=np.float32)
        with patch('datasets.crowd.random.random', side_effect=[.5, .5, .9]), \
             patch('datasets.crowd.random_crop', return_value=(0, 0, 16, 16)):
            _, retained, discrete = dataset.train_transform(Image.new('RGB', (16, 16)), points)
        self.assertEqual(retained.shape, (4, 2))
        self.assertEqual(discrete.sum().item(), 4.)
        self.assertTrue((retained >= 0).all() and (retained < 16).all())
        np.testing.assert_array_equal(discrete.numpy(), gen_discrete_map(16, 16, retained.numpy()))
        np.testing.assert_allclose(retained.numpy(), [[15., 0.], [0., 15.9], [10., 5.], [10., 5.]], atol=1e-6)

    def test_resize_and_offset_transform_coordinates_consistently(self):
        dataset = self.dataset('/tmp')
        points = np.array([[10., 10.], [12., 12.]], dtype=np.float32)
        # Scale 1.2 -> 24x24, then crop from (left=4, top=3).
        with patch('datasets.crowd.random.random', side_effect=[.5, .9, .1]), \
             patch('datasets.crowd.random_crop', return_value=(3, 4, 16, 16)):
            _, retained, discrete = dataset.train_transform(Image.new('RGB', (20, 20)), points)
        np.testing.assert_allclose(retained.numpy(), [[8., 9.], [10.4, 11.4]], atol=1e-5)
        self.assertEqual(discrete.sum().item(), 2.)

    def test_npy_xy_or_xy_distance_empty_and_collate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, points in [('xy', [[3., 3.]]), ('xyz', [[3., 3., 99.]]), ('empty', [])]:
                Image.new('RGB', (16, 16)).save(root / (name + '.jpg'))
                np.save(root / (name + '.npy'), np.array(points, dtype=np.float32))
            dataset = self.dataset(root)
            with patch('datasets.crowd.random.random', return_value=.5):
                batch = train_collate([dataset[i] for i in range(len(dataset))])
            images, points, discrete = batch
            self.assertEqual(images.shape, (3, 3, 16, 16))
            self.assertEqual(discrete.shape, (3, 1, 2, 2))
            self.assertEqual([len(x) for x in points], [0, 1, 1])
            torch.testing.assert_close(discrete.flatten(1).sum(1), torch.tensor([0., 1., 1.]))
            val = Crowd(str(root), 16, 8, method='val')
            self.assertEqual([val[i][1] for i in range(len(val))], [0, 1, 1])
            np.save(root / 'bad.npy', [[float('nan'), 0.]])
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                load_points(root / 'bad.npy')


class DMCountMigrationTest(unittest.TestCase):
    def test_resume_rejects_previous_objective_or_changed_settings(self):
        args = settings()
        checkpoint = dict(loss_type=LOSS_TYPE, loss_config=loss_config(args))
        validate_resume_loss(checkpoint, args)
        self.assertEqual(checkpoint['loss_config']['count_weight'], 1.)
        for key, value in [('count_loss_weight', .5), ('wot', .2), ('wtv', .02), ('reg', 5.),
                           ('num_of_iter_in_ot', 50), ('norm_cood', 1), ('crop_size', 32), ('batch_size', 4)]:
            changed = copy.copy(args)
            setattr(changed, key, value)
            with self.assertRaisesRegex(ValueError, 'settings'):
                validate_resume_loss(checkpoint, changed)
        with self.assertRaisesRegex(ValueError, '--init-checkpoint'):
            validate_resume_loss(dict(count_loss_weight=.1), args)

    def test_cli_flags_and_pipeline_clear_initialization_after_baseline(self):
        import train
        with patch('sys.argv', ['train.py', '--stage', 'all', '--init-checkpoint', 'old.pth']):
            args = train.parse_args()
        self.assertEqual((args.wot, args.wtv, args.reg, args.num_of_iter_in_ot, args.norm_cood),
                         (0.01, 0.01, 10., 100, 0))
        self.assertEqual(args.count_loss_weight, 1.)
        with patch('train.RegTrainer') as trainer:
            trainer.return_value.best_model_path = 'new.pth'
            train.run_stage(args, 'baseline')
            self.assertEqual(trainer.call_args.args[0].init_checkpoint, 'old.pth')
            train.run_stage(args, 'dora', parent_checkpoint='new_baseline.pth')
            stage_args = trainer.call_args.args[0]
            self.assertEqual(stage_args.init_checkpoint, '')
            self.assertEqual(stage_args.baseline_checkpoint, 'new_baseline.pth')
        for flags in [['--wot', '-1'], ['--wtv', 'nan'], ['--reg', '0'],
                      ['--num-of-iter-in-ot', '0'], ['--count-loss-weight', '-0.1'],
                      ['--count-loss-weight', 'nan'], ['--count-loss-weight', 'inf'],
                      ['--resume', 'old.pth'], ['--resume', 'old.tar', '--init-checkpoint', 'old.pth']]:
            with self.subTest(flags=flags), patch('sys.argv', ['train.py', '--stage', 'baseline'] + flags), \
                 patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:
                    train.parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_custom_count_weight_reaches_each_stage_and_resume_config(self):
        import train
        for weight in (0., .5, 2.):
            with patch('sys.argv', ['train.py', '--count-loss-weight', str(weight)]):
                args = train.parse_args()
            self.assertEqual(loss_config(args)['count_weight'], weight)
            validate_resume_loss(dict(loss_type=LOSS_TYPE, loss_config=loss_config(args)), args)
            with patch('train.RegTrainer') as trainer:
                trainer.return_value.best_model_path = 'weights.pth'
                for stage in ('baseline', 'dora', 'dora1', 'finetune'):
                    train.run_stage(args, stage)
                    self.assertEqual(trainer.call_args.args[0].count_loss_weight, weight)

    def test_weights_only_migration_loads_head_and_selected_adapter(self):
        torch.manual_seed(2)
        source = swin_t_trans(pretrained=False)
        args = settings(model_name='swin_t_trans', stage='baseline', init_checkpoint='old.tar',
                        dora_rank=4, dora_alpha=4., dora1_rank=4, dora1_alpha=4.)
        checkpoint = dict(architecture=ARCHITECTURE, output_stride=8,
            stage='baseline', model_name=args.model_name, count_loss_weight=.1,
            model_state_dict=source.state_dict(), epoch=9, best_mae=123.)
        target = swin_t_trans(pretrained=False)
        with patch('utils.counting_model.load_checkpoint', return_value=checkpoint):
            load_initial_weights(target, args)
        torch.testing.assert_close(source.regression.v1[0].weight, target.regression.v1[0].weight)
        source.enable_dual_dora()
        checkpoint.update(stage='dora1', model_state_dict=source.state_dict(),
                          dora_rank=4, dora_alpha=4., dora1_rank=4, dora1_alpha=4.)
        args.stage = 'dora1'
        target = swin_t_trans(pretrained=False)
        with patch('utils.counting_model.load_checkpoint', return_value=checkpoint):
            load_initial_weights(target, args)
        self.assertTrue(all(m.active_route == 'dora1' for m in target.modules()
                            if isinstance(m, DualQVDoRAAdapter)))
        self.assertTrue(all('.dora1.' in name for name, p in target.named_parameters() if p.requires_grad))
        self.assertTrue(all(not p.requires_grad for p in target.regression.parameters()))


if __name__ == '__main__':
    unittest.main()
