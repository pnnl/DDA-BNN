"""Exercise actual SVI updates before and after guide-scale warmup."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
import pyro
from pyro.optim import ClippedAdam
from torch.utils.data import DataLoader, TensorDataset
import training.config as cfg
from training.model import HybridNet
from training import train


class VariationalWarmupLifecycleTests(unittest.TestCase):
    def tearDown(self):
        pyro.clear_param_store()

    def test_scale_is_frozen_then_restored_and_trainable_with_same_optimizer(self):
        with patch.multiple(cfg, TARGETS=['A'], UNCERTAINTY_MODE='diag',
                            LAYER_SPEC=[dict(units=4, type='bnn', act='Tanh')]):
            pyro.clear_param_store(); pyro.set_rng_seed(12)
            net = HybridNet(2)
            x = torch.ones(8, 2); y = torch.ones(8, 1)
            net.guide(x)
            leaf = net.guide.scale.unconstrained()
            original = leaf.detach().clone()
            state = train._freeze_guide_scales(net)
            self.assertFalse(leaf.requires_grad)
            frozen = leaf.detach().clone()
            net.use_predicted_std = False
            optimizer = ClippedAdam({'lr': 0.01})
            svi = train.make_svi(net, 1, warmup=0)
            # #23 supplies the optional optimizer argument after integration;
            # assigning the same public SVI optimizer also works on #21 alone.
            svi.optim = optimizer
            for _ in range(2):
                svi.step(x, y)
            torch.testing.assert_close(leaf, frozen)
            self.assertIsNone(leaf.grad)
            loc_optimizer = optimizer.optim_objs[net.guide.loc]
            step_before = int(loc_optimizer.state[net.guide.loc]['step'])
            train._restore_guide_scales(state)
            self.assertIs(leaf, net.guide.scale.unconstrained())
            self.assertTrue(leaf.requires_grad)
            torch.testing.assert_close(leaf, original)
            net.use_predicted_std = True
            svi.step(x, y)
            self.assertFalse(torch.equal(leaf, original))
            self.assertEqual(int(loc_optimizer.state[net.guide.loc]['step']), step_before + 1)

    def test_warmup_initialization_is_independent_of_observation_variance_bounds(self):
        with patch.multiple(cfg, TARGETS=['A'], UNCERTAINTY_MODE='diag',
                            LAYER_SPEC=[dict(units=2, type='bnn', act='Tanh')]):
            pyro.clear_param_store(); net = HybridNet(1); net.guide(torch.zeros(1, 1))
            original = net.guide.scale.detach().clone()
            state = train._freeze_guide_scales(net)
            with patch.multiple(cfg, LOG_VAR_MIN=-30., LOG_VAR_MAX=-20.):
                train._restore_guide_scales(state)
            torch.testing.assert_close(net.guide.scale, original)

    def test_main_freezes_and_restores_scales_at_the_epoch_boundary(self):
        pyro.clear_param_store(); pyro.set_rng_seed(8)
        data = TensorDataset(torch.ones(8, 2), torch.ones(8, 1))
        loader = DataLoader(data, batch_size=4)
        meta = {'y_test_raw': np.ones((8, 1)), 'low_res_test_raw': np.ones((8, 1))}
        snapshots = []
        original_train = train.train_one_epoch

        def observe(model, *args, **kwargs):
            leaf = model.guide.scale.unconstrained()
            before = leaf.detach().clone()
            snapshots.append((leaf.requires_grad, model.use_predicted_std,
                              model.guide.scale.detach().clone()))
            result = original_train(model, *args, **kwargs)
            snapshots[-1] += (torch.equal(leaf, before),)
            return result

        with TemporaryDirectory() as tmp, patch.multiple(
                cfg, TARGETS=['A'], UNCERTAINTY_MODE='diag', DEVICE=torch.device('cpu'),
                LAYER_SPEC=[dict(units=3, type='bnn', act='Tanh')], ARTIFACT_DIR=Path(tmp),
                EPOCHS=2, DET_WARMUP=1, WARMUP=0, EARLY_STOP_PATIENCE=0), \
                patch.object(train, 'load_dataset', return_value=(loader, loader, loader, meta)), \
                patch.object(train, 'train_one_epoch', side_effect=observe), \
                patch.object(train, 'evaluate', return_value=(np.ones((8, 1)), None, None)), \
                patch.object(train, 'plot_loss'), patch.object(train, 'plot_results'), \
                patch('training.temp_tune.run_temp_tune', return_value={}):
            train.main()
        self.assertEqual([(s[0], s[1]) for s in snapshots], [(False, False), (True, True)])
        self.assertTrue(bool((snapshots[0][2] < 1e-4).all()))
        torch.testing.assert_close(snapshots[1][2], torch.full_like(snapshots[1][2], 0.1))
        self.assertTrue(snapshots[0][3])
        self.assertFalse(snapshots[1][3])
