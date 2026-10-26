"""Regression: SVI's optimizer and moment estimates survive epoch boundaries."""
import unittest

try:
    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.optim import ClippedAdam
    from training.train import make_svi
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"DDA-BNN dependencies/config unavailable: {_IMPORT_ERROR}")
class OptimizerReuseTests(unittest.TestCase):
    def test_moments_survive_an_epoch_boundary(self):
        pyro.clear_param_store()
        class Toy:
            def __call__(self, x, y=None):
                loc = pyro.param('toy_loc', torch.tensor(0.0))
                with pyro.plate('data', len(x)):
                    pyro.sample('obs', dist.Normal(loc, 1.0), obs=y)
            def guide(self, x, y=None):
                return None

        shared = ClippedAdam({'lr': 0.01})
        toy = Toy()
        data = torch.ones(8)
        first = make_svi(toy, epoch=1, warmup=0, optimizer=shared)
        first.step(data, data)
        opt = next(iter(shared.optim_objs.values()))
        state = next(iter(opt.state.values()))
        step_before = int(state['step'])

        second = make_svi(toy, epoch=2, warmup=0, optimizer=shared)
        self.assertIs(first.optim, second.optim)
        second.step(data, data)
        self.assertEqual(int(next(iter(opt.state.values()))['step']), step_before + 1)
        pyro.clear_param_store()
