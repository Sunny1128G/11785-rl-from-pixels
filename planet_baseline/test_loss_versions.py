"""Closed-form checks of the production loss, without environment rendering."""
import unittest
from types import SimpleNamespace
import torch
from torch.distributions import Independent, Normal
from planet import Planet


class LossVersionTests(unittest.TestCase):
    def test_nll_alignment_gradients_and_legacy_scale(self):
        images = torch.full((3, 2, 3, 4, 4), 0.25, requires_grad=True)
        rewards = torch.tensor([999., 3., 5.])[:, None, None].repeat(1, 2, 1).requires_grad_()
        latent = Independent(Normal(torch.zeros(2, 3), torch.ones(2, 3)), 1)
        pred = dict(recon_obs=[Independent(Normal(x, 1), 3) for x in images],
                    reward=[Independent(Normal(x, 1), 1) for x in rewards],
                    prior=[latent]*3, posterior=[latent]*3)
        target = dict(obs=torch.zeros_like(images),
                      reward=torch.tensor([2., 3., 888.])[:, None, None].repeat(1, 2, 1))
        model = SimpleNamespace(params=dict(loss_version="v2", free_nats=3.))
        loss, parts = Planet.compute_loss(model, target, pred)
        self.assertEqual(parts, (1.5, 3., 1.25))
        # Compare against actual Gaussian log_prob with constants removed.
        nll = -torch.stack([d.log_prob(x) for d, x in zip(pred['recon_obs'], target['obs'])]).mean()
        nll -= 48 * 0.5 * torch.log(torch.tensor(2 * torch.pi))
        self.assertAlmostEqual(parts[0], nll.item(), places=4)
        loss.backward()
        self.assertEqual(rewards.grad[0].abs().sum().item(), 0.)
        self.assertGreater(rewards.grad[1:].abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(images.grad).all())
        model.params['loss_version'] = 'aligned'
        _, old = Planet.compute_loss(model, target, pred)
        self.assertEqual(old, (3., 3., 2.5))
        model.params['loss_version'] = 'bad'
        with self.assertRaises(ValueError):
            Planet.compute_loss(model, target, pred)
        model.params['loss_version'] = 'v2'
        with self.assertRaises(ValueError):
            Planet.compute_loss(model, {k: v[:1] for k, v in target.items()},
                                {k: v[:1] for k, v in pred.items()})


if __name__ == '__main__':
    unittest.main()
