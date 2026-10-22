"""Regression checks for temporal alignment and the actual saved RSSM weights."""

import unittest

import numpy as np
import torch

from rssm_diagnostics import (HERE, aligned_reward_pairs, architecture_checks,
                              display_image, load_model, mean_objective, preprocess)


class RSSMDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model, _ = load_model(HERE / 'midterm_pilot_v2_checkpoint.pt')

    def test_reward_targets_exclude_unmatched_boundaries(self):
        predictions = torch.tensor([999., 10., 20., 30.])
        rewards = torch.tensor([10., 20., 30., 888.])
        predicted, target = aligned_reward_pairs(predictions, rewards)
        torch.testing.assert_close(predicted, target)
        self.assertEqual(len(predicted), 3)

    def test_reward_alignment_rejects_invalid_sequences(self):
        for prediction, reward in [([1], [1]), ([1, 2], [1, 2, 3])]:
            with self.assertRaises(ValueError):
                aligned_reward_pairs(prediction, reward)

    def test_historical_pixel_layout_round_trip(self):
        # Asymmetric image exposes H/W swaps that square-image shape tests miss.
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[2, 9] = [255, 128, 64]
        image = preprocess(rgb, 5)
        self.assertGreater(image[0, 9, 2].item(), 0.4)
        expected = np.floor(rgb.astype(np.float32) / 8) / 32
        np.testing.assert_allclose(display_image(image), expected)

    def test_checkpoint_shapes_and_all_module_gradients(self):
        trajectory = dict(rgb=np.zeros((4, 64, 64, 3), dtype=np.uint8),
                          actions=np.zeros((4, 1), dtype=np.float32),
                          rewards=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        original_params = self.model.params.copy()
        result = architecture_checks(self.model, trajectory)
        self.assertTrue(all(result['all_module_gradients_finite_nonzero'].values()))
        self.assertEqual(self.model.params, original_params)

    def test_likelihood_mean_objective_matches_unit_gaussian_nll(self):
        # A closed-form likelihood check catches sample-noise and loss scaling errors.
        from torch.distributions import Independent, Normal
        images = torch.ones(3, 2, 3, 4, 4) * 0.25
        rewards = torch.tensor([2., 3., 999.])[:, None, None].repeat(1, 2, 1)
        means = torch.tensor([123., 2., 3.])[:, None, None].repeat(1, 2, 1)
        latent = Independent(Normal(torch.zeros(2, 3), torch.ones(2, 3)), 1)
        pred = dict(recon_obs=[Independent(Normal(frame, 1), 3) for frame in images],
                    reward=[Independent(Normal(frame, 1), 1) for frame in means],
                    prior=[latent] * 3, posterior=[latent] * 3)
        target = dict(obs=torch.zeros_like(images), reward=rewards)
        loss, (recon, kl, reward) = mean_objective(target, pred, 3)
        self.assertAlmostEqual(recon.item(), 48 * 0.25 ** 2)
        self.assertEqual(kl.item(), 3.)
        self.assertEqual(reward.item(), 0.)
        self.assertAlmostEqual(loss.item(), recon.item() + 3.)
        nll = -pred['recon_obs'][0].log_prob(target['obs'][0]).mean()
        const = 48 * 0.5 * np.log(2 * np.pi)
        self.assertAlmostEqual(nll.item(), 0.5 * recon.item() + const, places=4)


if __name__ == '__main__':
    unittest.main()
