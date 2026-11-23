"""Focused checks for transition indexing and Dreamer optimizer boundaries."""
import unittest

import torch
from torch import nn
from torch.distributions import Normal

from dreamer import DreamerBehavior, lambda_returns


class TraceWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def rnn_model(self, h, z, action):
        return h + self.scale * action + 1

    def transition_model(self, h):
        return Normal(torch.zeros_like(h), torch.ones_like(h))

    def reward_model(self, h, z):
        return Normal(h, torch.ones_like(h))


class TestDreamer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.world = TraceWorldModel()
        self.behavior = DreamerBehavior(
            dict(h_dim=1, z_dim=1, imagination_horizon=3, discount=0.99,
                 lambda_return=0.95, grad_clip_norm=100, actor_lr=8e-5,
                 critic_lr=8e-5, adam_epsilon=1e-4),
            action_dim=1, device="cpu", dtype=torch.float32,
        )

    def test_lambda_indexes_successor_values_and_bootstrap(self):
        rewards = torch.tensor([2., 3., 5.])
        next_values = torch.tensor([7., 11., 13.])
        torch.testing.assert_close(
            lambda_returns(rewards, next_values, 0.5, 0.0), torch.tensor([5.5, 8.5, 11.5])
        )
        torch.testing.assert_close(
            lambda_returns(rewards, next_values, 0.5, 1.0), torch.tensor([6.375, 8.75, 11.5])
        )
        torch.testing.assert_close(
            lambda_returns(rewards, next_values, 0.5, 0.5), torch.tensor([5.90625, 8.625, 11.5])
        )

    def test_imagination_predicts_arrival_reward_with_action_gradients(self):
        features, rewards, _ = self.behavior.imagine(
            self.world, torch.zeros(8, 1), torch.zeros(8, 1)
        )
        torch.testing.assert_close(rewards[:-1, :, 0], features[1:, :, 0])
        self.assertTrue(torch.all(rewards[0] > 0))
        self.assertEqual(rewards.shape, (3, 8, 1))
        rewards[0].mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in self.behavior.actor.parameters()))

    def test_actor_and_critic_updates_are_isolated(self):
        actor = list(self.behavior.actor.parameters())
        critic = list(self.behavior.critic.parameters())
        actor_before = [p.detach().clone() for p in actor]
        critic_before = [p.detach().clone() for p in critic]
        seen_actor_gradients = []

        def after_actor_step(*_):
            self.assertTrue(all(p.grad is None for p in critic))
            self.assertIsNone(self.world.scale.grad)
            seen_actor_gradients.extend(p.grad.clone() for p in actor)

        with self.behavior.actor_optimizer.register_step_post_hook(after_actor_step):
            start = torch.zeros(2, 3, 1, requires_grad=True)
            losses = self.behavior.update(self.world, (start, start))
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertIsNone(start.grad)
        self.assertIsNone(self.world.scale.grad)
        self.assertEqual(self.world.scale.item(), 1.0)
        self.assertTrue(self.world.scale.requires_grad)
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(actor, actor_before)))
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(critic, critic_before)))
        for param, grad in zip(actor, seen_actor_gradients):
            torch.testing.assert_close(param.grad, grad)


if __name__ == "__main__":
    unittest.main()
