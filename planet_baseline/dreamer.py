"""Differentiable DreamerV1 behavior learning on the shared PlaNet RSSM."""
from contextlib import contextmanager

import torch
from torch import nn
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn.functional import softplus


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 2 * action_dim),
        )

    def forward(self, features):
        mean, raw_std = self.net(features).chunk(2, dim=-1)
        mean = 5.0 * (mean / 5.0).tanh()
        std = softplus(raw_std + 4.9932392506) + 1e-4  # inverse softplus(5)
        base = Independent(Normal(mean, std), 1)
        return TransformedDistribution(base, TanhTransform(cache_size=1))


def lambda_returns(rewards, successor_values, discount, lambda_):
    """Return at each source state, with rewards and values indexed by arrival."""
    target = successor_values[-1]
    returns = []
    for reward, value in zip(reversed(rewards), reversed(successor_values)):
        target = reward + discount * ((1 - lambda_) * value + lambda_ * target)
        returns.append(target)
    return torch.stack(returns[::-1])


@contextmanager
def frozen_parameters(*modules):
    parameters = [p for module in modules for p in module.parameters()]
    original = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original):
            parameter.requires_grad_(requires_grad)


class DreamerBehavior(nn.Module):
    def __init__(self, params, action_dim, device, dtype):
        super().__init__()
        feature_dim = params["h_dim"] + params["z_dim"]
        self.actor = Actor(feature_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 1),
        )
        self.horizon = params["imagination_horizon"]
        self.discount = params["discount"]
        self.lambda_ = params["lambda_return"]
        self.grad_clip_norm = params["grad_clip_norm"]
        self.to(device=device, dtype=dtype)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=params["actor_lr"], eps=params["adam_epsilon"]
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=params["critic_lr"], eps=params["adam_epsilon"]
        )

    def action(self, h, z, training):
        policy = self.actor(torch.cat((h, z), dim=-1))
        if training:
            return policy.rsample()
        return policy.base_dist.base_dist.loc.tanh()

    def imagine(self, world_model, h, z):
        features, rewards, values = [], [], []
        for _ in range(self.horizon):
            features.append(torch.cat((h, z), dim=-1))
            action = self.actor(features[-1].detach()).rsample()
            h = world_model.rnn_model(h, z, action)
            z = world_model.transition_model(h).rsample()
            rewards.append(world_model.reward_model(h, z).mean)
            values.append(self.critic(torch.cat((h, z), dim=-1)))
        return torch.stack(features), torch.stack(rewards), torch.stack(values)

    def update(self, world_model, posterior_states):
        h, z = (states.flatten(0, 1).detach() for states in posterior_states)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(world_model, self.critic):
            features, rewards, values = self.imagine(world_model, h, z)
            targets = lambda_returns(rewards, values, self.discount, self.lambda_)
            weights = self.discount ** torch.arange(
                self.horizon, device=targets.device, dtype=targets.dtype
            ).view(-1, 1, 1)
            actor_loss = -(weights * targets).mean()
            actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        error = self.critic(features.detach()) - targets.detach()
        critic_loss = 0.5 * (weights * error.square()).mean()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()
        return actor_loss.item(), critic_loss.item()
