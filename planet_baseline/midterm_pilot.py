import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import kl_divergence
from ruamel.yaml import YAML
from model_based_agent import ModelBasedLearner, Transition
from planet import Planet

torch.manual_seed(42)
np.random.seed(42)

with open("configs.yaml") as f:
    cfg = YAML(typ="safe").load(f)

params = {**cfg["defaults"], **cfg["dmc"]}
params.update({
    "device": "cpu",
    "rng_seed": 42,
    "domain_name": "cartpole",
    "task_name": "swingup",
    "action_repeat": 0,
    "batch_size": 4,
    "chunk_length": 16,
    "planning_horizon": 5,
    "plan_optimization_iter": 2,
    "n_plans": 32,
    "top_k": 8,
})

# Correct reward alignment and remove sampling
# noise from reconstruction/reward losses.
def aligned_loss(self, target, pred):
    obs_mean = torch.stack([
        d.mean for d in pred["recon_obs"]
    ])

    recon_loss = (
        (obs_mean - target["obs"]).square()
        .mean(dim=(0, 1)).sum()
    )

    kl = torch.stack([
        kl_divergence(q, p)
        for q, p in zip(
            pred["posterior"], pred["prior"]
        )
    ])

    kl_loss = torch.clamp(
        kl, min=self.params["free_nats"]
    ).mean()

    reward_mean = torch.stack([
        d.mean for d in pred["reward"]
    ])

    # State at t predicts reward from action t-1.
    reward_loss = (
        reward_mean[1:] - target["reward"][:-1]
    ).square().mean()

    loss = recon_loss + kl_loss + reward_loss
    return loss, (
        recon_loss.item(),
        kl_loss.item(),
        reward_loss.item()
    )

Planet.compute_loss = aligned_loss

agent = ModelBasedLearner(params)
env_steps = 0
losses = []
returns = []
return_steps = []

EP_STEPS = 120
UPDATES = 30
ROUNDS = 3

def collect(planning=False):
    global env_steps

    obs, _ = agent.env.reset()
    trajectory = []
    total_reward = 0.0
    h = agent.world_model.get_init_h_state(1)

    agent.world_model.eval()

    for t in range(EP_STEPS):
        if planning:
            with torch.no_grad():
                embed = agent.world_model.obs_encoder(
                    obs.unsqueeze(0)
                )
                z = agent.world_model.repr_model(
                    h, embed
                ).sample()

                action = agent.plan_action_with_cem(h, z)
                action = action + 0.3 * torch.randn_like(action)
                action = action.clamp(-1, 1)

                h = agent.world_model.rnn_model(
                    h, z, action.unsqueeze(0)
                )

                action_np = action.cpu().numpy()
        else:
            action_np = agent.env.action_space.sample()

        nxt, reward, done, trunc, _ = (
            agent.env.step(action_np)
        )

        trajectory.append(Transition(
            observation=obs.clone(),
            action=torch.as_tensor(
                action_np, dtype=torch.float32
            ),
            reward=torch.tensor(
                [reward], dtype=torch.float32
            )
        ))

        total_reward += float(reward)
        env_steps += 1
        obs = nxt

        if done or trunc:
            break

    agent.replay_buffer.push(trajectory)
    return total_reward

# Initial random experience
for i in range(2):
    r = collect(planning=False)
    print(
        f"Seed episode {i+1}: return={r:.4f}",
        flush=True
    )

# Alternating model training and CEM collection
for round_idx in range(ROUNDS):
    print(
        f"\nROUND {round_idx+1}/{ROUNDS}",
        flush=True
    )

    for update in range(UPDATES):
        agent.world_model.train()

        batch = agent.replay_buffer.sample(
            params["batch_size"]
        )
        pred = agent.world_model(batch)

        loss, components = (
            agent.world_model.compute_loss(
                batch, pred
            )
        )

        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            agent.world_model.parameters(), 100
        )
        agent.optimizer.step()

        losses.append(loss.item())

    # Collect with updated CEM policy
    reward = collect(planning=True)
    returns.append(reward)
    return_steps.append(env_steps)

    print(
        f"Environment Steps: {env_steps} | "
        f"120-step Return: {reward:.4f} | "
        f"Loss: {loss.item():.2f}",
        flush=True
    )

# Save results and model
np.savez(
    "midterm_pilot_metrics.npz",
    train_losses=np.array(losses),
    returns=np.array(returns),
    env_steps=np.array(return_steps)
)

torch.save({
    "model": agent.world_model.state_dict(),
    "optimizer": agent.optimizer.state_dict(),
    "env_steps": env_steps,
    "params": params
}, "midterm_pilot_checkpoint.pt")

plt.figure(figsize=(7, 4))
plt.plot(return_steps, returns, marker="o")
plt.xlabel("Training Environment Steps")
plt.ylabel("120-step Training Rollout Return")
plt.title("PlaNet Cartpole-Swingup Pilot")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("midterm_pilot_curve.png", dpi=150)
plt.close()

print("\nPILOT TRAINING COMPLETE!", flush=True)
print("Final Environment Steps:", env_steps, flush=True)
print("Returns:", returns, flush=True)

agent.logger.close()
