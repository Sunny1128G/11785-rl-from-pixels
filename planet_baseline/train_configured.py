"""Configuration-driven fresh training; historical pilots remain reproducible.

Run from any directory. Each seed gets a new output folder; no resume is implied.
Evaluation settings in the YAML are reserved for separate evaluation.
"""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def train(params, out):
    from model_based_agent import ModelBasedLearner, Transition
    torch.manual_seed(params["rng_seed"])
    np.random.seed(params["rng_seed"])
    random.seed(params["rng_seed"])
    out.mkdir(parents=True, exist_ok=False)
    print(f"seed={params['rng_seed']} loss_version={params['loss_version']}", flush=True)
    agent = ModelBasedLearner(params)
    agent.env.action_space.seed(params["rng_seed"])
    env_steps = 0
    losses = []
    returns = []
    return_steps = []

    EP_STEPS = params["collection_steps"]
    UPDATES = params["updates_per_round"]
    ROUNDS = params["rounds"]

    def collect(planning=False):
        nonlocal env_steps

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
                    action = action + params["action_epsilon"] * torch.randn_like(action)
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
    for i in range(params["n_seed_episodes"]):
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
                agent.world_model.parameters(), params["grad_clip_norm"]
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
        out / "metrics.npz",
        train_losses=np.array(losses),
        returns=np.array(returns),
        env_steps=np.array(return_steps),
        loss_version=params["loss_version"]
    )

    torch.save({
        "model": agent.world_model.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "env_steps": env_steps,
        "params": params,
        "loss_version": params["loss_version"]
    }, out / "checkpoint.pt")

    plt.figure(figsize=(7, 4))
    plt.plot(return_steps, returns, marker="o")
    plt.xlabel("Training Environment Steps")
    plt.ylabel("120-step Training Rollout Return")
    plt.title("PlaNet Cartpole-Swingup Pilot")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "curve.png", dpi=150)
    plt.close()

    print("\nPILOT TRAINING COMPLETE!", flush=True)
    print("Final Environment Steps:", env_steps, flush=True)
    print("Returns:", returns, flush=True)

    agent.logger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs_5seed.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="Run one configured training seed")
    args = parser.parse_args()
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with (HERE / "configs.yaml").open() as f:
        base = yaml.load(f)
    with args.config.open() as f:
        cfg = yaml.load(f)
    params = {**base["defaults"], **base["dmc"]}
    for section in ("environment", "training", "planner", "model"):
        params.update(cfg[section])
    if params.get("loss_version") not in ("original", "aligned", "v2"):
        parser.error("loss_version must be original, aligned, or v2")
    if params["collection_steps"] <= params["chunk_length"] or params["chunk_length"] < 2:
        parser.error("Require collection_steps > chunk_length >= 2")
    for key in ("rounds", "updates_per_round", "n_seed_episodes"):
        if params[key] < 1:
            parser.error(f"{key} must be positive")
    seeds = cfg["seeds"]["train_seeds"]
    if args.seed is not None:
        if args.seed not in seeds:
            parser.error("--seed must be listed in train_seeds")
        seeds = [args.seed]
    for seed in seeds:
        if (args.output / f"seed_{seed}").exists():
            parser.error(f"Output already exists for seed {seed}; use a fresh output directory")
    for seed in seeds:
        train(dict(params, rng_seed=seed), args.output / f"seed_{seed}")


if __name__ == "__main__":
    main()
