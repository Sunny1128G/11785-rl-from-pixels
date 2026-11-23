"""Train configured PlaNet or Dreamer runs from scratch, one folder per seed."""
import argparse
import math
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from ruamel.yaml import YAML

HERE = Path(__file__).resolve().parent


def train(params, out):
    from model_based_agent import ModelBasedLearner, Transition

    seed = params["rng_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    out.mkdir(parents=True, exist_ok=False)
    agent = ModelBasedLearner(params, log_dir=str(out / "tensorboard"))
    action_space = agent.env.action_space
    action_space.seed(seed)
    algorithm = params.get("algorithm", "planet")
    print(f"seed={seed} algorithm={algorithm} device={agent.device} "
          f"loss_version={params['loss_version']}", flush=True)

    env_steps = 0
    losses, behavior_losses = [], []
    returns, return_steps = [], []
    collection_steps = params["collection_steps"]
    step_budget = params.get("env_step_budget")

    @torch.no_grad()
    def collect(training_policy=False):
        nonlocal env_steps
        obs, _ = agent.env.reset()
        h = agent.world_model.get_init_h_state(1)
        agent.world_model.eval()
        trajectory = []
        total_reward = 0.0
        limit = collection_steps
        if step_budget is not None:
            limit = min(limit, step_budget - env_steps)
        for _ in range(limit):
            if training_policy:
                embed = agent.world_model.obs_encoder(obs.unsqueeze(0).to(agent.device))
                z = agent.world_model.repr_model(h, embed).sample()
                action = agent.select_action(h, z, training=True)
                action = action + params["action_epsilon"] * torch.randn_like(action)
                action = action.clamp(params["min_action"], params["max_action"])
                h = agent.world_model.rnn_model(h, z, action.unsqueeze(0))
                action_np = action.cpu().numpy()
            else:
                action_np = action_space.sample()

            nxt, reward, done, trunc, _ = agent.env.step(action_np)
            replay_obs = obs.clone()
            if step_budget is not None:
                # The preprocessed bins are exactly representable in one byte.
                replay_obs = ((obs + 0.5) * (2 ** params["pixel_bit"])).to(torch.uint8)
            trajectory.append(Transition(
                observation=replay_obs,
                action=torch.as_tensor(action_np, dtype=torch.float32),
                reward=torch.tensor([reward], dtype=torch.float32),
            ))
            total_reward += float(reward)
            env_steps += 1
            obs = nxt
            if done or trunc:
                break
        agent.replay_buffer.push(trajectory)
        return total_reward

    for i in range(params["n_seed_episodes"]):
        reward = collect()
        print(f"Seed episode {i+1}: return={reward:.4f}, env_steps={env_steps}", flush=True)

    if not len(agent.replay_buffer):
        raise RuntimeError("Random prefill produced no replay chunks")
    round_limit = params.get("rounds") if step_budget is None else None
    estimated_rounds = round_limit or math.ceil((step_budget - env_steps) / collection_steps)
    round_idx = 0
    while (round_idx < round_limit if step_budget is None else env_steps < step_budget):
        round_idx += 1
        agent.world_model.train()
        for _ in range(params["updates_per_round"]):
            batch = agent.replay_buffer.sample(params["batch_size"])
            if agent.behavior is None:
                pred = agent.world_model(batch)
            else:
                pred, states = agent.world_model(batch, return_states=True)
            loss, _ = agent.world_model.compute_loss(batch, pred)
            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.world_model.parameters(), params["grad_clip_norm"])
            agent.optimizer.step()
            losses.append(loss.item())
            if agent.behavior is not None:
                agent.optimizer.zero_grad(set_to_none=True)
                behavior_losses.append(agent.behavior.update(agent.world_model, states))

        reward = collect(training_policy=True)
        returns.append(reward)
        return_steps.append(env_steps)
        print(f"Round {round_idx}/{estimated_rounds} | Environment Steps: {env_steps} | "
              f"World Model Updates: {len(losses)} | Rollout Return: {reward:.4f} | "
              f"Loss: {losses[-1]:.2f}", flush=True)
        if not np.isfinite(losses[-1]) or (behavior_losses and not np.isfinite(behavior_losses[-1]).all()):
            raise FloatingPointError("Training produced non-finite losses")

    if step_budget is not None and env_steps != step_budget:
        raise RuntimeError(f"Expected {step_budget} environment steps, got {env_steps}")
    if not np.isfinite(losses).all() or not np.isfinite(behavior_losses).all():
        raise FloatingPointError("Training produced non-finite losses")
    np.savez(
        out / "metrics.npz",
        train_losses=np.array(losses),
        returns=np.array(returns),
        env_steps=np.array(return_steps),
        loss_version=params["loss_version"],
        behavior_losses=np.asarray(behavior_losses).reshape(-1, 2),
    )
    checkpoint = {
        "model": agent.world_model.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "env_steps": env_steps,
        "params": params,
        "loss_version": params["loss_version"],
        "world_model_updates": len(losses),
    }
    if agent.behavior is not None:
        checkpoint.update(
            behavior=agent.behavior.state_dict(),
            actor_optimizer=agent.behavior.actor_optimizer.state_dict(),
            critic_optimizer=agent.behavior.critic_optimizer.state_dict(),
        )
    staging = out / "checkpoint.tmp"
    torch.save(checkpoint, staging)
    staging.replace(out / "checkpoint.pt")

    plt.figure(figsize=(7, 4))
    plt.plot(return_steps, returns, marker="o")
    plt.xlabel("Training Environment Steps")
    plt.ylabel("Training Rollout Return")
    plt.title(f"{algorithm} Cartpole-Swingup Training")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "curve.png", dpi=150)
    plt.close()
    agent.logger.close()
    print(f"TRAINING COMPLETE | Environment Steps: {env_steps} | "
          f"World Model Updates: {len(losses)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs_5seed.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="Run one configured training seed")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Override the configured device")
    parser.add_argument("--env-steps", type=int, help="Override the exact environment-step budget")
    parser.add_argument("--threads", type=int, help="Set PyTorch CPU intra-op threads")
    args = parser.parse_args()

    yaml = YAML(typ="safe")
    with (HERE / "configs.yaml").open() as f:
        base = yaml.load(f)
    with args.config.open() as f:
        cfg = yaml.load(f)
    params = {**base["defaults"], **base["dmc"]}
    for section in ("environment", "training", "planner", "model", "behavior"):
        params.update(cfg.get(section, {}))
    if args.device is not None:
        params["device"] = args.device
    if args.env_steps is not None:
        params["env_step_budget"] = args.env_steps
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)

    if params.get("loss_version") not in ("original", "aligned", "v2"):
        parser.error("loss_version must be original, aligned, or v2")
    if params["collection_steps"] <= params["chunk_length"] or params["chunk_length"] < 2:
        parser.error("Require collection_steps > chunk_length >= 2")
    step_budget = params.get("env_step_budget")
    required_counts = ["updates_per_round", "n_seed_episodes"]
    if step_budget is None:
        required_counts.append("rounds")
    for key in required_counts:
        if params[key] < 1:
            parser.error(f"{key} must be positive")
    if step_budget is not None and step_budget <= params["collection_steps"] * params["n_seed_episodes"]:
        parser.error("Environment budget must exceed the maximum random prefill")
    if (args.device == "cuda" or (step_budget is not None and params["device"] == "cuda")):
        if not torch.cuda.is_available():
            parser.error("CUDA requested, but PyTorch cannot use it")
    algorithm = params.get("algorithm", "planet")
    if algorithm not in ("planet", "dreamer"):
        parser.error("algorithm must be planet or dreamer")
    if algorithm == "dreamer" and params["loss_version"] == "original":
        parser.error("Dreamer requires a world-model loss with successor reward alignment")

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
