"""Read-only model validation; all generated artifacts go to a separate directory."""
import argparse
import ast
import csv
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
from torch.distributions import Independent, Normal, kl_divergence
from dm_control import suite
from model_based_agent import ModelBasedLearner, ReplayBuffer, Transition
from planet import Planet

HERE = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def extract_loss(filename):
    # Load the exact checked-in function without executing training at import time.
    tree = ast.parse((HERE / filename).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "aligned_loss")
    scope = {"torch": torch, "kl_divergence": kl_divergence}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), filename, "exec"), scope)
    return scope["aligned_loss"]


def distribution(x):
    return Independent(Normal(x, torch.ones_like(x)), 1)


def verify():
    # Exercise production CEM with a known, action-dependent reward landscape.
    class Toy:
        def rnn_model(self, h, z, action):
            return action

        def transition_model(self, h):
            return distribution(torch.zeros_like(h))

        def reward_model(self, h, z):
            return distribution(1 - (h - 0.6).square())

    params = dict(planning_horizon=5, n_plans=32, plan_optimization_iter=2,
                  top_k=8, min_action=-1., max_action=1., min_reward=0.,
                  max_reward=1., action_repeat=0)
    toy = SimpleNamespace(params=params, action_dim=1, d_type=torch.float32,
                          device="cpu", world_model=Toy())
    actions = []
    for seed in range(100):
        torch.manual_seed(seed)
        action = ModelBasedLearner.plan_action_with_cem(toy, torch.zeros(1, 1), torch.zeros(1, 1))
        assert action.shape == (1,) and torch.isfinite(action).all()
        assert -1 <= action.item() <= 1
        actions.append(action.item())
    mae = float(np.mean(np.abs(np.array(actions) - 0.6)))
    assert mae < 0.2, f"CEM failed known optimum: MAE {mae}"

    # Trace the actual Planet.forward and ReplayBuffer serialization. A synthetic
    # environment has obs_t=t and transition reward_t=t+1; h_{t+1}=h_t+a_t.
    T, B = 5, 2
    buffer = ReplayBuffer(dict(fp_precision=32, device="cpu"))
    episode = [Transition(torch.tensor([float(t)]), torch.ones(1),
                          torch.tensor([float(t + 1)])) for t in range(T)]
    batch = buffer.serialize_episode([episode] * B)

    class TraceModel:
        params = dict(batch_size=B, chunk_length=T, pixel_bit=5, free_nats=3.)

        def get_init_h_state(self, batch_size):
            return torch.zeros(batch_size, 1)

        def obs_encoder(self, obs):
            return obs

        def transition_model(self, h):
            return distribution(h)

        def repr_model(self, h, encoded):
            # h must match current obs time before applying current action.
            return distribution(h)

        def decoder_model(self, h, z):
            return distribution(h)

        def reward_model(self, h, z):
            return distribution(h)

        def rnn_model(self, h, z, action):
            return h + action

    tracer = TraceModel()
    pred = Planet.forward(tracer, batch)
    reward_mean = torch.stack([d.mean for d in pred["reward"]])
    assert torch.equal(reward_mean, batch["obs"])
    assert torch.equal(reward_mean[1:], batch["reward"][:-1])
    unshifted_mse = float((reward_mean - batch["reward"]).square().mean())
    assert unshifted_mse == 1.
    losses = {}
    for filename in ("midterm_pilot.py", "midterm_pilot_v2.py"):
        loss_fn = extract_loss(filename)
        _, components = loss_fn(tracer, batch, pred)
        assert components[0] == 0 and components[2] == 0
        # Perturb each target: all but the final reward must affect aligned loss.
        for t in range(T):
            altered = {**batch, "reward": batch["reward"].clone()}
            altered["reward"][t] += 10
            _, changed = loss_fn(tracer, altered, pred)
            assert (changed[2] > 0) == (t < T - 1)
        losses[filename] = components[2]
    return dict(status="passed", cem_seeds=100, cem_optimal_action=0.6,
                cem_mean_absolute_error=mae, cem_actions=actions,
                aligned_reward_mse=losses, unshifted_reward_mse=unshifted_mse,
                trace_current_state=reward_mean[:, 0, 0].tolist(),
                trace_transition_rewards=batch["reward"][:, 0, 0].tolist(),
                boundary="First state prediction and final transition reward are excluded.",
                limitation="Synthetic indexing/optimizer checks; not evidence of learned-model accuracy or improved training.")


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    params = ckpt["params"].copy()
    params["device"] = "cpu"
    # Avoid constructing an unused rendering environment/logger for evaluation.
    model = Planet(params, action_dim=1).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    agent = SimpleNamespace(params=params, device="cpu", d_type=torch.float32,
                            action_dim=1, world_model=model)
    agent.plan_action_with_cem = lambda h, z: ModelBasedLearner.plan_action_with_cem(agent, h, z)
    return agent, dict(env_steps=ckpt["env_steps"], params=params,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_episode(seed, agent, max_steps, collect_states=False):
    env = suite.load("cartpole", "swingup", task_kwargs={"random": seed})
    try:
        env.reset()
        spec = env.action_spec()
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        h = agent.world_model.get_init_h_state(1) if agent else None
        total, prefix, states = 0., 0., []
        with torch.inference_mode():
            for step in range(max_steps):
                if agent is None:
                    action = rng.uniform(spec.minimum, spec.maximum).astype(spec.dtype)
                else:
                    image = env.physics.render(height=64, width=64, camera_id=0)
                    obs = ModelBasedLearner.process_dm_observation(agent, image).unsqueeze(0)
                    z = agent.world_model.repr_model(h, agent.world_model.obs_encoder(obs)).sample()
                    if collect_states and step in (0, 30, 60, 90):
                        states.append((h.clone(), z.clone()))
                    planned = agent.plan_action_with_cem(h, z)
                    action = np.clip(planned.numpy(), spec.minimum, spec.maximum).astype(spec.dtype)
                    h = agent.world_model.rnn_model(h, z, planned.unsqueeze(0))
                ts = env.step(action)
                total += float(ts.reward or 0.)
                if step < 120:
                    prefix = total
                if ts.last():
                    break
        return dict(seed=seed, steps=step + 1, episode_complete=bool(ts.last()),
                    return_120=prefix, episode_return=total), states
    finally:
        env.close()


def benchmark(agent, states, repeats):
    with torch.inference_mode():
        for i in range(10):
            agent.plan_action_with_cem(*states[i % len(states)])
        samples = []
        for i in range(repeats):
            start = perf_counter()
            agent.plan_action_with_cem(*states[i % len(states)])
            samples.append((perf_counter() - start) * 1000)
    return dict(scope="CEM only; excludes encoding, rendering and environment step",
                warmup=10, repeats=repeats, states=len(states), samples_ms=samples,
                mean_ms=float(np.mean(samples)), median_ms=float(np.median(samples)),
                p95_ms=float(np.percentile(samples, 95)), std_ms=float(np.std(samples)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "validation_results")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    write_json(args.output / "verification.json", verify())
    print("Synthetic CEM and reward alignment checks PASSED", flush=True)
    if args.verify_only:
        return
    metadata = dict(platform=platform.platform(), processor=os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
                    started_at_utc=datetime.now(timezone.utc).isoformat(),
                    packages={name: version(name) for name in ("torch", "numpy", "dm-control", "mujoco", "gymnasium")},
                    source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                   for name in ("validate_midterm.py", "model_based_agent.py", "planet.py", "midterm_pilot.py", "midterm_pilot_v2.py")},
                    python=platform.python_version(), torch=torch.__version__,
                    numpy=np.__version__, torch_threads=torch.get_num_threads(),
                    seeds=args.seeds, max_steps=args.max_steps,
                    training_seeds_note="Evaluation seeds are environment seeds, not independent training runs.")
    rows = []
    for policy, checkpoint in (("random", None), ("run1", "midterm_pilot_checkpoint.pt"),
                               ("run2", "midterm_pilot_v2_checkpoint.pt")):
        agent = None
        if checkpoint:
            agent, metadata[policy] = load_model(HERE / checkpoint)
        for seed in args.seeds:
            row, states = run_episode(seed, agent, args.max_steps,
                                      collect_states=seed == args.seeds[0] and agent is not None)
            row["policy"] = policy
            rows.append(row)
            # Save each completed episode immediately to support interrupted runs.
            write_json(args.output / "episodes.json", rows)
            print(f"{policy} seed={seed} steps={row['steps']} return={row['episode_return']:.6f}", flush=True)
            if states:
                metadata[policy]["benchmark"] = benchmark(agent, states, 200)
                print(f"{policy} CEM median={metadata[policy]['benchmark']['median_ms']:.3f}ms", flush=True)
                write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "metadata.json", metadata)
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for policy in ("random", "run1", "run2"):
        selected = [r for r in rows if r["policy"] == policy]
        summary[policy] = {}
        for metric in ("return_120", "episode_return"):
            values = np.array([r[metric] for r in selected])
            summary[policy][metric] = dict(mean=float(values.mean()), std=float(values.std()),
                                           min=float(values.min()), max=float(values.max()), n=len(values))
    write_json(args.output / "summary.json", summary)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    episode_title = "Complete episode" if all(r["episode_complete"] for r in rows) else f"Capped at {args.max_steps} steps"
    for ax, metric, title in zip(axes, ("return_120", "episode_return"), ("First 120 steps", episode_title)):
        for i, policy in enumerate(summary):
            vals = [r[metric] for r in rows if r["policy"] == policy]
            ax.bar(i, np.mean(vals), yerr=np.std(vals), alpha=.5, capsize=4)
            ax.scatter(np.full(len(vals), i), vals, s=15, color="black")
        ax.set_xticks(range(3), ["Random", "Run 1", "Run 2"])
        ax.set_title(title)
        ax.set_ylabel("Cumulative reward")
    fig.suptitle(f"Existing checkpoints; {len(args.seeds)} evaluation seeds (dots); error bars: population SD")
    fig.tight_layout()
    fig.savefig(args.output / "evaluation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
