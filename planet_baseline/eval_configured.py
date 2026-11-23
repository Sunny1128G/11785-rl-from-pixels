"""Evaluate configured PlaNet or Dreamer runs on held-out CPU episodes.

Run from planet_baseline/ after train_configured.py has produced seed folders:
    python eval_configured.py --runs ../runs/v2
    python eval_configured.py --runs ../runs/original --random-baseline

For every runs/seed_<s>/checkpoint.pt, writes runs/seed_<s>/eval.npz with per-episode
returns on each evaluation seed (100-109 by default). Existing eval.npz files are skipped.
Evaluation uses rendered camera-0 frames without exploration noise. Each episode
reseeds torch so CEM and posterior sampling are repeatable.
"""
import argparse
import hashlib
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np
import torch
from dm_control import suite
from ruamel.yaml import YAML

HERE = Path(__file__).resolve().parent


def load_checkpoint_agent(ckpt_path):
    from model_based_agent import ModelBasedLearner
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = dict(ckpt["params"])
    params["device"] = "cpu"
    agent = ModelBasedLearner(params)
    agent.world_model.load_state_dict(ckpt["model"])
    agent.world_model.eval()
    if agent.behavior is not None:
        agent.behavior.load_state_dict(ckpt["behavior"])
        agent.behavior.eval()
    return agent, ckpt


def run_episode(agent, eval_seed, policy, steps, report_steps, *, timing=None):
    params = agent.params
    resolution = params["observation_resolution"]
    env = suite.load(params["domain_name"], params["task_name"], task_kwargs={"random": eval_seed})
    spec = env.action_spec()
    rng = np.random.default_rng(eval_seed)
    torch.manual_seed(eval_seed)
    ts = env.reset()
    h = agent.world_model.get_init_h_state(1)
    total, total_at_report = 0.0, None

    with torch.no_grad():
        for t in range(steps):
            if policy == "random":
                action = rng.uniform(spec.minimum, spec.maximum).astype(spec.dtype)
            else:
                image = env.physics.render(height=resolution, width=resolution, camera_id=0)
                obs = agent.process_dm_observation(image).unsqueeze(0)
                embed = agent.world_model.obs_encoder(obs)
                z = agent.world_model.repr_model(h, embed).sample()
                if timing is not None:
                    start = perf_counter_ns()
                planned = agent.select_action(h, z)
                if timing is not None:
                    timing.append((perf_counter_ns() - start) / 1e6)
                action = np.clip(planned.cpu().numpy(), spec.minimum, spec.maximum).astype(spec.dtype)
                h = agent.world_model.rnn_model(h, z, planned.unsqueeze(0))

            ts = env.step(action)
            total += float(ts.reward or 0.0)
            if report_steps is not None and t + 1 == report_steps:
                total_at_report = total
            if ts.last():
                break

    return total, total_at_report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="Folder holding seed_<s>/ subfolders")
    parser.add_argument("--config", type=Path, default=HERE / "configs_5seed.yaml")
    parser.add_argument("--random-baseline", action="store_true", help="Also run uniform random actions")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU intra-op threads")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)

    with args.config.open() as f:
        cfg = YAML(typ="safe").load(f)
    eval_seeds = cfg["seeds"]["eval_seeds"]
    steps = cfg["evaluation"]["episode_steps"]
    report_steps = 120 if cfg["evaluation"]["report_120_step"] else None

    seed_dirs = sorted(args.runs.glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    if not seed_dirs:
        parser.error(f"No seed_* folders found under {args.runs}")

    summary = []
    for seed_dir in seed_dirs:
        out = seed_dir / "eval.npz"
        if out.exists():
            print(f"Skipping {seed_dir.name}: {out.name} already exists", flush=True)
            continue

        checkpoint = seed_dir / "checkpoint.pt"
        agent, ckpt = load_checkpoint_agent(checkpoint)
        algorithm = agent.params.get("algorithm", "planet")
        loss_version = ckpt.get("loss_version", agent.params.get("loss_version", "historical"))
        print(f"\n{seed_dir.name} loss_version={loss_version} env_steps={ckpt['env_steps']}", flush=True)

        full, at_report, timings = [], [], []
        rand_full, rand_report = [], []
        for es in eval_seeds:
            timing = []
            r_full, r_rep = run_episode(agent, es, algorithm, steps, report_steps, timing=timing)
            timings.append(np.array(timing))
            full.append(r_full)
            at_report.append(r_rep if r_rep is not None else np.nan)
            print(f"  eval seed {es}: full={r_full:.4f} report={r_rep}", flush=True)

            if args.random_baseline:
                rf, rr = run_episode(agent, es, "random", steps, report_steps)
                rand_full.append(rf)
                rand_report.append(rr if rr is not None else np.nan)

        lengths = np.array([len(t) for t in timings])
        latency = np.concatenate(timings)
        arrays = dict(
            algorithm=np.array(algorithm),
            eval_seeds=np.array(eval_seeds),
            **{f"{algorithm}_full": np.array(full), f"{algorithm}_120": np.array(at_report)},
            actual_episode_steps=lengths,
            action_latency_ms=latency,
            action_latency_median_ms=np.array([np.median(t) for t in timings]),
            action_latency_p95_ms=np.array([np.percentile(t, 95) for t in timings]),
            eval_cpu=np.array(platform.processor()),
            eval_threads=np.array(torch.get_num_threads()),
            eval_torch=np.array(str(torch.__version__)),
            checkpoint_sha256=np.array(hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
            train_seed=np.array(int(seed_dir.name.split("_")[1])),
            loss_version=loss_version,
            train_env_steps=np.array(ckpt["env_steps"]),
            episode_steps=np.array(steps),
        )
        if args.random_baseline:
            arrays["random_full"] = np.array(rand_full)
            arrays["random_120"] = np.array(rand_report)
        with out.with_suffix(".tmp").open("wb") as f:
            np.savez(f, **arrays)
        out.with_suffix(".tmp").replace(out)
        summary.append((seed_dir.name, np.mean(full), np.mean(at_report)))
        print(f"  action latency median={np.median(latency):.4f} ms "
              f"P95={np.percentile(latency, 95):.4f} ms", flush=True)

        agent.logger.close()

    if summary:
        print("\n=== PER TRAINING SEED (mean over eval seeds) ===")
        for name, m_full, m_120 in summary:
            print(f"{name}: full={m_full:.4f}  120-step={m_120:.4f}")

    print("\nEVALUATION COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
