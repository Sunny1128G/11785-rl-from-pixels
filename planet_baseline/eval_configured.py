"""Evaluate configured PlaNet runs on the held-out evaluation seeds.

Run from planet_baseline/ after train_configured.py has produced seed folders:
    python eval_configured.py --runs ../runs/v2
    python eval_configured.py --runs ../runs/original --random-baseline

For every runs/seed_<s>/checkpoint.pt, writes runs/seed_<s>/eval.npz with per-episode
returns on each evaluation seed (100-109 by default). Existing eval.npz files are skipped.
Evaluation follows the midterm protocol: rendered 64x64 camera-0 frames, CEM planning,
no exploration noise. Each episode reseeds torch so CEM sampling is repeatable.
"""
import argparse
from pathlib import Path

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
    return agent, ckpt


def run_episode(agent, eval_seed, policy, steps, report_steps):
    env = suite.load("cartpole", "swingup", task_kwargs={"random": eval_seed})
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
                image = env.physics.render(height=64, width=64, camera_id=0)
                obs = agent.process_dm_observation(image).unsqueeze(0)
                embed = agent.world_model.obs_encoder(obs)
                z = agent.world_model.repr_model(h, embed).sample()
                planned = agent.plan_action_with_cem(h, z)
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
    args = parser.parse_args()

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

        agent, ckpt = load_checkpoint_agent(seed_dir / "checkpoint.pt")
        print(f"\n{seed_dir.name} loss_version={ckpt['loss_version']} env_steps={ckpt['env_steps']}", flush=True)

        full, at_report = [], []
        rand_full, rand_report = [], []
        for es in eval_seeds:
            r_full, r_rep = run_episode(agent, es, "planet", steps, report_steps)
            full.append(r_full)
            at_report.append(r_rep if r_rep is not None else np.nan)
            print(f"  eval seed {es}: full={r_full:.4f} report={r_rep}", flush=True)

            if args.random_baseline:
                rf, rr = run_episode(agent, es, "random", steps, report_steps)
                rand_full.append(rf)
                rand_report.append(rr if rr is not None else np.nan)

        arrays = dict(
            eval_seeds=np.array(eval_seeds),
            planet_full=np.array(full),
            planet_120=np.array(at_report),
            train_seed=np.array(int(seed_dir.name.split("_")[1])),
            loss_version=ckpt["loss_version"],
            train_env_steps=np.array(ckpt["env_steps"]),
            episode_steps=np.array(steps),
        )
        if args.random_baseline:
            arrays["random_full"] = np.array(rand_full)
            arrays["random_120"] = np.array(rand_report)
        np.savez(out, **arrays)
        summary.append((seed_dir.name, np.mean(full), np.mean(at_report)))

        agent.logger.close()

    if summary:
        print("\n=== PER TRAINING SEED (mean over eval seeds) ===")
        for name, m_full, m_120 in summary:
            print(f"{name}: full={m_full:.4f}  120-step={m_120:.4f}")

    print("\nEVALUATION COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
