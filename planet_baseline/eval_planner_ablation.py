"""Compare PlaNet CEM settings on one fixed Cartpole-Swingup checkpoint."""
import argparse
import csv
import os
import platform
from pathlib import Path

import numpy as np
import torch

from validate_midterm import HERE, benchmark, load_model, run_episode, write_json


BASELINE = dict(planning_horizon=5, n_plans=32, plan_optimization_iter=2, top_k=8)
SWEEP = {
    "baseline": {},
    "horizon_3": {"planning_horizon": 3},
    "horizon_8": {"planning_horizon": 8},
    "candidates_16": {"n_plans": 16},
    "candidates_64": {"n_plans": 64},
    "iterations_1": {"plan_optimization_iter": 1},
    "iterations_4": {"plan_optimization_iter": 4},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "midterm_pilot_v2_checkpoint.pt")
    parser.add_argument("--output", required=True, type=Path, help="New results directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 110)))
    parser.add_argument("--max-steps", type=int, default=1000, help="Lower for a smoke run")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if not all(0 <= seed < 2**32 for seed in args.seeds):
        parser.error("--seeds must be between 0 and 4294967295")
    if args.max_steps < 1 or args.threads < 1:
        parser.error("--max-steps and --threads must be positive")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output already exists: {args.output}")

    torch.set_num_threads(args.threads)
    agent, checkpoint = load_model(args.checkpoint)
    write_json(args.output / "metadata.json", {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint["sha256"],
        "training_env_steps": checkpoint["env_steps"],
        "evaluation_env_seeds": args.seeds,
        "max_steps": args.max_steps,
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "timing_reference": {"configuration": "baseline", "seed": args.seeds[0],
                             "steps": [0, 30, 60, 90]},
        "configurations": {name: {**BASELINE, **changes} for name, changes in SWEEP.items()},
    })

    summaries = []
    timing_states = []
    for name, changes in SWEEP.items():
        settings = {**BASELINE, **changes}
        agent.params.update(settings)
        destination = args.output / name
        destination.mkdir()

        episodes = []
        for seed in args.seeds:
            episode, states = run_episode(
                seed, agent, args.max_steps,
                collect_states=name == "baseline" and seed == args.seeds[0],
            )
            timing_states.extend(states)
            episodes.append(episode)
            print(f"{name} seed={seed} steps={episode['steps']} "
                  f"return={episode['episode_return']:.6f}", flush=True)
        write_json(destination / "episodes.json", episodes)

        # Use identical reference states and reset the planner RNG for each timing run.
        torch.manual_seed(args.seeds[0])
        timing = benchmark(agent, timing_states, repeats=200)
        write_json(destination / "timing.json", timing)
        print(f"{name} CEM median={timing['median_ms']:.3f} ms "
              f"p95={timing['p95_ms']:.3f} ms", flush=True)

        returns = np.array([episode["episode_return"] for episode in episodes])
        summaries.append(dict(
            configuration=name, **settings,
            evaluation_seeds=len(episodes),
            complete_episodes=sum(episode["episode_complete"] for episode in episodes),
            mean_return=float(returns.mean()),
            return_population_sd=float(returns.std()),
            cem_median_ms=timing["median_ms"],
            cem_p95_ms=timing["p95_ms"],
        ))
        with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
