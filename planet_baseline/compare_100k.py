"""Validate and summarize the completed paired 100k Cartpole-Swingup runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

TRAIN_SEEDS = (42, 43, 44, 45, 46)
EVAL_SEEDS = tuple(range(100, 110))
ALGORITHMS = ("planet", "dreamer")
METRICS = ("return_1000", "return_120", "latency_median_ms", "latency_p95_ms")
T_95_DF4 = 2.7764451051977987


def read_seed(root, algorithm, seed):
    directory = root / algorithm / f"seed_{seed}"
    checkpoint = directory / "checkpoint.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with np.load(directory / "metrics.npz") as train, np.load(directory / "eval.npz") as data:
        if (data["algorithm"].item() != algorithm or data["train_seed"].item() != seed
                or data["loss_version"].item() != "v2" or data["train_env_steps"].item() != 100000
                or data["checkpoint_sha256"].item() != digest
                or tuple(data["eval_seeds"]) != EVAL_SEEDS
                or data["actual_episode_steps"].tolist() != [1000] * len(EVAL_SEEDS)):
            raise ValueError(f"Evaluation metadata or coverage mismatch in {directory}")
        expected_steps = np.minimum(np.arange(360, 100121, 120), 100000)
        if (not np.array_equal(train["env_steps"], expected_steps)
                or train["train_losses"].shape != (30 * len(expected_steps),)):
            raise ValueError(f"Training budget or update counts mismatch in {directory}")
        latencies = data["action_latency_ms"].reshape(len(EVAL_SEEDS), 1000)
        full, first = data[f"{algorithm}_full"], data[f"{algorithm}_120"]
        if not all(np.isfinite(x).all() for x in (latencies, full, first, train["train_losses"])):
            raise ValueError(f"Non-finite results in {directory}")
        if algorithm == "dreamer":
            behavior = train["behavior_losses"]
            if behavior.shape != (30 * len(expected_steps), 2) or not np.isfinite(behavior).all():
                raise ValueError(f"Missing or non-finite behavior updates in {directory}")
        environment = {key: data[key].item() for key in ("eval_cpu", "eval_threads", "eval_torch")}
        rows = [
            dict(algorithm=algorithm, train_seed=seed, eval_seed=es, steps=1000,
                 return_1000=float(full[i]), return_120=float(first[i]),
                 latency_median_ms=float(np.median(latencies[i])),
                 latency_p95_ms=float(np.percentile(latencies[i], 95)),
                 checkpoint_sha256=digest)
            for i, es in enumerate(EVAL_SEEDS)
        ]
    seed_metrics = dict(
        return_1000=float(full.mean()),
        return_120=float(first.mean()),
        latency_median_ms=float(np.median(latencies)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
    )
    return rows, seed_metrics, latencies.ravel(), environment


def compare(root):
    rows, environments = [], []
    seed_metrics = {algorithm: {} for algorithm in ALGORITHMS}
    latencies = {algorithm: [] for algorithm in ALGORITHMS}
    for algorithm in ALGORITHMS:
        for seed in TRAIN_SEEDS:
            episodes, metrics, timing, environment = read_seed(root, algorithm, seed)
            rows.extend(episodes)
            seed_metrics[algorithm][seed] = metrics
            latencies[algorithm].append(timing)
            environments.append(environment)
    if any(env != environments[0] for env in environments):
        raise ValueError("Evaluation CPU environments differ")

    paired = []
    intervals = {}
    for seed in TRAIN_SEEDS:
        paired.append(dict(
            train_seed=seed,
            **{metric: seed_metrics["dreamer"][seed][metric] - seed_metrics["planet"][seed][metric]
               for metric in METRICS},
        ))
    for metric in METRICS:
        differences = [pair[metric] for pair in paired]
        average = mean(differences)
        margin = T_95_DF4 * stdev(differences) / len(differences) ** 0.5
        intervals[metric] = dict(mean_difference=average, ci95=[average - margin, average + margin])

    aggregate = {}
    for algorithm in ALGORITHMS:
        timing = np.concatenate(latencies[algorithm])
        aggregate[algorithm] = dict(
            return_1000=mean(seed_metrics[algorithm][seed]["return_1000"] for seed in TRAIN_SEEDS),
            return_120=mean(seed_metrics[algorithm][seed]["return_120"] for seed in TRAIN_SEEDS),
            latency_median_ms=float(np.median(timing)),
            latency_p95_ms=float(np.percentile(timing, 95)),
        )
    summary = dict(
        evaluation_environment=environments[0],
        training_seeds=TRAIN_SEEDS,
        eval_seeds=EVAL_SEEDS,
        episodes=len(rows),
        checkpoint_count=len(TRAIN_SEEDS) * len(ALGORITHMS),
        aggregate=aggregate,
        per_training_seed=seed_metrics,
        paired_dreamer_minus_planet=paired,
        paired_t_intervals=intervals,
    )
    with (root / "episodes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))
    print(json.dumps(summary["paired_t_intervals"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    compare(parser.parse_args().runs)
