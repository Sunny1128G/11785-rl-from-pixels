"""Summarize eval.npz files from two run folders, paired by training seed.

Usage (from planet_baseline/):
    python summarize_eval.py --a ../runs/original --b ../runs/v2 --names original v2

Differences are B minus A, computed per training seed.
Random-baseline numbers are printed if the A folder has them (eval_configured.py --random-baseline).
"""
import argparse
from pathlib import Path

import numpy as np


def load(folder):
    rows = {}
    for f in sorted(Path(folder).glob("seed_*/eval.npz")):
        with np.load(f) as d:
            rows[int(d["train_seed"])] = {k: d[k] for k in d.files}
    return rows


def mean_sd(values):
    x = np.asarray(values, dtype=float)
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--names", nargs=2, default=["A", "B"])
    args = p.parse_args()

    A, B = load(args.a), load(args.b)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        p.error("No common training seeds found in the two folders")
    print(f"Paired training seeds: {seeds}\n")

    for name, rows in zip(args.names, (A, B)):
        full = [rows[s]["planet_full"].mean() for s in seeds]
        r120 = [rows[s]["planet_120"].mean() for s in seeds]
        m_f, s_f = mean_sd(full)
        m_r, s_r = mean_sd(r120)
        print(f"{name}: 1000-step {m_f:.3f} ± {s_f:.3f} | 120-step {m_r:.3f} ± {s_r:.3f}")

    if "random_120" in A[seeds[0]]:
        rand_120 = [A[s]["random_120"].mean() for s in seeds]
        rand_full = [A[s]["random_full"].mean() for s in seeds]
        m_rf, _ = mean_sd(rand_full)
        m_r, s_r = mean_sd(rand_120)
        print(f"random (same eval seeds, from {args.names[0]} folder): 1000-step {m_rf:.3f} | 120-step {m_r:.3f} ± {s_r:.3f}")

    d_full = [B[s]["planet_full"].mean() - A[s]["planet_full"].mean() for s in seeds]
    d_120 = [B[s]["planet_120"].mean() - A[s]["planet_120"].mean() for s in seeds]
    print(f"\nPaired difference ({args.names[1]} - {args.names[0]}):")
    print(f"{'seed':>6} {'d_1000':>10} {'d_120':>10}")
    for s, df, d1 in zip(seeds, d_full, d_120):
        print(f"{s:>6} {df:>10.3f} {d1:>10.3f}")
    m_df, s_df = mean_sd(d_full)
    m_d1, s_d1 = mean_sd(d_120)
    print(f"{'mean':>6} {m_df:>10.3f} {m_d1:>10.3f}")
    print(f"{'sd':>6} {s_df:>10.3f} {s_d1:>10.3f}")
    print(f"{args.names[1]} higher on {sum(d > 0 for d in d_120)}/{len(seeds)} seeds (120-step)")


if __name__ == "__main__":
    main()
