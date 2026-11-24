# DreamerV1 vs. PlaNet at 100K environment steps

On pixel-based Cartpole Swingup, DreamerV1 selected actions about **20× faster** on CPU and earned more reward in the first 120 steps on all five training seeds. The full 1,000-step evaluation did not establish a return advantage for either agent.

## Evaluation results

| Metric | PlaNet | DreamerV1 | Paired difference, DreamerV1 minus PlaNet (95% CI) |
| --- | ---: | ---: | ---: |
| **Mean return, 1,000 steps** | **74.00** | **77.24** | **+3.25 [−35.19, +41.69]** |
| Mean return, first 120 steps | 1.91 | 8.44 | +6.53 [+6.12, +6.94] |
| Median action-selection latency | 8.44 ms | 0.42 ms | −8.06 ms [−8.51, −7.60] |
| P95 action-selection latency | 10.80 ms | 0.58 ms | −10.05 ms [−11.05, −9.04] |

The full-episode return favored DreamerV1 on two of five training seeds. Across all seeds, its early lead of 6.53 points fell to 3.25 points by the end of the episode. On steps 121–1,000 alone, PlaNet averaged 3.28 points more. DreamerV1 seeds 43–45 also produced nearly identical mean full-episode returns (74.73 each), so these returns offer limited evidence about whether those policies learned distinct behaviors.

| Training seed | PlaNet 1,000-step return | DreamerV1 1,000-step return | Paired difference |
| --- | ---: | ---: | ---: |
| 42 | 45.88 | 92.53 | +46.65 |
| 43 | 100.11 | 74.73 | −25.38 |
| 44 | 95.88 | 74.73 | −21.15 |
| 45 | 51.04 | 74.73 | +23.69 |
| 46 | 77.08 | 69.50 | −7.57 |

Return intervals are two-sided paired Student t intervals over the five training-seed differences. Each seed's score averages the same ten held-out evaluation seeds. The table reports pooled action-selection quantiles for each algorithm, while the paired latency intervals use each training seed's own quantiles. Timing covers CEM or actor execution from an inferred latent state. Rendering, encoding, latent inference, and simulator stepping are outside the timed section.

## Experimental setup

- Both agents trained on 64×64 Cartpole Swingup pixels with the same RSSM v2 loss, two random 120-step rollouts, and 30 world-model updates per collection round. Each seed completed exactly 100,000 environment steps and 24,960 world-model updates.
- PlaNet used CEM with horizon 5, 32 candidates, 2 iterations, and 8 elites. DreamerV1 used a tanh-squashed Gaussian actor, a critic, and differentiable latent imagination with horizon 15, discount 0.99, and λ = 0.95.
- Training used the same RTX 3070 and PyTorch 2.14.1+cu130 environment for both agents. DreamerV1 seed 43 was restarted from step zero after an interruption. Its restarted run is the one included here.
- Each of the ten final checkpoints was evaluated on seeds 100–109 for full 1,000-step episodes, using the same CPU and one PyTorch thread with additive exploration noise disabled. DreamerV1 executed the tanh of the actor Gaussian location. PlaNet's CEM sampling and both agents' posterior sampling used seeded randomness.

## Results and reproduction

- [100 episode records](comparison-100k/episodes.csv)
- [Aggregate and per-seed statistics](comparison-100k/summary.json)
- [Formal checkpoints, raw timing, and training metrics](comparison-100k/)
- [Training environment](comparison-100k/logs/environment.txt)
- [Reproduction commands and evaluation protocol](../../planet_baseline/DREAMERV1_100K.md)
