# Lyra: Reward Alignment and Loss Audit for the November 20 Milestone

Completed: 2026-11-15. Scope: code timing, index boundaries, loss scaling, and training interfaces. This does not include a new five-seed training run or new control-performance conclusions.

## 1. Decisions

- Replay rows are `(o_t, a_t, R_t)`, with `R_t` returned after executing the action.
- Use incoming rewards: the successor state's reward head predicts `R_t`, pairing `reward_mean[1:]` with `target['reward'][:-1]`.
- The new objective is `loss_version: v2`: `0.5 * image_SSE + 0.5 * aligned_reward_MSE + KL_free_nats`, omitting parameter-independent Gaussian NLL constants.
- KL is posterior-to-prior. The Independent distribution sums latent dimensions; clamp each time/sample KL to 3 free nats before averaging. KL weight remains 1.
- Image/reward losses use output distribution means. Posterior latents retain reparameterized sampling. The 0.5 coefficient assumes the current fixed unit output variance.
- This unifies the objective; new training has not yet established a return improvement.

## 2. Code Path from Collection to Planning

| Location | Timing and checks |
| --- | --- |
| `train_configured.py::train.collect` | Calls `env.step(action)`, stores pre-action `obs` and returned reward, then assigns `obs = nxt`; stops at episode termination |
| `model_based_agent.py::collect_seed_episodes / collect_episode` | Historical collection also stores pre-action images and post-action rewards |
| `model_based_agent.py::ReplayBuffer.push/sample` | Stores separate trajectories; samples contiguous chunks within one trajectory without crossing resets |
| `model_based_agent.py::ReplayBuffer.serialize_episode` | Time-major images `[T,B,C,W,H]`, actions `[T,B,A]`, rewards `[T,B,1]` |
| `planet.py::Planet.forward` | Infers the posterior from `o_t`, decodes image/reward, then updates `h_(t+1)` using `a_t` |
| `planet.py::Planet.compute_loss` | Same-frame images; successor predictions paired with preceding row rewards |
| `model_based_agent.py::plan_action_with_cem` | Updates h with the action, samples the next prior z, then predicts reward; consistent with incoming rewards |
| `utils.py::ActionRepeatDM.step` | Sums reward within the wrapper transition; repeat=0 currently means one underlying step |

Example: observation identifiers `[0,1,2,3]`, rewards `[1,2,3,4]`, and state predictions `[0,1,2,3]` give aligned pairs `[1,2,3]` versus `[1,2,3]`. Reward MSE is 0; unshifted MSE is 1.

Boundary conventions:

1. Exclude the first prediction: its incoming reward is outside the chunk.
2. Exclude the final reward: its successor image is outside the chunk. Terminal transitions share this limitation; do not invent a successor observation by zero padding.
3. Dropping unmatched pairs is intentional for this storage format. Saving an extra successor image would require a revised, tested convention.
4. Require T >= 2 and exactly matching prediction/target shapes to prevent silent broadcasting.
5. The initial hidden state is zero without burn-in. This is a training limitation, not an alignment error.

## 3. Loss Versions and Scaling

| Version | Image/reward output | Reward target | Image/reward coefficient |
| --- | --- | --- | --- |
| original | Additional output-distribution samples | Same index | 1 |
| aligned | Output means | One-frame incoming shift | 1 |
| v2 | Output means | One-frame incoming shift | 0.5 |

Image SSE sums pixel channels and averages over T and B. Scalar reward MSE averages over valid T-1 and B. Image error is not averaged over pixels, and KL is not divided by latent dimension.

For a unit Gaussian, `-log p(x) = 0.5 * (x-mu)^2 + 0.5 * log(2*pi)`. Original output sampling adds expected errors of 12,288 per image and 1 per scalar reward, plus gradient noise. Aligned already removes this sampling; v2 additionally adopts NLL scaling.

Only reconstruction and reward terms are halved. KL stays unchanged, increasing its relative weight; this is not division of the whole loss by two. Total loss values across versions are not directly comparable measures of model quality.

## 4. Validation and Boundary Fix

Environment: repository `.venv-rssm`, CPU. Run from the repository root:

```powershell
.venv-rssm/Scripts/python.exe -m unittest discover -s planet_baseline -p 'test_*.py' -v
```

Result: **10/10 passed**.

- Five historical checks cover checkpoint/gradients, pixel ordering, reward pairing and invalid sequences, and historical loss versus NLL.
- `test_loss_versions.py` checks production v2 values, aligned scaling, output-mean gradients, invalid versions, and short sequences.
- Four `test_alignment_audit.py` checks cover production forward timing/boundaries; CEM reward after transition; exact-length replay, final windows, and episode isolation; KL direction/free nats and reward NLL.

Replay previously used `randint(high=L-T)`, failing for L=T and excluding the last window. The fix uses `high=L-T+1`, allowing starts 0 through L-T. Tests extract production ReplayBuffer/CEM bodies from source AST to avoid rendering dependencies. These are controlled tests, not full MuJoCo training tests.

Full training has not run in the current environment, which lacks ruamel.yaml. Historical short-checkpoint diagnostic results remain those of the original audit.

## 5. Integration and Experiment Plan for Runyu

`planet.py::Planet.compute_loss` reads `params['loss_version']`. `train_configured.py::main` loads YAML and merges training fields. `train` records the version in checkpoint params, top-level checkpoint metadata, and metrics.

`configs_5seed.yaml` already selects v2. Once the training environment is ready, run from the repository root:

```bash
python planet_baseline/train_configured.py --config planet_baseline/configs_5seed.yaml --output results/v2_run --seed 42
```

Omit `--seed` to run configured seeds 42?46. Use fresh output directories; each seed saves its checkpoint, metrics, and curve. This entry point starts fresh training and does not resume. YAML evaluation settings are reserved for separate evaluation; the training script does not automatically evaluate full episodes.

To isolate scaling, copy the same configuration for an aligned control and change only `training.loss_version: aligned`. Match training seeds, budgets, architecture, preprocessing, planner, and the replay fix in both groups; save outputs separately. Online trajectories can diverge as models change, so matched seeds do not guarantee identical subsequent data.

The current configuration remains a 1,440-step pilot per seed, not 500,000 steps. Runyu should set any expanded budget consistently across groups. Use full 1,000-step evaluation episodes with environment seeds 100?109, retain results for each training seed, and pass them to Michael for statistical analysis.

Both historical pilot checkpoints used aligned, including the filename containing v2. A filename alone cannot identify missing loss metadata. Explicitly select aligned when resuming a known historical pilot. Historical scripts, weights, and diagnostic `mean_objective` retain their meanings and scaling.

## 6. Message Ready to Send

Runyu, the reward-alignment and loss-scaling audit is complete. We use incoming rewards: reward_mean[1:] against target['reward'][:-1]. The formal loss_version is v2: 0.5 image SSE + 0.5 aligned reward MSE + the existing KL/free nats. Planet.compute_loss and the configured entry point are integrated, and the config selects v2. All ten checks passed. I also fixed replay sampling for exact-length trajectories and the final valid window. Please compare aligned against v2 with matched budgets and other settings, confirm the training version and budget, and share new checkpoints for multi-step and reward diagnostics.

Handoff status: conclusions and materials are ready. This document does not establish that Runyu has received or approved them; team confirmation of the budget and plan is still pending.
