# Midterm validation

The validation entry point calls the existing planner, world-model forward pass,
replay serialization, and each pilot's actual `aligned_loss` function. It does
not retrain or overwrite the original checkpoints, metrics, or plots.

## Environment and commands

From the repository root in PowerShell:

```powershell
$env:UV_CACHE_DIR = Join-Path (Split-Path (Get-Location)) '.uv-cache'
uv sync --locked
uv run --locked python planet_baseline/validate_midterm.py --verify-only
uv run --locked python planet_baseline/validate_midterm.py
```

Python 3.12 is required by `pyproject.toml`; `uv.lock` records exact dependency
versions. The local environment is `.venv`. Windows uses MuJoCo's default
rendering backend; do not set the Linux-only OSMesa environment variables here.
The full evaluation renders images and requires an available graphics backend.

To do a short smoke run without mixing it with the complete evaluation:

```powershell
uv run --locked python planet_baseline/validate_midterm.py --seeds 100 --max-steps 120 --output planet_baseline/validation_smoke
```

## Checks and interpretation

1. **Synthetic CEM:** execute the production planner with the pilot settings
   (horizon 5, 32 candidates, 2 iterations, 8 elites) on a known reward
   landscape whose optimal action is 0.6. Across 100 fixed RNG seeds, check
   finite, bounded, correctly shaped actions and mean absolute error below 0.2.
2. **Reward indexing:** serialize synthetic transitions with observation `t`,
   action `1`, and transition reward `t+1`. Execute `Planet.forward` with a
   traceable synthetic dynamics model. The prediction at state `t+1` must pair
   with the reward stored at transition `t`. Execute both pilot loss functions
   without importing their training side effects. Perturb each target separately
   to check the loss includes every interior target and excludes the last one.
3. **Timing:** load each real checkpoint strictly, gather four posterior states
   from its first evaluation trajectory, warm up for 10 planner calls, then
   measure 200 calls. Report mean, median, population SD and P95 in milliseconds.
   PyTorch uses one CPU thread by default. Encoding, rendering, environment
   stepping and posterior inference are excluded from planner timing.
4. **Evaluation:** compare random actions and both saved checkpoints on environment
   seeds 100–109, up to 1,000 steps or termination. Each policy gets a fresh
   environment with the same seed. Record both the first 120 steps and complete
   episode return, actual steps, and whether the episode terminated. These seeds
   differ from training environment seed 42. They are not independent training seeds.

The synthetic reward check establishes consistency of the index convention;
it does not establish that the learned model predicts reward accurately or that
the offset improves control. Chunk boundaries omit the initial state's reward
prediction (the preceding transition is unavailable) and the final stored reward
(the next observation/state is unavailable).

## Outputs

The default output directory is `validation_results/` next to this document:

- `verification.json`: synthetic checks and errors.
- `metadata.json`: package versions, CPU, thread count, source/checkpoint SHA256,
  checkpoint settings, and raw timing samples.
- `episodes.json`: saved after each episode so partial progress is inspectable.
- `episodes.csv`: per-policy, per-seed results after a completed run.
- `summary.json`: means, population standard deviations and ranges.
- `evaluation.png`: individual seed results and mean ± population SD.

Use a different `--output` directory when preserving a previous validation run.
The script reports failures with a nonzero exit code; results should only be
described as complete after all requested episodes finish.

## Planner ablation

From the repository root, run the seven configurations on the Run 2 checkpoint
with evaluation environment seeds 100–109:

```powershell
uv run --locked python planet_baseline/eval_planner_ablation.py --output ../planner-ablation-run2
```

Use `--checkpoint PATH` for another Cartpole-Swingup PlaNet checkpoint. Use
`--seeds 100 --max-steps 120` for a short smoke run. Each invocation requires a
new output directory.

The baseline is horizon 5, 32 candidates, and 2 optimization iterations. The six
variants independently change the horizon to 3 or 8, candidates to 16 or 64,
or iterations to 1 or 4. Each configuration uses 8 elites and the same checkpoint
and environment seeds. Evaluation runs up to 1,000 steps per episode by default.

Each configuration writes per-seed returns and completion flags to
`episodes.json`. Its `timing.json` contains 200 CEM samples after 10 warm-up
calls, with median and P95 in milliseconds. Timing uses the same posterior
states from the first baseline episode for all configurations. Rendering,
encoding, posterior inference, and environment stepping are outside the timed
interval. `summary.csv` aggregates completed configurations and
`metadata.json` records the checkpoint hash, seed list, and settings.
