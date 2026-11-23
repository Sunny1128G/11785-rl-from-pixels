# DreamerV1 and PlaNet at 100,000 environment steps

The comparison starts from PlaNet baseline commit 6ee095e. Both agents use
Cartpole Swingup with camera 0, 64×64 RGB observations, 5-bit pixels, one
environment step per action, and the same RSSM, replay sampling, and v2 loss.
Each seed starts with 240 random steps and alternates 30 world-model updates and
collection. With 120-step collection rounds, the last round collects 40 steps
for exactly 100,000 steps and 24,960 world-model updates per seed.

| Setting | PlaNet | DreamerV1 |
| --- | --- | --- |
| Action selection | CEM horizon 5, 32 candidates, 2 iterations, 8 elites | Tanh-squashed Gaussian actor |
| Behavior learning | CEM at collection time | Imagination horizon 15, discount 0.99, λ 0.95 |
| Training exploration | Additive Gaussian noise, SD 0.3 | Actor sample and additive Gaussian noise, SD 0.3 |

DreamerV1 starts 15 differentiable transitions from each detached posterior
state in the replay batch. The RSSM reward head scores the arrival state
because the v2 loss pairs that state with the preceding action's reward.
The λ-return at the source state bootstraps from the successor values, with
the last value used as the terminal bootstrap. The actor loss differentiates
through the imagined dynamics and value prediction with their parameters frozen.
The critic regresses from detached source features to detached λ-returns.
There is one actor update and one critic update per world-model update.
See [Hafner et al., Dream to Control](https://arxiv.org/abs/1912.01603).

## Set up and verify

From this repository root, using PowerShell and Python 3.12:

~~~powershell
uv sync --locked
$python = '.\.venv\Scripts\python.exe'
& $python -m unittest discover -s planet_baseline -p 'test_*.py'

foreach ($algorithm in @('planet', 'dreamer')) {
    $config = "planet_baseline/configs_$($algorithm)_100k.yaml"
    & $python planet_baseline/train_configured.py --config $config --output "runs/pilot/$algorithm" --seed 42 --device cpu --env-steps 1440 --threads 3
    if ($LASTEXITCODE) { throw "Pilot training failed for $algorithm" }
    & $python planet_baseline/eval_configured.py --config $config --runs "runs/pilot/$algorithm" --threads 1
    if ($LASTEXITCODE) { throw "Pilot evaluation failed for $algorithm" }
}
~~~

Check both training logs for 1,440 steps and 300 world-model updates. Evaluation
reloads the world-model weights and, for Dreamer, the actor and critic weights.
The checkpoints also store optimizer states. The configured trainer starts
fresh per seed. Use a separate output directory for retries of unfinished seeds.

## Train all five matched seeds

Select the same host, software environment, and device for both runs. The
configs default to CUDA. On this Windows system, the official CUDA build of
the locked PyTorch release can be installed in a local overlay. The CPU base
environment stays available for evaluation.

~~~powershell
uv pip install --python $python --target runs/cuda-overlay --no-deps --index-url https://download.pytorch.org/whl/cu130 'torch==2.14.1+cu130'
$env:PYTHONPATH = (Resolve-Path 'runs/cuda-overlay').Path
& $python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"
~~~

The following runs a complete seed pair at a time and records a separate log for
each training seed. Set $trainDevice to cpu and omit the CUDA overlay if running
the complete comparison on CPU instead. Set MUJOCO_GL=egl on a headless Linux GPU
host with EGL support.

~~~powershell
$trainDevice = 'cuda'
$run = 'runs/comparison-100k'
$env:MPLBACKEND = 'Agg'
New-Item -ItemType Directory -Force "$run/logs" | Out-Null
uv pip freeze --python $python > "$run/logs/environment.txt"
& $python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')" >> "$run/logs/environment.txt"
foreach ($seed in 42..46) {
    foreach ($algorithm in @('planet', 'dreamer')) {
        $config = "planet_baseline/configs_$($algorithm)_100k.yaml"
        & $python -u planet_baseline/train_configured.py --config $config --output "$run/$algorithm" --seed $seed --device $trainDevice --threads 2 *> "$run/logs/$algorithm-$seed.log"
        if ($LASTEXITCODE) { throw "Training failed for $algorithm seed $seed" }
    }
}
~~~

## Evaluate checkpoints and compare

Run the two evaluation commands sequentially on the same CPU. Both use one
PyTorch thread, held-out environment seeds 100–109, and no additive exploration
noise. Dreamer uses the tanh of the actor Gaussian location. PlaNet still samples
CEM candidates, and both sample the RSSM posterior with a fixed evaluation RNG seed.
Measure action selection on the inferred latent state. The timed section covers
the CEM planner or actor only. Rendering, observation encoding, latent inference,
environment stepping, and recurrent updates are outside the timed section.

~~~powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
foreach ($algorithm in @('planet', 'dreamer')) {
    $config = "planet_baseline/configs_$($algorithm)_100k.yaml"
    & $python -u planet_baseline/eval_configured.py --config $config --runs "$run/$algorithm" --threads 1 *> "$run/logs/$algorithm-eval.log"
    if ($LASTEXITCODE) { throw "Evaluation failed for $algorithm" }
}
& $python planet_baseline/compare_100k.py --runs $run
~~~

The comparison requires ten final checkpoints and 100 complete 1,000-step
episodes. Each seed directory contains checkpoint.pt, metrics.npz, and eval.npz
with raw timing samples and the checkpoint SHA-256. The comparison root contains
episodes.csv and summary.json. The summary reports mean returns at 1,000 and
120 steps and action latency median and P95. For each training seed, the paired
return difference is DreamerV1 minus PlaNet after averaging the ten held-out
episodes. The two-sided 95% Student t interval uses the five matched training
seeds and four degrees of freedom. The aggregate latency quantiles pool actions
within each algorithm. Paired latency differences use each seed's own quantiles.
