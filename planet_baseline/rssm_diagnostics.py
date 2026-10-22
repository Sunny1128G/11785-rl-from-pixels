"""Audit the unchanged pilot RSSM on real DMC frames; never train or overwrite weights.

Run from any directory. See rssm_audit/REPORT.md for the protocol and interpretation.
The historical model uses C,W,H (not C,H,W); preserve this for checkpoint fidelity.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import kl_divergence

from planet import Planet


HERE = Path(__file__).resolve().parent


def preprocess(rgb, bits):
    """Match process_dm_observation, including float16 rounding and axis order."""
    image = np.asarray(rgb, dtype=np.float16)
    if bits < 8:
        image = np.floor(image / 2 ** (8 - bits))
    image = image / (2 ** bits) - 0.5
    return torch.tensor(image, dtype=torch.float32).transpose(-3, -1)


def display_image(image):
    """Invert the checkpoint's C,W,H layout, for display only."""
    return np.clip(image.transpose(-3, -1).cpu().numpy() + 0.5, 0, 1)


def aligned_reward_pairs(predictions, rewards):
    """Buffer row t=(o_t,a_t,R_t); state t+1 predicts reward R_t.

    There is no incoming reward for state 0, and no observation of the final
    transition's successor in the old buffer; exclude both unmatched boundaries.
    """
    if len(predictions) < 2 or len(predictions) != len(rewards):
        raise ValueError("Need matching sequences with at least two states")
    return predictions[1:], rewards[:-1]


def mean_objective(target, predictions, free_nats):
    """Pilot's mean-based objective, with explicit incoming-reward alignment.

    Diagnostic only: unit Gaussian NLL has an additional factor 1/2 and constant.
    Preserve the pilot SSE scale here, rather than silently changing its objective.
    """
    images = torch.stack([d.mean for d in predictions['recon_obs']])
    reward = torch.stack([d.mean for d in predictions['reward']])
    reward_pred, reward_target = aligned_reward_pairs(reward, target['reward'])
    recon = (images - target['obs']).square().mean(dim=(0, 1)).sum()
    kl = torch.stack([kl_divergence(q, p) for q, p in
                      zip(predictions['posterior'], predictions['prior'])])
    kl = kl.clamp_min(free_nats).mean()
    reward_loss = (reward_pred - reward_target).square().mean()
    return recon + kl + reward_loss, (recon, kl, reward_loss)


def load_model(checkpoint):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    params = dict(ckpt['params'], device='cpu')
    # Action width is determined by saved GRU weights, not a task-specific guess.
    action_dim = ckpt['model']['rnn_model.gru_net.weight_ih'].shape[1] - params['z_dim']
    model = Planet(params, action_dim).float()
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model, ckpt


def collect_trajectory(params, seed, steps, upright=False):
    from dm_control import suite

    env = suite.load(params['domain_name'], params['task_name'],
                     task_kwargs={'random': seed})
    try:
        time_step = env.reset()
        if upright:
            if (params['domain_name'], params['task_name']) != ('cartpole', 'swingup'):
                raise ValueError('The upright stress probe only supports cartpole-swingup')
            # Engineered probe, NOT a standard environment reset or policy return.
            with env.physics.reset_context():
                env.physics.data.qpos[:] = 0
                env.physics.data.qvel[:] = 0
        rng = np.random.default_rng(seed)
        spec = env.action_spec()
        frames, actions, rewards = [], [], []
        for _ in range(steps):
            frames.append(env.physics.render(height=64, width=64, camera_id=0).copy())
            action = (np.zeros(spec.shape) if upright else
                      rng.uniform(spec.minimum, spec.maximum)).astype(spec.dtype)
            total_reward = 0.0
            for _ in range(1 + params['action_repeat']):
                time_step = env.step(action)
                total_reward += float(time_step.reward or 0.0)
                if time_step.last():
                    break
            actions.append(action.copy())
            rewards.append(total_reward)
            if time_step.last():
                break
        return dict(rgb=np.stack(frames), actions=np.asarray(actions, dtype=np.float32),
                    rewards=np.asarray(rewards, dtype=np.float32), seed=seed,
                    policy='upright_zero_action_probe' if upright else 'random')
    finally:
        env.close()


@torch.no_grad()
def reconstruct(model, trajectory, seed, use_mean=False):
    torch.manual_seed(seed)
    obs = preprocess(trajectory['rgb'], model.params['pixel_bit'])
    h = model.get_init_h_state(1)
    reconstructed, reward, prior_reward, kl_values = [], [], [], []
    states = []
    for frame, action in zip(obs, trajectory['actions']):
        prior = model.transition_model(h)
        # This estimate uses no current image: an extra check of the path used by CEM.
        prior_reward.append(model.reward_model(h, prior.mean).mean.item())
        posterior = model.repr_model(h, model.obs_encoder(frame.unsqueeze(0)))
        z = posterior.mean if use_mean else posterior.sample()
        reconstructed.append(model.decoder_model(h, z).mean[0])
        reward.append(model.reward_model(h, z).mean.item())
        kl_values.append(kl_divergence(posterior, prior).item())
        states.append((h.clone(), z.clone()))
        h = model.rnn_model(h, z, torch.as_tensor(action).unsqueeze(0))
    return dict(obs=obs, reconstruction=torch.stack(reconstructed),
                reward=np.asarray(reward), prior_reward=np.asarray(prior_reward),
                kl=np.asarray(kl_values), states=states)


@torch.no_grad()
def open_loop(model, trajectory, filtered, context=5, horizon=45):
    """Condition on first context images; then use actions and priors only."""
    torch.manual_seed(trajectory['seed'] + 20000)
    h, z = filtered['states'][context - 1]
    images = []
    end = min(len(trajectory['actions']), context + horizon)
    for t in range(context, end):
        action = torch.as_tensor(trajectory['actions'][t - 1]).unsqueeze(0)
        h = model.rnn_model(h, z, action)
        z = model.transition_model(h).sample()
        images.append(model.decoder_model(h, z).mean[0])
    return torch.stack(images)


def summarize(trajectory, result):
    error = (result['reconstruction'] - result['obs']).square()
    predicted, target = aligned_reward_pairs(result['reward'], trajectory['rewards'])
    reward_mse = float(np.mean((predicted - target) ** 2))
    variance = float(np.var(target))
    mse = error.mean().item()
    return dict(
        seed=int(trajectory['seed']), policy=trajectory['policy'], frames=len(error),
        reconstruction_mse=mse, reconstruction_sse_per_frame=error.flatten(1).sum(1).mean().item(),
        psnr_db=-10 * math.log10(max(mse, 1e-15)),
        first_frame_baseline_mse=(result['obs'] - result['obs'][0]).square().mean().item(),
        reward_aligned_mse=reward_mse,
        reward_aligned_mae=float(np.mean(np.abs(predicted - target))),
        reward_unshifted_mse=float(np.mean((result['reward'] - trajectory['rewards']) ** 2)),
        reward_prior_mean_mse=float(np.mean((result['prior_reward'][1:] - target) ** 2)),
        reward_zero_baseline_mse=float(np.mean(target ** 2)),
        reward_oracle_constant_baseline_mse=variance,
        reward_r2=1 - reward_mse / variance if variance > 1e-15 else None,
        reward_target_mean=float(np.mean(target)), reward_target_max=float(np.max(target)),
        reward_prediction_mean=float(np.mean(predicted)),
        reward_prediction_min=float(np.min(predicted)), reward_prediction_max=float(np.max(predicted)),
        reward_pairs=len(target), kl_mean_nats=float(np.mean(result['kl'])),
        kl_free_nats_mean=float(np.maximum(result['kl'], 3).mean()),
    )


def save_figures(out, trajectories, results, predictions, probe, probe_result):
    # Fixed evenly spaced frames, not hand-selected examples.
    columns = 6
    fig, axes = plt.subplots(3 * len(trajectories), columns,
                             figsize=(12, 4.4 * len(trajectories)), squeeze=False)
    for k, (trajectory, result) in enumerate(zip(trajectories, results)):
        indices = np.linspace(0, len(trajectory['rgb']) - 1, columns, dtype=int)
        for j, t in enumerate(indices):
            original = display_image(result['obs'][t])
            recon = display_image(result['reconstruction'][t])
            error = (result['reconstruction'][t] - result['obs'][t]).abs().mean(0).T.numpy()
            axes[3*k, j].imshow(original)
            axes[3*k+1, j].imshow(recon)
            axes[3*k+2, j].imshow(error, vmin=0, vmax=0.5, cmap='magma')
            axes[3*k, j].set_title(f"seed {trajectory['seed']}, t={t}", fontsize=9)
            for row in range(3):
                axes[3*k+row, j].set_xticks([])
                axes[3*k+row, j].set_yticks([])
        for row, label in enumerate(['Original (5-bit)', 'Reconstructed', 'Absolute error']):
            axes[3*k+row, 0].set_ylabel(label, fontsize=9)
    fig.suptitle('Held-out random trajectories | posterior samples, decoder means', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / 'reconstruction_comparison.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(trajectories)+1, 1, figsize=(11, 2.7 * (len(trajectories)+1)), squeeze=False)
    for ax, trajectory, result in zip(axes[:, 0], trajectories, results):
        pred, target = aligned_reward_pairs(result['reward'], trajectory['rewards'])
        ax.plot(target, label='Environment reward R_t')
        ax.plot(pred, label='Posterior reward from state t+1', alpha=0.8)
        ax.plot(result['prior_reward'][1:], label='Prior-mean reward (no next image)', alpha=0.7)
        ax.axhline(0, color='gray', linestyle=':', label='Zero baseline')
        ax.set(title=f"Random trajectory, seed {trajectory['seed']}", xlabel='Transition t', ylabel='Reward')
        ax.legend(fontsize=8, loc='upper right')
    ax = axes[-1, 0]
    pred, target = aligned_reward_pairs(probe_result['reward'], probe['rewards'])
    ax.plot(target, label='Environment reward = 1')
    ax.plot(pred, label='Posterior reward prediction')
    ax.set(title='Engineered upright stress probe (NOT a standard evaluation episode)',
           xlabel='Transition t', ylabel='Reward', ylim=(-0.05, 1.1))
    ax.legend(fontsize=8, loc='center right')
    fig.tight_layout()
    fig.savefig(out / 'reward_prediction.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    indices = np.linspace(0, len(predictions[0]) - 1, 6, dtype=int)
    result = results[0]
    for j, t in enumerate(indices):
        axes[0, j].imshow(display_image(result['obs'][5+t]))
        axes[1, j].imshow(display_image(result['reconstruction'][5+t]))
        axes[2, j].imshow(display_image(predictions[0][t]))
        axes[0, j].set_title(f't={5+t}')
        for row in range(3):
            axes[row, j].set_xticks([])
            axes[row, j].set_yticks([])
    for ax, label in zip(axes[:, 0], ['Original', 'Posterior recon.', 'Prior prediction']):
        ax.set_ylabel(label)
    fig.suptitle(f"Seed {trajectories[0]['seed']} | prior sees only first 5 frames, then actions")
    fig.tight_layout()
    fig.savefig(out / 'open_loop_prediction.png', dpi=160)
    plt.close(fig)


def architecture_checks(model, trajectory):
    """Check shapes, Normal scales, recurrence, and all six modules' gradients."""
    torch.manual_seed(1234)
    params = dict(model.params)
    model.params.update(batch_size=2, chunk_length=4)
    obs = preprocess(trajectory['rgb'][:4], params['pixel_bit'])
    batch = dict(obs=obs[:, None].repeat(1, 2, 1, 1, 1),
                 action=torch.as_tensor(trajectory['actions'][:4])[:, None].repeat(1, 2, 1),
                 reward=torch.as_tensor(trajectory['rewards'][:4])[:, None, None].repeat(1, 2, 1))
    try:
        model.zero_grad(set_to_none=True)
        pred = model(batch)
        assert model.obs_encoder(batch['obs'][0]).shape == (2, 1024)
        assert pred['prior'][0].batch_shape == (2,)
        assert pred['posterior'][0].event_shape == (30,)
        assert pred['recon_obs'][0].mean.shape == (2, 3, 64, 64)
        assert pred['reward'][0].mean.shape == (2, 1)
        assert all(torch.all(d.base_dist.scale >= params['min_std'])
                   for d in pred['prior'] + pred['posterior'])
        loss, components = mean_objective(batch, pred, params['free_nats'])
        assert torch.isfinite(loss)
        loss.backward()
        gradients = {}
        for name, module in model.named_children():
            grads = [p.grad for p in module.parameters()]
            gradients[name] = bool(all(g is not None and torch.isfinite(g).all() for g in grads)
                                   and any(torch.count_nonzero(g) for g in grads if g is not None))
        assert all(gradients.values()), gradients
        # Compare the old sampled loss against the mean loss on identical states.
        legacy = [model.compute_loss(batch, pred)[1] for _ in range(8)]
        result = dict(strict_checkpoint_load=True, encoder_shape=[2, 1024],
                      decoder_shape=[2, 3, 64, 64], latent_event_shape=[30],
                      all_module_gradients_finite_nonzero=gradients,
                      mean_objective_components=[float(v.detach()) for v in components],
                      legacy_loss_components_mean=np.mean(legacy, axis=0).tolist(),
                      legacy_loss_components_std=np.std(legacy, axis=0).tolist(),
                      legacy_reward_alignment='unshifted (different target from pilot)')
    finally:
        model.params.clear()
        model.params.update(params)
        model.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=HERE / 'midterm_pilot_v2_checkpoint.pt')
    parser.add_argument('--output', type=Path, default=HERE / 'rssm_audit')
    parser.add_argument('--seeds', type=int, nargs='+', default=[45, 46, 47])
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--data', type=Path, help='Replay saved trajectories, without DMC/rendering')
    args = parser.parse_args()
    if args.steps < 50:
        parser.error('--steps must be at least 50 for the 5+45 open-loop check')
    torch.set_num_threads(args.threads)
    model, ckpt = load_model(args.checkpoint)
    if model.params['observation_resolution'] != 64:
        parser.error('This checkpoint architecture requires 64x64 observations')
    args.output.mkdir(parents=True, exist_ok=True)
    trajectories = []
    if args.data:
        with np.load(args.data, allow_pickle=False) as data:
            for i, seed in enumerate(data['seeds']):
                trajectories.append(dict(seed=int(seed), policy='random',
                    rgb=data[f'rgb_{i}'], actions=data[f'actions_{i}'], rewards=data[f'rewards_{i}']))
            probe = dict(seed=int(data['probe_seed']), policy='upright_zero_action_probe',
                         rgb=data['probe_rgb'], actions=data['probe_actions'], rewards=data['probe_rewards'])
    else:
        for seed in args.seeds:
            print(f'Collecting held-out random trajectory: seed={seed}', flush=True)
            trajectories.append(collect_trajectory(model.params, seed, args.steps))
        probe = collect_trajectory(model.params, 48, args.steps, upright=True)
    results, predictions, metrics = [], [], []
    for trajectory in trajectories:
        seed = trajectory['seed'] + 10000
        result = reconstruct(model, trajectory, seed)
        mean_result = reconstruct(model, trajectory, seed, use_mean=True)
        prediction = open_loop(model, trajectory, result)
        summary = summarize(trajectory, result)
        summary['mean_latent_diagnostic'] = summarize(trajectory, mean_result)
        summary['open_loop_mse_5_context_45_future'] = (
            prediction - result['obs'][5:5+len(prediction)]).square().mean().item()
        results.append(result)
        predictions.append(prediction)
        metrics.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    probe_result = reconstruct(model, probe, 10048)
    checks = architecture_checks(model, trajectories[0])
    all_pred = np.concatenate([r['reward'][1:] for r in results])
    all_prior = np.concatenate([r['prior_reward'][1:] for r in results])
    all_target = np.concatenate([t['rewards'][:-1] for t in trajectories])
    pooled_mse = sum(m['reconstruction_mse'] * m['frames'] for m in metrics) / sum(m['frames'] for m in metrics)
    pooled_reward_mse = float(np.mean((all_pred - all_target) ** 2))
    aggregate = dict(frames=sum(m['frames'] for m in metrics), reward_pairs=len(all_target),
        reconstruction_mse=pooled_mse, psnr_db=-10 * math.log10(pooled_mse),
        reward_aligned_mse=pooled_reward_mse, reward_aligned_mae=float(np.mean(np.abs(all_pred-all_target))),
        reward_zero_baseline_mse=float(np.mean(all_target ** 2)),
        reward_oracle_constant_baseline_mse=float(np.var(all_target)),
        reward_prior_mean_mse=float(np.mean((all_prior - all_target) ** 2)),
        reward_r2=1 - pooled_reward_mse / float(np.var(all_target)) if np.var(all_target) > 1e-15 else None)
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'numpy', 'matplotlib', 'einops', 'gymnasium', 'moviepy']}
    for name in ['dm-control', 'mujoco']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed (offline replay)'
    summary = dict(checkpoint=args.checkpoint.name,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_training_env_steps=int(ckpt['env_steps']),
        parameter_count=sum(p.numel() for p in model.parameters()),
        python=platform.python_version(), platform=platform.platform(), versions=versions,
        params=model.params, protocol=dict(inference='posterior sample, decoder/reward means',
            seeds=[t['seed'] for t in trajectories], frames_per_trajectory=[len(t['rgb']) for t in trajectories],
            input_noise=False, pixel_layout='C,W,H (historical checkpoint)',
            reward_alignment='prediction[1:] versus rewards[:-1]',
            stress_probe='upright qpos=qvel=0, zero actions; engineered, not standard evaluation',
            psnr='-10 log10(global raw normalized-pixel MSE), data_range=1; no clipping'),
        architecture_checks=checks, aggregate=aggregate, trajectories=metrics,
        upright_stress_probe=summarize(probe, probe_result))
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    arrays = dict(seeds=np.asarray([t['seed'] for t in trajectories]), probe_seed=np.asarray(probe['seed']),
                  probe_rgb=probe['rgb'], probe_actions=probe['actions'], probe_rewards=probe['rewards'],
                  probe_predicted_rewards=probe_result['reward'])
    for i, (trajectory, result) in enumerate(zip(trajectories, results)):
        for key in ['rgb', 'actions', 'rewards']:
            arrays[f'{key}_{i}'] = trajectory[key]
        arrays[f'predicted_rewards_{i}'] = result['reward']
    np.savez_compressed(args.output / 'trajectories.npz', **arrays)
    save_figures(args.output, trajectories, results, predictions, probe, probe_result)
    print(f'Artifacts written to {args.output}', flush=True)


if __name__ == '__main__':
    main()
