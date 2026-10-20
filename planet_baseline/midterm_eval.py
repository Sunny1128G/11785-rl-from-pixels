import numpy as np
import torch
import matplotlib.pyplot as plt
from dm_control import suite
from model_based_agent import ModelBasedLearner

ckpt = torch.load(
    "midterm_pilot_checkpoint.pt",
    map_location="cpu",
    weights_only=True
)

params = ckpt["params"].copy()
params["device"] = "cpu"

agent = ModelBasedLearner(params)
agent.world_model.load_state_dict(ckpt["model"])
agent.world_model.eval()

SEEDS = [42, 43, 44]
MAX_STEPS = 120

def evaluate(seed, policy):
    env = suite.load(
        "cartpole", "swingup",
        task_kwargs={"random": seed}
    )

    ts = env.reset()
    rng = np.random.default_rng(seed)
    spec = env.action_spec()
    total = 0.0

    h = agent.world_model.get_init_h_state(1)
    torch.manual_seed(seed)

    with torch.no_grad():
        for t in range(MAX_STEPS):
            if policy == "random":
                action = rng.uniform(
                    spec.minimum, spec.maximum
                ).astype(spec.dtype)
            else:
                image = env.physics.render(
                    height=64, width=64,
                    camera_id=0
                )

                obs = agent.process_dm_observation(
                    image
                ).unsqueeze(0)

                embed = agent.world_model.obs_encoder(obs)
                z = agent.world_model.repr_model(
                    h, embed
                ).sample()

                planned = agent.plan_action_with_cem(
                    h, z
                )

                action = planned.cpu().numpy()
                action = np.clip(
                    action,
                    spec.minimum,
                    spec.maximum
                ).astype(spec.dtype)

                h = agent.world_model.rnn_model(
                    h, z, planned.unsqueeze(0)
                )

            ts = env.step(action)
            total += float(ts.reward or 0.0)

            if ts.last():
                break

    return total

random_returns = []
planet_returns = []

for seed in SEEDS:
    r = evaluate(seed, "random")
    p = evaluate(seed, "planet")

    random_returns.append(r)
    planet_returns.append(p)

    print(
        f"Seed {seed} | "
        f"Random: {r:.4f} | "
        f"PlaNet: {p:.4f}",
        flush=True
    )

random_returns = np.array(random_returns)
planet_returns = np.array(planet_returns)

print("\n=== RESULTS ===")
print("Random Mean:", random_returns.mean())
print("PlaNet Mean:", planet_returns.mean())
print("Random Std:", random_returns.std())
print("PlaNet Std:", planet_returns.std())

np.savez(
    "midterm_eval_results.npz",
    seeds=np.array(SEEDS),
    random_returns=random_returns,
    planet_returns=planet_returns,
    horizon=MAX_STEPS
)

names = ["Random", "PlaNet"]
means = [
    random_returns.mean(),
    planet_returns.mean()
]

plt.figure(figsize=(6, 4))
plt.bar(names, means)
plt.ylabel("120-step Return")
plt.title("Cartpole-Swingup Pilot Evaluation")
plt.tight_layout()
plt.savefig("midterm_eval.png", dpi=150)
plt.close()

agent.logger.close()
print("MIDTERM EVALUATION COMPLETE!")
