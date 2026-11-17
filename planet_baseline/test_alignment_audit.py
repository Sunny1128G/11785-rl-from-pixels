"""Exercise production replay/forward/CEM code with controlled states.

AST extraction avoids importing optional rendering dependencies; the executed
class and method bodies come directly from model_based_agent.py.
"""
import ast
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.distributions import Independent, Normal, kl_divergence
from planet import Planet
from utils import get_device, get_dtype


def production_nodes():
    source = Path(__file__).with_name('model_based_agent.py')
    tree = ast.parse(source.read_text())
    replay = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ReplayBuffer')
    learner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelBasedLearner')
    cem = next(n for n in learner.body if isinstance(n, ast.FunctionDef) and n.name == 'plan_action_with_cem')
    ns = dict(torch=torch, np=np, get_device=get_device, get_dtype=get_dtype,
              Independent=Independent, Normal=Normal)
    exec(compile(ast.Module(body=[replay, cem], type_ignores=[]), str(source), 'exec'), ns)
    return ns['ReplayBuffer'], ns['plan_action_with_cem']


class Point:
    def __init__(self, mean): self.mean = mean
    def sample(self): return self.mean
    def rsample(self): return self.mean


class AlignmentAuditTests(unittest.TestCase):
    def test_replay_exact_length_last_window_and_episode_isolation(self):
        Replay, _ = production_nodes()
        replay = Replay(dict(fp_precision=32, device='cpu', chunk_length=3))
        Row = namedtuple('Row', 'observation action reward')
        def episode(offset, n):
            return [Row(torch.tensor([float(offset+t)]), torch.ones(1),
                        torch.tensor([float(offset+t+1)])) for t in range(n)]
        replay.push(episode(0, 3))
        batch = replay.sample(4)
        torch.testing.assert_close(batch['obs'][:, 0, 0], torch.arange(3.).float())
        replay.push(episode(100, 5))
        with patch.object(np.random, 'choice', return_value=np.array([0, 1])), patch.object(
                np.random, 'randint', side_effect=lambda low, high: high-1):
            batch = replay.sample(2)
        torch.testing.assert_close(batch['obs'][:, 1, 0], torch.tensor([102., 103., 104.]))
        torch.testing.assert_close(batch['reward'][:-1], batch['obs'][1:])

    def test_actual_forward_order_and_boundary_gradients(self):
        latent = Independent(Normal(torch.zeros(2, 1), torch.ones(2, 1)), 1)
        model = SimpleNamespace(params=dict(batch_size=2, chunk_length=4, pixel_bit=5,
                                            loss_version='v2', free_nats=3.))
        model.get_init_h_state = lambda batch_size: torch.zeros(batch_size, 1)
        model.obs_encoder = lambda x: x
        model.transition_model = lambda h: latent
        model.repr_model = lambda h, x: latent
        model.decoder_model = lambda h, z: Point(h)
        model.reward_model = lambda h, z: Point(h)
        model.rnn_model = lambda h, z, a: h+a
        target = dict(obs=torch.arange(4.)[:, None, None].repeat(1, 2, 1),
                      action=torch.ones(4, 2, 1),
                      reward=torch.arange(1., 5.)[:, None, None].repeat(1, 2, 1))
        target['reward'].requires_grad_()
        pred = Planet.forward(model, target)
        loss, parts = Planet.compute_loss(model, target, pred)
        self.assertEqual(parts, (0., 3., 0.))
        means = torch.stack([d.mean for d in pred['reward']])
        self.assertEqual((means-target['reward']).square().mean().item(), 1.)
        changed = dict(target, reward=target['reward'].detach().clone())
        changed['reward'][-1] = 999.
        self.assertEqual(Planet.compute_loss(model, changed, pred)[1][2], 0.)
        changed['reward'][0] += 1.
        self.assertAlmostEqual(Planet.compute_loss(model, changed, pred)[1][2], 1/6)
        loss.backward()
        self.assertEqual(target['reward'].grad[-1].abs().sum().item(), 0.)

    def test_cem_rewards_are_read_after_transition(self):
        _, cem = production_nodes()
        seen = []
        wm = SimpleNamespace(rnn_model=lambda h,z,a: h+1,
                             transition_model=lambda h: Point(h))
        def reward(h,z):
            seen.append(h.clone())
            return Point(h * .1)
        wm.reward_model = reward
        agent = SimpleNamespace(world_model=wm, action_dim=1, d_type=torch.float32,
            device='cpu', params=dict(planning_horizon=3, plan_optimization_iter=1,
            n_plans=4, top_k=2, min_action=-1., max_action=1., min_reward=0.,
            max_reward=1., action_repeat=0))
        cem(agent, torch.zeros(1,1), torch.zeros(1,1))
        self.assertEqual([h[0,0].item() for h in seen], [1.,2.,3.])

    def test_kl_direction_free_nats_and_reward_nll(self):
        q = Independent(Normal(torch.tensor([[0.],[4.]]), torch.ones(2,1)), 1)
        p = Independent(Normal(torch.zeros(2,1), torch.ones(2,1)*2), 1)
        model = SimpleNamespace(params=dict(loss_version='v2', free_nats=3.))
        obs = torch.zeros(3,2,1)
        pred = dict(recon_obs=[Point(x) for x in obs], prior=[p]*3, posterior=[q]*3,
                    reward=[Independent(Normal(torch.ones(2,1)*v,1),1) for v in (999.,2.,4.)])
        target = dict(obs=obs, reward=torch.zeros(3,2,1))
        _, parts = Planet.compute_loss(model,target,pred)
        self.assertAlmostEqual(parts[1], kl_divergence(q,p).clamp_min(3.).mean().item())
        expected = -torch.stack([d.log_prob(torch.zeros(2,1)) for d in pred['reward'][1:]]).mean()
        expected -= .5*np.log(2*np.pi)
        self.assertAlmostEqual(parts[2], expected.item(), places=5)


if __name__ == '__main__': unittest.main()
