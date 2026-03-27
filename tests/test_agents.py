"""Tests for MADDPG agents."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from src.utils.config import Config
from src.environment import EdgeComputingEnv
from src.agents import MADDPG, ReplayBuffer
from src.models.networks import Actor, Critic


@pytest.fixture
def config():
    cfg = Config()
    cfg.agent.warmup_steps = 0  # skip warmup in tests
    cfg.agent.batch_size = 8
    cfg.agent.buffer_size = 100
    cfg.agent.hidden_dim = 64
    cfg.agent.num_layers = 2
    return cfg


@pytest.fixture
def env(config):
    return EdgeComputingEnv(config, seed=0)


@pytest.fixture
def maddpg(config, env):
    return MADDPG(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        config=config,
        device=torch.device("cpu"),
    )


class TestNetworks:
    def test_actor_output_shape(self, env):
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        actor = Actor(obs_dim, act_dim, hidden_dim=64, num_layers=2)
        obs = torch.randn(4, obs_dim)
        out = actor(obs)
        assert out.shape == (4, act_dim)

    def test_critic_output_shape(self, env):
        n = env.n_agents
        critic = Critic(
            total_obs_dim=env.obs_dim * n,
            total_act_dim=env.act_dim * n,
            hidden_dim=64,
            num_layers=2,
        )
        batch = 4
        obs = torch.randn(batch, env.obs_dim * n)
        acts = torch.randn(batch, env.act_dim * n)
        out = critic(obs, acts)
        assert out.shape == (batch, 1)

    def test_actor_gumbel_sample(self, env):
        actor = Actor(env.obs_dim, env.act_dim, hidden_dim=64, num_layers=2)
        obs = torch.randn(4, env.obs_dim)
        sample = actor.sample_gumbel(obs)
        assert sample.shape == (4, env.act_dim)
        # Gumbel-softmax outputs should sum to ~1
        assert torch.allclose(sample.sum(dim=-1), torch.ones(4), atol=1e-5)


class TestReplayBuffer:
    def test_push_and_sample(self, env):
        buf = ReplayBuffer(capacity=50, n_agents=env.n_agents,
                           obs_dim=env.obs_dim, act_dim=env.act_dim)
        for _ in range(20):
            obs = np.random.randn(env.n_agents, env.obs_dim).astype(np.float32)
            acts = np.zeros((env.n_agents, env.act_dim), dtype=np.float32)
            rews = np.random.randn(env.n_agents).astype(np.float32)
            next_obs = np.random.randn(env.n_agents, env.obs_dim).astype(np.float32)
            buf.push(obs, acts, rews, next_obs, False)
        assert len(buf) == 20
        batch = buf.sample(8, torch.device("cpu"))
        obs_b, act_b, rew_b, nobs_b, done_b = batch
        assert obs_b.shape == (8, env.n_agents, env.obs_dim)
        assert act_b.shape == (8, env.n_agents, env.act_dim)

    def test_capacity_ring_buffer(self, env):
        buf = ReplayBuffer(capacity=10, n_agents=env.n_agents,
                           obs_dim=env.obs_dim, act_dim=env.act_dim)
        for _ in range(25):
            obs = np.zeros((env.n_agents, env.obs_dim), dtype=np.float32)
            acts = np.zeros((env.n_agents, env.act_dim), dtype=np.float32)
            rews = np.zeros(env.n_agents, dtype=np.float32)
            buf.push(obs, acts, rews, obs, False)
        assert len(buf) == 10  # capped at capacity


class TestMADDPG:
    def test_select_actions_shape(self, maddpg, env):
        obs_n = env.reset()
        actions = maddpg.select_actions(obs_n, explore=False)
        assert len(actions) == env.n_agents
        for i, a in actions.items():
            assert 0 <= a < env.act_dim

    def test_store_transition(self, maddpg, env):
        obs_n = env.reset()
        actions = maddpg.select_actions(obs_n)
        next_obs_n, rewards, done, _ = env.step(actions)
        maddpg.store_transition(obs_n, actions, rewards, next_obs_n, done)
        assert len(maddpg.buffer) == 1

    def test_update_runs_without_error(self, maddpg, env):
        obs_n = env.reset()
        for _ in range(20):
            actions = maddpg.select_actions(obs_n, explore=True)
            next_obs_n, rewards, done, _ = env.step(actions)
            maddpg.store_transition(obs_n, actions, rewards, next_obs_n, done)
            obs_n = next_obs_n
            if done:
                obs_n = env.reset()
        losses = maddpg.update()
        # losses may be None if update_every not met, but should not crash
        if losses is not None:
            for key in losses:
                assert np.isfinite(losses[key]), f"Non-finite loss: {key}={losses[key]}"

    def test_save_load(self, maddpg, tmp_path):
        maddpg.save(str(tmp_path))
        maddpg.load(str(tmp_path))   # should not raise

    def test_warmup_random_actions(self, env, config):
        config.agent.warmup_steps = 10_000
        m = MADDPG(n_agents=env.n_agents, obs_dim=env.obs_dim,
                   act_dim=env.act_dim, config=config,
                   device=torch.device("cpu"))
        obs_n = env.reset()
        # During warmup all actions should be in valid range
        actions = m.select_actions(obs_n, explore=True)
        for a in actions.values():
            assert 0 <= a < env.act_dim
