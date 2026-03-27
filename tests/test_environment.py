"""Tests for edge computing environment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.utils.config import Config
from src.environment import EdgeComputingEnv
from src.environment.task import Task, TaskStatus, make_task
from src.environment.edge_node import EdgeNode


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def env(config):
    return EdgeComputingEnv(config, seed=0)


class TestTask:
    def test_feature_vector_shape(self):
        cfg = Config()
        rng = np.random.default_rng(0)
        task = make_task(0, 0, cfg.task.task_types[0], 0, rng)
        fv = task.feature_vector()
        assert fv.shape == (Task.FEATURE_DIM,)
        assert fv.dtype == np.float32

    def test_normalisation_bounds(self):
        cfg = Config()
        rng = np.random.default_rng(1)
        for t in range(cfg.env.num_task_types):
            task = make_task(0, t, cfg.task.task_types[t], 0, rng)
            fv = task.feature_vector()
            assert np.all(fv >= 0.0), f"negative feature: {fv}"
            assert np.all(fv <= 1.0 + 1e-6), f"feature > 1: {fv}"

    def test_tick_decrements_deadline(self):
        cfg = Config()
        rng = np.random.default_rng(2)
        task = make_task(0, 0, cfg.task.task_types[0], 0, rng)
        before = task.remaining_deadline
        task.tick()
        assert task.remaining_deadline == before - 1.0

    def test_urgency_range(self):
        cfg = Config()
        rng = np.random.default_rng(3)
        task = make_task(0, 0, cfg.task.task_types[0], 0, rng)
        assert 0.0 <= task.urgency <= 1.0


class TestEdgeNode:
    def test_feature_vector_shape(self):
        node = EdgeNode(node_id=0, cpu_capacity=16.0, memory_capacity=32.0, bandwidth=100.0)
        fv = node.feature_vector()
        assert fv.shape == (EdgeNode.FEATURE_DIM,)

    def test_can_accept_healthy(self):
        node = EdgeNode(node_id=0, cpu_capacity=16.0, memory_capacity=32.0, bandwidth=100.0)
        cfg = Config()
        rng = np.random.default_rng(0)
        task = make_task(0, 0, cfg.task.task_types[0], 0, rng)
        task.cpu_demand = 2.0
        task.memory_demand = 2.0
        assert node.can_accept(task)

    def test_cannot_accept_when_failed(self):
        node = EdgeNode(node_id=0, cpu_capacity=16.0, memory_capacity=32.0, bandwidth=100.0)
        node.is_failed = True
        cfg = Config()
        rng = np.random.default_rng(0)
        task = make_task(0, 0, cfg.task.task_types[0], 0, rng)
        assert not node.can_accept(task)

    def test_assign_updates_resources(self):
        node = EdgeNode(node_id=0, cpu_capacity=16.0, memory_capacity=32.0, bandwidth=100.0)
        cfg = Config()
        rng = np.random.default_rng(0)
        task = make_task(0, 0, cfg.task.task_types[0], 0, rng)
        task.cpu_demand = 2.0
        task.memory_demand = 4.0
        node.assign(task, current_step=0)
        assert node.cpu_used == 2.0
        assert node.memory_used == 4.0
        assert task.status == TaskStatus.RUNNING

    def test_fail_drops_tasks(self):
        node = EdgeNode(node_id=0, cpu_capacity=16.0, memory_capacity=32.0, bandwidth=100.0)
        cfg = Config()
        rng = np.random.default_rng(0)
        task = make_task(0, 0, cfg.task.task_types[0], 0, rng)
        task.cpu_demand = 2.0
        task.memory_demand = 2.0
        node.assign(task, current_step=0)
        dropped = node.fail()
        assert len(dropped) == 1
        assert dropped[0].status == TaskStatus.DROPPED
        assert node.is_failed


class TestEdgeEnv:
    def test_reset_obs_shape(self, env):
        obs_n = env.reset()
        assert len(obs_n) == env.n_agents
        for i, obs in obs_n.items():
            assert obs.shape == (env.obs_dim,), f"agent {i}: {obs.shape}"
            assert obs.dtype == np.float32

    def test_step_returns_correct_keys(self, env):
        obs_n = env.reset()
        actions = {i: env.act_dim - 1 for i in range(env.n_agents)}  # all idle
        next_obs_n, rewards, done, info = env.step(actions)
        assert set(next_obs_n.keys()) == set(range(env.n_agents))
        assert set(rewards.keys()) == set(range(env.n_agents))
        assert isinstance(done, bool)
        assert "queue_length" in info

    def test_episode_terminates(self, env):
        obs_n = env.reset()
        done = False
        steps = 0
        while not done:
            actions = {i: env.act_dim - 1 for i in range(env.n_agents)}
            obs_n, rewards, done, info = env.step(actions)
            steps += 1
            assert steps <= env.config.env.max_steps + 1, "Episode did not terminate"

    def test_valid_actions_include_idle(self, env):
        env.reset()
        for i in range(env.n_agents):
            valid = env.get_valid_actions(i)
            assert env.act_dim - 1 in valid, "Idle action always valid"

    def test_schedule_action_removes_from_queue(self, env):
        obs_n = env.reset()
        # Force a task into queue
        assert len(env.task_queue) > 0, "Queue should have tasks after reset"
        # Find a node that can accept the first task
        task = env.task_queue[0]
        target_node = None
        for node in env.nodes:
            if node.can_accept(task):
                target_node = node.node_id
                break
        if target_node is None:
            pytest.skip("No node can accept task in this random state")
        original_task_id = env.task_queue[0].task_id
        actions = {i: (0 if i == target_node else env.act_dim - 1)
                   for i in range(env.n_agents)}
        env.step(actions)
        # The originally scheduled task should no longer be at the front
        remaining_ids = {t.task_id for t in env.task_queue}
        assert original_task_id not in remaining_ids, \
            "Scheduled task should have been removed from queue"

    def test_obs_values_are_finite(self, env):
        obs_n = env.reset()
        for i, obs in obs_n.items():
            assert np.all(np.isfinite(obs)), f"Non-finite obs for agent {i}"
