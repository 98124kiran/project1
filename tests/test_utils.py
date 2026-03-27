"""Tests for training utilities."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.utils.config import Config
from src.utils.metrics import MetricsTracker, EpisodeMetrics


class TestMetricsTracker:
    def test_mean_after_updates(self):
        tracker = MetricsTracker(window=10)
        for i in range(5):
            tracker.update(reward=float(i))
        assert abs(tracker.mean("reward") - 2.0) < 1e-6

    def test_windowing(self):
        tracker = MetricsTracker(window=3)
        for v in [1, 2, 3, 4, 5]:
            tracker.update(x=float(v))
        # window=3, last 3 values = [3,4,5], mean=4
        assert abs(tracker.mean("x") - 4.0) < 1e-6

    def test_summary_keys(self):
        tracker = MetricsTracker()
        tracker.update(a=1.0, b=2.0)
        s = tracker.summary()
        assert "a" in s and "b" in s

    def test_reset(self):
        tracker = MetricsTracker()
        tracker.update(x=5.0)
        tracker.reset()
        assert tracker.mean("x") == 0.0


class TestEpisodeMetrics:
    def test_to_dict_success_rate(self):
        ep = EpisodeMetrics()
        ep.tasks_completed = 8
        ep.tasks_dropped = 2
        ep.tasks_deadline_missed = 0
        ep.total_reward = 100.0
        ep.finalize(makespan=50.0)
        d = ep.to_dict()
        assert abs(d["success_rate"] - 0.8) < 1e-6
        assert d["makespan"] == 50.0


class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.env.num_edge_nodes == 5
        assert cfg.agent.gamma == 0.95

    def test_yaml_roundtrip(self, tmp_path):
        cfg = Config()
        path = str(tmp_path / "cfg.yaml")
        cfg.to_yaml(path)
        cfg2 = Config.from_yaml(path)
        assert cfg2.env.num_edge_nodes == cfg.env.num_edge_nodes
        assert cfg2.agent.actor_lr == cfg.agent.actor_lr
