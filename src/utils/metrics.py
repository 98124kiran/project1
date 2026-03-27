"""
Performance metrics tracking for the MADRL scheduler.
"""

from collections import deque
from typing import Dict, List
import numpy as np


class MetricsTracker:
    """Tracks and computes rolling averages for scheduler performance metrics."""

    def __init__(self, window: int = 100):
        self.window = window
        self._buffers: Dict[str, deque] = {}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            if key not in self._buffers:
                self._buffers[key] = deque(maxlen=self.window)
            self._buffers[key].append(float(value))

    def mean(self, key: str) -> float:
        buf = self._buffers.get(key)
        if not buf:
            return 0.0
        return float(np.mean(buf))

    def summary(self) -> Dict[str, float]:
        return {k: self.mean(k) for k in self._buffers}

    def reset(self) -> None:
        self._buffers.clear()


class EpisodeMetrics:
    """Collects per-episode metrics for one episode."""

    def __init__(self) -> None:
        self.total_reward: float = 0.0
        self.tasks_completed: int = 0
        self.tasks_dropped: int = 0
        self.tasks_deadline_missed: int = 0
        self.makespan: float = 0.0
        self.avg_utilization: List[float] = []
        self.step_rewards: List[float] = []

    def add_step(self, reward: float, completed: int, dropped: int,
                 deadline_missed: int, utilization: List[float]) -> None:
        self.total_reward += reward
        self.tasks_completed += completed
        self.tasks_dropped += dropped
        self.tasks_deadline_missed += deadline_missed
        self.avg_utilization.extend(utilization)
        self.step_rewards.append(reward)

    def finalize(self, makespan: float) -> None:
        self.makespan = makespan

    def to_dict(self) -> Dict[str, float]:
        n = self.tasks_completed + self.tasks_dropped + self.tasks_deadline_missed
        success_rate = self.tasks_completed / max(n, 1)
        return {
            "episode_reward": self.total_reward,
            "tasks_completed": float(self.tasks_completed),
            "tasks_dropped": float(self.tasks_dropped),
            "deadline_miss_rate": self.tasks_deadline_missed / max(n, 1),
            "success_rate": success_rate,
            "avg_utilization": float(np.mean(self.avg_utilization)) if self.avg_utilization else 0.0,
            "makespan": self.makespan,
        }
