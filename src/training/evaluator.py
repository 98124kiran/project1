"""
Evaluator: loads a trained MADDPG checkpoint and benchmarks it against
baseline heuristics.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from src.environment import EdgeComputingEnv
from src.agents import MADDPG
from src.utils import Config, get_logger, EpisodeMetrics


class RandomPolicy:
    """Uniformly random scheduling baseline."""

    def __init__(self, n_agents: int, act_dim: int):
        self.n_agents = n_agents
        self.act_dim = act_dim

    def select_actions(self, obs_n: Dict, **kwargs) -> Dict[int, int]:
        return {i: int(np.random.randint(0, self.act_dim))
                for i in range(self.n_agents)}


class GreedyPolicy:
    """
    Greedy baseline: each node greedily picks the highest-priority task
    it can currently execute. Falls back to idle if nothing fits.
    """

    def __init__(self, env: EdgeComputingEnv):
        self.env = env

    def select_actions(self, obs_n: Dict, **kwargs) -> Dict[int, int]:
        actions: Dict[int, int] = {}
        scheduled: set = set()
        for i in range(self.env.n_agents):
            queue = self.env.task_queue
            best_idx = self.env.act_dim - 1   # idle
            best_priority = -1
            for j, task in enumerate(queue):
                if j in scheduled:
                    continue
                if self.env.nodes[i].can_accept(task) and task.priority > best_priority:
                    best_priority = task.priority
                    best_idx = j
            if best_idx < len(queue):
                scheduled.add(best_idx)
            actions[i] = best_idx
        return actions


class Evaluator:
    """Runs evaluation episodes and prints/returns comparison tables."""

    def __init__(self, config: Config, checkpoint_dir: Optional[str] = None):
        self.config = config
        self.logger = get_logger("evaluator", config.training.log_dir)

        self.env = EdgeComputingEnv(config, seed=config.training.seed + 1000)

        # MADDPG policy
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.maddpg = MADDPG(
            n_agents=self.env.n_agents,
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            config=config,
            device=device,
        )
        if checkpoint_dir:
            self.maddpg.load(checkpoint_dir)
            self.logger.info(f"Loaded checkpoint from {checkpoint_dir}")

        self.random_policy = RandomPolicy(self.env.n_agents, self.env.act_dim)
        self.greedy_policy = GreedyPolicy(self.env)

    def evaluate_all(self, n_episodes: int = 20) -> Dict[str, Dict]:
        policies = {
            "MADDPG": self.maddpg,
            "Greedy": self.greedy_policy,
            "Random": self.random_policy,
        }
        results = {}
        for name, policy in policies.items():
            metrics_list: List[Dict] = []
            for _ in range(n_episodes):
                obs_n = self.env.reset()
                ep = EpisodeMetrics()
                for _ in range(self.config.env.max_steps):
                    actions = policy.select_actions(obs_n)
                    next_obs_n, rewards, done, info = self.env.step(actions)
                    ep.add_step(
                        reward=float(np.mean(list(rewards.values()))),
                        completed=0, dropped=0, deadline_missed=0,
                        utilization=info["node_utilizations"],
                    )
                    obs_n = next_obs_n
                    if done:
                        break
                ep.tasks_completed = info.get("completed_tasks", 0)
                ep.tasks_dropped = info.get("dropped_tasks", 0)
                ep.total_reward = sum(ep.step_rewards)
                ep.finalize(float(info.get("step", 0)))
                metrics_list.append(ep.to_dict())

            # Aggregate
            agg: Dict[str, float] = {}
            for key in metrics_list[0]:
                agg[key] = float(np.mean([m[key] for m in metrics_list]))
            results[name] = agg
            self.logger.info(f"{name}: {agg}")

        self._print_table(results)
        return results

    def _print_table(self, results: Dict[str, Dict]) -> None:
        header_keys = ["episode_reward", "success_rate", "deadline_miss_rate",
                       "avg_utilization", "tasks_completed"]
        header = f"{'Policy':<10} " + " ".join(f"{k:>22}" for k in header_keys)
        self.logger.info("\n" + "=" * len(header))
        self.logger.info(header)
        self.logger.info("-" * len(header))
        for name, metrics in results.items():
            row = f"{name:<10} " + " ".join(
                f"{metrics.get(k, 0.0):>22.4f}" for k in header_keys
            )
            self.logger.info(row)
        self.logger.info("=" * len(header))
