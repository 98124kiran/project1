"""
Training loop for MADDPG on the edge computing environment.
"""

from __future__ import annotations

import os
import time
import numpy as np
from typing import Optional

from src.environment import EdgeComputingEnv
from src.agents import MADDPG
from src.utils import Config, get_logger, MetricsTracker, EpisodeMetrics


class Trainer:
    """
    Orchestrates the MADDPG training loop.

    Key design decisions:
      - Adaptive replanning: the agents continuously update their policies
        as the environment changes (node failures, load spikes, new task arrivals).
      - The centralised critic uses joint observations to learn coordinated
        scheduling policies.
    """

    def __init__(self, config: Config, device: Optional[str] = None):
        import torch
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger = get_logger("trainer", config.training.log_dir)

        # Build environment
        self.env = EdgeComputingEnv(config, seed=config.training.seed)

        # Build MADDPG
        self.maddpg = MADDPG(
            n_agents=self.env.n_agents,
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            config=config,
            device=self.device,
        )

        self.metrics = MetricsTracker(window=100)
        self.best_reward = -float("inf")

        self.logger.info(
            f"Environment: {self.env.n_agents} agents, "
            f"obs_dim={self.env.obs_dim}, act_dim={self.env.act_dim}"
        )
        self.logger.info(f"Device: {self.device}")

    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.config.training
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        self.logger.info("=" * 60)
        self.logger.info("Starting MADDPG training for edge computing scheduler")
        self.logger.info("=" * 60)

        start_time = time.time()

        for episode in range(1, cfg.num_episodes + 1):
            ep_metrics = self._run_episode(explore=True)

            self.metrics.update(**ep_metrics.to_dict())

            # Logging
            if episode % cfg.log_interval == 0:
                summary = self.metrics.summary()
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Episode {episode:4d}/{cfg.num_episodes} | "
                    f"reward={summary['episode_reward']:+.2f} | "
                    f"success={summary['success_rate']:.2%} | "
                    f"util={summary['avg_utilization']:.2%} | "
                    f"dl_miss={summary['deadline_miss_rate']:.2%} | "
                    f"elapsed={elapsed:.0f}s"
                )

            # Evaluation
            if episode % cfg.eval_interval == 0:
                eval_reward = self._evaluate(cfg.eval_episodes)
                self.logger.info(
                    f"  [EVAL] episode={episode}, "
                    f"avg_reward={eval_reward:.3f}"
                )
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.maddpg.save(os.path.join(cfg.checkpoint_dir, "best"))
                    self.logger.info("  [EVAL] New best model saved.")

            # Periodic checkpoints
            if episode % cfg.save_interval == 0:
                self.maddpg.save(
                    os.path.join(cfg.checkpoint_dir, f"episode_{episode}")
                )

        self.logger.info("Training complete.")
        self.maddpg.save(os.path.join(cfg.checkpoint_dir, "final"))

    # ------------------------------------------------------------------

    def _run_episode(self, explore: bool = True) -> EpisodeMetrics:
        obs_n = self.env.reset()
        ep = EpisodeMetrics()
        total_reward = 0.0

        for _ in range(self.config.env.max_steps):
            actions = self.maddpg.select_actions(obs_n, explore=explore)
            next_obs_n, rewards, done, info = self.env.step(actions)

            # Store transition
            self.maddpg.store_transition(obs_n, actions, rewards, next_obs_n, done)

            # Update networks
            self.maddpg.update()

            step_reward = float(np.mean(list(rewards.values())))
            total_reward += step_reward

            ep.add_step(
                reward=step_reward,
                # Per-step completion/drop counts are not tracked here;
                # cumulative totals are read from info at episode end.
                completed=0,
                dropped=0,
                deadline_missed=0,
                utilization=info["node_utilizations"],
            )

            obs_n = next_obs_n
            if done:
                break

        # Fill cumulative counts from final info
        ep.tasks_completed = info.get("completed_tasks", 0)
        ep.tasks_dropped = info.get("dropped_tasks", 0)
        ep.total_reward = total_reward
        ep.finalize(makespan=float(info.get("step", 0)))
        return ep

    def _evaluate(self, n_episodes: int) -> float:
        """Run n_episodes without exploration, return mean episode reward."""
        total = 0.0
        for _ in range(n_episodes):
            ep = self._run_episode(explore=False)
            total += ep.total_reward
        return total / max(n_episodes, 1)
