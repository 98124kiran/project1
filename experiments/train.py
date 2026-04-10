"""
Training script — Phase 2 & 3 (Objectives 3 & 4).

Trains a MAPPO or GNN agent on the ManufacturingEnv and logs progress.
Supports loading a config from configs/default.yaml.

Quick-start
-----------
    # From the project root:
    python -m experiments.train --agent mappo --total-steps 100000

    # With GNN policy:
    python -m experiments.train --agent gnn --total-steps 200000

    # With meta-learning agent:
    python -m experiments.train --agent meta --total-steps 100000

Output
------
Checkpoint files are saved under ``checkpoints/<agent>/<timestamp>/``.
Training logs are printed to stdout.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Allow running as ``python -m experiments.train`` from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.manufacturing_env import ManufacturingEnv
from agents.ppo_agent import MAPPOAgent
from agents.gnn_policy import GNNPolicyAgent
from agents.meta_agent import MetaAgent


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_env(cfg: Dict[str, Any]) -> ManufacturingEnv:
    env_cfg = cfg.get("env", {})
    return ManufacturingEnv(config=env_cfg)


def build_agent(agent_type: str, env: ManufacturingEnv, cfg: Dict[str, Any]):
    """Instantiate the requested agent type."""
    obs_size = env.obs_size
    action_size = env.action_size
    num_agents = env.num_agents
    mappo_cfg = cfg.get("mappo", {})
    device = cfg.get("training", {}).get("device", "cpu")

    if agent_type == "mappo":
        return MAPPOAgent.from_config(obs_size, action_size, num_agents, {**mappo_cfg, "device": device})

    if agent_type == "gnn":
        gnn_cfg = cfg.get("gnn", {})
        return GNNPolicyAgent(
            obs_size=obs_size,
            action_size=action_size,
            num_agents=num_agents,
            num_machines=env.M,
            num_observable_jobs=env.K,
            d_model=gnn_cfg.get("d_model", 64),
            n_heads=gnn_cfg.get("n_heads", 4),
            n_layers=gnn_cfg.get("n_layers", 2),
            hidden_size=gnn_cfg.get("hidden_size", 128),
            dropout=gnn_cfg.get("dropout", 0.1),
            critic_hidden_size=mappo_cfg.get("critic_hidden_size", 256),
            lr_actor=mappo_cfg.get("lr_actor", 3e-4),
            lr_critic=mappo_cfg.get("lr_critic", 1e-3),
            gamma=mappo_cfg.get("gamma", 0.99),
            gae_lambda=mappo_cfg.get("gae_lambda", 0.95),
            clip_eps=mappo_cfg.get("clip_eps", 0.2),
            entropy_coef=mappo_cfg.get("entropy_coef", 0.01),
            value_coef=mappo_cfg.get("value_coef", 0.5),
            max_grad_norm=mappo_cfg.get("max_grad_norm", 0.5),
            n_epochs=mappo_cfg.get("n_epochs", 10),
            batch_size=mappo_cfg.get("batch_size", 64),
            rollout_steps=mappo_cfg.get("rollout_steps", 2048),
            device=device,
        )

    if agent_type == "meta":
        return MetaAgent.from_config(obs_size, action_size, num_agents, cfg)

    raise ValueError(f"Unknown agent type: {agent_type!r}. Choose mappo | gnn | meta.")


def evaluate_agent(
    agent,
    env: ManufacturingEnv,
    n_episodes: int = 5,
) -> Dict[str, float]:
    """
    Run *n_episodes* evaluation rollouts and return aggregate KPIs.

    Returns
    -------
    Dict with keys:
        mean_reward, mean_jobs_completed, mean_deadline_miss_rate,
        mean_cpu_util, mean_latency
    """
    total_reward = 0.0
    total_jobs = 0
    total_steps = 0
    total_violations = 0
    total_cpu = 0.0
    total_latency = 0.0

    for _ in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False
        while not done:
            actions, _, _ = agent.select_actions(obs, deterministic=True)
            obs, rewards, dones, info = env.step(actions)
            ep_reward += sum(rewards)
            ep_steps += 1
            done = all(dones)
        total_reward += ep_reward
        total_jobs += info.get("total_jobs_completed", 0)
        total_steps += ep_steps
        total_cpu += float(np.mean(info.get("cpu_utilizations", [0.0])))
        total_latency += float(np.mean(info.get("latencies", [0.0])))

    n = max(n_episodes, 1)
    return {
        "mean_reward": total_reward / n,
        "mean_jobs_completed": total_jobs / n,
        "mean_cpu_util": total_cpu / n,
        "mean_latency_ms": total_latency / n,
    }


# --------------------------------------------------------------------------- #
# Training loop                                                                 #
# --------------------------------------------------------------------------- #

def train(
    agent,
    env: ManufacturingEnv,
    cfg: Dict[str, Any],
    total_timesteps: int,
    rollout_steps: int,
    save_dir: str,
    log_interval: int = 5000,
    save_interval: int = 50000,
    n_eval_episodes: int = 5,
) -> List[Dict[str, Any]]:
    """
    On-policy training loop compatible with MAPPO, GNNPolicyAgent, and MetaAgent.

    Returns
    -------
    history : List[Dict]  — per-update log entries
    """
    os.makedirs(save_dir, exist_ok=True)

    history: List[Dict[str, Any]] = []
    obs = env.reset(seed=cfg.get("training", {}).get("seed"))
    env_steps = 0
    ep_rewards: List[float] = []
    ep_reward_acc: float = 0.0
    last_log_step = 0
    last_save_step = 0
    update_count = 0

    print(f"\n{'=' * 60}")
    print(f"  Training {type(agent).__name__}")
    print(f"  obs_size={env.obs_size}  action_size={env.action_size}  "
          f"agents={env.num_agents}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"{'=' * 60}\n")

    start_time = time.perf_counter()

    while env_steps < total_timesteps:
        # ---- Collect rollout -------------------------------------------
        for _ in range(rollout_steps):
            actions, log_probs, values = agent.select_actions(obs)
            global_obs = np.concatenate(obs)  # (N * obs_size,)

            next_obs, rewards, dones, info = env.step(actions)
            ep_reward_acc += sum(rewards)

            agent.store_transition(
                observations=obs,
                global_obs=global_obs,
                actions=actions,
                log_probs=log_probs,
                rewards=rewards,
                dones=dones,
                values=values,
            )
            obs = next_obs
            env_steps += env.num_agents

            if all(dones):
                ep_rewards.append(ep_reward_acc)
                ep_reward_acc = 0.0
                obs = env.reset()

        # ---- PPO update ------------------------------------------------
        update_metrics = agent.update(last_observations=obs, last_dones=dones)
        update_count += 1

        # ---- Logging ---------------------------------------------------
        if env_steps - last_log_step >= log_interval:
            elapsed = time.perf_counter() - start_time
            fps = env_steps / max(elapsed, 1e-9)
            mean_ep_reward = float(np.mean(ep_rewards[-20:])) if ep_rewards else 0.0
            log_entry = {
                "env_steps": env_steps,
                "updates": update_count,
                "mean_ep_reward": mean_ep_reward,
                "elapsed_s": elapsed,
                "fps": fps,
                **update_metrics,
            }
            history.append(log_entry)
            print(
                f"  step={env_steps:>8,}  "
                f"updates={update_count:>5}  "
                f"ep_reward={mean_ep_reward:>8.2f}  "
                f"actor_loss={update_metrics.get('actor_loss', 0):.4f}  "
                f"fps={fps:.0f}"
            )
            last_log_step = env_steps

        # ---- Save checkpoint -------------------------------------------
        if env_steps - last_save_step >= save_interval:
            ckpt_path = os.path.join(save_dir, f"checkpoint_{env_steps}.pt")
            agent.save(ckpt_path)
            print(f"  [checkpoint] Saved to {ckpt_path}")
            last_save_step = env_steps

    # ---- Final checkpoint ----------------------------------------------
    final_path = os.path.join(save_dir, "final.pt")
    agent.save(final_path)
    print(f"\n  [done] Final checkpoint saved to {final_path}")

    # ---- Final evaluation ----------------------------------------------
    print("\n  Running final evaluation ...")
    eval_metrics = evaluate_agent(agent, env, n_episodes=n_eval_episodes)
    print(f"  Eval metrics: {eval_metrics}")

    return history


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DRL agent on ManufacturingEnv."
    )
    parser.add_argument(
        "--agent",
        choices=["mappo", "gnn", "meta"],
        default="mappo",
        help="Agent type to train (default: mappo).",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override total_timesteps from config.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Checkpoint save directory (default: checkpoints/<agent>/).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (cpu / cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    mappo_cfg = cfg.get("mappo", {})
    train_cfg = cfg.get("training", {})

    if args.device:
        train_cfg["device"] = args.device
        cfg["training"] = train_cfg

    total_steps = args.total_steps or mappo_cfg.get("total_timesteps", 500_000)
    rollout_steps = mappo_cfg.get("rollout_steps", 2048)
    save_dir = args.save_dir or os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", args.agent
    )

    env = build_env(cfg)
    agent = build_agent(args.agent, env, cfg)

    history = train(
        agent=agent,
        env=env,
        cfg=cfg,
        total_timesteps=total_steps,
        rollout_steps=rollout_steps,
        save_dir=save_dir,
        log_interval=train_cfg.get("log_interval", 5000),
        save_interval=train_cfg.get("save_interval", 50000),
        n_eval_episodes=train_cfg.get("n_eval_episodes", 5),
    )

    print(f"\nTraining complete. {len(history)} log entries recorded.")


if __name__ == "__main__":
    main()
