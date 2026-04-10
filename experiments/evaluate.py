"""
Evaluation script — Phase 5 (Objective 5).

Benchmarks trained DRL agents against classical baselines across
multiple episodes and reports scheduling KPIs.

Usage
-----
    # Evaluate a saved MAPPO checkpoint:
    python -m experiments.evaluate \\
        --agent-type mappo \\
        --checkpoint checkpoints/mappo/final.pt \\
        --n-episodes 50

    # Compare all baselines (no checkpoint needed):
    python -m experiments.evaluate --baselines-only --n-episodes 50

    # Save comparison bar chart:
    python -m experiments.evaluate --n-episodes 50 --save-plots results/
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.manufacturing_env import ManufacturingEnv
from agents.baselines import EDDAgent, FIFOAgent, GreedyAgent, RandomAgent, SPTAgent
from agents.ppo_agent import MAPPOAgent
from agents.gnn_policy import GNNPolicyAgent
from agents.meta_agent import MetaAgent
from visualization.gantt import plot_metrics_comparison, save_figure


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_env(cfg: Dict[str, Any], seed: int = 42) -> ManufacturingEnv:
    env_cfg = {**cfg.get("env", {}), "seed": seed}
    return ManufacturingEnv(config=env_cfg)


def load_drl_agent(
    agent_type: str,
    checkpoint: str,
    env: ManufacturingEnv,
    cfg: Dict[str, Any],
):
    """Load a trained DRL agent from a checkpoint file."""
    device = cfg.get("training", {}).get("device", "cpu")
    mappo_cfg = {**cfg.get("mappo", {}), "device": device}

    if agent_type == "mappo":
        agent = MAPPOAgent.from_config(
            env.obs_size, env.action_size, env.num_agents, mappo_cfg
        )
    elif agent_type == "gnn":
        gnn_cfg = cfg.get("gnn", {})
        agent = GNNPolicyAgent(
            obs_size=env.obs_size,
            action_size=env.action_size,
            num_agents=env.num_agents,
            num_machines=env.M,
            num_observable_jobs=env.K,
            d_model=gnn_cfg.get("d_model", 64),
            n_heads=gnn_cfg.get("n_heads", 4),
            n_layers=gnn_cfg.get("n_layers", 2),
            hidden_size=gnn_cfg.get("hidden_size", 128),
            device=device,
        )
    elif agent_type == "meta":
        agent = MetaAgent.from_config(
            env.obs_size, env.action_size, env.num_agents, cfg
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type!r}")

    agent.load(checkpoint)
    return agent


def build_baselines(env: ManufacturingEnv) -> Dict[str, Any]:
    """Construct all classical baseline agents."""
    return {
        "Random": RandomAgent(env.action_size, env.num_agents, rng_seed=0),
        "FIFO": FIFOAgent(env.action_size, env.num_agents, env.M, rng_seed=0),
        "SPT": SPTAgent(env.action_size, env.num_agents, env.M, env.K, rng_seed=0),
        "EDD": EDDAgent(env.action_size, env.num_agents, env.M, env.K, rng_seed=0),
        "Greedy": GreedyAgent(env.action_size, env.num_agents, env.M, rng_seed=0),
    }


# --------------------------------------------------------------------------- #
# Rollout evaluation                                                            #
# --------------------------------------------------------------------------- #

def run_episode(agent, env: ManufacturingEnv, deterministic: bool = True) -> Dict[str, float]:
    """
    Run one episode and return per-episode KPIs.

    Returns
    -------
    Dict with keys:
        total_reward, jobs_completed, deadline_miss_rate,
        mean_cpu_util, mean_latency_ms, episode_length
    """
    obs = env.reset()
    total_reward = 0.0
    step = 0
    deadline_violations_total = 0
    cpu_util_sum = 0.0
    latency_sum = 0.0
    done = False

    while not done:
        actions, _, _ = agent.select_actions(obs, deterministic=deterministic)
        obs, rewards, dones, info = env.step(actions)
        total_reward += sum(rewards)
        cpu_util_sum += float(np.mean(info.get("cpu_utilizations", [0.0])))
        latency_sum += float(np.mean(info.get("latencies", [0.0])))
        step += 1
        done = all(dones)

    jobs_completed = info.get("total_jobs_completed", 0)

    return {
        "total_reward": total_reward,
        "jobs_completed": float(jobs_completed),
        "mean_cpu_util": cpu_util_sum / max(step, 1),
        "mean_latency_ms": latency_sum / max(step, 1),
        "episode_length": float(step),
    }


def evaluate_agent(
    agent,
    env: ManufacturingEnv,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed_offset: int = 0,
) -> Dict[str, float]:
    """
    Evaluate *agent* over *n_episodes* and return aggregated KPIs.

    Uses different seeds per episode to reduce variance.
    """
    results: Dict[str, List[float]] = {
        "total_reward": [],
        "jobs_completed": [],
        "mean_cpu_util": [],
        "mean_latency_ms": [],
        "episode_length": [],
    }

    for ep in range(n_episodes):
        ep_seed = seed_offset + ep * 13  # deterministic seed sequence
        # Override env seed per episode for reproducibility
        env.reset(seed=ep_seed)
        ep_metrics = run_episode(agent, env, deterministic=deterministic)
        for k, v in ep_metrics.items():
            results[k].append(v)

    agg: Dict[str, float] = {}
    for k, vals in results.items():
        arr = np.array(vals)
        agg[f"mean_{k}"] = float(arr.mean())
        agg[f"std_{k}"] = float(arr.std())

    return agg


# --------------------------------------------------------------------------- #
# Main evaluation                                                               #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    env = build_env(cfg, seed=42)

    agents_to_eval: Dict[str, Any] = {}

    # Baselines (always included)
    baselines = build_baselines(env)
    agents_to_eval.update(baselines)

    # DRL agent (if checkpoint provided)
    if args.checkpoint and not args.baselines_only:
        print(f"\nLoading {args.agent_type} agent from {args.checkpoint} ...")
        drl_agent = load_drl_agent(args.agent_type, args.checkpoint, env, cfg)
        label = f"{args.agent_type.upper()} (trained)"
        agents_to_eval[label] = drl_agent

    print(f"\nEvaluating {len(agents_to_eval)} agents over {args.n_episodes} episodes each...")
    print("-" * 60)

    all_metrics: Dict[str, Dict[str, float]] = {}
    for name, agent in agents_to_eval.items():
        print(f"  {name:<20}", end="", flush=True)
        metrics = evaluate_agent(agent, env, n_episodes=args.n_episodes)
        all_metrics[name] = metrics
        print(
            f"  reward={metrics['mean_total_reward']:>8.2f} ± {metrics['std_total_reward']:.2f}"
            f"  jobs={metrics['mean_jobs_completed']:>6.1f}"
            f"  cpu={metrics['mean_mean_cpu_util']:.2f}"
        )

    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    header = f"{'Agent':<22} {'Mean Reward':>12} {'Jobs':>8} {'CPU Util':>10} {'Latency ms':>12}"
    print(header)
    print("-" * 66)
    for name, m in all_metrics.items():
        print(
            f"{name:<22}"
            f" {m['mean_total_reward']:>12.2f}"
            f" {m['mean_jobs_completed']:>8.1f}"
            f" {m['mean_mean_cpu_util']:>10.3f}"
            f" {m['mean_mean_latency_ms']:>12.2f}"
        )

    # Plot comparison
    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)
        plot_metrics = ["mean_total_reward", "mean_jobs_completed", "mean_mean_cpu_util"]
        fig = plot_metrics_comparison(
            {n: {k: m[k] for k in plot_metrics} for n, m in all_metrics.items()},
            metrics_to_plot=plot_metrics,
            title="Agent Performance Comparison",
        )
        out_path = os.path.join(args.save_plots, "comparison.png")
        save_figure(fig, out_path)
        print(f"\nPlot saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DRL and baseline scheduling agents."
    )
    parser.add_argument(
        "--agent-type",
        choices=["mappo", "gnn", "meta"],
        default="mappo",
        help="Type of DRL agent to load.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained DRL agent checkpoint (.pt file).",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Only evaluate classical baselines (no DRL agent).",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per agent.",
    )
    parser.add_argument(
        "--save-plots",
        default=None,
        help="Directory to save comparison plots (optional).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
