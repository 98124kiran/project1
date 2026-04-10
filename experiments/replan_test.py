"""
Disruption & Replanning Test — Phase 3 (Objective 3).

Tests how well agents recover from sudden disruptions mid-episode.
A disruption (e.g., forced machine failure on all nodes) is injected
at a configurable step, and recovery metrics are measured.

Metrics
-------
- Pre-disruption reward  : mean reward in steps [0, disruption_step)
- Post-disruption reward : mean reward in steps [disruption_step, end)
- Recovery drop          : (pre - post) / |pre|  — % degradation
- Recovery speed         : steps until post-disruption reward ≥ 0.9 × pre
- Makespan increase      : increase in total remaining work after disruption

Usage
-----
    # Test MAPPO checkpoint against all baselines:
    python -m experiments.replan_test \\
        --checkpoint checkpoints/mappo/final.pt \\
        --agent-type mappo \\
        --disruption-step 100 \\
        --n-episodes 20

    # MAML adaptation test:
    python -m experiments.replan_test \\
        --checkpoint checkpoints/meta/final.pt \\
        --agent-type meta \\
        --disruption-step 100 \\
        --adapt         # apply inner-loop adaptation after disruption
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.manufacturing_env import ManufacturingEnv, DEFAULT_CONFIG
from src.env.machine import MachineStatus
from agents.baselines import EDDAgent, FIFOAgent, GreedyAgent, RandomAgent, SPTAgent
from agents.ppo_agent import MAPPOAgent
from agents.gnn_policy import GNNPolicyAgent
from agents.meta_agent import MetaAgent
from visualization.gantt import plot_disruption_timeline, save_figure


# --------------------------------------------------------------------------- #
# Disruption injection                                                          #
# --------------------------------------------------------------------------- #

def inject_machine_failures(env: ManufacturingEnv, failure_fraction: float = 0.5) -> int:
    """
    Forcibly fail a fraction of all machines across all nodes.

    Parameters
    ----------
    env              : ManufacturingEnv
    failure_fraction : float — fraction of machines to fail (default 0.5)

    Returns
    -------
    int — number of machines failed
    """
    count = 0
    for node in env.nodes:
        for machine in node.machines:
            if machine.is_busy or machine.is_idle:
                if np.random.random() < failure_fraction:
                    machine.fail(repair_time=np.random.exponential(20.0))
                    count += 1
    return count


# --------------------------------------------------------------------------- #
# Replanning episode runner                                                     #
# --------------------------------------------------------------------------- #

def run_replan_episode(
    agent,
    env: ManufacturingEnv,
    disruption_step: int,
    failure_fraction: float = 0.5,
    adapt: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run one episode with a forced disruption at *disruption_step*.

    Parameters
    ----------
    agent           : any scheduling agent
    env             : ManufacturingEnv
    disruption_step : int   — step at which to inject machine failures
    failure_fraction: float — fraction of machines to fail
    adapt           : bool  — if True and agent is MetaAgent, run inner-loop
                              adaptation after the disruption
    seed            : int

    Returns
    -------
    Dict with episode metrics and per-step rewards.
    """
    obs = env.reset(seed=seed)

    pre_rewards: List[float] = []
    post_rewards: List[float] = []
    n_failed = 0
    disruption_occurred = False

    # For MAML adaptation: collect recent transitions before disruption
    adapt_obs_buffer: List[np.ndarray] = []
    adapt_act_buffer: List[int] = []
    adapt_rew_buffer: List[float] = []

    step = 0
    done = False
    while not done:
        # ---- Disruption injection -------------------------------------
        if step == disruption_step and not disruption_occurred:
            n_failed = inject_machine_failures(env, failure_fraction)
            disruption_occurred = True

            # MAML adaptation: adapt policy using pre-disruption data
            if adapt and isinstance(agent, MetaAgent) and len(adapt_obs_buffer) > 0:
                obs_arr = np.array(adapt_obs_buffer)          # (T, obs_size)
                act_arr = np.array(adapt_act_buffer)           # (T,)
                rew_arr = np.array(adapt_rew_buffer)           # (T,)
                # Compute simple discounted returns
                returns = np.zeros_like(rew_arr)
                G = 0.0
                for t in reversed(range(len(rew_arr))):
                    G = rew_arr[t] + 0.99 * G
                    returns[t] = G
                # Reshape to (T, 1, obs_size) for adapt interface
                agent.adapt(
                    observations=obs_arr[:, np.newaxis, :],
                    actions=act_arr[:, np.newaxis],
                    returns=returns[:, np.newaxis],
                    steps=agent.adapt_steps,
                )

        # ---- Action selection ----------------------------------------
        actions, _, _ = agent.select_actions(obs, deterministic=True)
        next_obs, rewards, dones, info = env.step(actions)

        step_reward = sum(rewards)

        # ---- Buffer for MAML -----------------------------------------
        if not disruption_occurred:
            for i, o in enumerate(obs):
                adapt_obs_buffer.append(o)
                adapt_act_buffer.append(actions[i])
                adapt_rew_buffer.append(rewards[i])

        # ---- Record rewards ------------------------------------------
        if step < disruption_step:
            pre_rewards.append(step_reward)
        else:
            post_rewards.append(step_reward)

        obs = next_obs
        step += 1
        done = all(dones)

    # ---- Restore meta weights after episode --------------------------
    if adapt and isinstance(agent, MetaAgent):
        agent.restore_meta_weights()

    # ---- Compute KPIs ------------------------------------------------
    pre_mean = float(np.mean(pre_rewards)) if pre_rewards else 0.0
    post_mean = float(np.mean(post_rewards)) if post_rewards else 0.0
    recovery_drop = (
        (pre_mean - post_mean) / (abs(pre_mean) + 1e-8)
        if abs(pre_mean) > 1e-8
        else 0.0
    )

    # Recovery speed: first post-disruption window where reward ≥ 0.9 * pre
    threshold = 0.9 * pre_mean
    window = 10
    recovery_speed = len(post_rewards)  # default: never recovered
    for i in range(len(post_rewards) - window + 1):
        if float(np.mean(post_rewards[i: i + window])) >= threshold:
            recovery_speed = i
            break

    return {
        "pre_mean_reward": pre_mean,
        "post_mean_reward": post_mean,
        "recovery_drop_pct": recovery_drop * 100,
        "recovery_speed_steps": recovery_speed,
        "n_machines_failed": n_failed,
        "total_jobs_completed": info.get("total_jobs_completed", 0),
        "pre_rewards": pre_rewards,
        "post_rewards": post_rewards,
        "disruption_step": disruption_step,
    }


# --------------------------------------------------------------------------- #
# Multi-episode evaluation                                                      #
# --------------------------------------------------------------------------- #

def evaluate_replanning(
    agent,
    env: ManufacturingEnv,
    disruption_step: int,
    n_episodes: int = 20,
    failure_fraction: float = 0.5,
    adapt: bool = False,
) -> Dict[str, float]:
    """Run *n_episodes* disruption tests and aggregate results."""
    all_drops = []
    all_speeds = []
    all_pre = []
    all_post = []

    for ep in range(n_episodes):
        result = run_replan_episode(
            agent, env, disruption_step, failure_fraction, adapt, seed=ep * 7
        )
        all_drops.append(result["recovery_drop_pct"])
        all_speeds.append(result["recovery_speed_steps"])
        all_pre.append(result["pre_mean_reward"])
        all_post.append(result["post_mean_reward"])

    return {
        "mean_pre_reward": float(np.mean(all_pre)),
        "mean_post_reward": float(np.mean(all_post)),
        "mean_recovery_drop_pct": float(np.mean(all_drops)),
        "std_recovery_drop_pct": float(np.std(all_drops)),
        "mean_recovery_speed_steps": float(np.mean(all_speeds)),
        "std_recovery_speed_steps": float(np.std(all_speeds)),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test agent replanning performance after mid-episode disruptions."
    )
    parser.add_argument("--agent-type", choices=["mappo", "gnn", "meta"], default="mappo")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint.")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "configs", "default.yaml"
    ))
    parser.add_argument("--disruption-step", type=int, default=100)
    parser.add_argument("--failure-fraction", type=float, default=0.5)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="Apply MAML inner-loop adaptation after disruption (MetaAgent only).",
    )
    parser.add_argument("--save-plots", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    env = ManufacturingEnv(config=cfg.get("env", {}))

    # Build agents to test
    agents: Dict[str, Any] = {
        "Random": RandomAgent(env.action_size, env.num_agents),
        "FIFO": FIFOAgent(env.action_size, env.num_agents, env.M),
        "Greedy": GreedyAgent(env.action_size, env.num_agents, env.M),
    }

    # Load DRL agent if provided
    if args.checkpoint:
        device = cfg.get("training", {}).get("device", "cpu")
        mappo_cfg = {**cfg.get("mappo", {}), "device": device}

        if args.agent_type == "mappo":
            drl = MAPPOAgent.from_config(env.obs_size, env.action_size, env.num_agents, mappo_cfg)
        elif args.agent_type == "gnn":
            gnn_cfg = cfg.get("gnn", {})
            drl = GNNPolicyAgent(
                obs_size=env.obs_size, action_size=env.action_size,
                num_agents=env.num_agents, num_machines=env.M,
                num_observable_jobs=env.K,
                d_model=gnn_cfg.get("d_model", 64), device=device,
            )
        else:
            drl = MetaAgent.from_config(env.obs_size, env.action_size, env.num_agents, cfg)

        drl.load(args.checkpoint)
        label = f"{args.agent_type.upper()}"
        if args.adapt and args.agent_type == "meta":
            label += "+MAML"
        agents[label] = drl

    print(f"\nReplanning test — disruption at step {args.disruption_step}")
    print(f"Failure fraction: {args.failure_fraction:.0%}  |  Episodes: {args.n_episodes}")
    print("=" * 70)

    all_results: Dict[str, Dict[str, float]] = {}
    for name, agent in agents.items():
        adapt = args.adapt and isinstance(agent, MetaAgent)
        result = evaluate_replanning(
            agent, env,
            disruption_step=args.disruption_step,
            n_episodes=args.n_episodes,
            failure_fraction=args.failure_fraction,
            adapt=adapt,
        )
        all_results[name] = result
        print(
            f"  {name:<22}"
            f"  pre={result['mean_pre_reward']:>8.2f}"
            f"  post={result['mean_post_reward']:>8.2f}"
            f"  drop={result['mean_recovery_drop_pct']:>6.1f}%"
            f"  recovery={result['mean_recovery_speed_steps']:>5.1f} steps"
        )

    if args.save_plots:
        # Save disruption timeline for the last episode of the first DRL agent
        drl_name = [n for n in agents if n not in {"Random", "FIFO", "Greedy"}]
        if drl_name:
            agent = agents[drl_name[0]]
            ep = run_replan_episode(
                agent, env, args.disruption_step, args.failure_fraction,
                adapt=args.adapt and isinstance(agent, MetaAgent), seed=99
            )
            fig = plot_disruption_timeline(
                reward_before=ep["pre_rewards"],
                reward_after=ep["post_rewards"],
                disruption_step=args.disruption_step,
                title=f"Disruption Timeline — {drl_name[0]}",
            )
            out = os.path.join(args.save_plots, "disruption_timeline.png")
            save_figure(fig, out)
            print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
