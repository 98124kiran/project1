# MADRL Edge Computing Scheduler for Smart Manufacturing

**Multi-Agent Deep Reinforcement Learning for Adaptive Scheduling and Replanning in Edge Computing at Dynamic Environments**

---

## Overview

This project implements a **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** system that learns to adaptively schedule and replan manufacturing tasks across a fleet of edge computing nodes in a smart factory setting.

Each edge node acts as an independent RL agent that:
- Observes the full task queue and the state of all peer nodes (centralised information during training)
- Decides which pending task to execute locally at each time step (decentralised execution)
- Continuously adapts its policy as conditions change: node failures, fluctuating background loads, and new task arrivals

---

## Architecture

```
project1/
├── src/
│   ├── environment/
│   │   ├── edge_env.py       # Multi-agent Gym-style environment
│   │   ├── edge_node.py      # Edge node model (capacity, failure, recovery)
│   │   └── task.py           # Manufacturing task definition & features
│   ├── agents/
│   │   ├── maddpg.py         # MADDPG controller + per-agent logic
│   │   └── replay_buffer.py  # Centralised experience replay
│   ├── models/
│   │   └── networks.py       # Actor (decentralised) & Critic (centralised) NNs
│   ├── training/
│   │   ├── trainer.py        # Training loop
│   │   └── evaluator.py      # Evaluation + baseline comparison
│   └── utils/
│       ├── config.py         # Dataclass-based configuration
│       ├── logger.py         # File + console logging
│       └── metrics.py        # Rolling metrics & episode tracking
├── tests/                    # 32 pytest unit tests
├── main.py                   # CLI entry point
├── config.yaml               # Default hyperparameters
└── requirements.txt
```

---

## Key Features

| Feature | Details |
|---|---|
| **Algorithm** | MADDPG – centralised training, decentralised execution |
| **Dynamic environment** | Node failures/recovery, Poisson task arrivals, fluctuating background load |
| **Task types** | Welding control, vision inspection, assembly planning, anomaly detection |
| **Reward shaping** | Priority-weighted completion bonus, deadline penalty, utilisation reward |
| **Adaptive replanning** | Policies update online every `update_every` steps |
| **Exploration** | Gaussian noise in logit space with exponential decay |
| **Baselines** | Random and Greedy (priority-based) policies for comparison |

---

## Environment Details

### Observation Space (per agent, dim = 230)
- Own node state: CPU/memory utilisation, bandwidth, failure flag, queue length (dim 5)
- Global task queue (padded to 20 tasks × 10 features each = 200)
- All-node summary (5 nodes × 5 features = 25)

### Action Space (per agent, dim = 21)
- Actions 0–19: schedule the task at queue position *i* onto this node
- Action 20: idle (do nothing this step)

### Reward Function
```
R = Σ priority × W_complete   (for each finished task)
  + Σ priority × W_miss       (for each deadline violation)
  + W_util                     (if CPU utilisation in healthy range 40–80%)
  + W_drop                     (for queue overflow or node-failure drops)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0.

---

## Quick Start

### Train
```bash
python main.py train                        # default config
python main.py train --config config.yaml --episodes 5000 --seed 123
```

### Evaluate (compare MADDPG vs Greedy vs Random)
```bash
python main.py eval --checkpoint checkpoints/best --episodes 30
```

### Run Tests
```bash
python -m pytest tests/ -v
```

---

## Configuration

All hyperparameters live in `config.yaml` and can be overridden at the CLI.

| Parameter | Default | Description |
|---|---|---|
| `env.num_edge_nodes` | 5 | Number of edge nodes (= number of agents) |
| `env.max_queue_size` | 20 | Global pending task queue capacity |
| `env.node_failure_prob` | 0.02 | Per-step node failure probability |
| `agent.actor_lr` | 1e-4 | Actor learning rate |
| `agent.critic_lr` | 3e-4 | Centralised critic learning rate |
| `agent.gamma` | 0.95 | Discount factor |
| `agent.warmup_steps` | 1000 | Random exploration before training starts |
| `training.num_episodes` | 3000 | Total training episodes |

---

## Algorithm: MADDPG

```
For each episode:
  Reset environment → obs_n (one observation per agent)
  For each time step t:
    1. Each agent i selects action a_i = π_i(obs_i) + ε  (actor + noise)
    2. Environment executes {a_i}, returns {obs_i'}, {r_i}, done
    3. Store (obs_n, a_n, r_n, obs_n', done) in shared replay buffer
    4. Every K steps, sample mini-batch and update:
       - Critic loss:  MSE( Q_i(obs_n, a_n),  r_i + γ Q̂_i(obs_n', â_n') )
       - Actor loss:  -E[ Q_i(obs_n, π̃_n) ]   (only π_i is differentiated)
       - Soft-update target networks: θ̂ ← τθ + (1-τ)θ̂
```

---

## Performance Metrics

- **Success rate**: fraction of tasks completed before deadline
- **Deadline miss rate**: fraction of tasks that expired in queue or during execution
- **Average CPU utilisation**: across all non-failed nodes
- **Episode reward**: cumulative shaped reward per episode
