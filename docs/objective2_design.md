# Objective 2 — Environment / Simulator Design

**Project:** Multiagent Deep Reinforcement Learning for Adaptive Scheduling and Replanning in Dynamic Environments

---

## Overview

Objective 2 delivers a fully self-contained, Gym-compatible multi-agent simulation environment that models the Dec-POMDP defined in `problem_formulation.md`.  All open questions from Objective 1 are resolved here.

---

## Design Decisions (Resolving Objective 1 Open Questions)

| Question | Decision | Rationale |
|----------|----------|-----------|
| What simulator? | Custom Python (no SimPy) | Zero external dependency; full control over step semantics; trivial Gym wrapping |
| How many machines M and nodes N? | **N = 3 edge nodes**, **M = 5 machines/node** | Small enough to train quickly; large enough to exhibit interesting coordination |
| Discrete vs continuous actions? | **Discrete** | Direct DQN/PPO compatibility; action semantics are naturally categorical |
| How to encode job graphs? | **Flat vector** (top-K observable jobs) | Simple baseline; GNN encoding is deferred to Objective 3 as an optional extension |

---

## Source Code Structure

```
src/
├── __init__.py
├── env/
│   ├── __init__.py
│   ├── job.py                  # Job and Operation data structures
│   ├── machine.py              # Machine model (IDLE / BUSY / FAILED lifecycle)
│   ├── edge_node.py            # Edge node with priority job queue
│   ├── disturbances.py         # Stochastic disturbance generator
│   └── manufacturing_env.py    # Main multi-agent Gym-compatible environment
└── utils/
    ├── __init__.py
    └── workload_generator.py   # Poisson job-stream generator
```

---

## Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_nodes` N | 3 | Number of edge nodes / agents |
| `num_machines_per_node` M | 5 | Machines per node |
| `num_machine_types` | 3 | Distinct machine types |
| `max_queue_length` | 20 | Maximum jobs buffered per node |
| `num_observable_jobs` K | 5 | Jobs visible per observation |
| `dt` | 1.0 | Time step duration (minutes) |
| `max_steps` | 500 | Episode length |
| `lambda_fail` | 0.01 | Per-machine failure probability/step |
| `mean_repair_time` | 20.0 | Mean repair time (time units) |
| `lambda_urgent` | 0.05 | Urgent job injection probability/step/node |
| `latency_sigma` | 0.5 | Latency random-walk std (ms) |
| `lambda_job` | 0.5 | Mean normal job arrivals per step |
| `min/max_processing_time` | 5–30 | Operation processing time range |
| `min/max_deadline_slack` | 20–100 | Slack added to processing time for deadline |
| `reward_alpha` α | 1.0 | Makespan growth penalty |
| `reward_beta` β | 0.5 | Machine utilisation reward |
| `reward_gamma` γ | 2.0 | Deadline violation penalty |
| `reward_delta` δ | 0.3 | Network latency penalty |
| `reward_epsilon` ε | 1.0 | Job completion reward |

---

## Observation Space

Each agent `n` receives a local observation vector of size **`M + 1 + 2K + 3`** = **19** (with defaults):

| Slice | Size | Description |
|-------|------|-------------|
| `machine_statuses` | M = 5 | Status of each machine: 0=IDLE, 1=BUSY, 2=FAILED (normalised ÷ 2) |
| `queue_length` | 1 | Number of jobs in queue (normalised by max_queue_length) |
| `job_i_remaining` | K = 5 | Remaining processing time for top-K jobs (normalised) |
| `job_i_deadline` | K = 5 | Absolute deadline for top-K jobs (normalised) |
| `cpu_utilization` | 1 | Fraction of busy machines [0, 1] |
| `memory_utilization` | 1 | Queue fill ratio [0, 1] |
| `network_latency_norm` | 1 | Normalised edge latency [0, 1] |

---

## Action Space

Each agent has **`M + N + 1 = 9`** discrete actions (with defaults M=5, N=3):

| Action index | Semantics |
|-------------|-----------|
| 0 | **IDLE** — no action this step |
| 1 … M | **ASSIGN** queue-head to machine `(action - 1)` |
| M+1 … M+N-1 | **MIGRATE** queue-head to peer node `peers[action - M - 1]` |
| M+N | **DEFER** queue-head (re-append at back of queue) |

---

## Reward Function

At each time step *t*, agent *n* receives:

```
r_t^n = - α · makespan_delta_t
        + β · avg_machine_utilisation_t^n
        - γ · deadline_violations_t^n
        - δ · norm_latency_t^n
        + ε · jobs_completed_t^n
```

`makespan_delta_t` is computed as the change in total remaining work across the entire system (global signal), while the other terms are node-local.

---

## Disturbance Model

| Event | Mechanism | Rate / Distribution |
|-------|-----------|-------------------|
| Machine failure | Bernoulli per machine per step | P = `lambda_fail` = 0.01 |
| Machine repair | Repair time sampled per failure | Exp(mean = 20 time units) |
| Urgent job arrival | Bernoulli per node per step | P = `lambda_urgent` = 0.05 |
| Network latency | Gaussian random walk | Δ ~ N(0, 0.5 ms), clipped [0.5, 50] ms |

---

## Quick-Start

```python
from src.env.manufacturing_env import ManufacturingEnv

env = ManufacturingEnv()                          # default config
obs = env.reset(seed=42)                          # list of N arrays, each shape (19,)

for step in range(10):
    actions = [env.action_spaces[i].sample() for i in range(env.num_agents)]
    obs, rewards, dones, info = env.step(actions)
    print(env.render())
    if all(dones):
        break
```

---

## Next Steps → Objective 3

- Implement MAPPO Actor-Critic networks (see `docs/thesis_structure.md` Chapter 5)
- Feed `obs` into shared-parameter Actor; feed concatenated `obs` into Centralised Critic
- Optionally upgrade job encoding to a GNN for richer feature extraction
