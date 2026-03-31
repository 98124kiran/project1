# Problem Formulation — MDP Definition

**Project:** Multiagent Deep Reinforcement Learning for Adaptive Scheduling and Replanning in Dynamic Environments

---

## Overview

We model the adaptive scheduling problem as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** where multiple edge-node agents cooperate to schedule and replan manufacturing jobs under dynamic disruptions.

Formally, the Dec-POMDP is defined as the tuple:

```
M = (S, A, R, P, γ, N)
```

| Symbol | Meaning |
|--------|---------|
| `S`    | State space |
| `A`    | Joint action space |
| `R`    | Reward function |
| `P`    | Transition probability |
| `γ`    | Discount factor (e.g., 0.99) |
| `N`    | Number of agents (edge nodes) |

---

## 1. State Space `S`

Each agent `n` observes a **local state** `o_t^n` (partial observation):

```
o_t^n = [
  machine_status[i]        ∈ {0=idle, 1=busy, 2=failed}   ∀ machine i at node n
  job_queue_length[j]      ∈ ℤ≥0                           ∀ queue j at node n
  job_remaining_time[k]    ∈ ℝ≥0   (processing time left)
  job_deadline[k]          ∈ ℝ≥0   (due date / time units)
  edge_cpu_utilization[n]  ∈ [0, 1]
  edge_memory_util[n]      ∈ [0, 1]
  network_latency[n]       ∈ ℝ≥0   (ms)
]
```

The **global state** `s_t` is the concatenation of all local observations:

```
s_t = [ o_t^1, o_t^2, ..., o_t^N ]
```

> **TODO:** Define exact feature dimensions once the simulation environment is implemented.

---

## 2. Action Space `A`

Each agent `n` selects one action per time step from:

```
a_t^n ∈ {
  ASSIGN(job_k → machine_i)       # Schedule job k on machine i
  MIGRATE(task → node_n')         # Offload task to another edge node n'
  DEFER(job_k, δ)                 # Postpone job k by δ time units
  IDLE                            # Take no action this step
}
```

The **joint action** is:

```
a_t = (a_t^1, a_t^2, ..., a_t^N)
```

> **TODO:** Decide whether actions are discrete or continuous. Start with discrete for DQN/PPO compatibility.

---

## 3. Reward Function `R`

The reward at time step `t` is:

```
r_t = - α · makespan_delta_t
      + β · avg_machine_utilization_t
      - γ · deadline_violations_t
      - δ · avg_edge_latency_t
      + ε · jobs_completed_t
```

| Coefficient | Meaning | Initial Value (to tune) |
|-------------|---------|------------------------|
| `α` | Penalty for increasing makespan | 1.0 |
| `β` | Reward for high machine utilization | 0.5 |
| `γ` | Penalty per deadline violation | 2.0 |
| `δ` | Penalty for edge network latency cost | 0.3 |
| `ε` | Reward per job successfully completed | 1.0 |

> **TODO:** Tune coefficients during experiments. Log individual components separately for analysis.

---

## 4. Transition Model `P`

The environment transitions stochastically according to:

```
P(s_{t+1} | s_t, a_t)
```

**Dynamic disturbances included in the transition model:**

| Disturbance | Distribution |
|-------------|-------------|
| Machine failure | Poisson process with rate λ_fail |
| Machine repair | Exponential distribution with rate λ_repair |
| Urgent job injection | Poisson process with rate λ_job |
| Network latency spikes | Random walk on latency values |

> **TODO:** Calibrate λ values from real or synthetic workload data.

---

## 5. Agent Architecture

Each agent `n` is a neural network with the following structure (preliminary):

```
Input: o_t^n  (local observation vector)
  │
  ▼
[Actor Network]  →  policy π^n(a | o^n)   (for action selection)
[Critic Network] →  value V(s_t)           (for centralized training)
```

**Training paradigm:** Centralized Training, Decentralized Execution (CTDE)
- During training: agents share global state through a centralized critic
- During execution: each agent acts only on its local observation

**Algorithm candidate:** MAPPO (Multi-Agent PPO)

---

## 6. Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| Makespan | Total time to complete all jobs in a batch |
| Machine Utilization | Average % of time machines are processing jobs |
| Deadline Miss Rate | % of jobs that exceed their due date |
| Average Latency | Mean edge node response time |
| Replanning Frequency | How often agents replan after disruptions |
| Convergence Speed | Episodes needed to reach stable policy |

---

## 7. Assumptions & Scope (Objective 1)

- Finite set of `M` machines and `N` edge nodes
- Jobs are preemptable (can be interrupted and resumed)
- Communication between edge nodes is possible but incurs latency cost
- Time is discretized into fixed-length steps `Δt`
- No prior knowledge of future job arrivals (online setting)

---

## Open Questions (resolve in Objective 2)

- [ ] What simulator will be used? (SimPy, custom, OpenAI Gym wrapper?)
- [ ] How many machines `M` and edge nodes `N` for initial experiments?
- [ ] Discrete vs continuous action space?
- [ ] How to encode job graphs — flat vector or GNN?
