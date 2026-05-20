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

Each agent `n` observes a **local state** `o_t^n` (partial observation). In our implementation, the observation vector has **dimension 19** and is structured as follows:

```
o_t^n = [
  machine_status[0..M-1]   ∈ [0, 0.5, 1]          (normalised: IDLE/BUSY/FAILED)
  queue_length             ∈ [0, 1]               (normalised by max_queue_length)
  job_remaining[0..K-1]    ∈ [0, 1]               (normalised by max total processing time)
  job_deadline[0..K-1]     ∈ [0, 1]               (normalised by max deadline bound)
  cpu_utilization          ∈ [0, 1]               (fraction of busy machines)
  memory_utilization       ∈ [0, 1]               (queue fill ratio)
  network_latency_norm     ∈ [0, 1]               (normalised latency)
]
```

**Default Dimensions (see configs/default.yaml):**
- M = 5 machines per node
- K = 5 observable jobs (top-K by priority + EDF)
- N = 3 edge nodes
- obs_size = M + 1 + 2K + 3 = 5 + 1 + 10 + 3 = **19**

The **global state** `s_t` is the concatenation of all local observations:

```
s_t = concat(o_t^1, o_t^2, ..., o_t^N)  ∈ ℝ^(N·obs_size)
```

Used only during centralized critic training; agents never observe full global state during execution.

---

## 2. Action Space `A`

Each agent `n` selects one action per time step from a **discrete action set** of size **|A_n| = 9** (default):

```
a_t^n ∈ {
  0                        # IDLE — take no action this step
  1 … M                    # ASSIGN — assign queue-head to machine (action - 1)
  M+1 … M+N-2              # MIGRATE — send queue-head to peer node
  M+N-1                    # DEFER — re-append queue-head to back of queue
}
```

**Default Action Space (M=5 machines, N=3 nodes):**
- Action 0: IDLE
- Actions 1–5: ASSIGN to machines 0–4
- Actions 6–7: MIGRATE to peers (other 2 nodes)
- Action 8: DEFER

**Total actions: 1 + M + (N-1) + 1 = 1 + 5 + 2 + 1 = 9**

The **joint action** is:

```
a_t = (a_t^1, a_t^2, ..., a_t^N)  ∈ A_1 × A_2 × ... × A_N
```

Action space is **discrete** (not continuous) for compatibility with PPO and practical deployment on edge devices (no floating-point action scaling needed).

---

## 3. Reward Function `R`

At each time step `t`, agent `n` receives a composite reward:

```
r_t^n = - α · makespan_delta_t
        + β · avg_machine_utilization_t^n
        - γ · deadline_violations_t^n
        - δ · norm_latency_t^n
        + ε · jobs_completed_t^n
```

**Default Coefficients (from configs/default.yaml):**

| Coefficient | Meaning | Value | Rationale |
|-------------|---------|-------|-----------|
| `α` | Penalty for increasing makespan | 1.0 | Directly penalizes system-wide slowdown |
| `β` | Reward for machine utilization | 0.5 | Encourages keeping machines busy (efficiency) |
| `γ` | Penalty per deadline violation | 2.0 | Critical KPI; high penalty ensures priority |
| `δ` | Penalty for normalized latency | 0.3 | Lighter penalty; latency important but secondary |
| `ε` | Reward per job completed | 1.0 | Positive reinforcement for throughput |

**Computation Details:**
- `makespan_delta_t`: Increase in total remaining work across all nodes
- `avg_machine_utilization_t^n`: Fraction of busy machines at node n
- `deadline_violations_t^n`: Count of jobs completed after deadline at node n
- `norm_latency_t^n`: Network latency at node n, normalized by `latency_max` = 50 ms
- `jobs_completed_t^n`: Jobs completed at node n in this step

**Implementation Note:** Individual reward components are logged separately during training for post-hoc analysis and ablation studies (see experiments/train.py).

---

## 4. Transition Model `P`

The environment transitions stochastically according to:

```
P(s_{t+1} | s_t, a_t)
```

**Dynamic disturbances included in the transition model:**

| Disturbance | Distribution | Default Rate | Source |
|-------------|-------------|--------------|--------|
| Machine failure | Bernoulli per machine per step (Poisson approx) | λ_fail = 0.01 | DisturbanceGenerator |
| Machine repair | Exponential duration upon failure | E[repair] = 20.0 time units | Machine.fail() |
| Urgent job injection | Bernoulli per node per step (Poisson approx) | λ_urgent = 0.05 | DisturbanceGenerator |
| Network latency walk | Gaussian random walk, clipped [0.5, 50] ms | σ = 0.5 ms | DisturbanceGenerator |
| Normal job arrivals | Poisson process per time step | λ_job = 0.5 jobs/step | WorkloadGenerator |

**Calibration Status:**
- λ values are set in `configs/default.yaml` (confirmed implemented; see src/env/disturbances.py)
- Machine failure + repair dynamics validated in experiments/replan_test.py
- Job arrival stream generated by src/utils/workload_generator.py using Poisson sampling
- Disturbance intensities chosen to create moderately dynamic (but solvable) environment for 500-step episodes

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

## 7. Assumptions & Scope

**Model Assumptions:**
- Finite set of **M = 5 machines per node** and **N = 3 edge nodes** (configurable in default.yaml)
- **Jobs are preemptable** (can be interrupted on machine failure and rescheduled)
- **Communication between edge nodes is possible but incurs simulated latency** (network_latency ∈ [0.5, 50] ms)
- **Time is discretized into fixed-length steps** `Δt = 1.0` minute (configurable)
- **No prior knowledge of future job arrivals** (online setting; only see current queue + K observable jobs)
- **Single-operation visibility**: Agents see top-K jobs by priority (normal jobs: priority=1, urgent: priority=2)

**Implementation Status:**
All assumptions are implemented and validated in the code:
- Topology: see src/env/manufacturing_env.py (num_nodes, num_machines_per_node)
- Preemption: src/env/machine.py (fail() method resets operation)
- Latency: src/env/disturbances.py (_apply_latency_walk)
- Time stepping: src/env/manufacturing_env.py (dt, max_steps)
- Job arrival: src/utils/workload_generator.py (Poisson sampling)
- Partial observability: src/env/manufacturing_env.py (_observe method)

---

## 8. Research Objectives → Measurable KPIs

| Research Objective | Implementation | Measured By | Target Metric |
|-------------------|---------------|-----------|-|
| **Obj 1:** Formalize Dec-POMDP problem | Sections 1–6 above | None (theoretical) | Self-contained definitions |
| **Obj 2:** Build realistic simulator | src/env/ + src/utils/ | experiments/evaluate.py | Agent policies learn non-random behavior |
| **Obj 3:** Develop MARL architecture | agents/ppo_agent.py, agents/gnn_policy.py | experiments/train.py | Convergence to stable policy in 500k+ steps |
| **Obj 4:** Train & benchmark agents | experiments/train.py + experiments/evaluate.py | Completion rate | MAPPO/GNN reward > all baselines |
| **Obj 5:** Evaluate disruption recovery | experiments/replan_test.py | Recovery metrics | Meta-agent pre/post drop < 50% |
| **Ext:** Hybrid edge-cloud | hybrid_compute/edge_inference.py + cloud_trainer.py | Latency simulation | Feasibility demonstration (not real deployment) |

---

## 9. Symbol Summary Table

| Symbol | Definition | Range/Type | Notes |
|--------|-----------|-----------|-------|
| `N` | Number of edge nodes (agents) | 3 (default) | Configurable; tested with N ∈ {2,3,4} |
| `M` | Machines per node | 5 (default) | Configurable; action space scales with M |
| `K` | Observable jobs in local state | 5 (default) | Top-K by priority then EDF ordering |
| `obs_size` | Local observation dimension | 19 (default) | M + 1 + 2K + 3 |
| `action_size` | Discrete actions per agent | 9 (default) | 1 + M + (N-1) + 1 |
| `dt` | Time step duration | 1.0 min | Simulation time unit |
| `T` | Episode length | 500 steps | max_steps in config |
| `γ` | Discount factor | 0.99 | PPO hyperparameter |
| `λ_fail` | Machine failure rate | 0.01 per step | ~1% per machine per step |
| `λ_urgent` | Urgent job rate | 0.05 per step per node | ~5% injection probability |
| `λ_job` | Normal job arrival rate | 0.5 jobs/step (Poisson) | Moderate workload intensity |
| `α, β, γ, δ, ε` | Reward coefficients | 1.0, 0.5, 2.0, 0.3, 1.0 | Tunable; see reward_* in default.yaml |
