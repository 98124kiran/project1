# Thesis Structure

**Title:** Multiagent Deep Reinforcement Learning for Adaptive Scheduling and Replanning in Dynamic Manufacturing Environments

---

## Abstract

A concise summary (~300 words) covering:
- The industrial motivation (smart manufacturing, edge computing)
- The core problem (real-time adaptive scheduling under disruptions)
- The proposed approach (MARL with Dec-POMDP, CTDE, MAPPO)
- Key results and contributions

---

## Chapter 1 — Introduction

### 1.1 Motivation
- Industry 4.0 and the rise of smart manufacturing
- Dynamic disruptions: machine failures, urgent order injections, resource variability
- Limitations of classical scheduling (OR methods, heuristics) under uncertainty
- Edge computing as the deployment substrate

### 1.2 Problem Statement
- Gap between static optimal scheduling and real-time adaptive replanning
- Need for decentralised, low-latency decision-making at the edge
- Multi-agent coordination under partial observability

### 1.3 Research Objectives
1. Formalise the adaptive scheduling problem as a Dec-POMDP
2. Design and implement a realistic simulator (edge nodes, machines, disturbances)
3. Develop a MARL architecture (MAPPO with CTDE) for cooperative scheduling
4. Train and benchmark against classical and single-agent baselines
5. Evaluate on scheduling KPIs: makespan, utilisation, deadline miss rate, latency

### 1.4 Contributions
- A Dec-POMDP formulation of distributed adaptive manufacturing scheduling
- An open-source Gym-compatible multi-agent simulator
- A MAPPO-based cooperative scheduling framework with centralized critic
- Comprehensive experimental evaluation with ablation studies

### 1.5 Thesis Organisation
- Summary of each chapter with forward references

---

## Chapter 2 — Background and Related Work

### 2.1 Reinforcement Learning Fundamentals
- MDPs: state, action, reward, transition, policy
- Bellman equations and value functions
- Policy gradient theorem

### 2.2 Deep Reinforcement Learning
- Deep Q-Network (DQN) and extensions (Double DQN, Dueling DQN)
- Actor-Critic methods (A3C, A2C)
- Proximal Policy Optimisation (PPO): clipped surrogate objective, advantages

### 2.3 Multi-Agent Reinforcement Learning
- Cooperative, competitive, and mixed settings
- Dec-POMDP formulation and tractability challenges
- Centralised Training with Decentralised Execution (CTDE)
- Key algorithms: QMIX, MADDPG, MAPPO, HAPPO

### 2.4 Job-Shop Scheduling
- Classical JSP and Flexible JSP (FJSP) formulations
- Complexity (NP-hardness) and dispatching rules (SPT, EDD, FIFO)
- DRL-based dispatching: L2D, FJSP-DRL, graph-based approaches

### 2.5 Edge Computing and Task Offloading
- Mobile Edge Computing (MEC) architecture
- Latency-aware offloading formulations
- DRL for MEC task scheduling

### 2.6 Related Work Summary and Research Gap
- Table comparing existing works on: multi-agent, edge-aware, dynamic disruptions, replanning
- Identification of the combined gap addressed by this thesis

---

## Chapter 3 — Problem Formulation

### 3.1 System Model
- Smart manufacturing floor: M machines, N edge nodes
- Job arrival model (Poisson process)
- Communication topology between edge nodes

### 3.2 Dec-POMDP Formulation
Formal definition: `M = (S, A, R, P, γ, N)`

### 3.3 State Space
- Global state `s_t` and local observation `o_t^n` per agent
- Feature vector components: machine statuses, queue state, resource utilisation, latency

### 3.4 Action Space
- Discrete action set: ASSIGN, MIGRATE, DEFER, IDLE
- Action space size analysis: `|A_n| = M + N + 1`

### 3.5 Reward Function
- Composite reward: makespan delta, utilisation, deadline violations, latency, completions
- Reward coefficient rationale and tuning strategy

### 3.6 Transition Model and Dynamic Disturbances
- Machine failure: Poisson process (rate λ_fail)
- Machine repair: Exponential distribution (rate λ_repair)
- Urgent job injection: Poisson process (rate λ_urgent)
- Network latency: Gaussian random walk

### 3.7 Assumptions and Scope

---

## Chapter 4 — Environment and Simulator Design  *(Objective 2)*

### 4.1 Design Decisions
- Simulator choice: custom Python environment (no external simulator dependency)
- Gymnasium-compatible interface for RL framework interoperability
- Discrete action space (suitable for PPO and DQN)
- Flat vector observations (GNN encoding deferred to Objective 3)

### 4.2 Architecture Overview
```
ManufacturingEnv
├── EdgeNode × N
│   ├── Machine × M
│   └── Job Queue (priority + EDF ordering)
├── WorkloadGenerator  (Poisson job arrivals)
└── DisturbanceGenerator (failures, urgent jobs, latency walk)
```

### 4.3 Job and Operation Model
- Multi-operation jobs with machine-type constraints
- Preemption and re-queuing on machine failure
- Deadline, priority, and slack tracking

### 4.4 Machine Model
- Three statuses: IDLE, BUSY, FAILED
- Exponential repair-time sampling
- Per-machine utilisation tracking

### 4.5 Edge Node Model
- Priority queue with EDF tie-breaking
- Greedy auto-assignment of idle machines
- CPU, memory, and latency metrics

### 4.6 Disturbance Generator
- Bernoulli-approximated Poisson failures per step
- Gaussian random-walk latency model
- Urgent job injection per node

### 4.7 Workload Generator
- Configurable Poisson arrival rate λ_job
- Parameterised processing times and deadline slack
- Urgent job generation (tight deadlines, priority = 2)

### 4.8 Observation and Action Interface
- Observation vector breakdown (size: M + 1 + 2K + 3 per agent)
- Action decoding logic (IDLE, ASSIGN, MIGRATE, DEFER)
- Reward computation per step per agent

### 4.9 Validation and Sanity Checks
- Random-policy rollout statistics
- Queue saturation analysis
- Failure injection verification

---

## Chapter 5 — Multi-Agent DRL Architecture  *(Objective 3)*

### 5.1 Algorithm Selection: MAPPO
- Rationale over QMIX and MADDPG for continuous shared reward
- PPO clipping for training stability
- Shared vs independent networks trade-off

### 5.2 Actor Network
- Input: local observation `o_t^n`
- Architecture: MLP (e.g., 128-128) → softmax over action logits
- Optional: attention mechanism over observable jobs

### 5.3 Critic Network (Centralised)
- Input: global state `s_t` (concatenated observations)
- Architecture: MLP → scalar value estimate `V(s_t)`
- Role: reduces variance during training only

### 5.4 Training Paradigm: CTDE
- Training loop: collect trajectories → compute GAE advantages → PPO update
- Decentralised execution: each agent uses only its actor and local observation

### 5.5 Hyperparameters
- Learning rates, clip ε, GAE λ, discount γ, entropy coefficient
- Batch size and rollout length

### 5.6 Communication (Optional Extension)
- Emergent communication via shared critic
- Explicit message passing (future work)

---

## Chapter 6 — Training and Experiments  *(Objective 4)*

### 6.1 Experimental Setup
- Hardware: GPU training environment
- Environment configuration: N=3 nodes, M=5 machines, λ_fail=0.01, …
- Training budget: X million environment steps

### 6.2 Baselines
| Baseline | Description |
|----------|-------------|
| Random | Uniformly random action selection |
| FIFO | First-in-first-out queue dispatching |
| EDD | Earliest Due Date dispatching rule |
| SPT | Shortest Processing Time heuristic |
| Single-agent PPO | Central PPO with full state access |

### 6.3 Hyperparameter Tuning
- Grid search / Bayesian optimisation over key hyperparameters
- Training curves and convergence analysis

### 6.4 Reward Coefficient Sensitivity
- Ablation on α, β, γ, δ, ε values
- Effect on individual KPI components

### 6.5 Scalability Experiments
- Varying N (nodes) and M (machines)
- Effect of increasing λ_fail and λ_job

---

## Chapter 7 — Evaluation and Results  *(Objective 5)*

### 7.1 KPI Definitions
| Metric | Formula |
|--------|---------|
| Makespan | Total completion time for a job batch |
| Machine Utilisation | Mean fraction of busy machine-time |
| Deadline Miss Rate | % jobs completed after deadline |
| Average Latency | Mean edge node network latency |
| Replanning Frequency | Mean actions ≠ IDLE per episode |
| Convergence Speed | Steps to reach 95 % of peak reward |

### 7.2 Main Results
- Comparison table: MAPPO vs all baselines across all KPIs
- Training reward curves (mean ± std across seeds)

### 7.3 Ablation Studies
- No disturbances vs. with disturbances
- Single-agent vs. multi-agent critic
- Observation feature ablation (remove latency / queue / machine status)

### 7.4 Qualitative Analysis
- Case study: agent behaviour during a machine failure event
- Migration patterns between edge nodes under high load
- Deadline-aware deferral behaviour for low-priority jobs

### 7.5 Discussion
- Strengths and limitations of the proposed approach
- Failure modes and edge cases
- Comparison with non-learning methods on fairness grounds

---

## Chapter 8 — Conclusion and Future Work

### 8.1 Summary of Contributions

### 8.2 Limitations
- Simulation-to-reality gap
- Scalability beyond N=10 nodes
- Communication overhead not yet modelled

### 8.3 Future Work
- GNN-based job encoding for richer state representation (originally flagged in Objective 2)
- Hierarchical MARL for multi-level scheduling
- Transfer learning to real manufacturing datasets
- Explicit inter-agent communication via message passing
- Sim-to-real transfer with a digital twin

---

## Appendices

### Appendix A — Hyperparameter Tables
Full tables of all hyperparameters used in each experiment.

### Appendix B — Additional Results and Plots
Training curves, KPI breakdowns by node, latency histograms.

### Appendix C — Environment API Reference
Auto-generated or hand-written documentation for `ManufacturingEnv`,
`EdgeNode`, `Machine`, `Job`, `WorkloadGenerator`, and `DisturbanceGenerator`.

### Appendix D — Bibliography
All cited works in APA / IEEE format (see `references/references.bib`).

---

*Document version: Objective 2 complete. Chapters 1–4 and 8 are fully
scoped. Chapters 5–7 will be populated during Objectives 3–5.*
