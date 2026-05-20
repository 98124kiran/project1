# Literature Review

**Project:** Multiagent Deep Reinforcement Learning for Adaptive Scheduling and Replanning in Dynamic Environments

---

## How to Use This File

For each paper below, fill in the fields under its heading. A completed Gap Analysis section at the bottom ties everything together into your research contribution.

---

## Paper 1: Deep Reinforcement Learning for Job-Shop Scheduling Problems

- **Authors & Year:** Zhang et al., 2020
- **Problem Solved:** Job-shop scheduling problem (JSSP) with static job sets, makespan minimization
- **Method Used:** Deep Q-Network (DQN) with feature engineering on job and machine state representations
- **Key Results:** Achieves competitive makespan on standard benchmarks (Lawson 10×10, 15×15); demonstrates faster convergence than pure heuristics
- **Relevance to My Project:** Foundational work on applying DRL to manufacturing scheduling; focuses on single-agent centralized control
- **Gap / Limitation:** No multi-agent decomposition, no dynamic job arrivals, no real-time replanning after disruptions, no edge computing constraints

---

## Paper 2: Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning (L2D)

- **Authors & Year:** Zhang et al., 2020
- **Problem Solved:** Job shop scheduling with explicit job-machine interaction graph encoding
- **Method Used:** Graph Neural Networks (GNN) + PPO; node embeddings for jobs and machines with attention-based aggregation
- **Key Results:** Strong generalization to unseen problem sizes; outperforms supervised learning and heuristics on multiple benchmarks
- **Relevance to My Project:** Demonstrates GNN effectiveness for scheduling; provides architectural precedent for my GNN policy variant
- **Gap / Limitation:** Single-agent, no multi-node coordination, no edge computing or bandwidth constraints, static problem instances

---

## Paper 3: Smart Manufacturing Scheduling with Edge Computing Using Multiclass Deep Q-Network

- **Authors & Year:** Shiue et al., 2018
- **Problem Solved:** Task offloading and scheduling across multiple edge nodes in smart manufacturing
- **Method Used:** Multiclass DQN where each node maintains local Q-value functions for resource allocation
- **Key Results:** Reduced average task latency and improved resource utilization vs. greedy baselines in simulated edge network
- **Relevance to My Project:** Directly relevant; combines edge computing + DRL for manufacturing; demonstrates importance of edge-aware scheduling
- **Gap / Limitation:** Single DQN per node (no true multi-agent coordination), no meta-learning or replanning mechanisms, limited treatment of machine failures

---

## Paper 4: A Deep Reinforcement Learning Approach for Real-Time Online Shop Scheduling

- **Authors & Year:** Han & Yang, 2021
- **Problem Solved:** Dynamic job arrivals and machine breakdowns in online scheduling with real-time decision constraints
- **Method Used:** Hybrid actor-critic architecture with state encoding capturing current queue depth, machine availability, and failure history
- **Key Results:** Demonstrates adaptive behavior to disruptions; competitive average job completion time with low latency variance
- **Relevance to My Project:** Directly addresses dynamic disruptions and replanning; motivates need for online adaptive policies
- **Gap / Limitation:** Single-agent centralized approach, no multi-node cooperation, no explicit meta-learning mechanism for rapid disruption recovery

---

## Paper 5: Dynamic Job-Shop Scheduling Using Deep Reinforcement Learning

- **Authors & Year:** Park et al., 2021
- **Problem Solved:** Flexible job-shop scheduling (FJSP) with AND/OR precedence constraints and machine alternatives
- **Method Used:** Deep actor-critic network with LSTM for capturing temporal job dependencies and machine state history
- **Key Results:** Competitive makespan and reduced tardiness on standard benchmark sets; handles complex job precedence constraints
- **Relevance to My Project:** Extends to multi-operation jobs (relevant to my Job/Operation model); LSTM recurrence provides temporal context
- **Gap / Limitation:** No multi-agent distribution, no explicit handling of edge computing latency, no meta-learning for rapid adaptation

---

## Paper 6: Deep Reinforcement Learning for Mobile Edge Computing

- **Authors & Year:** Huang et al., 2019
- **Problem Solved:** Computation offloading decisions in mobile edge computing with network bandwidth and latency constraints
- **Method Used:** Deep Q-learning with state encoding capturing wireless channel quality, task queue depth, and edge server load
- **Key Results:** Reduces task latency and energy consumption compared to local-only or cloud-only execution; adapts to time-varying channel conditions
- **Relevance to My Project:** Foundational work on edge-aware DRL; directly motivates hybrid edge-cloud simulator in my framework
- **Gap / Limitation:** Task offloading only (no job scheduling/ordering), single-node optimization, no multi-agent coordination or manufacturing-specific constraints

---

## Paper 7: Multi-Agent Deep Reinforcement Learning for Edge Computing

- **Authors & Year:** Chen et al., 2021
- **Problem Solved:** Cooperative task offloading across multiple edge nodes using distributed multi-agent DRL
- **Method Used:** Multi-agent Deep Deterministic Policy Gradient (MADDPG) with centralized critic and decentralized execution for task placement
- **Key Results:** Better network congestion management and lower overall latency vs. single-agent or greedy approaches across distributed edge testbed
- **Relevance to My Project:** Directly relevant; demonstrates MARL effectiveness for edge computing; validates CTDE paradigm (centralized training, decentralized execution)
- **Gap / Limitation:** Focus on offloading decisions only, no job scheduling/ordering on edge nodes, limited treatment of hardware failures and disruptions, no explicit meta-learning

---

## Paper 8: Proximal Policy Optimization Algorithms

- **Authors & Year:** Schulman et al., 2017 (OpenAI)
- **Problem Solved:** Stable, sample-efficient on-policy policy gradient training with monotonic improvement guarantees
- **Method Used:** PPO with clipped surrogate loss objective; eliminates manual learning rate scheduling; supports continuous and discrete action spaces
- **Key Results:** Achieves strong performance on diverse continuous control benchmarks (Atari, locomotion); simpler and more stable than TRPO
- **Relevance to My Project:** Core training algorithm; PPO provides stable baseline for both single-agent and multi-agent (MAPPO) variants in my framework
- **Gap / Limitation:** General-purpose algorithm; not manufacturing or scheduling-specific; requires careful reward design and hyperparameter tuning for domain

---

## Gap Analysis

**Research Gap:** Existing works in DRL-based scheduling either focus on:
1. **Static single-agent scheduling** (Zhang 2020, Park 2021): Assume fixed job sets, no dynamic arrivals, centralized decision-making
2. **Edge computing offloading in isolation** (Huang 2019, Chen 2021): Optimize task placement but ignore scheduling/ordering on edge machines
3. **Real-time reactive scheduling** (Han & Yang 2021): Adapt to disruptions but remain single-agent without distributed coordination
4. **Manufacturing scheduling without edge constraints** (Shiue 2018): Address manufacturing domain but without sophisticated multi-agent protocols or meta-learning

**This Project's Contribution:** Addresses the combined gap by proposing a **multi-agent cooperative framework** that:
- **Combines multi-agent DRL (MAPPO)** with **edge-aware adaptive scheduling** to leverage distributed intelligence
- **Handles real-time dynamic disruptions** (machine failures, urgent job injections) with explicit **replanning via MAML meta-learning**
- **Provides multiple architectural variants** (flat MAPPO, GNN-enhanced policy) to trade off expressiveness vs. scalability
- **Integrates hybrid edge-cloud simulation** to validate practical deployment feasibility (latency, bandwidth, federated learning)
- **Evaluates empirically** against classical baselines (FIFO/SPT/EDD) and single-agent control under realistic manufacturing workloads

**Novelty**: No prior work combines all four elements—(1) multi-agent cooperative scheduling, (2) edge computing resource constraints, (3) real-time disruption recovery, (4) meta-learning adaptation—in a unified simulation and evaluation framework.

---

## Reading Checklist & Status

| Paper | Authors | Status | Summary |
|-------|---------|--------|---------|
| 1 | Zhang et al. 2020 (DRL for JSP) | ✅ Documented | Single-agent DQN; strong foundation but lacks multi-agent & disruption handling |
| 2 | Zhang et al. 2020 (L2D) | ✅ Documented | GNN + PPO; inspired GNN policy variant; no edge constraints or replanning |
| 3 | Shiue et al. 2018 (Edge + DQN) | ✅ Documented | Multiclass DQN for edge; lacks true coordination & failure recovery |
| 4 | Han & Yang 2021 (Real-time) | ✅ Documented | Dynamic arrivals & breakdowns; single-agent, no meta-learning |
| 5 | Park et al. 2021 (Dynamic FJSP) | ✅ Documented | Multi-operation jobs with LSTM; no multi-agent, no edge awareness |
| 6 | Huang et al. 2019 (MEC + DRL) | ✅ Documented | Foundational edge computing work; offloading focus, not scheduling |
| 7 | Chen et al. 2021 (MARL + Edge) | ✅ Documented | MADDPG for edge coordination; limited manufacturing focus, no replanning |
| 8 | Schulman et al. 2017 (PPO) | ✅ Documented | Core algorithm; stable but general-purpose, requires domain tuning |
