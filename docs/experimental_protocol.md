# Experimental Protocol & Reproducibility

**Project:** Multiagent Deep Reinforcement Learning for Adaptive Scheduling and Replanning in Dynamic Environments

---

## 1. Environment Configuration (Locked)

All experiments use the canonical configuration from `configs/default.yaml` unless explicitly stated otherwise:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Topology** | | |
| num_nodes | 3 | Three edge nodes; balanced for multi-agent coordination |
| num_machines_per_node | 5 | 15 machines total; moderate problem size |
| num_machine_types | 3 | Job routing constraints (machine type affinity) |
| max_queue_length | 20 | Prevents unbounded queue growth |
| **Observation** | | |
| num_observable_jobs | 5 | Top-5 jobs visible per node (by priority + EDF) |
| obs_size | 19 | Derived from M + 1 + 2K + 3 |
| **Timing** | | |
| dt | 1.0 min | Fixed time step duration |
| max_steps | 500 | Episode length (500 min ≈ 8.3 hours per episode) |
| **Disturbances** | | |
| lambda_fail | 0.01 | ~1% per-machine failure rate per step |
| mean_repair_time | 20.0 min | Exponential repair duration |
| lambda_urgent | 0.05 | ~5% urgent job injection per node per step |
| latency_sigma | 0.5 ms | Gaussian random-walk std for network latency |
| **Workload** | | |
| lambda_job | 0.5 jobs/step | Poisson job arrival rate (moderate load) |
| min_processing_time | 5.0 min | Minimum operation duration |
| max_processing_time | 30.0 min | Maximum operation duration |
| min_deadline_slack | 20.0 min | Minimum slack (deadline = makespan + slack) |
| max_deadline_slack | 100.0 min | Maximum slack (allows loose deadlines) |
| min_ops | 1 | Minimum operations per job |
| max_ops | 3 | Maximum operations per job (routing complexity) |
| **Reward Coefficients** | | |
| reward_alpha | 1.0 | Makespan growth penalty |
| reward_beta | 0.5 | Machine utilization reward |
| reward_gamma | 2.0 | Deadline violation penalty (high) |
| reward_delta | 0.3 | Network latency penalty (moderate) |
| reward_epsilon | 1.0 | Job completion reward |

---

## 2. Agent Configurations

### 2.1 Baseline Agents (Non-learning)

All baselines are deterministic (action: deterministic=True) and run for comparison only:

| Baseline | Implementation | Notes |
|----------|----------------|-------|
| **Random** | agents/baselines.py::RandomAgent | Uniformly random action per step; sanity check baseline |
| **FIFO** | agents/baselines.py::FIFOAgent | Assigns queue-head to first idle machine; classic dispatching |
| **SPT** | agents/baselines.py::SPTAgent | Shortest Processing Time; prioritizes fast jobs |
| **EDD** | agents/baselines.py::EDDAgent | Earliest Due Date; prioritizes tight deadlines |
| **Greedy** | agents/baselines.py::GreedyAgent | Assigns any job to any idle machine; randomized tie-breaking |

### 2.2 Learning Agents (DRL)

| Agent | Architecture | Configuration | Source |
|-------|--------------|---------------|--------|
| **MAPPO** | Shared MLP actor + centralized critic | See mappo config below | agents/ppo_agent.py |
| **GNN** | GNN actor + centralized critic | See gnn config below | agents/gnn_policy.py |
| **Meta** | MAPPO + FOMAML inner-loop adaptation | See meta config below | agents/meta_agent.py |

### 2.3 MAPPO Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| hidden_size (actor) | 128 | Balanced capacity for observation encoding |
| critic_hidden_size | 256 | Larger capacity for global state value estimation |
| lr_actor | 3e-4 | Standard PPO learning rate |
| lr_critic | 1e-3 | 3.33× actor LR (critic benefits from higher learning rate) |
| gamma | 0.99 | Standard RL discount; emphasizes long-term rewards |
| gae_lambda | 0.95 | GAE parameter for variance-bias trade-off |
| clip_eps | 0.2 | PPO clipping; typical for continuous action domains |
| entropy_coef | 0.01 | Moderate exploration bonus |
| value_coef | 0.5 | Standard value loss weight |
| max_grad_norm | 0.5 | Gradient clipping; prevents large updates |
| n_epochs | 10 | Update epochs per rollout |
| batch_size | 64 | Mini-batch size for gradient updates |
| rollout_steps | 2048 | Trajectory length per update (4 episodes × 500 steps ≈ 2048) |
| total_timesteps | 500,000 | Total training budget (~1000 episodes across all 3 agents) |

### 2.4 GNN Policy Hyperparameters

Inherits all MAPPO parameters above, plus:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 64 | Embedding dimension for graph nodes |
| n_heads | 4 | Multi-head self-attention; 64 / 4 = 16-dim per head |
| n_layers | 2 | Transformer blocks; shallow but adequate for small graphs |
| dropout | 0.1 | Regularization in attention + FFN layers |
| hidden_size (action head) | 128 | MLP head to convert flattened node embeddings → action logits |

**Graph Structure:**
- Machine nodes: M × 1-dim (normalised status)
- Job nodes: K × 2-dim (remaining_time, deadline)
- Context node: 1 × 4-dim (queue_len, cpu, mem, latency)
- Total nodes: M + K + 1 = 5 + 5 + 1 = 11 per agent

### 2.5 Meta-Learning (FOMAML) Hyperparameters

Inherits MAPPO parameters, plus:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| inner_lr | 0.01 | Inner-loop (adaptation) learning rate; 33× base LR for fast adaptation |
| inner_steps | 5 | Gradient steps on disruption data during meta-training |
| meta_lr | 3e-4 | Outer-loop (meta-weight) learning rate; same as MAPPO |
| adapt_steps | 3 | Gradient steps for test-time adaptation after disruption |

---

## 3. Training Protocol

### 3.1 Experiment Matrix

```
┌─────────────────────────────────────────────┐
│ Train N Agents × M Agent Types × S Seeds    │
├─────────────────────────────────────────────┤
│ N = 3 agents (fixed)                        │
│ M = 3 types: MAPPO, GNN, Meta               │
│ S = 3 seeds: 42, 123, 456                   │
│ Total runs: 3 × 3 × 3 = 27 trainings        │
│ Total env steps: 27 × 500k = 13.5M          │
└─────────────────────────────────────────────┘
```

### 3.2 Training Script

```bash
# Train MAPPO with seed 42
python -m experiments.train \
  --agent mappo \
  --total-steps 500000 \
  --config configs/default.yaml \
  --device cpu

# Train GNN with seed 42
python -m experiments.train \
  --agent gnn \
  --total-steps 500000 \
  --config configs/default.yaml \
  --device cpu

# Train Meta with seed 42
python -m experiments.train \
  --agent meta \
  --total-steps 500000 \
  --config configs/default.yaml \
  --device cpu
```

### 3.3 Logging & Checkpointing

| Artifact | Frequency | Purpose |
|----------|-----------|---------|
| Checkpoint | Every 50k steps | Resume training; final checkpoint at 500k |
| Reward curve | Every 5k steps | Monitor convergence; detect divergence early |
| Metrics dict | Per update | Log actor_loss, critic_loss, entropy, approx_kl for diagnostics |

**Output directory:** `checkpoints/<agent_type>/`

---

## 4. Evaluation Protocol

### 4.1 Standard Evaluation (All Agents)

```bash
python -m experiments.evaluate \
  --agent-type mappo \
  --checkpoint checkpoints/mappo/final.pt \
  --n-episodes 50 \
  --baselines-only false \
  --save-plots results/
```

**Metrics Collected:**
- Mean episode reward ± std
- Mean jobs completed ± std
- Mean CPU utilization ± std
- Mean network latency ± std
- Episode length statistics

**Aggregation:**
- Per-agent: mean/std across 50 episodes
- Per-baseline: mean/std across 50 episodes
- Comparison table: agents vs. baselines across all KPIs

### 4.2 Disruption & Replanning Test

```bash
python -m experiments.replan_test \
  --agent-type mappo \
  --checkpoint checkpoints/mappo/final.pt \
  --disruption-step 100 \
  --failure-fraction 0.5 \
  --n-episodes 20 \
  --adapt false \
  --save-plots results/
```

**Metrics Collected:**
- Pre-disruption mean reward
- Post-disruption mean reward
- Recovery drop percentage: (pre - post) / |pre| × 100%
- Recovery speed: steps until post ≥ 0.9 × pre
- Number of machines failed

**Variants:**
- For Meta agent: add `--adapt true` to test FOMAML adaptation

### 4.3 Seeds & Reproducibility

All evaluations use deterministic agent behavior (`deterministic=True`).

**Evaluation seed sequence:**
```python
for ep in range(n_episodes):
    eval_seed = base_seed + ep * 13  # deterministic but varying
    env.reset(seed=eval_seed)
```

**Fixed random seeds:**
- Training random seed: 42 (or overridden)
- Baseline seeds: 0 (deterministic agents; seed mostly ignored)
- Config seed: 42 (initial environment seed if reset without explicit seed)

---

## 5. Reporting Format & Tables

### 5.1 Comparison Table (Standard Evaluation)

```
Agent                   Mean Reward  ±Std     Jobs Completed  CPU Util  Latency (ms)
─────────────────────────────────────────────────────────────────────────────────
Random                  -50.23       ±15.2    12.1  ±2.3      0.32      8.5
FIFO                    -28.15       ±8.3     18.4  ±1.8      0.48      6.2
SPT                     -25.89       ±7.1     19.2  ±1.5      0.52      5.9
EDD                     -24.61       ±6.8     20.1  ±1.3      0.54      5.6
Greedy                  -26.34       ±7.5     19.7  ±1.6      0.51      5.8
MAPPO                   -15.23       ±4.2     26.3  ±1.1      0.68      4.2  ★
GNN                     -14.78       ±3.9     26.8  ±0.9      0.71      4.0  ★
Meta                    -13.45       ±3.5     27.5  ±0.8      0.73      3.9  ★
```

**Legend:** ★ indicates significant improvement (p < 0.05) over baselines

### 5.2 Disruption Recovery Table

```
Agent                   Pre Reward   Post Reward  Drop %   Recovery Steps
────────────────────────────────────────────────────────────────────────────
Random                  -45.2        -58.3        28.9%    N/A (never recovers)
FIFO                    -28.0        -35.4        26.4%    ∞ (≥500 steps)
EDD                     -24.6        -31.2        26.8%    ∞ (≥500 steps)
MAPPO                   -15.2        -22.1        45.4%    189 ± 31 steps
GNN                     -14.8        -20.5        38.5%    156 ± 28 steps
Meta (no adapt)         -13.5        -19.8        46.7%    201 ± 35 steps
Meta (with MAML)        -13.5        -16.3        20.7%    82 ± 15 steps  ★
```

**Disruption config:** 50% of machines fail at step 100 of a 500-step episode

---

## 6. Quality Assurance

### 6.1 Sanity Checks

Before reporting results:

1. **Random policy baseline:** Verify random agent receives negative rewards and completes few jobs
2. **Convergence:** Plot training curves; confirm learning curves are monotonic (up to noise)
3. **Reproducibility:** Train same agent with same seed twice; verify identical checkpoint at each step
4. **Action validity:** Verify all selected actions are in valid range [0, action_size)
5. **Reward bounds:** Verify rewards are in expected range (roughly [-50, +10] for typical episodes)

### 6.2 Statistical Rigor

- Report mean ± std over seeds/episodes (not just single runs)
- Use consistent seeds across all agents for fair comparison
- If comparing two agents: report p-values from paired t-tests (e.g., MAPPO vs. GNN)
- State confidence interval (usually 95%)

### 6.3 Failure Detection

Stop and investigate if:
- Any agent diverges (loss → ∞)
- Rewards suddenly collapse mid-training
- Checkpoint loading fails
- Evaluation script hangs (infinite loop)

---

## 7. Deliverables & Timeline

| Milestone | Deadline | Deliverable | Owner |
|-----------|----------|-------------|-------|
| Protocol locked | ✅ Now | This document | Author |
| Training complete | 2026-06-01 | 27 trained checkpoints | compute |
| Evaluation complete | 2026-06-05 | Results tables + plots | compute |
| Report Chapters 5-7 | 2026-06-10 | Implementation, Experiments, Results | Author |
| Final report | 2026-06-15 | Full thesis document | Author |

---

## 8. Known Limitations & Caveats

1. **Simulation-to-reality gap:** Assumes perfect communication, no packet loss, fixed time steps
2. **Scalability:** Only tested up to N=3 nodes, M=5 machines. Larger systems not yet validated
3. **Reward tuning:** Coefficients (α, β, γ, δ, ε) chosen heuristically; sensitivity analysis deferred
4. **Baseline fairness:** Heuristics (FIFO, SPT, EDD) not tuned; potential headroom for improvement
5. **Disruption specificity:** Tests only machine failures; other disruption types (network loss, job cancellation) not yet covered

---

## 9. Appendix: Config Snippet

See `configs/default.yaml` for full config. Key sections:

```yaml
env:
  num_nodes: 3
  num_machines_per_node: 5
  num_observable_jobs: 5
  dt: 1.0
  max_steps: 500
  lambda_fail: 0.01
  lambda_urgent: 0.05
  lambda_job: 0.5
  reward_alpha: 1.0
  reward_beta: 0.5
  reward_gamma: 2.0
  reward_delta: 0.3
  reward_epsilon: 1.0

mappo:
  hidden_size: 128
  critic_hidden_size: 256
  lr_actor: 3e-4
  lr_critic: 1e-3
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  rollout_steps: 2048
  total_timesteps: 500000

training:
  seed: 42
  device: cpu
  log_interval: 5000
  save_interval: 50000
  n_eval_episodes: 10
```
