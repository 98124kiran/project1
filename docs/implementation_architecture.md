# Implementation Architecture

**Project:** Multiagent Deep Reinforcement Learning for Adaptive Scheduling and Replanning in Dynamic Environments

---

## Overview

The codebase is organized into five main layers:

```
┌──────────────────────────────────────┐
│  Experiment Scripts (train/eval)     │  experiments/
├──────────────────────────────────────┤
│  Agents & Training Algorithms        │  agents/
├──────────────────────────────────────┤
│  RL Environment & Simulator          │  src/env/ + src/utils/
├──────────────────────────────────────┤
│  Hybrid Edge-Cloud Simulation        │  hybrid_compute/
└──────────────────────────────────────┘
```

---

## Layer 1: Environment & Simulation (`src/env/` + `src/utils/`)

### 1.1 ManufacturingEnv (src/env/manufacturing_env.py)

**Purpose:** Main Gym-compatible environment; orchestrates all environment dynamics.

**Key Classes:**
- `ManufacturingEnv`: Top-level environment class
  - Attributes:
    - `num_agents`, `obs_size`, `action_size` — dimensions
    - `_nodes: List[EdgeNode]` — managed nodes
    - `_disturbances: DisturbanceGenerator` — dynamic events
    - `_workload: WorkloadGenerator` — job arrivals
    - `action_spaces: List[DiscreteSpace]` — action space per agent
  - Methods:
    - `reset(seed)` → `List[np.ndarray]` — resets all nodes + generators
    - `step(actions)` → `(obs, rewards, dones, info)` — applies actions, advances time
    - `_apply_action(agent_id, action)` — decodes discrete action → node operation
    - `_compute_reward(agent_id)` → `float` — composite reward (see problem_formulation.md)
    - `_observe(agent_id)` → `np.ndarray` — builds local observation vector
    - `render()` → `str` — text summary of current state

**Data Flow (per step):**
```
Actions → Apply (ASSIGN/MIGRATE/DEFER) → Tick machines → Inject jobs/disturbances
  ↓
Compute rewards (makespan_delta, util, violations, latency, completions)
  ↓
Build observations (machine status, queue depth, top-K jobs, metrics)
  ↓
Return (obs, rewards, dones, info)
```

**Action Decoding:**
- Action 0: IDLE (no-op)
- Actions 1–M: ASSIGN queue-head to machine (m = action - 1)
- Actions M+1 to M+N-2: MIGRATE queue-head to peer node
- Action M+N-1: DEFER queue-head to back of queue

### 1.2 EdgeNode (src/env/edge_node.py)

**Purpose:** Represents one edge computing node with local machines and job queue.

**Key Components:**
- `machines: List[Machine]` — M machines (distributed round-robin by type)
- `job_queue: deque` — FIFO with priority + EDF tie-breaking
- `cpu_utilization, memory_utilization` — resource metrics
- `network_latency` — latency for this node (updated by disturbance generator)

**Key Methods:**
- `enqueue(job) → bool` — add job to queue (returns False if full)
- `dequeue() → Optional[Job]` — remove highest-priority job from queue
- `peek() → Optional[Job]` — view queue head without removal
- `try_assign_to_machine(machine_index) → bool` — schedule queue head on specific machine
- `try_assign_any_idle() → int` — greedy assignment to any compatible idle machine
- `tick(dt, current_time)` — advance machines, track completions & violations

**Queue Ordering:**
```python
sorted_queue = sorted(
    job_queue,
    key=lambda j: (-j.priority, j.deadline)  # High priority first, then EDF
)
```

### 1.3 Machine (src/env/machine.py)

**Purpose:** Models a single machine with three states: IDLE, BUSY, FAILED.

**Lifecycle:**
```
┌───────┐  assign(job, op)  ┌──────┐
│ IDLE  │ ──────────────────→ │ BUSY │
└───────┘                    └──────┘
   ↑                            │
   │ repair_time_remaining=0    │ tick() completes op
   │                            ↓
┌──────────┐  tick()      ┌─────────┐
│ FAILED   │ ←─────────── │ IDLE(2) │
└──────────┘              └─────────┘
   ↑
   │ fail(repair_time)
  BUSY or IDLE
```

**Key Attributes:**
- `status: int` — 0=IDLE, 1=BUSY, 2=FAILED
- `current_operation: Optional[Operation]` — job operation being executed
- `repair_time_remaining: float` — time until FAILED→IDLE transition
- `_busy_time, _total_time` — utilization tracking

**Key Methods:**
- `assign(job, operation) → bool` — start processing operation (check type match)
- `fail(repair_time)` — interrupt operation, mark as FAILED
- `tick(dt) → Optional[Job]` — advance by dt; return completed job if operation finished
- `utilization` property — _busy_time / _total_time

### 1.4 Job & Operation (src/env/job.py)

**Purpose:** Data structures for multi-operation jobs with deadline tracking.

**Job Structure:**
```python
@dataclass
class Job:
    job_id: int
    operations: List[Operation]  # Sequential task decomposition
    deadline: float               # Absolute deadline (time units)
    arrival_time: float
    priority: int                 # 1=normal, 2=urgent
    current_op_index: int        # Which operation is current/next
    status: JobStatus            # WAITING, PROCESSING, COMPLETED, DEFERRED, MIGRATED
    completion_time: Optional[float]
```

**Operation Structure:**
```python
@dataclass
class Operation:
    op_id: int
    processing_time: float        # Total time required
    machine_type: int             # Required machine type (0–2)
    remaining_time: float         # Time left (decreases in Machine.tick)
    status: OperationStatus       # PENDING, PROCESSING, COMPLETED
```

**Key Job Methods:**
- `advance_operation() → bool` — move to next operation; return True if job complete
- `total_remaining_time` property — sum of remaining times for current + future operations
- `slack(current_time)` property — laxity (deadline - current_time - remaining_time)

### 1.5 Disturbance Generator (src/env/disturbances.py)

**Purpose:** Injects stochastic dynamic events each time step.

**Mechanisms:**
- **Machine failures:** Bernoulli(λ_fail) per machine → sample exponential repair time
- **Urgent job injection:** Bernoulli(λ_urgent) per node → caller creates urgent job
- **Network latency walk:** Gaussian random walk clipped to [0.5, 50] ms

**Key Method:**
```python
def apply(nodes, current_time) → (num_failures, urgent_node_ids):
    # Apply all disturbances; return counts + IDs for caller to handle
```

### 1.6 Workload Generator (src/utils/workload_generator.py)

**Purpose:** Generates synthetic job streams with configurable arrival rates and characteristics.

**Key Method:**
```python
def step(current_time, priority=1) → List[Job]:
    # Sample Poisson(lambda_job) arrivals
    # Generate operations with random machine types + processing times
    # Compute deadline = current_time + total_proc_time + slack
```

**Parameters:**
- `lambda_job` — Poisson rate (0.5 jobs/step = moderate load)
- `min_ops, max_ops` — operations per job (1–3 for complexity)
- `min_processing_time, max_processing_time` — uniform distribution
- `min_deadline_slack, max_deadline_slack` — deadline looseness

---

## Layer 2: Agents & Learning (`agents/`)

### 2.1 MAPPO Agent (agents/ppo_agent.py)

**Architecture:** Shared MLP actor + centralized critic (CTDE paradigm)

**Components:**

#### ActorNetwork (shared across all agents)
```
Input: o_t^n (obs_size,)
  ↓
Dense(obs_size → 128), Tanh
Dense(128 → 128), Tanh
Dense(128 → action_size) [logits]
  ↓
Output: Categorical(logits) for action sampling
```

#### CriticNetwork (centralized, used only during training)
```
Input: s_t (N·obs_size,)  [global state]
  ↓
Dense(N·obs_size → 256), Tanh
Dense(256 → 256), Tanh
Dense(256 → 1) [value]
  ↓
Output: V(s_t) ∈ ℝ (scalar value estimate)
```

#### RolloutBuffer (trajectory storage)
```python
# Stores T time steps × N agents
observations: (T, N, obs_size)
global_obs: (T, N·obs_size)
actions: (T, N)
log_probs: (T, N)
rewards: (T, N)
dones: (T, N)
values: (T, N)

# Methods:
compute_advantages_and_returns(last_values) → (advantages, returns)  # GAE
iterate_batches(advantages, returns, batch_size) → generator  # Yield mini-batches
```

**Training Loop (per update):**
```
1. Collect rollout_steps environment interactions
2. Compute GAE(γ, λ) advantages and discounted returns
3. For n_epochs:
    For each mini-batch:
        Forward: dist = actor(obs), values = critic(global_obs)
        Loss: actor_loss = -min(surr1, surr2) - entropy_coef * entropy
              critic_loss = MSE(values, returns)
        Backward + update
```

**Decentralized Execution (inference):**
```python
def select_actions(observations):
    # observations: List[np.ndarray], one per agent (local obs only!)
    actions, log_probs, values = actor(observations)
    # Values computed from global state, but only for logging
    return actions, log_probs, values
```

### 2.2 GNN Policy Agent (agents/gnn_policy.py)

**Drop-in replacement for MAPPO actor.** Same critic + training loop.

**GNN Actor Architecture:**

```
Observation Parsing:
  o_t^n → machine_feats (M, 1)
          job_feats (K, 2)
          context_feats (1, 4)  [queue, cpu, mem, latency]

Node Embedding:
  machine_feats → embed (M, d_model=64)
  job_feats → embed (K, d_model=64)
  context_feats → embed (1, d_model=64)
  
  all_nodes = concat([machine_embed, job_embed, context_embed])  # (M+K+1, 64)

Multi-Head Self-Attention (n_layers=2):
  for _ in range(2):
      all_nodes = TransformerBlock(all_nodes)
      # LN → MultiHeadAttn → residual + FFN → residual
      # Output: (M+K+1, 64)

Action Head:
  flat = flatten(all_nodes)  # ((M+K+1)·d_model,)
  logits = MLP(flat)  # → (action_size,)
  
Output: Categorical(logits)
```

**Rationale:**
- Explicit graph structure captures job-machine relationships
- Attention allows agents to focus on relevant jobs/machines
- Flattened readout aggregates information across entire graph

### 2.3 Meta-Learning Agent (agents/meta_agent.py)

**FOMAML (First-Order MAML) for adaptive replanning.**

**Architecture:** Dual network pairs for meta + adapted weights.

```
Meta Networks (θ):
  actor: ActorNetwork
  critic: CriticNetwork

Adapted Networks (θ'):
  Cloned during inner loop for task-specific updates
```

**Inner Loop (adaptation at test time):**
```python
def adapt(observations, actions, returns, steps=3):
    # observations: (T, N, obs_size)  [disruption trajectory]
    # actions: (T, N)
    # returns: (T, N)  [discounted]
    
    # Clone meta networks
    adapted_actor = deepcopy(self.actor)
    adapted_critic = deepcopy(self.critic)
    
    # SGD(inner_lr=0.01) for steps iterations
    for _ in range(steps):
        dist = adapted_actor(obs_flat)
        log_probs = dist.log_prob(act_flat)
        values = adapted_critic(gobs_flat)
        
        loss = -(log_probs * (returns - values.detach())).mean() \
               + value_coef * MSE(values, returns)
        
        loss.backward()
        # Update adapted weights
    
    self._adapted_actor = adapted_actor
    self._is_adapted = True
```

**Outer Loop (meta-training):**
```
For each episode:
    1. Split rollout buffer in half
    2. Inner loop: adapt θ → θ' using first half
    3. Outer loss: evaluate θ' on second half
    4. Meta-update θ using ∇_θ(outer_loss)  [FOMAML: first-order only]
```

### 2.4 Baseline Agents (agents/baselines.py)

All baselines share interface: `select_actions(observations) → (actions, log_probs=zeros, values=zeros)`

| Baseline | Strategy |
|----------|----------|
| Random | Uniform random action |
| FIFO | Assign queue-head to first idle machine |
| SPT | Assign shortest-processing-time job to idle machine |
| EDD | Assign earliest-deadline job to idle machine |
| Greedy | Assign any queued job to any idle machine (randomized) |

---

## Layer 3: Training & Evaluation Scripts (`experiments/`)

### 3.1 train.py

**Purpose:** Main training loop for MAPPO/GNN/Meta agents.

**Key Functions:**
```python
def train(agent, env, cfg, total_timesteps, rollout_steps, ...):
    """
    On-policy training loop.
    
    While env_steps < total_timesteps:
        # Collect rollout
        for _ in range(rollout_steps):
            actions, log_probs, values = agent.select_actions(obs)
            obs, rewards, dones, info = env.step(actions)
            agent.store_transition(...)
            if all(dones): obs = env.reset()
        
        # Update
        metrics = agent.update(last_observations=obs, last_dones=dones)
        
        # Log & checkpoint periodically
    """
```

**CLI:**
```bash
python -m experiments.train --agent mappo --total-steps 500000 --device cpu
```

### 3.2 evaluate.py

**Purpose:** Benchmark trained agent against baselines on standard evaluation.

**Key Functions:**
```python
def run_episode(agent, env, deterministic=True):
    # Single episode with deterministic agent behavior
    # Return KPIs: reward, jobs_completed, cpu_util, latency

def evaluate_agent(agent, env, n_episodes=50):
    # n_episodes runs; return aggregated mean ± std
```

**Baselines always evaluated:** Random, FIFO, SPT, EDD, Greedy

**CLI:**
```bash
python -m experiments.evaluate \
  --agent-type mappo \
  --checkpoint checkpoints/mappo/final.pt \
  --n-episodes 50 \
  --save-plots results/
```

### 3.3 replan_test.py

**Purpose:** Test disruption recovery; compare agent vs. baseline + MAML adaptation.

**Key Functions:**
```python
def run_replan_episode(agent, env, disruption_step=100, adapt=False):
    # Run episode with forced disruption (50% machine failures) at step 100
    # If adapt=True and agent is MetaAgent: call agent.adapt() with pre-disruption data
    # Return pre/post rewards + recovery metrics

def evaluate_replanning(agent, env, disruption_step, n_episodes=20, adapt=False):
    # Run n_episodes disruption tests; aggregate recovery metrics
```

**Recovery Metrics:**
- `recovery_drop_pct`: (pre - post) / |pre| × 100%
- `recovery_speed_steps`: steps until post ≥ 0.9 × pre
- `pre_mean_reward`, `post_mean_reward`: averages

**CLI:**
```bash
python -m experiments.replan_test \
  --agent-type meta \
  --checkpoint checkpoints/meta/final.pt \
  --disruption-step 100 \
  --adapt true \
  --n-episodes 20
```

---

## Layer 4: Hybrid Edge-Cloud Simulation (`hybrid_compute/`)

### 4.1 EdgeInferenceEngine (hybrid_compute/edge_inference.py)

**Purpose:** Simulates lightweight policy deployment on edge nodes.

**Key Features:**
- Fast inference (no training)
- Simulated latency: base_latency_ms + network_latency_ms
- Experience buffer for uploading to cloud
- Bandwidth-limited upload simulation
- Weight synchronization (download from cloud)

**Interface:**
```python
def infer(observations) → (actions, log_probs, values):
    # Forward pass through actor (frozen weights)
    # Simulate latency
    # Return actions

def store_experience(transition_dict):
    # Buffer for cloud upload

def upload_experience(max_size=None) → (batch, transfer_time_ms):
    # Simulate bandwidth-limited upload

def sync_weights(state_dict) → sync_latency_ms:
    # Simulate weight download from cloud
```

### 4.2 CloudTrainer (hybrid_compute/cloud_trainer.py)

**Purpose:** Simulates central cloud training on aggregated edge experience.

**Components:**
- `CloudTrainer`: receives experience from all edges, runs PPO updates
- `FederatedAggregator`: implements FedAvg (weighted averaging of edge weights)

**Key Methods:**
```python
def receive_experience(node_id, batch):
    # Accumulate experience from edge node

def train_step() → metrics:
    # Run PPO on aggregated experience pool
    # Return: actor_loss, critic_loss, entropy

def federated_aggregate() → bool:
    # Merge weights from all edges via FedAvg if ready
    # Returns True if aggregation performed

def get_actor_weights() → state_dict:
    # Broadcast updated actor to edges
```

**Workflow:**
```
Edge 1: infer → store_exp  ─→ upload_exp
Edge 2: infer → store_exp  ─→ upload_exp    ──→ Cloud: receive_experience
Edge 3: infer → store_exp  ─→ upload_exp  ──→        train_step()
                                                      federated_aggregate()
                                              ←────── broadcast weights
```

---

## Layer 5: Visualization (`visualization/`)

### 5.1 gantt.py

**Plotting Functions:**
- `plot_gantt(schedule)` — Gantt chart of job execution timeline
- `plot_learning_curves(rewards_dict)` — Training reward curves with smoothing
- `plot_metrics_comparison(metrics_dict)` — Grouped bar charts (agents vs. KPIs)
- `plot_disruption_timeline(reward_before, reward_after)` — Pre/post disruption visualization
- `save_figure(fig, path)` — Save Matplotlib figure to disk

---

## Data Flow Summary

### Training Loop
```
env.reset()
└─ obs = [o_1, o_2, o_3]  # local obs per agent

for step in range(rollout_steps):
    agent.select_actions(obs)
    └─ actions, log_probs, values = actor(obs), critic(concat(obs))
    
    env.step(actions)
    ├─ Apply actions (ASSIGN/MIGRATE/DEFER)
    ├─ Tick machines (advance operations)
    ├─ Inject jobs & disturbances
    └─ obs', rewards, dones, info
    
    agent.store_transition(obs, global_obs, actions, log_probs, rewards, dones, values)
    └─ Accumulate in RolloutBuffer

agent.update(last_obs, last_dones)
├─ Bootstrap last values
├─ Compute GAE advantages & returns
└─ For n_epochs:
    ├─ For mini-batch:
    │   ├─ Forward: dist, values
    │   ├─ Compute losses
    │   └─ Backward + update
    └─ (Reset buffer)
```

### Inference Loop
```
obs = env.reset()

for step in range(episode_length):
    actions, _, _ = agent.select_actions(obs)
    └─ Use actor.eval() (no gradient, no critic)
    
    obs, rewards, dones, info = env.step(actions)
    
    if all(dones): obs = env.reset()
```

---

## Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Shared actor across agents** | Simplicity; reduces parameters | Less specialization per agent |
| **Centralized critic (training only)** | Variance reduction; captures global state | Not deployable; requires coordination |
| **Discrete actions (1 per step)** | Practical edge deployment; no scaling | Limited action expressiveness |
| **Flat observation + GNN variants** | Baseline + enhanced versions | GNN adds complexity |
| **Separate Meta networks** | Clean adaptation logic | Memory overhead (2 × networks) |
| **Simulation-only hybrid stack** | Fast iteration; no real infrastructure | Sim-to-reality gap |

---

## Extensibility

**Adding new baselines:**
- Subclass `BaselineAgent` in agents/baselines.py
- Implement `_choose_actions(observations) → actions`
- Will be automatically picked up by evaluate.py

**Adding new disturbances:**
- Extend `DisturbanceGenerator.apply()` in src/env/disturbances.py
- Ensure disturbance is applied each step + returns metadata

**Adding new agent variants:**
- Copy agents/ppo_agent.py → agents/new_agent.py
- Implement interface: `select_actions()`, `store_transition()`, `update()`, `save()`, `load()`
- Add to `build_agent()` in experiments/train.py

**Changing environment scale:**
- Update `configs/default.yaml`: num_nodes, num_machines_per_node
- Recompute obs_size = M + 1 + 2K + 3
- Recompute action_size = 1 + M + (N-1) + 1
- Retrain agents
