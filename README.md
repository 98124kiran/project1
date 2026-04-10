# project1

## Multiagent Deep Reinforcement Learning for Adaptive Scheduling and Replanning in Dynamic Environments

### Project Structure

```
project1/
├── configs/
│   └── default.yaml           # All hyperparameters (env, MAPPO, GNN, meta, hybrid)
├── docs/
│   ├── literature_review.md   # Paper summaries and gap analysis
│   ├── problem_formulation.md # MDP problem definition (Dec-POMDP)
│   ├── objective2_design.md   # Simulator design decisions and API reference
│   └── thesis_structure.md    # Full thesis chapter outline
├── references/
│   └── references.bib         # BibTeX bibliography
├── src/                       # Phase 1 — Environment (complete)
│   ├── env/
│   │   ├── job.py             # Job and Operation data structures
│   │   ├── machine.py         # Machine model (IDLE/BUSY/FAILED lifecycle)
│   │   ├── edge_node.py       # Edge node with priority queue
│   │   ├── disturbances.py    # Stochastic disturbance generator
│   │   └── manufacturing_env.py  # Main multi-agent Gym-compatible environment
│   └── utils/
│       └── workload_generator.py  # Poisson job-stream generator
├── agents/                    # Phase 2 & 3 — DRL Agents
│   ├── ppo_agent.py           # MAPPO — shared actor + centralised critic + GAE
│   ├── gnn_policy.py          # Graph Attention Network policy (Set-Transformer)
│   ├── meta_agent.py          # FOMAML meta-learning agent for adaptive replanning
│   └── baselines.py           # FIFO, SPT, EDD, Greedy, Random baselines
├── hybrid_compute/            # Phase 4 — Hybrid Edge-Cloud Computing
│   ├── edge_inference.py      # Edge inference engine (latency + bandwidth simulation)
│   └── cloud_trainer.py       # Cloud trainer + FedAvg federated aggregation
├── experiments/               # Phase 5 — Experiment Scripts
│   ├── train.py               # CLI training script (MAPPO / GNN / Meta)
│   ├── evaluate.py            # Evaluation & baseline benchmarks
│   └── replan_test.py         # Mid-episode disruption & recovery test
├── visualization/             # Phase 5 — Visualization
│   └── gantt.py               # Gantt charts, learning curves, comparison plots
├── notebooks/
│   └── results_analysis.ipynb # Step-by-step analysis notebook
└── requirements.txt           # numpy, torch, matplotlib, plotly, pandas, pyyaml
```

### Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Literature Review + MDP Problem Formulation | 🔄 In Progress |
| 2 | Environment / Simulator Design | ✅ Complete |
| 3 | Multi-Agent DRL Architecture (MAPPO + GNN) | ✅ Complete |
| 4 | Training & Experiments (train.py + evaluate.py) | ✅ Complete |
| 5 | Evaluation & Results (replan_test + visualization) | ✅ Complete |

---

### Phase 1 — Environment Quick-Start

```python
from src.env.manufacturing_env import ManufacturingEnv

env = ManufacturingEnv()                          # default: 3 nodes, 5 machines each
obs = env.reset(seed=42)                          # list of 3 arrays, each shape (19,)

for step in range(500):
    actions = [env.action_spaces[i].sample() for i in range(env.num_agents)]
    obs, rewards, dones, info = env.step(actions)
    if all(dones):
        break

print(env.render())
```

---

### Phase 2 — MAPPO Agent Quick-Start

```python
from src.env.manufacturing_env import ManufacturingEnv
from agents.ppo_agent import MAPPOAgent
import numpy as np

env = ManufacturingEnv()
agent = MAPPOAgent(
    obs_size=env.obs_size,        # 19
    action_size=env.action_size,  # 9
    num_agents=env.num_agents,    # 3
    rollout_steps=2048,
)

obs = env.reset(seed=42)
for step in range(2048):
    actions, log_probs, values = agent.select_actions(obs)
    global_obs = np.concatenate(obs)
    next_obs, rewards, dones, info = env.step(actions)
    agent.store_transition(obs, global_obs, actions, log_probs, rewards, dones, values)
    obs = next_obs
    if all(dones):
        obs = env.reset()

metrics = agent.update(last_observations=obs)
print(metrics)
```

---

### Phase 2 — GNN Policy (drop-in replacement)

```python
from agents.gnn_policy import GNNPolicyAgent

agent = GNNPolicyAgent(
    obs_size=env.obs_size,
    action_size=env.action_size,
    num_agents=env.num_agents,
    num_machines=env.M,            # needed for graph construction
    num_observable_jobs=env.K,
    d_model=64, n_heads=4, n_layers=2,
)
# Same interface as MAPPOAgent
```

---

### Phase 3 — MAML Adaptive Replanning

```python
from agents.meta_agent import MetaAgent
from experiments.replan_test import run_replan_episode

agent = MetaAgent(obs_size=env.obs_size, action_size=env.action_size, num_agents=env.num_agents)
# ... (train as usual) ...

# At test time, after a disruption occurs:
agent.adapt(observations, actions, returns, steps=5)   # inner-loop adaptation
# Agent now uses adapted weights for the rest of the disrupted episode
agent.restore_meta_weights()                           # reset for next episode
```

---

### Phase 4 — Hybrid Edge-Cloud Simulation

```python
from hybrid_compute.edge_inference import EdgeInferenceEngine
from hybrid_compute.cloud_trainer import CloudTrainer, FederatedAggregator

# Edge node inference (low-latency, no training)
engine = EdgeInferenceEngine(actor=agent.actor, node_id=0, bandwidth_mbps=10.0)
actions, lp, _ = engine.infer(observations)
engine.store_experience({"obs": ..., "action": ..., "reward": ..., ...})

# Upload to cloud and train
cloud = CloudTrainer(obs_size=19, action_size=9, num_agents=3)
batch, _ = engine.upload_experience()
cloud.receive_experience(node_id=0, batch=batch)
metrics = cloud.train_step()

# FedAvg: merge weights from all edge nodes
fed = FederatedAggregator(n_nodes=3)
fed.submit_local_weights(node_id=0, state_dict=agent.actor.state_dict(), n_samples=100)
agg_weights = fed.aggregate()
```

---

### Phase 5 — Training & Evaluation CLI

```bash
# Train MAPPO
python -m experiments.train --agent mappo --total-steps 500000

# Train GNN
python -m experiments.train --agent gnn --total-steps 500000

# Train with MAML meta-learning
python -m experiments.train --agent meta --total-steps 500000

# Evaluate against all baselines
python -m experiments.evaluate --checkpoint checkpoints/mappo/final.pt --n-episodes 50

# Disruption / replanning test
python -m experiments.replan_test --checkpoint checkpoints/mappo/final.pt --disruption-step 100 --n-episodes 20
```

---

### Installation

```bash
pip install -r requirements.txt
```

### Thesis Structure

See [`docs/thesis_structure.md`](docs/thesis_structure.md) for the complete 8-chapter thesis outline.

