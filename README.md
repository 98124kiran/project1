# project1

## Multiagent Deep Reinforcement Learning for Adaptive Scheduling and Replanning in Dynamic Environments

### Project Structure

```
project1/
├── docs/
│   ├── literature_review.md       # Paper summaries and gap analysis
│   └── problem_formulation.md     # MDP problem definition (Dec-POMDP)
├── references/
│   └── references.bib             # BibTeX bibliography
└── src/                           # Source code (populated from Objective 2)
```

### Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Literature Review + MDP Problem Formulation | 🔄 In Progress |
| 2 | Environment / Simulator Design | ✅ Complete |
| 3 | Multi-Agent DRL Architecture | ⏳ Pending |
| 4 | Training & Experiments | ⏳ Pending |
| 5 | Evaluation & Results | ⏳ Pending |

### Project Structure (Updated)

```
project1/
├── docs/
│   ├── literature_review.md       # Paper summaries and gap analysis
│   ├── problem_formulation.md     # MDP problem definition (Dec-POMDP)
│   ├── objective2_design.md       # Simulator design decisions and API reference
│   └── thesis_structure.md        # Full thesis chapter outline
├── references/
│   └── references.bib             # BibTeX bibliography
├── requirements.txt               # Python dependencies (numpy)
└── src/
    ├── env/
    │   ├── job.py                 # Job and Operation data structures
    │   ├── machine.py             # Machine model (IDLE/BUSY/FAILED lifecycle)
    │   ├── edge_node.py           # Edge node with priority queue
    │   ├── disturbances.py        # Stochastic disturbance generator
    │   └── manufacturing_env.py   # Main multi-agent Gym-compatible environment
    └── utils/
        └── workload_generator.py  # Poisson job-stream generator
```

### Objective 1 — Getting Started

1. Read the 8 papers listed in [`docs/literature_review.md`](docs/literature_review.md)
2. Fill in each paper's summary template
3. Complete the Gap Analysis section
4. Review and extend [`docs/problem_formulation.md`](docs/problem_formulation.md) with your notes

### Objective 2 — Quick-Start

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

See [`docs/objective2_design.md`](docs/objective2_design.md) for full API reference and design decisions.

### Thesis Structure

See [`docs/thesis_structure.md`](docs/thesis_structure.md) for the complete 8-chapter thesis outline.

