# Literature Review

**Project:** Multiagent Deep Reinforcement Learning for Adaptive Scheduling and Replanning in Dynamic Environments

---

## How to Use This File

For each paper below, fill in the fields under its heading. A completed Gap Analysis section at the bottom ties everything together into your research contribution.

---

## Paper 1: Deep Reinforcement Learning for Job-Shop Scheduling Problems

- **Authors & Year:** Zhang et al., 2020
- **Problem Solved:** _[Fill in: what scheduling problem does this solve?]_
- **Method Used:** _[Fill in: algorithm, network architecture]_
- **Key Results:** _[Fill in: metrics, benchmarks beaten]_
- **Relevance to My Project:** _[Fill in: how does this relate to your work?]_
- **Gap / Limitation:** _[Fill in: what does this paper NOT address?]_

---

## Paper 2: Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning (L2D)

- **Authors & Year:** Zhang et al., 2020
- **Problem Solved:** _[Fill in]_
- **Method Used:** Graph Neural Networks + PPO
- **Key Results:** _[Fill in]_
- **Relevance to My Project:** _[Fill in]_
- **Gap / Limitation:** _[Fill in]_

---

## Paper 3: Smart Manufacturing Scheduling with Edge Computing Using Multiclass Deep Q-Network

- **Authors & Year:** Shiue et al., 2018
- **Problem Solved:** _[Fill in]_
- **Method Used:** Deep Q-Network (DQN) at edge nodes
- **Key Results:** _[Fill in]_
- **Relevance to My Project:** _[Fill in]_
- **Gap / Limitation:** _[Fill in]_

---

## Paper 4: A Deep Reinforcement Learning Approach for Real-Time Online Shop Scheduling

- **Authors & Year:** Han & Yang, 2021
- **Problem Solved:** Dynamic job arrivals and machine breakdowns
- **Method Used:** _[Fill in]_
- **Key Results:** _[Fill in]_
- **Relevance to My Project:** _[Fill in]_
- **Gap / Limitation:** _[Fill in]_

---

## Paper 5: Dynamic Job-Shop Scheduling Using Deep Reinforcement Learning

- **Authors & Year:** Park et al., 2021
- **Problem Solved:** _[Fill in]_
- **Method Used:** _[Fill in]_
- **Key Results:** _[Fill in]_
- **Relevance to My Project:** _[Fill in]_
- **Gap / Limitation:** _[Fill in]_

---

## Paper 6: Deep Reinforcement Learning for Mobile Edge Computing

- **Authors & Year:** Huang et al., 2019
- **Problem Solved:** Task offloading decisions in mobile edge computing
- **Method Used:** _[Fill in]_
- **Key Results:** _[Fill in]_
- **Relevance to My Project:** _[Fill in]_
- **Gap / Limitation:** _[Fill in]_

---

## Paper 7: Multi-Agent Deep Reinforcement Learning for Edge Computing

- **Authors & Year:** Chen et al., 2021
- **Problem Solved:** _[Fill in]_
- **Method Used:** Multi-agent DRL
- **Key Results:** _[Fill in]_
- **Relevance to My Project:** _[Fill in]_
- **Gap / Limitation:** _[Fill in]_

---

## Paper 8: Proximal Policy Optimization Algorithms

- **Authors & Year:** Schulman et al., 2017 (OpenAI)
- **Problem Solved:** Stable, sample-efficient policy gradient training
- **Method Used:** PPO (clipped surrogate objective)
- **Key Results:** _[Fill in]_
- **Relevance to My Project:** Core training algorithm likely to be used
- **Gap / Limitation:** General-purpose; not scheduling-specific

---

## Gap Analysis

> _Fill in after reading all papers. Template below:_

Existing works in DRL-based scheduling either focus on static job-shop problems, edge computing offloading in isolation, or single-agent formulations. None combine **multi-agent DRL** with **edge-aware adaptive replanning** under **real-time dynamic disruptions** (machine failures, urgent job injections) in smart manufacturing environments. This project addresses that combined gap by proposing a multi-agent framework where distributed edge nodes act as cooperative agents, each making local scheduling decisions that adapt globally to environmental disturbances.

---

## Reading Checklist

- [ ] Paper 1 — Zhang et al. 2020 (DRL for JSP)
- [ ] Paper 2 — Zhang et al. 2020 (L2D)
- [ ] Paper 3 — Shiue et al. 2018 (Edge + DQN)
- [ ] Paper 4 — Han & Yang 2021 (Real-time online scheduling)
- [ ] Paper 5 — Park et al. 2021 (Dynamic JSP)
- [ ] Paper 6 — Huang et al. 2019 (MEC + DRL)
- [ ] Paper 7 — Chen et al. 2021 (MARL + Edge)
- [ ] Paper 8 — Schulman et al. 2017 (PPO)
