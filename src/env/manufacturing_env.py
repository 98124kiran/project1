"""
Multi-Agent Manufacturing Scheduling Environment.

Implements a Decentralized Partially Observable Markov Decision Process
(Dec-POMDP) for adaptive job scheduling across distributed edge nodes,
as defined in docs/problem_formulation.md.

Interface mirrors OpenAI Gym / Gymnasium but returns per-agent lists,
making it directly compatible with MAPPO and similar multi-agent
training frameworks.

Usage
-----
>>> from src.env.manufacturing_env import ManufacturingEnv
>>> env = ManufacturingEnv()
>>> obs = env.reset()
>>> actions = [env.action_spaces[i].sample() for i in range(env.num_agents)]
>>> obs, rewards, dones, info = env.step(actions)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.env.disturbances import DisturbanceGenerator
from src.env.edge_node import EdgeNode
from src.env.job import JobStatus
from src.utils.workload_generator import WorkloadGenerator

# --------------------------------------------------------------------------- #
# Default configuration                                                         #
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG: Dict[str, Any] = {
    # Topology
    "num_nodes": 3,                 # N — number of edge nodes / agents
    "num_machines_per_node": 5,     # M — machines per node
    "num_machine_types": 3,         # distinct machine types
    "max_queue_length": 20,         # max jobs buffered per node
    "num_observable_jobs": 5,       # K — jobs visible in observation vector

    # Simulation timing
    "dt": 1.0,                      # time step duration (minutes)
    "max_steps": 500,               # episode length

    # Disturbance rates
    "lambda_fail": 0.01,            # machine failure probability per step
    "mean_repair_time": 20.0,       # mean repair duration (time units)
    "lambda_urgent": 0.05,          # urgent job injection probability per step per node
    "latency_sigma": 0.5,           # random-walk std for network latency (ms)

    # Workload
    "lambda_job": 0.5,              # mean normal jobs arriving per step (Poisson)
    "min_processing_time": 5.0,
    "max_processing_time": 30.0,
    "min_deadline_slack": 20.0,
    "max_deadline_slack": 100.0,
    "min_ops": 1,
    "max_ops": 3,

    # Reward shaping coefficients (see problem_formulation.md §3)
    "reward_alpha": 1.0,            # penalty coefficient for makespan growth
    "reward_beta": 0.5,             # reward coefficient for machine utilisation
    "reward_gamma": 2.0,            # penalty per deadline violation
    "reward_delta": 0.3,            # penalty for average edge latency
    "reward_epsilon": 1.0,          # reward per completed job

    # Reproducibility
    "seed": None,
}


# --------------------------------------------------------------------------- #
# Discrete action space descriptor (lightweight, no gymnasium dependency)      #
# --------------------------------------------------------------------------- #

class DiscreteSpace:
    """Minimal discrete action space compatible with numpy sampling."""

    def __init__(self, n: int, rng: np.random.Generator) -> None:
        self.n = n
        self._rng = rng

    def sample(self) -> int:
        return int(self._rng.integers(0, self.n))

    def contains(self, x: int) -> bool:
        return 0 <= x < self.n


# --------------------------------------------------------------------------- #
# Action indices                                                                #
# --------------------------------------------------------------------------- #
#
# For each agent (edge node n) with M machines and N total nodes:
#
#   0              → IDLE
#   1 … M          → ASSIGN queue-head to machine (action - 1)
#   M+1 … M+(N-2)  → MIGRATE queue-head to peer node (skipping self)
#   M+(N-1)        → DEFER queue-head job by one planning horizon
#
# Total actions = 1 + M + (N-1) + 1 = M + N + 1
# --------------------------------------------------------------------------- #


class ManufacturingEnv:
    """
    Dec-POMDP environment for multi-agent adaptive job scheduling.

    Attributes
    ----------
    num_agents : int
        Number of cooperative agents (= number of edge nodes).
    obs_size : int
        Dimension of each agent's local observation vector.
    action_size : int
        Number of discrete actions available to each agent.
    action_spaces : List[DiscreteSpace]
        Per-agent action spaces (for sampling during random rollouts).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg

        # Topology
        self.num_agents: int = cfg["num_nodes"]
        self.M: int = cfg["num_machines_per_node"]
        self.num_machine_types: int = cfg["num_machine_types"]
        self.K: int = cfg["num_observable_jobs"]
        self.dt: float = cfg["dt"]
        self.max_steps: int = cfg["max_steps"]

        # Action space size: IDLE + M machine assigns + (N-1) migrates + DEFER
        self.action_size: int = 1 + self.M + (self.num_agents - 1) + 1

        # Observation size per agent:
        #   M machine statuses  +  1 queue length  +  K*(remaining_time + deadline)
        #   +  cpu_util  +  mem_util  +  latency
        self.obs_size: int = self.M + 1 + self.K * 2 + 3

        # RNG
        seed = cfg["seed"]
        self._rng = np.random.default_rng(seed)

        # Sub-components
        self._nodes: List[EdgeNode] = [
            EdgeNode(
                node_id=i,
                num_machines=self.M,
                num_machine_types=self.num_machine_types,
                max_queue_length=cfg["max_queue_length"],
                rng=self._rng,
            )
            for i in range(self.num_agents)
        ]

        self._disturbances = DisturbanceGenerator(
            lambda_fail=cfg["lambda_fail"],
            mean_repair_time=cfg["mean_repair_time"],
            lambda_urgent=cfg["lambda_urgent"],
            latency_sigma=cfg["latency_sigma"],
            rng=self._rng,
        )

        self._workload = WorkloadGenerator(
            lambda_job=cfg["lambda_job"],
            min_processing_time=cfg["min_processing_time"],
            max_processing_time=cfg["max_processing_time"],
            min_deadline_slack=cfg["min_deadline_slack"],
            max_deadline_slack=cfg["max_deadline_slack"],
            min_ops=cfg["min_ops"],
            max_ops=cfg["max_ops"],
            num_machine_types=self.num_machine_types,
            rng=self._rng,
        )

        # Action spaces (for random rollout / sanity checks)
        self.action_spaces: List[DiscreteSpace] = [
            DiscreteSpace(self.action_size, self._rng)
            for _ in range(self.num_agents)
        ]

        # Episode state
        self._current_step: int = 0
        self._current_time: float = 0.0
        self._prev_makespan: float = 0.0
        self._total_jobs_completed: int = 0

        # Observation normalisation constants (computed once, reused every step)
        self._obs_max_proc = cfg["max_processing_time"] * cfg["max_ops"]
        self._obs_max_deadline = (
            cfg["max_steps"] * cfg["dt"]        # latest possible current_time
            + self._obs_max_proc                # + max processing
            + cfg["max_deadline_slack"]         # + max slack
        )

    # ------------------------------------------------------------------ #
    # Gym-like API                                                         #
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> List[np.ndarray]:
        """
        Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Override the RNG seed for this episode.

        Returns
        -------
        observations : List[np.ndarray]
            One observation vector per agent (shape: [obs_size]).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            # Propagate new RNG to sub-components
            for node in self._nodes:
                node.rng = self._rng
            self._disturbances.rng = self._rng
            self._workload.rng = self._rng

        for node in self._nodes:
            node.reset()
        self._workload.reset()

        self._current_step = 0
        self._current_time = 0.0
        self._prev_makespan = 0.0
        self._total_jobs_completed = 0

        # Warm-start: generate initial jobs and distribute round-robin
        initial_jobs = self._workload.step(self._current_time)
        for idx, job in enumerate(initial_jobs):
            self._nodes[idx % self.num_agents].enqueue(job)

        return [self._observe(i) for i in range(self.num_agents)]

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """
        Apply one action per agent and advance the environment by *dt*.

        Parameters
        ----------
        actions : List[int]
            One discrete action per agent. Must satisfy
            ``0 <= actions[i] < action_size``.

        Returns
        -------
        observations : List[np.ndarray]
            Updated local observation per agent.
        rewards : List[float]
            Scalar reward per agent.
        dones : List[bool]
            True for each agent when the episode has ended.
        info : Dict[str, Any]
            Diagnostic information (shared across agents).
        """
        assert len(actions) == self.num_agents, (
            f"Expected {self.num_agents} actions, got {len(actions)}"
        )

        # 1. Apply agent actions
        for agent_id, action in enumerate(actions):
            self._apply_action(agent_id, action)

        # 2. Advance machines / resource metrics
        for node in self._nodes:
            node.tick(self.dt, self._current_time)

        # 3. Inject new jobs from workload stream
        new_jobs = self._workload.step(self._current_time)
        for idx, job in enumerate(new_jobs):
            self._nodes[idx % self.num_agents].enqueue(job)

        # 4. Apply disturbances (failures, urgent jobs, latency)
        _, urgent_node_ids = self._disturbances.apply(self._nodes, self._current_time)
        for node_id in urgent_node_ids:
            urgent_job = self._workload.step_urgent(self._current_time)
            self._nodes[node_id].enqueue(urgent_job)

        # 5. Advance time
        self._current_step += 1
        self._current_time += self.dt

        # 6. Compute rewards and check terminal condition
        rewards = [self._compute_reward(i) for i in range(self.num_agents)]
        done = self._current_step >= self.max_steps
        dones = [done] * self.num_agents

        # 7. Build observations
        observations = [self._observe(i) for i in range(self.num_agents)]

        info = self._build_info()
        return observations, rewards, dones, info

    # ------------------------------------------------------------------ #
    # Action application                                                   #
    # ------------------------------------------------------------------ #

    def _apply_action(self, agent_id: int, action: int) -> None:
        """Decode and apply a single agent's action."""
        node = self._nodes[agent_id]

        if action == 0:
            # IDLE — do nothing
            return

        if 1 <= action <= self.M:
            # ASSIGN queue-head to machine (action - 1)
            machine_index = action - 1
            node.try_assign_to_machine(machine_index)
            return

        migrate_start = self.M + 1
        migrate_end = self.M + self.num_agents - 1   # last migrate action index (inclusive)

        if migrate_start <= action <= migrate_end:
            # MIGRATE queue-head to a peer node
            # Build ordered list of peer node IDs (skip self)
            peers = [n for n in range(self.num_agents) if n != agent_id]
            peer_index = action - migrate_start
            if peer_index < len(peers):
                target_node_id = peers[peer_index]
                job = node.dequeue()
                if job is not None:
                    job.status = JobStatus.MIGRATED
                    accepted = self._nodes[target_node_id].enqueue(job)
                    if not accepted:
                        # Target queue full — return job to sender
                        node.enqueue(job)
            return

        # DEFER — push queue-head to back of queue (simple deferral)
        defer_action = self.M + self.num_agents
        if action == defer_action:
            job = node.dequeue()
            if job is not None:
                job.status = JobStatus.DEFERRED
                node.requeue(job)

    # ------------------------------------------------------------------ #
    # Reward                                                               #
    # ------------------------------------------------------------------ #

    def _compute_reward(self, agent_id: int) -> float:
        """
        Compute the reward for *agent_id* at the current step.

        Follows the reward function from problem_formulation.md §3:
            r_t = - α·makespan_delta
                  + β·avg_machine_utilisation
                  - γ·deadline_violations
                  - δ·avg_edge_latency
                  + ε·jobs_completed
        """
        cfg = self.cfg
        node = self._nodes[agent_id]

        # Makespan delta: proxy using remaining work in the system
        current_makespan = sum(
            sum(m.current_operation.remaining_time
                for m in n.machines if m.current_operation is not None)
            + sum(j.total_remaining_time for j in n.job_queue)
            for n in self._nodes
        )
        makespan_delta = max(0.0, current_makespan - self._prev_makespan)
        self._prev_makespan = current_makespan

        # Average machine utilisation at this node
        avg_util = float(np.mean([m.utilization for m in node.machines]))

        # Deadline violations and completed jobs this step (from node accounting)
        deadline_violations = node._deadline_violations_this_step
        jobs_completed = node._jobs_completed_this_step
        self._total_jobs_completed += jobs_completed

        # Network latency (normalised by latency_max)
        latency_max = self._disturbances.latency_max
        norm_latency = node.network_latency / latency_max

        reward = (
            - cfg["reward_alpha"] * makespan_delta
            + cfg["reward_beta"] * avg_util
            - cfg["reward_gamma"] * deadline_violations
            - cfg["reward_delta"] * norm_latency
            + cfg["reward_epsilon"] * jobs_completed
        )
        return float(reward)

    # ------------------------------------------------------------------ #
    # Observation                                                          #
    # ------------------------------------------------------------------ #

    def _observe(self, agent_id: int) -> np.ndarray:
        """
        Build the local observation vector for *agent_id*.

        Structure:
            [machine_status_0, …, machine_status_{M-1},   # M values (int 0/1/2)
             queue_length,                                 # 1 value
             job_0_remaining, job_0_deadline,             # K × 2 values
             …
             job_{K-1}_remaining, job_{K-1}_deadline,
             cpu_utilization, memory_utilization,         # 2 values
             network_latency_norm]                        # 1 value
        """
        node = self._nodes[agent_id]
        obs: List[float] = []

        # Machine statuses (normalised to [0, 1] by dividing by 2)
        obs.extend(s / 2.0 for s in node.get_machine_statuses())

        # Queue length (normalised)
        obs.append(node.queue_length / node.max_queue_length)

        # Top-K jobs in queue (sorted by priority then earliest deadline)
        sorted_jobs = sorted(
            node.job_queue,
            key=lambda j: (-j.priority, j.deadline),
        )[: self.K]

        for job in sorted_jobs:
            obs.append(min(job.total_remaining_time / max(self._obs_max_proc, 1.0), 1.0))
            obs.append(min(job.deadline / max(self._obs_max_deadline, 1.0), 1.0))

        # Pad with zeros if fewer than K jobs are visible
        padding = self.K - len(sorted_jobs)
        obs.extend([0.0, 0.0] * padding)

        # Node-level metrics
        obs.append(node.cpu_utilization)
        obs.append(node.memory_utilization)
        obs.append(node.network_latency / self._disturbances.latency_max)

        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def _build_info(self) -> Dict[str, Any]:
        return {
            "step": self._current_step,
            "time": self._current_time,
            "total_jobs_completed": self._total_jobs_completed,
            "queue_lengths": [n.queue_length for n in self._nodes],
            "cpu_utilizations": [n.cpu_utilization for n in self._nodes],
            "latencies": [n.network_latency for n in self._nodes],
            "machine_statuses": [
                [m.status for m in n.machines] for n in self._nodes
            ],
        }

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def render(self) -> str:
        """Return a human-readable summary of the current environment state."""
        lines = [
            f"Step {self._current_step}/{self.max_steps}  "
            f"Time={self._current_time:.1f}  "
            f"Jobs completed={self._total_jobs_completed}"
        ]
        for node in self._nodes:
            statuses = node.get_machine_statuses()
            status_str = " ".join(["I", "B", "F"][s] for s in statuses)
            lines.append(
                f"  Node {node.node_id}: machines=[{status_str}]  "
                f"queue={node.queue_length}  "
                f"cpu={node.cpu_utilization:.2f}  "
                f"lat={node.network_latency:.1f}ms"
            )
        return "\n".join(lines)

    @property
    def nodes(self) -> List[EdgeNode]:
        """Read-only access to edge node objects (for inspection / tests)."""
        return self._nodes
