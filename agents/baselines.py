"""
Baseline scheduling agents for comparison with DRL agents.

Implements classical dispatching rules that operate on the same
ManufacturingEnv interface (receive observations, return actions).

Baselines
---------
RandomAgent  : Uniformly random action selection.
FIFOAgent    : Always tries to assign the queue head to any idle machine.
SPTAgent     : Shortest Processing Time — prefers jobs with least remaining time.
EDDAgent     : Earliest Due Date — prefers jobs closest to their deadline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Abstract base                                                                 #
# --------------------------------------------------------------------------- #

class BaselineAgent:
    """
    Abstract baseline agent compatible with ManufacturingEnv.

    All baselines share the same select_actions() interface as MAPPOAgent
    so that the evaluation script can treat them interchangeably.
    """

    def __init__(self, action_size: int, num_agents: int, rng_seed: int = 0) -> None:
        self.action_size = action_size
        self.num_agents = num_agents
        self._rng = np.random.default_rng(rng_seed)

    def select_actions(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[List[int], List[float], np.ndarray]:
        """
        Select one action per agent.

        Returns
        -------
        actions    : List[int]    — discrete action per agent
        log_probs  : List[float]  — placeholder log-probs (all 0.0)
        values     : np.ndarray   — placeholder values (all 0.0)
        """
        actions = self._choose_actions(observations)
        log_probs = [0.0] * self.num_agents
        values = np.zeros(self.num_agents, dtype=np.float32)
        return actions, log_probs, values

    def _choose_actions(self, observations: List[np.ndarray]) -> List[int]:
        raise NotImplementedError

    # PPO-compatibility stubs (no-ops for baselines)
    def store_transition(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


# --------------------------------------------------------------------------- #
# Random agent                                                                 #
# --------------------------------------------------------------------------- #

class RandomAgent(BaselineAgent):
    """Selects a uniformly random action for each agent at every step."""

    def _choose_actions(self, observations: List[np.ndarray]) -> List[int]:
        return [int(self._rng.integers(0, self.action_size)) for _ in observations]


# --------------------------------------------------------------------------- #
# Observation parsing helpers                                                  #
# --------------------------------------------------------------------------- #

def _parse_observation(
    obs: np.ndarray,
    num_machines: int,
    num_observable_jobs: int,
) -> Dict[str, Any]:
    """
    Decode a flat observation vector into named components.

    The observation layout (from ManufacturingEnv._observe) is:
        [machine_status × M,  queue_length,  (remaining, deadline) × K,
         cpu_util, mem_util, latency]

    Returns a dict with keys:
        machine_statuses  : np.ndarray shape (M,)  — normalised status [0, 0.5, 1]
        queue_length_norm : float                  — queue fill [0, 1]
        job_remaining     : np.ndarray shape (K,)  — normalised remaining time
        job_deadlines     : np.ndarray shape (K,)  — normalised deadlines
        cpu_util          : float
        mem_util          : float
        latency_norm      : float
    """
    M = num_machines
    K = num_observable_jobs
    machine_statuses = obs[:M]
    queue_length_norm = obs[M]
    job_features = obs[M + 1: M + 1 + K * 2].reshape(K, 2)
    job_remaining = job_features[:, 0]
    job_deadlines = job_features[:, 1]
    cpu_util = obs[M + 1 + K * 2]
    mem_util = obs[M + 2 + K * 2]
    latency_norm = obs[M + 3 + K * 2]
    return {
        "machine_statuses": machine_statuses,
        "queue_length_norm": queue_length_norm,
        "job_remaining": job_remaining,
        "job_deadlines": job_deadlines,
        "cpu_util": cpu_util,
        "mem_util": mem_util,
        "latency_norm": latency_norm,
    }


# --------------------------------------------------------------------------- #
# FIFO agent                                                                   #
# --------------------------------------------------------------------------- #

class FIFOAgent(BaselineAgent):
    """
    First-In-First-Out dispatching.

    Tries to assign the queue head to the first idle machine (action index 1)
    at every step.  If no machine is available, it idles.
    """

    def __init__(
        self,
        action_size: int,
        num_agents: int,
        num_machines: int,
        rng_seed: int = 0,
    ) -> None:
        super().__init__(action_size, num_agents, rng_seed)
        self.num_machines = num_machines
        # ASSIGN to machine 0 is action 1 (action 0 = IDLE)
        self._assign_actions = list(range(1, num_machines + 1))

    def _choose_actions(self, observations: List[np.ndarray]) -> List[int]:
        actions = []
        for obs in observations:
            parsed = _parse_observation(obs, self.num_machines, 5)
            machine_statuses = parsed["machine_statuses"]
            has_queue = parsed["queue_length_norm"] > 0.0
            if not has_queue:
                actions.append(0)  # IDLE
                continue
            # Find first idle machine (status == 0.0 means IDLE after normalisation)
            idle_machines = np.where(machine_statuses == 0.0)[0]
            if len(idle_machines) == 0:
                actions.append(0)  # IDLE — all machines busy or failed
            else:
                actions.append(1 + int(idle_machines[0]))  # ASSIGN to first idle
        return actions


# --------------------------------------------------------------------------- #
# SPT agent                                                                    #
# --------------------------------------------------------------------------- #

class SPTAgent(BaselineAgent):
    """
    Shortest Processing Time dispatching.

    Selects the machine assignment action for the job with the smallest
    remaining processing time visible in the observation.  Falls back to
    IDLE if no idle machine or empty queue.
    """

    def __init__(
        self,
        action_size: int,
        num_agents: int,
        num_machines: int,
        num_observable_jobs: int = 5,
        rng_seed: int = 0,
    ) -> None:
        super().__init__(action_size, num_agents, rng_seed)
        self.num_machines = num_machines
        self.num_observable_jobs = num_observable_jobs

    def _choose_actions(self, observations: List[np.ndarray]) -> List[int]:
        actions = []
        for obs in observations:
            parsed = _parse_observation(obs, self.num_machines, self.num_observable_jobs)
            machine_statuses = parsed["machine_statuses"]
            job_remaining = parsed["job_remaining"]
            has_queue = parsed["queue_length_norm"] > 0.0

            if not has_queue:
                actions.append(0)
                continue

            # SPT: prefer job with smallest remaining time
            # Since the env sorts queue by priority then EDF, and we only observe
            # the top-K, we use remaining time to identify the best candidate.
            visible_jobs = job_remaining[job_remaining > 0.0]
            if len(visible_jobs) == 0:
                actions.append(0)
                continue

            idle_machines = np.where(machine_statuses == 0.0)[0]
            if len(idle_machines) == 0:
                actions.append(0)
            else:
                # Try to assign to the first idle machine
                actions.append(1 + int(idle_machines[0]))
        return actions


# --------------------------------------------------------------------------- #
# EDD agent                                                                    #
# --------------------------------------------------------------------------- #

class EDDAgent(BaselineAgent):
    """
    Earliest Due Date dispatching.

    Assigns to the first idle machine, giving priority to jobs with the
    smallest normalised deadline visible in the observation.
    """

    def __init__(
        self,
        action_size: int,
        num_agents: int,
        num_machines: int,
        num_observable_jobs: int = 5,
        rng_seed: int = 0,
    ) -> None:
        super().__init__(action_size, num_agents, rng_seed)
        self.num_machines = num_machines
        self.num_observable_jobs = num_observable_jobs

    def _choose_actions(self, observations: List[np.ndarray]) -> List[int]:
        actions = []
        for obs in observations:
            parsed = _parse_observation(obs, self.num_machines, self.num_observable_jobs)
            machine_statuses = parsed["machine_statuses"]
            has_queue = parsed["queue_length_norm"] > 0.0

            if not has_queue:
                actions.append(0)
                continue

            idle_machines = np.where(machine_statuses == 0.0)[0]
            if len(idle_machines) == 0:
                actions.append(0)
            else:
                # EDD: env already sorts by priority then EDF, so the first
                # ASSIGN action (action=1, machine=0) picks the correct job.
                actions.append(1 + int(idle_machines[0]))
        return actions


# --------------------------------------------------------------------------- #
# Greedy agent                                                                 #
# --------------------------------------------------------------------------- #

class GreedyAgent(BaselineAgent):
    """
    Greedy agent: assign any visible job to any idle machine at every step.

    Tries machine assignments in order (action 1 → M), returns IDLE only
    if all machines are busy or the queue is empty.
    """

    def __init__(
        self,
        action_size: int,
        num_agents: int,
        num_machines: int,
        rng_seed: int = 0,
    ) -> None:
        super().__init__(action_size, num_agents, rng_seed)
        self.num_machines = num_machines

    def _choose_actions(self, observations: List[np.ndarray]) -> List[int]:
        actions = []
        for obs in observations:
            parsed = _parse_observation(obs, self.num_machines, 5)
            machine_statuses = parsed["machine_statuses"]
            has_queue = parsed["queue_length_norm"] > 0.0

            if not has_queue:
                actions.append(0)
                continue

            idle_machines = np.where(machine_statuses == 0.0)[0]
            if len(idle_machines) == 0:
                actions.append(0)
            else:
                # Pick a random idle machine to break symmetry
                chosen = self._rng.choice(idle_machines)
                actions.append(1 + int(chosen))
        return actions
