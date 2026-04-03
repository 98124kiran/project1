"""
Dynamic disturbance generator for the manufacturing scheduling environment.

Implements stochastic events as described in the Dec-POMDP formulation:
  - Machine failures    : Poisson process with rate λ_fail
  - Machine repairs     : Exponential distribution with rate λ_repair (handled by Machine.tick)
  - Urgent job injection: Poisson process with rate λ_urgent
  - Network latency     : Random walk
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from src.env.edge_node import EdgeNode


class DisturbanceGenerator:
    """
    Generates stochastic disturbances and applies them to edge nodes each step.

    Parameters
    ----------
    lambda_fail : float
        Per-machine failure probability per time step (Poisson rate).
    mean_repair_time : float
        Mean repair time (time units). Repair duration ~ Exp(1/mean_repair_time).
    lambda_urgent : float
        Probability of injecting an urgent job into a random node per step.
    latency_sigma : float
        Std-dev of the random-walk increment applied to network_latency each step.
    latency_min : float
        Minimum latency clamp (ms).
    latency_max : float
        Maximum latency clamp (ms).
    rng : np.random.Generator, optional
        Shared random number generator for reproducibility.
    """

    def __init__(
        self,
        lambda_fail: float = 0.01,
        mean_repair_time: float = 20.0,
        lambda_urgent: float = 0.05,
        latency_sigma: float = 0.5,
        latency_min: float = 0.5,
        latency_max: float = 50.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.lambda_fail = lambda_fail
        self.mean_repair_time = mean_repair_time
        self.lambda_urgent = lambda_urgent
        self.latency_sigma = latency_sigma
        self.latency_min = latency_min
        self.latency_max = latency_max
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Per-step application                                                 #
    # ------------------------------------------------------------------ #

    def apply(
        self, nodes: List["EdgeNode"], current_time: float
    ) -> Tuple[int, List[int]]:
        """
        Apply all disturbances to *nodes* for the current time step.

        Returns
        -------
        num_failures : int
            Total number of machine failures injected this step.
        urgent_node_ids : List[int]
            Node IDs that received an urgent job injection (caller must
            create and enqueue the actual job).
        """
        num_failures = self._apply_machine_failures(nodes)
        urgent_node_ids = self._apply_urgent_injections(nodes)
        self._apply_latency_walk(nodes)
        return num_failures, urgent_node_ids

    # ------------------------------------------------------------------ #
    # Individual disturbance methods                                       #
    # ------------------------------------------------------------------ #

    def _apply_machine_failures(self, nodes: List["EdgeNode"]) -> int:
        """Randomly fail machines across all nodes using a Poisson process."""
        num_failures = 0
        for node in nodes:
            for machine in node.machines:
                if machine.is_failed:
                    continue
                # Bernoulli approximation of Poisson: P(failure) = lambda_fail * dt
                if self.rng.random() < self.lambda_fail:
                    repair_time = self.rng.exponential(self.mean_repair_time)
                    machine.fail(repair_time)
                    num_failures += 1
        return num_failures

    def _apply_urgent_injections(self, nodes: List["EdgeNode"]) -> List[int]:
        """
        Decide which nodes receive urgent job arrivals this step.

        Returns a list of node IDs that should receive an urgent job.
        The caller (ManufacturingEnv) is responsible for generating and
        enqueuing the actual Job object.
        """
        urgent_node_ids: List[int] = []
        for node in nodes:
            if self.rng.random() < self.lambda_urgent:
                urgent_node_ids.append(node.node_id)
        return urgent_node_ids

    def _apply_latency_walk(self, nodes: List["EdgeNode"]) -> None:
        """Apply a Gaussian random walk to each node's network latency."""
        for node in nodes:
            delta = self.rng.normal(0.0, self.latency_sigma)
            node.network_latency = float(
                np.clip(node.network_latency + delta, self.latency_min, self.latency_max)
            )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def sample_repair_time(self) -> float:
        """Sample a repair duration from Exp(1/mean_repair_time)."""
        return float(self.rng.exponential(self.mean_repair_time))
