"""
Workload generator for the manufacturing scheduling environment.

Generates synthetic job streams with configurable arrival rates,
processing time distributions, deadline tightness, and job sizes.
"""

from __future__ import annotations

import itertools
from typing import List, Optional

import numpy as np

from src.env.job import Job, Operation


class WorkloadGenerator:
    """
    Generates a stream of *Job* objects for the simulator.

    Parameters
    ----------
    lambda_job : float
        Mean number of normal jobs arriving per time step (Poisson rate).
    min_processing_time : float
        Minimum processing time for a single operation (time units).
    max_processing_time : float
        Maximum processing time for a single operation (time units).
    min_deadline_slack : float
        Minimum slack added on top of total processing time for deadline.
    max_deadline_slack : float
        Maximum slack added on top of total processing time for deadline.
    min_ops : int
        Minimum number of operations per job.
    max_ops : int
        Maximum number of operations per job.
    num_machine_types : int
        Number of distinct machine types available.
    rng : np.random.Generator, optional
        Shared random number generator for reproducibility.
    """

    def __init__(
        self,
        lambda_job: float = 0.5,
        min_processing_time: float = 5.0,
        max_processing_time: float = 30.0,
        min_deadline_slack: float = 20.0,
        max_deadline_slack: float = 100.0,
        min_ops: int = 1,
        max_ops: int = 3,
        num_machine_types: int = 3,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.lambda_job = lambda_job
        self.min_processing_time = min_processing_time
        self.max_processing_time = max_processing_time
        self.min_deadline_slack = min_deadline_slack
        self.max_deadline_slack = max_deadline_slack
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.num_machine_types = num_machine_types
        self.rng = rng or np.random.default_rng()

        self._job_counter = itertools.count()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def step(self, current_time: float, priority: int = 1) -> List[Job]:
        """
        Sample jobs that arrive this time step.

        Parameters
        ----------
        current_time : float
            The current simulation time (used to set arrival_time and deadline).
        priority : int
            Priority to assign to all jobs in this batch (1=normal, 2=urgent).

        Returns
        -------
        List[Job]
            Zero or more new Job objects.
        """
        num_arrivals = self.rng.poisson(self.lambda_job)
        return [self._make_job(current_time, priority) for _ in range(num_arrivals)]

    def step_urgent(self, current_time: float) -> Job:
        """
        Generate a single urgent job (priority=2, tight deadline).

        Called by the environment when the disturbance generator injects an
        urgent job into a node.
        """
        return self._make_job(current_time, priority=2, tight=True)

    def reset(self) -> None:
        """Reset the job ID counter (call alongside env.reset())."""
        self._job_counter = itertools.count()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _make_job(
        self, current_time: float, priority: int = 1, tight: bool = False
    ) -> Job:
        job_id = next(self._job_counter)
        num_ops = int(self.rng.integers(self.min_ops, self.max_ops + 1))
        operations = [self._make_operation(i) for i in range(num_ops)]

        total_proc_time = sum(op.processing_time for op in operations)

        if tight:
            slack = self.rng.uniform(
                self.min_deadline_slack, self.min_deadline_slack * 2
            )
        else:
            slack = self.rng.uniform(self.min_deadline_slack, self.max_deadline_slack)

        deadline = current_time + total_proc_time + slack

        return Job(
            job_id=job_id,
            operations=operations,
            deadline=deadline,
            arrival_time=current_time,
            priority=priority,
        )

    def _make_operation(self, op_id: int) -> Operation:
        processing_time = float(
            self.rng.uniform(self.min_processing_time, self.max_processing_time)
        )
        machine_type = int(self.rng.integers(0, self.num_machine_types))
        return Operation(
            op_id=op_id,
            processing_time=processing_time,
            machine_type=machine_type,
        )
