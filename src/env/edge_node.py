"""
Edge node model for the manufacturing scheduling environment.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np

from src.env.job import Job, JobStatus
from src.env.machine import Machine, MachineStatus


class EdgeNode:
    """
    An edge computing node that hosts a set of machines and manages a job queue.

    Each EdgeNode corresponds to one agent in the multi-agent setting.
    It maintains its own job queue and resource utilisation metrics.
    """

    def __init__(
        self,
        node_id: int,
        num_machines: int,
        num_machine_types: int,
        max_queue_length: int = 20,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.node_id = node_id
        self.max_queue_length = max_queue_length
        self.rng = rng or np.random.default_rng()

        # Machines: distribute types round-robin so every type is represented
        self.machines: List[Machine] = [
            Machine(
                machine_id=i,
                machine_type=i % num_machine_types,
                node_id=node_id,
            )
            for i in range(num_machines)
        ]

        # Job queue (FIFO with priority tie-breaking handled externally)
        self.job_queue: Deque[Job] = deque()

        # Node-level resource metrics
        self.cpu_utilization: float = 0.0      # [0, 1]
        self.memory_utilization: float = 0.0   # [0, 1]
        self.network_latency: float = 1.0      # milliseconds

        # Per-step accounting
        self._jobs_completed_this_step: int = 0
        self._deadline_violations_this_step: int = 0

    # ------------------------------------------------------------------ #
    # Queue helpers                                                        #
    # ------------------------------------------------------------------ #

    @property
    def queue_length(self) -> int:
        return len(self.job_queue)

    @property
    def is_queue_full(self) -> bool:
        return len(self.job_queue) >= self.max_queue_length

    def enqueue(self, job: Job) -> bool:
        """Add *job* to the queue. Returns False if queue is full."""
        if self.is_queue_full:
            return False
        job.assigned_node = self.node_id
        self.job_queue.append(job)
        return True

    def dequeue(self) -> Optional[Job]:
        """Remove and return the highest-priority job from the queue."""
        if not self.job_queue:
            return None
        # Sort by priority (desc) then by earliest deadline first
        sorted_queue = sorted(
            self.job_queue,
            key=lambda j: (-j.priority, j.deadline),
        )
        best = sorted_queue[0]
        self.job_queue.remove(best)
        return best

    def peek(self) -> Optional[Job]:
        """Return the highest-priority job without removing it."""
        if not self.job_queue:
            return None
        return min(self.job_queue, key=lambda j: (-j.priority, j.deadline))

    # ------------------------------------------------------------------ #
    # Actions                                                              #
    # ------------------------------------------------------------------ #

    def try_assign_to_machine(self, machine_index: int) -> bool:
        """
        Attempt to assign the queue head to machine *machine_index*.

        Returns True on success.
        """
        if not self.job_queue:
            return False
        machine = self.machines[machine_index]
        if not machine.is_idle:
            return False
        job = self.peek()
        if job is None:
            return False
        op = job.current_operation
        if op is None:
            return False
        if op.machine_type != machine.machine_type:
            return False
        self.dequeue()
        job.status = JobStatus.PROCESSING
        machine.assign(job, op)
        return True

    def try_assign_any_idle(self) -> int:
        """
        Greedily assign queued jobs to any compatible idle machine.

        Returns the number of successful assignments.
        """
        assigned = 0
        for machine in self.machines:
            if not machine.is_idle or not self.job_queue:
                continue
            job = self.peek()
            if job is None:
                break
            op = job.current_operation
            if op is None:
                continue
            if op.machine_type == machine.machine_type:
                self.dequeue()
                job.status = JobStatus.PROCESSING
                machine.assign(job, op)
                assigned += 1
        return assigned

    # ------------------------------------------------------------------ #
    # Time-step update                                                     #
    # ------------------------------------------------------------------ #

    def tick(self, dt: float, current_time: float) -> None:
        """Advance all machines and update resource metrics."""
        self._jobs_completed_this_step = 0
        self._deadline_violations_this_step = 0

        for machine in self.machines:
            completed_job = machine.tick(dt)
            if completed_job is not None:
                completed_job.completion_time = current_time
                self._jobs_completed_this_step += 1
                if current_time > completed_job.deadline:
                    self._deadline_violations_this_step += 1
                # If job has more operations, re-enqueue for next operation
                if not completed_job.is_complete:
                    completed_job.status = JobStatus.WAITING
                    self.enqueue(completed_job)

        # Simulate CPU utilisation as fraction of busy machines
        busy = sum(1 for m in self.machines if m.is_busy)
        self.cpu_utilization = busy / len(self.machines)

        # Memory utilisation proxy: queue fill ratio
        self.memory_utilization = self.queue_length / self.max_queue_length

    # ------------------------------------------------------------------ #
    # Observation                                                          #
    # ------------------------------------------------------------------ #

    def get_machine_statuses(self) -> List[int]:
        return [m.status for m in self.machines]

    def get_machine_utilizations(self) -> List[float]:
        return [m.utilization for m in self.machines]

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        for machine in self.machines:
            machine.reset()
        self.job_queue.clear()
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.network_latency = 1.0
        self._jobs_completed_this_step = 0
        self._deadline_violations_this_step = 0
