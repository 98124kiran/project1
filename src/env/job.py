"""
Job and Operation data structures for the manufacturing scheduling environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class OperationStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"


class JobStatus:
    WAITING = "waiting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    MIGRATED = "migrated"


@dataclass
class Operation:
    """A single processing step within a job."""

    op_id: int
    processing_time: float          # total time required on a machine (minutes)
    machine_type: int               # type of machine required (0-indexed)
    remaining_time: float = field(init=False)
    machine_assigned: Optional[int] = field(default=None, init=False)
    status: str = field(default=OperationStatus.PENDING, init=False)

    def __post_init__(self) -> None:
        self.remaining_time = self.processing_time

    def start(self, machine_id: int) -> None:
        self.machine_assigned = machine_id
        self.status = OperationStatus.PROCESSING

    def tick(self, dt: float) -> bool:
        """Advance one time step. Returns True if operation just completed."""
        if self.status != OperationStatus.PROCESSING:
            return False
        self.remaining_time = max(0.0, self.remaining_time - dt)
        if self.remaining_time == 0.0:
            self.status = OperationStatus.COMPLETED
            return True
        return False

    def reset(self) -> None:
        self.remaining_time = self.processing_time
        self.machine_assigned = None
        self.status = OperationStatus.PENDING


@dataclass
class Job:
    """A job consisting of one or more sequential operations."""

    job_id: int
    operations: List[Operation]
    deadline: float                 # absolute deadline (time units from env start)
    arrival_time: float             # time step at which job enters the system
    priority: int = 1               # 1 = normal, 2 = urgent
    assigned_node: Optional[int] = field(default=None, init=False)
    current_op_index: int = field(default=0, init=False)
    status: str = field(default=JobStatus.WAITING, init=False)
    completion_time: Optional[float] = field(default=None, init=False)
    defer_until: float = field(default=0.0, init=False)

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def current_operation(self) -> Optional[Operation]:
        if self.current_op_index < len(self.operations):
            return self.operations[self.current_op_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_op_index >= len(self.operations)

    @property
    def total_remaining_time(self) -> float:
        return sum(op.remaining_time for op in self.operations[self.current_op_index:])

    def slack(self, current_time: float) -> float:
        """
        Laxity: how much extra time remains beyond the minimum needed to finish.

        slack = (deadline - current_time) - total_remaining_processing_time

        A negative value means the job cannot meet its deadline even if it
        starts all remaining operations immediately.
        """
        return (self.deadline - current_time) - self.total_remaining_time

    # ------------------------------------------------------------------ #
    # Mutations                                                            #
    # ------------------------------------------------------------------ #

    def advance_operation(self) -> bool:
        """Move to the next operation. Returns True if the job is now complete."""
        self.current_op_index += 1
        if self.is_complete:
            self.status = JobStatus.COMPLETED
            return True
        return False

    def reset(self) -> None:
        for op in self.operations:
            op.reset()
        self.current_op_index = 0
        self.assigned_node = None
        self.status = JobStatus.WAITING
        self.completion_time = None
        self.defer_until = 0.0
