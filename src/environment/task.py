"""
Task definition for smart manufacturing edge computing.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np


class TaskStatus(IntEnum):
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    DROPPED = 3        # queue overflow or rejected
    DEADLINE_MISSED = 4


@dataclass
class Task:
    """Represents a manufacturing task to be scheduled on an edge node."""

    task_id: int
    task_type: int          # index into task_types config
    type_name: str

    cpu_demand: float       # CPU cores required
    memory_demand: float    # GB RAM required
    size_mb: float          # data size in MB (affects transfer time)
    deadline: float         # remaining time steps until deadline
    priority: int           # higher = more critical (1-5)
    arrival_step: int       # environment step at which task arrived

    # assigned fields (set when scheduled)
    assigned_node: Optional[int] = None
    start_step: Optional[int] = None
    finish_step: Optional[int] = None
    processing_time: float = 0.0    # computed at assignment time
    status: TaskStatus = TaskStatus.PENDING

    def __post_init__(self):
        self.remaining_deadline = float(self.deadline)

    def tick(self) -> None:
        """Advance one time slot."""
        self.remaining_deadline -= 1.0
        if self.status == TaskStatus.RUNNING:
            self.processing_time = max(0.0, self.processing_time - 1.0)

    @property
    def is_deadline_violated(self) -> bool:
        return self.remaining_deadline <= 0 and self.status not in (
            TaskStatus.COMPLETED, TaskStatus.DROPPED
        )

    @property
    def urgency(self) -> float:
        """Normalised urgency score in [0, 1]; higher = more urgent."""
        if self.remaining_deadline <= 0:
            return 1.0
        return 1.0 / (1.0 + self.remaining_deadline)

    def feature_vector(self, max_cpu: float = 20.0, max_mem: float = 48.0,
                       max_deadline: float = 30.0, max_size: float = 200.0) -> np.ndarray:
        """
        Returns a normalised feature vector for use as NN input.
        [task_type_onehot(4), cpu, memory, deadline, priority, size, urgency]
        """
        num_types = 4
        onehot = np.zeros(num_types, dtype=np.float32)
        onehot[min(self.task_type, num_types - 1)] = 1.0
        return np.concatenate([
            onehot,
            [
                self.cpu_demand / max_cpu,
                self.memory_demand / max_mem,
                self.remaining_deadline / max_deadline,
                self.priority / 5.0,
                self.size_mb / max_size,
                self.urgency,
            ],
        ]).astype(np.float32)

    # Dimension of feature vector
    FEATURE_DIM: int = 4 + 6  # 10


def make_task(task_id: int, task_type: int, task_cfg: dict,
              arrival_step: int, rng: np.random.Generator) -> Task:
    """Factory: sample a task from a task type config dict."""

    def sample(lo_hi):
        return float(rng.uniform(lo_hi[0], lo_hi[1]))

    return Task(
        task_id=task_id,
        task_type=task_type,
        type_name=task_cfg["name"],
        cpu_demand=sample(task_cfg["cpu"]),
        memory_demand=sample(task_cfg["memory"]),
        deadline=sample(task_cfg["deadline"]),
        priority=int(task_cfg["priority"]),
        size_mb=sample(task_cfg["size_mb"]),
        arrival_step=arrival_step,
    )
