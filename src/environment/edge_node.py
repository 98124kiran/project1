"""
Edge node model for smart manufacturing.
"""

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .task import Task


@dataclass
class EdgeNode:
    """
    Represents an edge computing node in the manufacturing plant.
    Each node has a fixed CPU / memory capacity and may fail dynamically.
    """

    node_id: int
    cpu_capacity: float     # total CPU cores
    memory_capacity: float  # total RAM in GB
    bandwidth: float        # uplink to cloud in Mbps

    # runtime state
    cpu_used: float = 0.0
    memory_used: float = 0.0
    is_failed: bool = False
    running_tasks: List["Task"] = field(default_factory=list)
    completed_count: int = 0
    dropped_count: int = 0

    # background load (non-schedulable processes)
    bg_cpu_load: float = 0.0
    bg_mem_load: float = 0.0

    @property
    def cpu_available(self) -> float:
        return max(0.0, self.cpu_capacity - self.cpu_used - self.bg_cpu_load)

    @property
    def memory_available(self) -> float:
        return max(0.0, self.memory_capacity - self.memory_used - self.bg_mem_load)

    @property
    def cpu_utilization(self) -> float:
        return (self.cpu_used + self.bg_cpu_load) / self.cpu_capacity

    @property
    def memory_utilization(self) -> float:
        return (self.memory_used + self.bg_mem_load) / self.memory_capacity

    def can_accept(self, task: "Task") -> bool:
        if self.is_failed:
            return False
        return (task.cpu_demand <= self.cpu_available and
                task.memory_demand <= self.memory_available)

    def assign(self, task: "Task", current_step: int) -> float:
        """
        Assign a task to this node. Returns estimated processing time.
        Processing time is proportional to CPU demand and bandwidth (for data transfer).
        """
        from .task import TaskStatus
        transfer_time = task.size_mb / self.bandwidth  # seconds (~ time slots)
        compute_time = task.cpu_demand / max(self.cpu_available, 0.1)
        proc_time = max(1.0, transfer_time + compute_time)

        task.assigned_node = self.node_id
        task.start_step = current_step
        task.processing_time = proc_time
        task.status = TaskStatus.RUNNING

        self.cpu_used += task.cpu_demand
        self.memory_used += task.memory_demand
        self.running_tasks.append(task)
        return proc_time

    def tick(self, current_step: int) -> List["Task"]:
        """
        Advance one time step. Returns list of newly completed tasks.
        """
        from .task import TaskStatus
        finished = []
        still_running = []
        for task in self.running_tasks:
            task.processing_time = max(0.0, task.processing_time - 1.0)
            if task.processing_time <= 0:
                task.status = TaskStatus.COMPLETED
                task.finish_step = current_step
                self.cpu_used = max(0.0, self.cpu_used - task.cpu_demand)
                self.memory_used = max(0.0, self.memory_used - task.memory_demand)
                self.completed_count += 1
                finished.append(task)
            else:
                still_running.append(task)
        self.running_tasks = still_running
        return finished

    def fail(self) -> List["Task"]:
        """Simulate node failure; running tasks are dropped."""
        from .task import TaskStatus
        dropped = []
        for task in self.running_tasks:
            task.status = TaskStatus.DROPPED
            self.dropped_count += 1
            dropped.append(task)
        self.running_tasks = []
        self.cpu_used = 0.0
        self.memory_used = 0.0
        self.is_failed = True
        return dropped

    def recover(self) -> None:
        self.is_failed = False

    def feature_vector(self) -> np.ndarray:
        """
        Normalised state vector for the node.
        [cpu_util, mem_util, bw_norm, is_failed, queue_len_norm]
        """
        return np.array([
            self.cpu_utilization,
            self.memory_utilization,
            min(self.bandwidth / 200.0, 1.0),
            float(self.is_failed),
            min(len(self.running_tasks) / 10.0, 1.0),
        ], dtype=np.float32)

    # Dimension of feature vector
    FEATURE_DIM: int = 5
