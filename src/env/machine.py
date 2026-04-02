"""
Machine model for the manufacturing scheduling environment.
"""

from __future__ import annotations

from typing import Optional

from src.env.job import Job, Operation, OperationStatus


class MachineStatus:
    IDLE = 0
    BUSY = 1
    FAILED = 2


class Machine:
    """
    Represents a physical machine at an edge node.

    A machine can process one operation at a time.  It may fail randomly
    (handled externally by the disturbance generator) and requires repair
    before resuming work.
    """

    def __init__(self, machine_id: int, machine_type: int, node_id: int) -> None:
        self.machine_id = machine_id
        self.machine_type = machine_type
        self.node_id = node_id

        self.status: int = MachineStatus.IDLE
        self.current_job: Optional[Job] = None
        self.current_operation: Optional[Operation] = None

        # Repair countdown (time steps remaining until machine is back online)
        self.repair_time_remaining: float = 0.0

        # Utilization tracking
        self._busy_time: float = 0.0
        self._total_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def is_idle(self) -> bool:
        return self.status == MachineStatus.IDLE

    @property
    def is_busy(self) -> bool:
        return self.status == MachineStatus.BUSY

    @property
    def is_failed(self) -> bool:
        return self.status == MachineStatus.FAILED

    @property
    def utilization(self) -> float:
        """Fraction of time the machine was busy (0.0–1.0)."""
        if self._total_time == 0.0:
            return 0.0
        return self._busy_time / self._total_time

    # ------------------------------------------------------------------ #
    # Actions                                                              #
    # ------------------------------------------------------------------ #

    def assign(self, job: Job, operation: Operation) -> bool:
        """
        Assign an operation from *job* to this machine.

        Returns False if the machine is not idle or the machine type does
        not match the operation's required type.
        """
        if not self.is_idle:
            return False
        if operation.machine_type != self.machine_type:
            return False
        self.current_job = job
        self.current_operation = operation
        self.status = MachineStatus.BUSY
        operation.start(self.machine_id)
        return True

    def fail(self, repair_time: float) -> None:
        """Mark this machine as failed; interrupt any in-progress operation."""
        self.status = MachineStatus.FAILED
        self.repair_time_remaining = repair_time
        if self.current_operation is not None:
            # Reset the operation so it can be rescheduled
            self.current_operation.reset()
        self.current_job = None
        self.current_operation = None

    # ------------------------------------------------------------------ #
    # Time-step update                                                     #
    # ------------------------------------------------------------------ #

    def tick(self, dt: float) -> Optional[Job]:
        """
        Advance machine state by *dt* time units.

        Returns the completed *Job* if the current operation finished this
        step and the entire job is now complete; otherwise returns None.
        """
        self._total_time += dt

        if self.is_failed:
            self.repair_time_remaining = max(0.0, self.repair_time_remaining - dt)
            if self.repair_time_remaining == 0.0:
                self.status = MachineStatus.IDLE
            return None

        if self.is_busy:
            self._busy_time += dt
            op_done = self.current_operation.tick(dt)
            if op_done:
                completed_job = self.current_job
                job_done = completed_job.advance_operation()
                self.current_job = None
                self.current_operation = None
                self.status = MachineStatus.IDLE
                if job_done:
                    return completed_job
        return None

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self.status = MachineStatus.IDLE
        self.current_job = None
        self.current_operation = None
        self.repair_time_remaining = 0.0
        self._busy_time = 0.0
        self._total_time = 0.0
