"""
Edge computing environment for smart manufacturing scheduling.

Observation space (per agent / node):
  - Own node features          : EdgeNode.FEATURE_DIM = 5
  - Global task queue features : max_queue_size * Task.FEATURE_DIM = 20 * 10 = 200
  - All nodes features         : num_nodes * EdgeNode.FEATURE_DIM = 5 * 5 = 25
  Total: 5 + 200 + 25 = 230 (padded with zeros when queue is not full)

Action space (per agent):
  Discrete action: which task (by queue position 0..max_queue_size-1) to schedule
  on this node, or action = max_queue_size meaning "do nothing".
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .task import Task, TaskStatus, make_task
from .edge_node import EdgeNode
from src.utils.config import Config


class EdgeComputingEnv:
    """
    Multi-agent environment where each edge node acts as an independent agent.

    Follows a gym-like interface but returns dicts keyed by agent index.
    """

    # Reward shaping weights
    W_COMPLETE = 2.0        # reward per completed task (scaled by priority)
    W_DEADLINE_MISS = -3.0  # penalty for deadline miss (scaled by priority)
    W_DROP = -1.0           # penalty for queue overflow drop
    W_UTIL = 0.1            # reward for healthy utilisation (0.4-0.8)
    W_INVALID = -0.5        # penalty for invalid action

    def __init__(self, config: Config, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

        self.num_nodes = config.env.num_edge_nodes
        self.max_queue = config.env.max_queue_size
        self.max_steps = config.env.max_steps

        self._task_counter = 0
        self.current_step = 0
        self.nodes: List[EdgeNode] = []
        self.task_queue: List[Task] = []    # global pending queue

        self._build_nodes()

        # computed once for efficiency
        self._obs_dim = self._compute_obs_dim()
        self._act_dim = self.max_queue + 1   # 0..max_queue-1 = pick task, max_queue = idle

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self) -> Dict[int, np.ndarray]:
        self._task_counter = 0
        self.current_step = 0
        self.task_queue = []
        self._build_nodes()
        self._spawn_tasks()
        return self._get_observations()

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, dict]:
        """
        Execute one time step.

        actions: {agent_id: action_index}
        Returns: observations, rewards, done, info
        """
        rewards: Dict[int, float] = {i: 0.0 for i in range(self.num_nodes)}
        info: Dict = {}

        # 1. Execute scheduling actions
        for agent_id, action in actions.items():
            node = self.nodes[agent_id]
            if action < len(self.task_queue):        # schedule a task
                task = self.task_queue[action]
                if node.can_accept(task):
                    node.assign(task, self.current_step)
                    self.task_queue.pop(action)
                    # small bonus proportional to priority
                    rewards[agent_id] += 0.1 * task.priority
                else:
                    rewards[agent_id] += self.W_INVALID  # can't fit → invalid action
            # else: idle action, no immediate reward

        # 2. Tick all nodes (advance running tasks)
        for node in self.nodes:
            if not node.is_failed:
                finished = node.tick(self.current_step)
                for task in finished:
                    if task.remaining_deadline >= 0:
                        rewards[node.node_id] += self.W_COMPLETE * task.priority
                    else:
                        rewards[node.node_id] += self.W_DEADLINE_MISS * task.priority

        # 3. Tick queued tasks (advance deadlines)
        expired = []
        for task in self.task_queue:
            task.tick()
            if task.is_deadline_violated:
                task.status = TaskStatus.DEADLINE_MISSED
                expired.append(task)
        for task in expired:
            self.task_queue.remove(task)
            # broadcast deadline-miss penalty to all agents
            for i in range(self.num_nodes):
                rewards[i] += self.W_DEADLINE_MISS * task.priority / self.num_nodes

        # 4. Node failures / recoveries
        for node in self.nodes:
            if not node.is_failed and self.rng.random() < self.config.env.node_failure_prob:
                dropped = node.fail()
                for task in dropped:
                    for i in range(self.num_nodes):
                        rewards[i] += self.W_DROP / self.num_nodes
            elif node.is_failed and self.rng.random() < self.config.env.node_recovery_prob:
                node.recover()

        # 5. Dynamic background load fluctuation
        for node in self.nodes:
            node.bg_cpu_load = max(0.0, node.bg_cpu_load + float(
                self.rng.normal(0, self.config.env.dynamic_load_std * node.cpu_capacity)
            ))
            node.bg_cpu_load = min(node.bg_cpu_load, node.cpu_capacity * 0.4)
            node.bg_mem_load = max(0.0, node.bg_mem_load + float(
                self.rng.normal(0, self.config.env.dynamic_load_std * node.memory_capacity)
            ))
            node.bg_mem_load = min(node.bg_mem_load, node.memory_capacity * 0.4)

        # 6. Utilisation reward
        for node in self.nodes:
            util = node.cpu_utilization
            if 0.4 <= util <= 0.8:
                rewards[node.node_id] += self.W_UTIL

        # 7. Queue overflow: drop oldest low-priority tasks
        while len(self.task_queue) > self.max_queue:
            dropped_task = self.task_queue.pop(0)  # drop oldest
            dropped_task.status = TaskStatus.DROPPED
            for i in range(self.num_nodes):
                rewards[i] += self.W_DROP / self.num_nodes

        # 8. Spawn new tasks
        self._spawn_tasks()

        self.current_step += 1
        done = self.current_step >= self.max_steps

        info = self._collect_info()
        return self._get_observations(), rewards, done, info

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        return self._act_dim

    @property
    def n_agents(self) -> int:
        return self.num_nodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_nodes(self) -> None:
        nc = self.config.node
        self.nodes = [
            EdgeNode(
                node_id=i,
                cpu_capacity=nc.cpu_capacities[i],
                memory_capacity=nc.memory_capacities[i],
                bandwidth=nc.bandwidth[i],
            )
            for i in range(self.num_nodes)
        ]

    def _spawn_tasks(self) -> None:
        """Poisson-distributed task arrivals."""
        n_arrivals = int(self.rng.poisson(self.config.env.task_arrival_rate *
                                          self.config.env.max_tasks_per_step))
        for _ in range(n_arrivals):
            if len(self.task_queue) >= self.max_queue:
                break
            t_type = int(self.rng.integers(0, self.config.env.num_task_types))
            task_cfg = self.config.task.task_types[t_type]
            task = make_task(self._task_counter, t_type, task_cfg,
                             self.current_step, self.rng)
            self._task_counter += 1
            self.task_queue.append(task)

    def _compute_obs_dim(self) -> int:
        return (
            EdgeNode.FEATURE_DIM                            # own node
            + self.max_queue * Task.FEATURE_DIM             # full queue
            + self.num_nodes * EdgeNode.FEATURE_DIM         # all nodes
        )

    def _get_observations(self) -> Dict[int, np.ndarray]:
        # Build padded queue feature matrix once
        queue_features = np.zeros(
            (self.max_queue, Task.FEATURE_DIM), dtype=np.float32
        )
        for i, task in enumerate(self.task_queue[:self.max_queue]):
            queue_features[i] = task.feature_vector()
        queue_flat = queue_features.flatten()

        all_node_features = np.concatenate(
            [n.feature_vector() for n in self.nodes]
        )

        obs = {}
        for node in self.nodes:
            own = node.feature_vector()
            obs[node.node_id] = np.concatenate([own, queue_flat, all_node_features])
        return obs

    def _collect_info(self) -> dict:
        total_running = sum(len(n.running_tasks) for n in self.nodes)
        total_completed = sum(n.completed_count for n in self.nodes)
        total_dropped = sum(n.dropped_count for n in self.nodes)
        utilizations = [n.cpu_utilization for n in self.nodes]
        return {
            "step": self.current_step,
            "queue_length": len(self.task_queue),
            "running_tasks": total_running,
            "completed_tasks": total_completed,
            "dropped_tasks": total_dropped,
            "node_utilizations": utilizations,
            "failed_nodes": [n.node_id for n in self.nodes if n.is_failed],
        }

    def get_valid_actions(self, agent_id: int) -> List[int]:
        """
        Return list of valid action indices for this agent at current state.
        An action is valid if the node can accept the task at that queue index,
        or it is the idle action.
        """
        node = self.nodes[agent_id]
        valid = []
        for i, task in enumerate(self.task_queue[:self.max_queue]):
            if node.can_accept(task):
                valid.append(i)
        valid.append(self.max_queue)   # idle is always valid
        return valid
