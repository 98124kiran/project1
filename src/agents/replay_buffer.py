"""
Experience replay buffer for MADDPG.
Stores transitions as flat numpy arrays for memory efficiency.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """
    Centralised replay buffer shared across all agents.

    Each transition stores:
      obs_n     – observations for all agents  (n_agents, obs_dim)
      act_n     – one-hot actions for all agents (n_agents, act_dim)
      rew_n     – rewards for all agents         (n_agents,)
      next_obs_n – next observations             (n_agents, obs_dim)
      done      – terminal flag                  scalar
    """

    def __init__(self, capacity: int, n_agents: int,
                 obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ptr = 0
        self.size = 0

        self.obs_n = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.act_n = np.zeros((capacity, n_agents, act_dim), dtype=np.float32)
        self.rew_n = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs_n = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, obs_n: np.ndarray, act_n: np.ndarray, rew_n: np.ndarray,
             next_obs_n: np.ndarray, done: bool) -> None:
        self.obs_n[self.ptr] = obs_n
        self.act_n[self.ptr] = act_n
        self.rew_n[self.ptr] = rew_n
        self.next_obs_n[self.ptr] = next_obs_n
        self.done[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        idx = np.random.randint(0, self.size, size=batch_size)

        def t(arr: np.ndarray) -> torch.Tensor:
            return torch.FloatTensor(arr[idx]).to(device)

        return t(self.obs_n), t(self.act_n), t(self.rew_n), \
               t(self.next_obs_n), t(self.done)

    def __len__(self) -> int:
        return self.size
