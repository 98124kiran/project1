"""
Actor and Critic neural networks for MADDPG.

Architecture:
  - Actor: obs → softmax over actions  (each agent has its own actor)
  - Critic: (all obs, all actions) → Q-value  (centralised critic)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def _mlp(dims: List[int], activation: nn.Module = nn.ReLU(),
         output_activation: nn.Module | None = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation)
            layers.append(nn.LayerNorm(dims[i + 1]))
        elif output_activation is not None:
            layers.append(output_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Decentralised actor: maps local observation to a probability distribution
    over discrete actions (using Gumbel-Softmax for differentiable sampling).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3):
        super().__init__()
        dims = [obs_dim] + [hidden_dim] * (num_layers - 1) + [act_dim]
        self.net = _mlp(dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logits over actions."""
        return self.net(obs)

    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)

    def sample_gumbel(self, obs: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
        """Differentiable discrete action using Gumbel-Softmax trick."""
        logits = self.forward(obs)
        return F.gumbel_softmax(logits, tau=temperature, hard=False)


class Critic(nn.Module):
    """
    Centralised critic: maps (all observations concatenated, all one-hot actions
    concatenated) to a scalar Q-value for one agent.
    """

    def __init__(self, total_obs_dim: int, total_act_dim: int,
                 hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        dims = [total_obs_dim + total_act_dim] + [hidden_dim] * (num_layers - 1) + [1]
        self.net = _mlp(dims)

    def forward(self, all_obs: torch.Tensor,
                all_actions: torch.Tensor) -> torch.Tensor:
        """
        all_obs:     (batch, total_obs_dim)
        all_actions: (batch, total_act_dim)  — one-hot concatenated
        Returns:     (batch, 1)
        """
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)
