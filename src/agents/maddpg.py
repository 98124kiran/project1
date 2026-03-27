"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) implementation.

Paper: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"

Key features:
  - Decentralised execution: each agent uses only its local observation at inference.
  - Centralised training: the critic receives observations and actions of ALL agents.
  - Adaptive replanning: agents continuously re-optimise their policies as conditions change.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.models.networks import Actor, Critic
from src.agents.replay_buffer import ReplayBuffer
from src.utils.config import Config


class MADDPGAgent:
    """Single agent within the MADDPG framework."""

    def __init__(self, agent_id: int, obs_dim: int, act_dim: int,
                 total_obs_dim: int, total_act_dim: int, config: Config,
                 device: torch.device):
        self.agent_id = agent_id
        self.act_dim = act_dim
        self.device = device
        cfg = config.agent

        # Networks
        self.actor = Actor(obs_dim, act_dim, cfg.hidden_dim, cfg.num_layers).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = Critic(total_obs_dim, total_act_dim,
                             cfg.hidden_dim, cfg.num_layers).to(device)
        self.target_critic = copy.deepcopy(self.critic)

        # Optimisers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.noise_std = cfg.noise_std
        self.noise_decay = cfg.noise_decay
        self.min_noise_std = cfg.min_noise_std

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        """Return action index for this agent given its local observation."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits = self.actor(obs_t).squeeze(0)

        if explore:
            # Add Gaussian noise in logit space and renormalise
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        return int(torch.argmax(logits).item())

    @torch.no_grad()
    def act_target(self, obs: torch.Tensor) -> torch.Tensor:
        """Gumbel-softmax action from target actor (for critic update)."""
        logits = self.target_actor(obs)
        return F.gumbel_softmax(logits, tau=1.0, hard=False)

    def decay_noise(self) -> None:
        self.noise_std = max(self.min_noise_std, self.noise_std * self.noise_decay)

    # ------------------------------------------------------------------
    # Soft update
    # ------------------------------------------------------------------

    def soft_update(self) -> None:
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic.load_state_dict(ckpt["target_critic"])


class MADDPG:
    """
    Multi-Agent DDPG controller.

    Coordinates a group of MADDPGAgents that share a single replay buffer.
    Each agent has its own decentralised actor but uses a centralised critic
    that sees the full joint observation-action space.
    """

    def __init__(self, n_agents: int, obs_dim: int, act_dim: int,
                 config: Config, device: Optional[torch.device] = None):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        total_obs_dim = obs_dim * n_agents
        total_act_dim = act_dim * n_agents

        self.agents: List[MADDPGAgent] = [
            MADDPGAgent(
                agent_id=i,
                obs_dim=obs_dim,
                act_dim=act_dim,
                total_obs_dim=total_obs_dim,
                total_act_dim=total_act_dim,
                config=config,
                device=self.device,
            )
            for i in range(n_agents)
        ]

        self.buffer = ReplayBuffer(
            capacity=config.agent.buffer_size,
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )

        self.total_steps = 0
        self.batch_size = config.agent.batch_size
        self.warmup_steps = config.agent.warmup_steps
        self.update_every = config.agent.update_every

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def select_actions(self, obs_n: Dict[int, np.ndarray],
                       explore: bool = True) -> Dict[int, int]:
        """Select actions for all agents."""
        if self.total_steps < self.warmup_steps:
            # Random exploration during warmup
            return {i: int(np.random.randint(0, self.act_dim))
                    for i in range(self.n_agents)}
        return {i: self.agents[i].act(obs_n[i], explore=explore)
                for i in range(self.n_agents)}

    def store_transition(self, obs_n: Dict[int, np.ndarray],
                         actions: Dict[int, int],
                         rewards: Dict[int, float],
                         next_obs_n: Dict[int, np.ndarray],
                         done: bool) -> None:
        """Convert dicts → arrays and push to replay buffer."""
        obs_arr = np.stack([obs_n[i] for i in range(self.n_agents)])
        act_arr = np.zeros((self.n_agents, self.act_dim), dtype=np.float32)
        for i in range(self.n_agents):
            act_arr[i, actions[i]] = 1.0
        rew_arr = np.array([rewards[i] for i in range(self.n_agents)], dtype=np.float32)
        next_obs_arr = np.stack([next_obs_n[i] for i in range(self.n_agents)])
        self.buffer.push(obs_arr, act_arr, rew_arr, next_obs_arr, done)
        self.total_steps += 1

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one gradient update step for all agents.
        Returns loss dict or None if conditions not met.
        """
        if (len(self.buffer) < self.batch_size or
                self.total_steps % self.update_every != 0 or
                self.total_steps < self.warmup_steps):
            return None

        obs_n, act_n, rew_n, next_obs_n, done = self.buffer.sample(
            self.batch_size, self.device
        )
        # obs_n:  (B, n_agents, obs_dim)
        # act_n:  (B, n_agents, act_dim)
        # rew_n:  (B, n_agents)
        # next_obs_n: (B, n_agents, obs_dim)
        # done:   (B, 1)

        losses = {}

        # Flatten joint tensors for centralised critic
        all_obs = obs_n.view(self.batch_size, -1)             # (B, n*obs)
        all_next_obs = next_obs_n.view(self.batch_size, -1)   # (B, n*obs)
        all_acts = act_n.view(self.batch_size, -1)            # (B, n*act)

        # ---- Compute joint next actions from target actors ----
        with torch.no_grad():
            next_acts = []
            for i, agent in enumerate(self.agents):
                ni_obs = next_obs_n[:, i, :]    # (B, obs_dim)
                next_acts.append(agent.act_target(ni_obs))  # (B, act_dim)
            all_next_acts = torch.cat(next_acts, dim=-1)    # (B, n*act)

        # ---- Update each agent's critic and actor ----
        for i, agent in enumerate(self.agents):
            # ---- Critic update ----
            with torch.no_grad():
                q_next = agent.target_critic(all_next_obs, all_next_acts)  # (B,1)
                q_target = rew_n[:, i:i+1] + agent.gamma * (1 - done) * q_next

            q_pred = agent.critic(all_obs, all_acts)    # (B,1)
            critic_loss = F.mse_loss(q_pred, q_target)

            agent.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
            agent.critic_optim.step()

            # ---- Actor update ----
            curr_acts = []
            for j, a in enumerate(self.agents):
                oj = obs_n[:, j, :]
                logits_j = a.actor(oj)
                if j == i:
                    curr_acts.append(F.gumbel_softmax(logits_j, tau=1.0, hard=False))
                else:
                    with torch.no_grad():
                        curr_acts.append(F.gumbel_softmax(logits_j, tau=1.0, hard=False))
            all_curr_acts = torch.cat(curr_acts, dim=-1)   # (B, n*act)

            actor_loss = -agent.critic(all_obs, all_curr_acts).mean()

            agent.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optim.step()

            losses[f"actor_loss_{i}"] = float(actor_loss.item())
            losses[f"critic_loss_{i}"] = float(critic_loss.item())

        # ---- Soft update target networks ----
        for agent in self.agents:
            agent.soft_update()
            agent.decay_noise()

        return losses

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        import os
        os.makedirs(directory, exist_ok=True)
        for agent in self.agents:
            agent.save(os.path.join(directory, f"agent_{agent.agent_id}.pt"))

    def load(self, directory: str) -> None:
        import os
        for agent in self.agents:
            path = os.path.join(directory, f"agent_{agent.agent_id}.pt")
            if os.path.exists(path):
                agent.load(path)
