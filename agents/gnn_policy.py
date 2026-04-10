"""
Graph Attention Network (GAT) policy for manufacturing scheduling.

Replaces the flat MLP actor in MAPPOAgent with a graph-structured
neural network that explicitly models relationships between jobs
(nodes) and machines (nodes) via multi-head attention.

Architecture
------------
Observation → Parse → Node embeddings
                           ↓
                    Multi-head Self-Attention × L layers
                           ↓
                    Global average pool (readout)
                           ↓
                    MLP head → action logits

Node types
----------
- Machine nodes : features = [normalised status]  (1-dim)
- Job nodes     : features = [remaining_time, deadline]  (2-dim)
- Context node  : features = [queue_len, cpu, mem, latency]  (4-dim)
  (one "summary" node aggregates global env metrics)

All node features are linearly projected to *d_model* before attention.

Usage (drop-in replacement for ActorNetwork inside MAPPOAgent)
-----
>>> from agents.gnn_policy import GNNPolicyAgent
>>> agent = GNNPolicyAgent(obs_size=19, action_size=9, num_agents=3,
...                        num_machines=5, num_observable_jobs=5)
>>> obs = env.reset()
>>> actions, log_probs, values = agent.select_actions(obs)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.ppo_agent import CriticNetwork, RolloutBuffer


# --------------------------------------------------------------------------- #
# Multi-head self-attention layer                                               #
# --------------------------------------------------------------------------- #

class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.

    Input  : (batch, n_nodes, d_model)
    Output : (batch, n_nodes, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        K = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        scale = math.sqrt(Dh)
        attn = (Q @ K.transpose(-2, -1)) / scale        # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


# --------------------------------------------------------------------------- #
# Transformer encoder block                                                    #
# --------------------------------------------------------------------------- #

class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN → MHSA → residual → LN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# --------------------------------------------------------------------------- #
# GNN / Set-Transformer actor                                                  #
# --------------------------------------------------------------------------- #

class GNNActor(nn.Module):
    """
    Graph-structured policy network using multi-head self-attention.

    Parses a flat observation vector into per-node feature tensors,
    applies L Transformer blocks for message-passing, then pools to
    produce action logits.

    Parameters
    ----------
    obs_size         : int   — flat observation dimension
    action_size      : int   — number of discrete actions
    num_machines     : int   — M (machines per node)
    num_observable_jobs : int — K (jobs visible in obs)
    d_model          : int   — node embedding dimension
    n_heads          : int   — attention heads
    n_layers         : int   — number of Transformer blocks
    hidden_size      : int   — MLP head hidden dim
    dropout          : float
    """

    # Dimensions of each node-type's raw features
    _MACHINE_FEAT_DIM = 1   # normalised status
    _JOB_FEAT_DIM = 2       # (remaining_time, deadline)
    _CONTEXT_FEAT_DIM = 4   # (queue_len, cpu, mem, latency)

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        num_machines: int,
        num_observable_jobs: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.M = num_machines
        self.K = num_observable_jobs
        self.d_model = d_model

        # Node-type embedding projections
        self.machine_embed = nn.Linear(self._MACHINE_FEAT_DIM, d_model)
        self.job_embed = nn.Linear(self._JOB_FEAT_DIM, d_model)
        self.context_embed = nn.Linear(self._CONTEXT_FEAT_DIM, d_model)

        # Transformer blocks (message passing)
        self.transformer = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Action head: global readout → logits
        n_nodes = num_machines + num_observable_jobs + 1  # +1 for context node
        self.action_head = nn.Sequential(
            nn.Linear(d_model * n_nodes, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.action_head[-1].weight, gain=0.01)

    def _parse_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode flat obs into machine, job, and context feature tensors.

        Parameters
        ----------
        obs : (batch, obs_size)

        Returns
        -------
        machine_feats : (batch, M, 1)
        job_feats     : (batch, K, 2)
        context_feats : (batch, 1, 4)
        """
        M, K = self.M, self.K
        machine_feats = obs[:, :M].unsqueeze(-1)                        # (B, M, 1)
        queue_len = obs[:, M:M + 1]                                      # (B, 1)
        job_feats = obs[:, M + 1: M + 1 + K * 2].view(-1, K, 2)        # (B, K, 2)
        global_feats = obs[:, M + 1 + K * 2:]                           # (B, 3)
        # Context node: [queue_len, cpu, mem, latency]
        context_feats = torch.cat([queue_len, global_feats], dim=-1).unsqueeze(1)  # (B, 1, 4)
        return machine_feats, job_feats, context_feats

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action logits. obs shape: (batch, obs_size) or (obs_size,)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # add batch dim
            squeeze = True
        else:
            squeeze = False

        machine_feats, job_feats, context_feats = self._parse_obs(obs)

        # Project each node type to d_model
        m_emb = self.machine_embed(machine_feats)      # (B, M, d_model)
        j_emb = self.job_embed(job_feats)              # (B, K, d_model)
        c_emb = self.context_embed(context_feats)      # (B, 1, d_model)

        # Stack all nodes along node dimension: (B, M+K+1, d_model)
        nodes = torch.cat([m_emb, j_emb, c_emb], dim=1)

        # Multi-head self-attention message passing
        for block in self.transformer:
            nodes = block(nodes)

        # Flatten all node embeddings for the action head
        B = nodes.shape[0]
        flat = nodes.view(B, -1)   # (B, n_nodes * d_model)
        logits = self.action_head(flat)

        if squeeze:
            logits = logits.squeeze(0)
        return logits

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))


# --------------------------------------------------------------------------- #
# GNN Policy Agent (MAPPO with GNN actor)                                      #
# --------------------------------------------------------------------------- #

class GNNPolicyAgent:
    """
    MAPPO agent that uses a GNN actor instead of a flat MLP.

    Identical training interface to MAPPOAgent so it can be swapped in
    the training script with a single flag.

    Parameters
    ----------
    obs_size, action_size, num_agents : int
    num_machines, num_observable_jobs : int
        Used by the GNN to parse the flat observation vector.
    d_model, n_heads, n_layers, hidden_size, dropout : GNN hyper-params.
    critic_hidden_size : int   — centralised critic hidden size.
    lr_actor, lr_critic : float
    gamma, gae_lambda, clip_eps, entropy_coef, value_coef : PPO params.
    max_grad_norm, n_epochs, batch_size, rollout_steps : training params.
    device : str
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        num_agents: int,
        num_machines: int,
        num_observable_jobs: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        hidden_size: int = 128,
        dropout: float = 0.1,
        critic_hidden_size: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        device: str = "cpu",
    ) -> None:
        self.obs_size = obs_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.global_obs_size = obs_size * num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.device = torch.device(device)

        # GNN actor
        self.actor = GNNActor(
            obs_size=obs_size,
            action_size=action_size,
            num_machines=num_machines,
            num_observable_jobs=num_observable_jobs,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            hidden_size=hidden_size,
            dropout=dropout,
        ).to(self.device)

        # Centralised critic (same as MAPPO)
        self.critic = CriticNetwork(self.global_obs_size, critic_hidden_size).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        self.buffer = RolloutBuffer(
            num_agents=num_agents,
            obs_size=obs_size,
            rollout_steps=rollout_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        self.train_step: int = 0
        self.total_env_steps: int = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Same interface as MAPPOAgent                                         #
    # ------------------------------------------------------------------ #

    def select_actions(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[List[int], List[float], np.ndarray]:
        obs_np = np.array(observations, dtype=np.float32)
        obs_t = torch.as_tensor(obs_np).to(self.device)
        with torch.no_grad():
            dist = self.actor.get_distribution(obs_t)
            if deterministic:
                actions_t = dist.probs.argmax(dim=-1)
            else:
                actions_t = dist.sample()
            log_probs_t = dist.log_prob(actions_t)
            global_obs_t = obs_t.flatten().unsqueeze(0)
            value_scalar = self.critic(global_obs_t).item()
            values = np.full(self.num_agents, value_scalar, dtype=np.float32)
        return actions_t.cpu().tolist(), log_probs_t.cpu().tolist(), values

    def store_transition(
        self,
        observations: List[np.ndarray],
        global_obs: np.ndarray,
        actions: List[int],
        log_probs: List[float],
        rewards: List[float],
        dones: List[bool],
        values: np.ndarray,
    ) -> None:
        self.buffer.add(
            observations=np.array(observations, dtype=np.float32),
            global_obs=np.array(global_obs, dtype=np.float32),
            actions=np.array(actions, dtype=np.int64),
            log_probs=np.array(log_probs, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
            values=values,
        )
        self.total_env_steps += self.num_agents

    def update(
        self,
        last_observations: Optional[List[np.ndarray]] = None,
        last_dones: Optional[List[bool]] = None,
    ) -> Dict[str, float]:
        if last_observations is not None:
            obs_t = torch.as_tensor(np.array(last_observations, dtype=np.float32)).to(self.device)
            with torch.no_grad():
                g_t = obs_t.flatten().unsqueeze(0)
                last_v = self.critic(g_t).item()
                if last_dones is not None and all(last_dones):
                    last_v = 0.0
            last_values = np.full(self.num_agents, last_v, dtype=np.float32)
        else:
            last_values = np.zeros(self.num_agents, dtype=np.float32)

        advantages, returns = self.buffer.compute_advantages_and_returns(last_values)

        metrics: Dict[str, float] = {
            "actor_loss": 0.0, "critic_loss": 0.0,
            "entropy": 0.0, "approx_kl": 0.0,
        }
        n_batches = 0

        self.actor.train()
        self.critic.train()

        for _ in range(self.n_epochs):
            for batch in self.buffer.iterate_batches(
                advantages, returns, self.batch_size, self._rng
            ):
                obs_b, gobs_b, act_b, old_lp_b, adv_b, ret_b = [
                    x.to(self.device) for x in batch
                ]
                dist = self.actor.get_distribution(obs_b)
                new_lp = dist.log_prob(act_b)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(gobs_b)
                critic_loss = F.mse_loss(values_pred, ret_b)

                total_loss = (
                    actor_loss - self.entropy_coef * entropy + self.value_coef * critic_loss
                )

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                metrics["actor_loss"] += actor_loss.item()
                metrics["critic_loss"] += critic_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += approx_kl
                n_batches += 1

        self.buffer.reset()
        self.train_step += 1
        self.actor.eval()
        self.critic.eval()

        if n_batches > 0:
            for k in metrics:
                metrics[k] /= n_batches
        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "train_step": self.train_step,
                "total_env_steps": self.total_env_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_optim" in ckpt:
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.train_step = ckpt.get("train_step", 0)
        self.total_env_steps = ckpt.get("total_env_steps", 0)
