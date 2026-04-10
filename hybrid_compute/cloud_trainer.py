"""
Cloud Trainer — simulates cloud-side training and federated aggregation.

In a hybrid edge-cloud deployment:
- Edge nodes run inference locally and collect experience.
- Periodically, edge nodes upload their experience to the cloud.
- The cloud runs a full PPO update using the aggregated experience.
- Updated weights are broadcast back to all edge nodes.
- Federated averaging (FedAvg) merges locally-updated models before
  cloud aggregation to reduce communication rounds.

This module provides:
- ``CloudTrainer``        : central training loop over aggregated experience.
- ``FederatedAggregator`` : FedAvg weight merging from multiple edge agents.

Reference
---------
McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks
from Decentralized Data" (FedAvg).

Usage
-----
>>> from hybrid_compute.cloud_trainer import CloudTrainer, FederatedAggregator
>>> trainer = CloudTrainer(obs_size=19, action_size=9, num_agents=3)
>>> # Receive experience from edge nodes
>>> trainer.receive_experience(node_id=0, batch=edge0_exp)
>>> trainer.receive_experience(node_id=1, batch=edge1_exp)
>>> metrics = trainer.train_step()
>>> new_weights = trainer.get_actor_weights()
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.ppo_agent import ActorNetwork, CriticNetwork


# --------------------------------------------------------------------------- #
# Federated Aggregator (FedAvg)                                                #
# --------------------------------------------------------------------------- #

class FederatedAggregator:
    """
    Implements Federated Averaging (FedAvg) to merge model weights
    from multiple edge agents.

    FedAvg computes a weighted average of edge-local model weights,
    where each node's contribution is weighted by its dataset size:

        θ_global = Σ_n (|D_n| / |D|) · θ_n

    Parameters
    ----------
    n_nodes : int
        Number of edge nodes participating in federation.
    """

    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes
        self._local_weights: Dict[int, Tuple[dict, int]] = {}

    def submit_local_weights(
        self,
        node_id: int,
        state_dict: dict,
        n_samples: int,
    ) -> None:
        """
        Register local model weights from one edge node.

        Parameters
        ----------
        node_id   : int   — edge node identifier
        state_dict: dict  — PyTorch state dict of the local model
        n_samples : int   — number of training samples this node used
        """
        self._local_weights[node_id] = (copy.deepcopy(state_dict), n_samples)

    def aggregate(self) -> Optional[dict]:
        """
        Compute FedAvg over all submitted local weights.

        Returns the aggregated global state dict, or None if no weights
        have been submitted.
        """
        if not self._local_weights:
            return None

        total_samples = sum(n for _, n in self._local_weights.values())
        if total_samples == 0:
            return None

        global_dict: dict = {}
        for node_id, (sd, n) in self._local_weights.items():
            weight = n / total_samples
            for key, param in sd.items():
                if key in global_dict:
                    global_dict[key] += weight * param.float()
                else:
                    global_dict[key] = weight * param.float()

        return global_dict

    def is_ready(self, min_nodes: int = 1) -> bool:
        """True if at least *min_nodes* nodes have submitted weights."""
        return len(self._local_weights) >= min_nodes

    def reset(self) -> None:
        """Clear submitted weights (call after each aggregation round)."""
        self._local_weights.clear()


# --------------------------------------------------------------------------- #
# Cloud Trainer                                                                 #
# --------------------------------------------------------------------------- #

class CloudTrainer:
    """
    Cloud-side training controller.

    Aggregates experience from multiple edge nodes, runs PPO updates on
    the combined data, and manages federated weight aggregation.

    Parameters
    ----------
    obs_size, action_size, num_agents : int — environment dimensions
    hidden_size      : int   — actor MLP hidden size
    critic_hidden_size : int — critic MLP hidden size
    lr_actor, lr_critic : float — learning rates
    gamma            : float — discount factor
    clip_eps         : float — PPO clip epsilon
    entropy_coef     : float — entropy bonus coefficient
    value_coef       : float — value loss coefficient
    max_grad_norm    : float — gradient clip norm
    n_epochs         : int   — update epochs per training step
    batch_size       : int   — mini-batch size
    bandwidth_mbps   : float — simulated cloud download bandwidth
    model_size_mb    : float — approximate model size for latency calc
    device           : str   — "cpu" or "cuda"
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        num_agents: int,
        hidden_size: int = 128,
        critic_hidden_size: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        bandwidth_mbps: float = 100.0,
        model_size_mb: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.obs_size = obs_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.global_obs_size = obs_size * num_agents
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.bandwidth_mbps = bandwidth_mbps
        self.model_size_mb = model_size_mb
        self.device = torch.device(device)

        # Global (cloud) networks
        self.actor = ActorNetwork(obs_size, action_size, hidden_size).to(self.device)
        self.critic = CriticNetwork(self.global_obs_size, critic_hidden_size).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        # Experience pool: list of transition dicts from all edges
        self._experience_pool: List[Dict[str, np.ndarray]] = []
        self._node_experience_counts: Dict[int, int] = {}

        # Federated aggregator
        self.fed_aggregator = FederatedAggregator(num_agents)

        # Metrics
        self.train_step: int = 0
        self._metrics_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ #
    # Experience reception                                                 #
    # ------------------------------------------------------------------ #

    def receive_experience(
        self,
        node_id: int,
        batch: List[Dict[str, Any]],
    ) -> None:
        """
        Accept a batch of experience transitions from an edge node.

        Parameters
        ----------
        node_id : int
            Source edge node.
        batch   : List[Dict]
            Each dict must contain:
                obs       : np.ndarray (obs_size,)
                action    : int
                reward    : float
                next_obs  : np.ndarray (obs_size,)
                done      : bool
                log_prob  : float
        """
        self._experience_pool.extend(batch)
        self._node_experience_counts[node_id] = (
            self._node_experience_counts.get(node_id, 0) + len(batch)
        )

    def receive_federated_weights(
        self,
        node_id: int,
        state_dict: dict,
        n_samples: int,
    ) -> None:
        """Accept locally-updated actor weights from an edge node for FedAvg."""
        self.fed_aggregator.submit_local_weights(node_id, state_dict, n_samples)

    # ------------------------------------------------------------------ #
    # Federated aggregation                                                #
    # ------------------------------------------------------------------ #

    def federated_aggregate(self, min_nodes: int = 1) -> bool:
        """
        Run FedAvg if enough nodes have submitted weights.

        Updates the cloud actor with the aggregated weights, then resets
        the aggregator for the next round.

        Returns True if aggregation was performed.
        """
        if not self.fed_aggregator.is_ready(min_nodes):
            return False

        agg_weights = self.fed_aggregator.aggregate()
        if agg_weights is not None:
            self.actor.load_state_dict(agg_weights)
        self.fed_aggregator.reset()
        return True

    # ------------------------------------------------------------------ #
    # Cloud PPO training                                                   #
    # ------------------------------------------------------------------ #

    def train_step(self) -> Dict[str, float]:
        """
        Run one PPO training step on the accumulated experience pool.

        Returns
        -------
        Dict with keys: actor_loss, critic_loss, entropy, n_samples
        """
        if not self._experience_pool:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "n_samples": 0.0}

        # Convert experience pool to tensors
        obs = np.array([e["obs"] for e in self._experience_pool], dtype=np.float32)
        actions = np.array([e["action"] for e in self._experience_pool], dtype=np.int64)
        rewards = np.array([e["reward"] for e in self._experience_pool], dtype=np.float32)
        next_obs = np.array([e["next_obs"] for e in self._experience_pool], dtype=np.float32)
        dones = np.array([float(e["done"]) for e in self._experience_pool], dtype=np.float32)
        old_log_probs = np.array([e["log_prob"] for e in self._experience_pool], dtype=np.float32)

        # Compute discounted returns (simple Monte-Carlo without GAE for cloud)
        returns = self._compute_returns(rewards, dones)

        obs_t = torch.as_tensor(obs).to(self.device)
        act_t = torch.as_tensor(actions).to(self.device)
        ret_t = torch.as_tensor(returns).to(self.device)
        old_lp_t = torch.as_tensor(old_log_probs).to(self.device)
        next_obs_t = torch.as_tensor(next_obs).to(self.device)

        # Build global obs: pad obs with zeros to full global_obs_size
        # (edge nodes send local obs only; cloud approximates global state)
        pad_size = self.global_obs_size - self.obs_size
        gobs_np = np.pad(obs, ((0, 0), (0, pad_size)))
        gobs_t = torch.as_tensor(gobs_np.astype(np.float32)).to(self.device)

        n_samples = len(obs_t)
        metrics: Dict[str, float] = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "n_samples": float(n_samples),
        }
        n_batches = 0

        self.actor.train()
        self.critic.train()

        indices = np.arange(n_samples)
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                idx = indices[start: start + self.batch_size]
                obs_b = obs_t[idx]
                gobs_b = gobs_t[idx]
                act_b = act_t[idx]
                old_lp_b = old_lp_t[idx]
                ret_b = ret_t[idx]

                dist = Categorical(logits=self.actor(obs_b))
                new_lp = dist.log_prob(act_b)
                entropy = dist.entropy().mean()

                values = self.critic(gobs_b)
                adv = ret_b - values.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, ret_b)
                total_loss = actor_loss - self.entropy_coef * entropy + self.value_coef * critic_loss

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                metrics["actor_loss"] += actor_loss.item()
                metrics["critic_loss"] += critic_loss.item()
                metrics["entropy"] += entropy.item()
                n_batches += 1

        self.actor.eval()
        self.critic.eval()

        if n_batches > 0:
            metrics["actor_loss"] /= n_batches
            metrics["critic_loss"] /= n_batches
            metrics["entropy"] /= n_batches

        # Clear the pool after training
        self._experience_pool.clear()
        self._node_experience_counts.clear()
        self.train_step += 1
        self._metrics_history.append(metrics)

        return metrics

    # ------------------------------------------------------------------ #
    # Weight broadcast                                                     #
    # ------------------------------------------------------------------ #

    def get_actor_weights(self) -> dict:
        """Return actor state dict for broadcasting to edge nodes."""
        return copy.deepcopy(self.actor.state_dict())

    def broadcast_latency_ms(self) -> float:
        """
        Estimate the time to broadcast updated weights to all edge nodes.

        Returns simulated total latency in milliseconds.
        """
        data_bytes = int(self.model_size_mb * 1024 * 1024)
        bits = data_bytes * 8
        bandwidth_bps = self.bandwidth_mbps * 1_000_000
        transfer_ms = (bits / bandwidth_bps) * 1000.0
        return transfer_ms * self.num_agents  # broadcast to all nodes

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _compute_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute discounted returns (Monte-Carlo) from rewards and done flags."""
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G * (1.0 - dones[t])
            returns[t] = G
        return returns

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Return list of training step metrics dictionaries."""
        return list(self._metrics_history)

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "train_step": self.train_step,
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
