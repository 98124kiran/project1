"""
MAPPO — Multi-Agent Proximal Policy Optimisation.

Implements Centralised Training with Decentralised Execution (CTDE):
  - Shared Actor  : all agents use the same MLP policy network.
                    At execution time each agent acts only on its
                    *local* observation (decentralised).
  - Central Critic: a single value network conditioned on the global
                    state (concatenation of all local observations).
                    Used only during training (never deployed at edge).

Training algorithm
------------------
1. Collect a rollout of *rollout_steps* environment steps.
2. Compute Generalised Advantage Estimates (GAE).
3. Run *n_epochs* of mini-batch PPO updates on (actor + critic).
4. Clear buffer and repeat.

Reference: Schulman et al. (2017) "Proximal Policy Optimization
Algorithms"; Yu et al. (2022) "The Surprising Effectiveness of PPO
in Cooperative Multi-Agent Games" (MAPPO paper).

Usage
-----
>>> from agents.ppo_agent import MAPPOAgent
>>> agent = MAPPOAgent(obs_size=19, action_size=9, num_agents=3)
>>> obs = env.reset()
>>> actions, log_probs, values = agent.select_actions(obs)
>>> obs2, rewards, dones, info = env.step(actions)
>>> agent.store_transition(obs, np.concatenate(obs), actions,
...                        log_probs, rewards, dones, values)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# --------------------------------------------------------------------------- #
# Actor network (shared across all agents)                                     #
# --------------------------------------------------------------------------- #

class ActorNetwork(nn.Module):
    """
    MLP policy network.  Outputs un-normalised action logits.

    Architecture: Linear → Tanh → Linear → Tanh → Linear (logits)
    """

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for the final output layer
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action logits (not probabilities)."""
        return self.net(obs)

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        """Return a Categorical distribution over actions."""
        return Categorical(logits=self.forward(obs))


# --------------------------------------------------------------------------- #
# Centralised critic network                                                   #
# --------------------------------------------------------------------------- #

class CriticNetwork(nn.Module):
    """
    Centralised value network.  Takes the global state (concatenation of
    all agents' local observations) and outputs a scalar value estimate.

    Used only during training — never deployed on edge nodes.
    """

    def __init__(self, global_obs_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate, shape (batch,)."""
        return self.net(global_obs).squeeze(-1)


# --------------------------------------------------------------------------- #
# Rollout buffer with GAE                                                       #
# --------------------------------------------------------------------------- #

class RolloutBuffer:
    """
    Fixed-size buffer that stores on-policy trajectories from all agents.

    After *rollout_steps* have been added, call
    ``compute_advantages_and_returns()`` followed by ``iterate_batches()``
    to yield mini-batches for PPO updates.

    Storage layout
    --------------
    observations : (T, N, obs_size)   — local obs per agent
    global_obs   : (T, N*obs_size)    — global state (same for all agents)
    actions      : (T, N)             — chosen action per agent
    log_probs    : (T, N)             — log π(a|o) at collection time
    rewards      : (T, N)             — individual rewards
    dones        : (T, N)             — episode-done flags
    values       : (T, N)             — critic value estimates
    """

    def __init__(
        self,
        num_agents: int,
        obs_size: int,
        rollout_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.num_agents = num_agents
        self.obs_size = obs_size
        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._ptr = 0
        self._full = False
        self._allocate()

    def _allocate(self) -> None:
        T, N, D = self.rollout_steps, self.num_agents, self.obs_size
        self.observations = np.zeros((T, N, D), dtype=np.float32)
        self.global_obs = np.zeros((T, N * D), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)

    def add(
        self,
        observations: np.ndarray,   # (N, obs_size)
        global_obs: np.ndarray,     # (N * obs_size,)
        actions: np.ndarray,        # (N,)
        log_probs: np.ndarray,      # (N,)
        rewards: np.ndarray,        # (N,)
        dones: np.ndarray,          # (N,)
        values: np.ndarray,         # (N,)
    ) -> None:
        t = self._ptr
        self.observations[t] = observations
        self.global_obs[t] = global_obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self._ptr += 1
        if self._ptr >= self.rollout_steps:
            self._full = True

    @property
    def is_ready(self) -> bool:
        return self._full

    def reset(self) -> None:
        self._ptr = 0
        self._full = False

    def compute_advantages_and_returns(
        self,
        last_values: np.ndarray,  # (N,)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE(γ, λ) advantages and discounted returns.

        Returns
        -------
        advantages : (T, N) float32
        returns    : (T, N) float32
        """
        T = self.rollout_steps
        advantages = np.zeros((T, self.num_agents), dtype=np.float32)
        last_gae = np.zeros(self.num_agents, dtype=np.float32)

        next_values = last_values.copy()
        for t in reversed(range(T)):
            non_terminal = 1.0 - self.dones[t]
            next_v = next_values if t == T - 1 else self.values[t + 1]
            delta = (
                self.rewards[t]
                + self.gamma * next_v * non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns

    def iterate_batches(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
    ):
        """
        Yield mini-batches by flattening the (T, N) time×agent dimensions.

        Each yielded batch is a tuple:
            (obs, global_obs, actions, old_log_probs, advantages, returns)
        all as torch.Tensor on CPU.
        """
        T, N = self.rollout_steps, self.num_agents
        total = T * N

        # Flatten time × agent
        obs_flat = self.observations.reshape(total, self.obs_size)
        # For the centralised critic, each agent at time t sees the same
        # global obs.  Repeat global_obs[t] for each of the N agents.
        gobs_flat = np.repeat(self.global_obs, N, axis=0).reshape(total, N * self.obs_size)
        # Correct interleaving: row i belongs to time t=i//N, agent n=i%N
        # We rebuild with correct ordering: (t*N + n) → global_obs[t]
        gobs_flat2 = np.zeros((total, N * self.obs_size), dtype=np.float32)
        for t in range(T):
            gobs_flat2[t * N: t * N + N] = self.global_obs[t]
        actions_flat = self.actions.reshape(total)
        lp_flat = self.log_probs.reshape(total)
        adv_flat = advantages.reshape(total)
        ret_flat = returns.reshape(total)

        # Normalise advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        indices = rng.permutation(total)
        for start in range(0, total, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                torch.as_tensor(obs_flat[idx]),
                torch.as_tensor(gobs_flat2[idx]),
                torch.as_tensor(actions_flat[idx]),
                torch.as_tensor(lp_flat[idx]),
                torch.as_tensor(adv_flat[idx]),
                torch.as_tensor(ret_flat[idx]),
            )


# --------------------------------------------------------------------------- #
# MAPPO Agent                                                                  #
# --------------------------------------------------------------------------- #

class MAPPOAgent:
    """
    Multi-Agent PPO with shared actor and centralised critic.

    Parameters
    ----------
    obs_size : int
        Dimension of each agent's local observation vector.
    action_size : int
        Number of discrete actions per agent.
    num_agents : int
        Number of cooperative agents (edge nodes).
    hidden_size : int
        Hidden layer width for the actor MLP.
    critic_hidden_size : int
        Hidden layer width for the critic MLP.
    lr_actor, lr_critic : float
        Learning rates for actor and critic Adam optimisers.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda parameter.
    clip_eps : float
        PPO clipping epsilon.
    entropy_coef : float
        Entropy bonus coefficient (encourages exploration).
    value_coef : float
        Value loss coefficient in the combined PPO loss.
    max_grad_norm : float
        Gradient clipping max norm.
    n_epochs : int
        Number of PPO update epochs per rollout.
    batch_size : int
        Mini-batch size.
    rollout_steps : int
        Number of environment steps collected before each update.
    device : str
        PyTorch device string ("cpu" or "cuda").
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

        # Networks
        self.actor = ActorNetwork(obs_size, action_size, hidden_size).to(self.device)
        self.critic = CriticNetwork(self.global_obs_size, critic_hidden_size).to(self.device)

        # Optimisers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        # Buffer
        self.buffer = RolloutBuffer(
            num_agents=num_agents,
            obs_size=obs_size,
            rollout_steps=rollout_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Tracking
        self.train_step: int = 0
        self.total_env_steps: int = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Decentralised execution                                              #
    # ------------------------------------------------------------------ #

    def select_actions(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[List[int], List[float], np.ndarray]:
        """
        Select actions for all agents given their local observations.

        Parameters
        ----------
        observations : List[np.ndarray]
            One (obs_size,) array per agent.
        deterministic : bool
            If True, take argmax actions (for evaluation).

        Returns
        -------
        actions   : List[int]
        log_probs : List[float]
        values    : np.ndarray shape (num_agents,)
        """
        obs_np = np.array(observations, dtype=np.float32)  # (N, obs_size)
        obs_t = torch.as_tensor(obs_np).to(self.device)

        with torch.no_grad():
            dist = self.actor.get_distribution(obs_t)
            if deterministic:
                actions_t = dist.probs.argmax(dim=-1)
            else:
                actions_t = dist.sample()
            log_probs_t = dist.log_prob(actions_t)

            # Centralised critic: global state = concatenated local obs
            global_obs_t = obs_t.flatten().unsqueeze(0)  # (1, N*obs_size)
            value_scalar = self.critic(global_obs_t).item()
            values = np.full(self.num_agents, value_scalar, dtype=np.float32)

        return (
            actions_t.cpu().tolist(),
            log_probs_t.cpu().tolist(),
            values,
        )

    # ------------------------------------------------------------------ #
    # Buffer management                                                    #
    # ------------------------------------------------------------------ #

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
        """Store one environment step in the rollout buffer."""
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

    # ------------------------------------------------------------------ #
    # Training update                                                      #
    # ------------------------------------------------------------------ #

    def update(
        self,
        last_observations: Optional[List[np.ndarray]] = None,
        last_dones: Optional[List[bool]] = None,
    ) -> Dict[str, float]:
        """
        Run PPO update on the collected rollout buffer.

        Parameters
        ----------
        last_observations : List[np.ndarray], optional
            Observations after the final step (for bootstrap value).
            If None, terminal bootstrapping uses value = 0.
        last_dones : List[bool], optional
            Done flags for the last step.

        Returns
        -------
        Dict with keys: actor_loss, critic_loss, entropy, approx_kl
        """
        # Bootstrap last value
        if last_observations is not None:
            obs_np = np.array(last_observations, dtype=np.float32)
            obs_t = torch.as_tensor(obs_np).to(self.device)
            with torch.no_grad():
                global_t = obs_t.flatten().unsqueeze(0)
                last_v = self.critic(global_t).item()
                if last_dones is not None and all(last_dones):
                    last_v = 0.0
            last_values = np.full(self.num_agents, last_v, dtype=np.float32)
        else:
            last_values = np.zeros(self.num_agents, dtype=np.float32)

        advantages, returns = self.buffer.compute_advantages_and_returns(last_values)

        metrics: Dict[str, float] = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
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

                # Actor update
                dist = self.actor.get_distribution(obs_b)
                new_lp = dist.log_prob(act_b)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic update
                values_pred = self.critic(gobs_b)
                critic_loss = F.mse_loss(values_pred, ret_b)

                # Combined loss
                total_loss = (
                    actor_loss
                    - self.entropy_coef * entropy
                    + self.value_coef * critic_loss
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

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save actor and critic weights to *path*."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "train_step": self.train_step,
                "total_env_steps": self.total_env_steps,
                "obs_size": self.obs_size,
                "action_size": self.action_size,
                "num_agents": self.num_agents,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load actor and critic weights from *path*."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "actor_optim" in ckpt:
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.train_step = ckpt.get("train_step", 0)
        self.total_env_steps = ckpt.get("total_env_steps", 0)

    @classmethod
    def from_config(cls, obs_size: int, action_size: int, num_agents: int, cfg: dict) -> "MAPPOAgent":
        """Construct from a config dict (e.g. loaded from default.yaml mappo section)."""
        return cls(
            obs_size=obs_size,
            action_size=action_size,
            num_agents=num_agents,
            hidden_size=cfg.get("hidden_size", 128),
            critic_hidden_size=cfg.get("critic_hidden_size", 256),
            lr_actor=cfg.get("lr_actor", 3e-4),
            lr_critic=cfg.get("lr_critic", 1e-3),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            clip_eps=cfg.get("clip_eps", 0.2),
            entropy_coef=cfg.get("entropy_coef", 0.01),
            value_coef=cfg.get("value_coef", 0.5),
            max_grad_norm=cfg.get("max_grad_norm", 0.5),
            n_epochs=cfg.get("n_epochs", 10),
            batch_size=cfg.get("batch_size", 64),
            rollout_steps=cfg.get("rollout_steps", 2048),
            device=cfg.get("device", "cpu"),
        )
