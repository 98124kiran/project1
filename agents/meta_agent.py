"""
First-Order MAML (FOMAML) meta-learning agent for adaptive replanning.

Motivation
----------
When a disruption occurs (e.g. machine failure at step 100), the current
policy may be sub-optimal for the new environment dynamics.  MAML trains
the *meta-weights* θ such that a small number of gradient steps on
disruption-episode data produces an adapted policy θ' that performs well
under the disruption.

Algorithm (FOMAML)
------------------
Outer loop  (meta-training):
    For each episode:
        1. Run K inner gradient steps using recent trajectory → θ'
        2. Collect a new trajectory with θ'
        3. Update meta-weights θ using the gradient of the outer loss
           evaluated at θ' (first-order approximation — no 2nd derivatives).

Inner loop  (adaptation at test time):
    adapt(disruption_trajectories, steps=5) → θ'
    Use θ' for the remainder of the disrupted episode.

Reference
---------
Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast Adaptation of
Deep Neural Networks."  (FOMAML variant, Algorithm 2.)

Usage
-----
>>> from agents.meta_agent import MetaAgent
>>> agent = MetaAgent(obs_size=19, action_size=9, num_agents=3)
>>> # Collect disruption trajectory
>>> agent.adapt(disruption_obs, disruption_actions, disruption_returns, steps=5)
>>> # Now agent uses adapted weights for inference
>>> actions, lp, v = agent.select_actions(obs)
>>> # Restore meta-weights for the next episode
>>> agent.restore_meta_weights()
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.ppo_agent import ActorNetwork, CriticNetwork, RolloutBuffer


# --------------------------------------------------------------------------- #
# MetaAgent                                                                    #
# --------------------------------------------------------------------------- #

class MetaAgent:
    """
    FOMAML-based meta-learning agent for adaptive replanning.

    Maintains two sets of actor weights:
    - *meta_weights* (θ): the base policy trained by the outer loop.
    - *adapted_weights* (θ'): task-specific weights produced by the
      inner loop during adaptation.  Inference uses adapted_weights when
      available, otherwise falls back to meta_weights.

    The centralized critic uses a separate meta-weight/adapted-weight
    pair for value estimation.

    Parameters
    ----------
    obs_size, action_size, num_agents : int
    hidden_size : int        — actor MLP hidden size
    critic_hidden_size : int — critic MLP hidden size
    inner_lr : float         — inner-loop (adaptation) learning rate
    inner_steps : int        — gradient steps in inner loop during meta-training
    meta_lr : float          — outer-loop (meta) learning rate
    adapt_steps : int        — gradient steps during test-time adaptation
    gamma : float            — discount factor
    gae_lambda : float       — GAE lambda
    clip_eps : float         — PPO clip epsilon
    entropy_coef : float
    value_coef : float
    max_grad_norm : float
    rollout_steps : int
    batch_size : int
    device : str
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        num_agents: int,
        hidden_size: int = 128,
        critic_hidden_size: int = 256,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        meta_lr: float = 3e-4,
        adapt_steps: int = 3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 2048,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.obs_size = obs_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.global_obs_size = obs_size * num_agents
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.adapt_steps = adapt_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Meta (base) networks
        self.actor = ActorNetwork(obs_size, action_size, hidden_size).to(self.device)
        self.critic = CriticNetwork(self.global_obs_size, critic_hidden_size).to(self.device)

        # Meta optimisers (outer loop)
        self.meta_actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=meta_lr, eps=1e-5
        )
        self.meta_critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=meta_lr, eps=1e-5
        )

        # Adapted networks (inner loop); initially None (use meta weights)
        self._adapted_actor: Optional[ActorNetwork] = None
        self._adapted_critic: Optional[CriticNetwork] = None
        self._is_adapted: bool = False

        # Rollout buffer (shared)
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
    # Properties: active networks                                          #
    # ------------------------------------------------------------------ #

    @property
    def _active_actor(self) -> ActorNetwork:
        return self._adapted_actor if self._is_adapted else self.actor

    @property
    def _active_critic(self) -> CriticNetwork:
        return self._adapted_critic if self._is_adapted else self.critic

    # ------------------------------------------------------------------ #
    # Decentralised execution                                              #
    # ------------------------------------------------------------------ #

    def select_actions(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[List[int], List[float], np.ndarray]:
        """Use adapted weights if available, otherwise meta weights."""
        obs_np = np.array(observations, dtype=np.float32)
        obs_t = torch.as_tensor(obs_np).to(self.device)

        with torch.no_grad():
            dist = self._active_actor.get_distribution(obs_t)
            if deterministic:
                actions_t = dist.probs.argmax(dim=-1)
            else:
                actions_t = dist.sample()
            log_probs_t = dist.log_prob(actions_t)
            global_obs_t = obs_t.flatten().unsqueeze(0)
            value_scalar = self._active_critic(global_obs_t).item()
            values = np.full(self.num_agents, value_scalar, dtype=np.float32)

        return actions_t.cpu().tolist(), log_probs_t.cpu().tolist(), values

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
    # Inner loop — adaptation                                              #
    # ------------------------------------------------------------------ #

    def adapt(
        self,
        observations: np.ndarray,   # (T, N, obs_size)
        actions: np.ndarray,        # (T, N)
        returns: np.ndarray,        # (T, N)
        steps: Optional[int] = None,
    ) -> None:
        """
        Adapt the policy to a disruption context using inner-loop gradient steps.

        Parameters
        ----------
        observations : np.ndarray  shape (T, N, obs_size)
        actions      : np.ndarray  shape (T, N)
        returns      : np.ndarray  shape (T, N) — discounted returns for PG loss
        steps        : int, optional — override adapt_steps
        """
        n_steps = steps if steps is not None else self.adapt_steps

        # Clone meta networks for adaptation
        adapted_actor = copy.deepcopy(self.actor)
        adapted_critic = copy.deepcopy(self.critic)

        # Inner-loop optimisers (SGD for clean gradient flow)
        inner_actor_optim = torch.optim.SGD(
            adapted_actor.parameters(), lr=self.inner_lr
        )
        inner_critic_optim = torch.optim.SGD(
            adapted_critic.parameters(), lr=self.inner_lr
        )

        T, N = observations.shape[:2]
        obs_flat = observations.reshape(T * N, self.obs_size)
        act_flat = actions.reshape(T * N)
        ret_flat = returns.reshape(T * N)
        # Global obs: concatenate all agent obs per time step, zero-pad to global_obs_size
        gobs_flat = np.zeros((T * N, self.global_obs_size), dtype=np.float32)
        for t in range(T):
            row_obs = observations[t].flatten()  # shape (N * obs_size,) or smaller
            fill_len = min(len(row_obs), self.global_obs_size)
            for n in range(N):
                gobs_flat[t * N + n, :fill_len] = row_obs[:fill_len]

        obs_t = torch.as_tensor(obs_flat).to(self.device)
        act_t = torch.as_tensor(act_flat).to(self.device)
        ret_t = torch.as_tensor(ret_flat).to(self.device)
        gobs_t = torch.as_tensor(gobs_flat).to(self.device)

        adapted_actor.train()
        adapted_critic.train()

        for _ in range(n_steps):
            dist = adapted_actor.get_distribution(obs_t)
            log_probs = dist.log_prob(act_t)
            values = adapted_critic(gobs_t)

            # Simple policy gradient loss + value loss
            actor_loss = -(log_probs * (ret_t - values.detach())).mean()
            critic_loss = F.mse_loss(values, ret_t)
            entropy = dist.entropy().mean()

            total_loss = (
                actor_loss
                - self.entropy_coef * entropy
                + self.value_coef * critic_loss
            )

            inner_actor_optim.zero_grad()
            inner_critic_optim.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(adapted_actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(adapted_critic.parameters(), self.max_grad_norm)
            inner_actor_optim.step()
            inner_critic_optim.step()

        adapted_actor.eval()
        adapted_critic.eval()

        self._adapted_actor = adapted_actor
        self._adapted_critic = adapted_critic
        self._is_adapted = True

    def restore_meta_weights(self) -> None:
        """Discard adapted weights and revert to meta weights."""
        self._adapted_actor = None
        self._adapted_critic = None
        self._is_adapted = False

    # ------------------------------------------------------------------ #
    # Outer loop — meta update (FOMAML)                                   #
    # ------------------------------------------------------------------ #

    def update(
        self,
        last_observations: Optional[List[np.ndarray]] = None,
        last_dones: Optional[List[bool]] = None,
    ) -> Dict[str, float]:
        """
        FOMAML outer-loop update.

        Collects the current rollout, runs inner-loop adaptation on the
        first half, then evaluates on the second half (outer loss) and
        updates the meta weights.  Falls back to standard PPO update
        when no disruption data is available.
        """
        if last_observations is not None:
            obs_t = torch.as_tensor(
                np.array(last_observations, dtype=np.float32)
            ).to(self.device)
            with torch.no_grad():
                g_t = obs_t.flatten().unsqueeze(0)
                last_v = self._active_critic(g_t).item()
                if last_dones is not None and all(last_dones):
                    last_v = 0.0
            last_values = np.full(self.num_agents, last_v, dtype=np.float32)
        else:
            last_values = np.zeros(self.num_agents, dtype=np.float32)

        advantages, returns = self.buffer.compute_advantages_and_returns(last_values)

        # --- Inner loop on first half of buffer --------------------------
        T = self.rollout_steps
        half = T // 2
        inner_obs = self.buffer.observations[:half]    # (T//2, N, obs_size)
        inner_act = self.buffer.actions[:half]          # (T//2, N)
        inner_ret = returns[:half]                      # (T//2, N)

        # Clone meta weights for inner-loop update
        tmp_actor = copy.deepcopy(self.actor)
        tmp_critic = copy.deepcopy(self.critic)
        inner_a_opt = torch.optim.SGD(tmp_actor.parameters(), lr=self.inner_lr)
        inner_c_opt = torch.optim.SGD(tmp_critic.parameters(), lr=self.inner_lr)

        obs_i = torch.as_tensor(inner_obs.reshape(-1, self.obs_size)).to(self.device)
        act_i = torch.as_tensor(inner_act.reshape(-1)).to(self.device)
        ret_i = torch.as_tensor(inner_ret.reshape(-1)).to(self.device)
        gobs_i = torch.zeros(len(obs_i), self.global_obs_size, device=self.device)
        for t in range(half):
            for n in range(self.num_agents):
                gobs_i[t * self.num_agents + n] = torch.as_tensor(
                    inner_obs[t].flatten()
                ).to(self.device)

        tmp_actor.train()
        tmp_critic.train()
        for _ in range(self.inner_steps):
            dist_i = tmp_actor.get_distribution(obs_i)
            lp_i = dist_i.log_prob(act_i)
            v_i = tmp_critic(gobs_i)
            loss_i = -(lp_i * (ret_i - v_i.detach())).mean() + self.value_coef * F.mse_loss(v_i, ret_i)
            inner_a_opt.zero_grad()
            inner_c_opt.zero_grad()
            loss_i.backward()
            inner_a_opt.step()
            inner_c_opt.step()

        # --- Outer loss on second half -----------------------------------
        outer_obs = self.buffer.observations[half:]
        outer_act = self.buffer.actions[half:]
        outer_adv = advantages[half:]
        outer_ret = returns[half:]

        obs_o = torch.as_tensor(outer_obs.reshape(-1, self.obs_size)).to(self.device)
        act_o = torch.as_tensor(outer_act.reshape(-1)).to(self.device)
        adv_o = torch.as_tensor(outer_adv.reshape(-1)).to(self.device)
        ret_o = torch.as_tensor(outer_ret.reshape(-1)).to(self.device)
        # Normalise advantages
        adv_o = (adv_o - adv_o.mean()) / (adv_o.std() + 1e-8)

        gobs_o = torch.zeros(len(obs_o), self.global_obs_size, device=self.device)
        T_outer = len(outer_obs)
        for t in range(T_outer):
            for n in range(self.num_agents):
                gobs_o[t * self.num_agents + n] = torch.as_tensor(
                    outer_obs[t].flatten()
                ).to(self.device)

        tmp_actor.eval()
        tmp_critic.eval()

        # Evaluate outer loss with adapted weights (FOMAML: no 2nd order)
        dist_o = tmp_actor.get_distribution(obs_o)
        new_lp_o = dist_o.log_prob(act_o)
        # Need old log-probs from collection
        with torch.no_grad():
            dist_meta = self.actor.get_distribution(obs_o)
            old_lp_o = dist_meta.log_prob(act_o)

        ratio = torch.exp(new_lp_o - old_lp_o.detach())
        surr1 = ratio * adv_o
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_o
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy = dist_o.entropy().mean()
        values_o = tmp_critic(gobs_o)
        critic_loss = F.mse_loss(values_o, ret_o)

        meta_loss = actor_loss - self.entropy_coef * entropy + self.value_coef * critic_loss

        # Update meta weights using gradients of meta_loss w.r.t. θ_meta
        # In FOMAML we treat ∇θ'(L_outer) ≈ ∇θ(L_outer) (ignore second order)
        self.meta_actor_optim.zero_grad()
        self.meta_critic_optim.zero_grad()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(tmp_actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(tmp_critic.parameters(), self.max_grad_norm)
        # Copy gradients from adapted to meta weights
        for meta_p, adapted_p in zip(self.actor.parameters(), tmp_actor.parameters()):
            if adapted_p.grad is not None:
                meta_p.grad = adapted_p.grad.clone()
        for meta_p, adapted_p in zip(self.critic.parameters(), tmp_critic.parameters()):
            if adapted_p.grad is not None:
                meta_p.grad = adapted_p.grad.clone()
        self.meta_actor_optim.step()
        self.meta_critic_optim.step()

        self.buffer.reset()
        self.train_step += 1

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "meta_loss": meta_loss.item(),
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "meta_actor_optim": self.meta_actor_optim.state_dict(),
                "meta_critic_optim": self.meta_critic_optim.state_dict(),
                "train_step": self.train_step,
                "total_env_steps": self.total_env_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "meta_actor_optim" in ckpt:
            self.meta_actor_optim.load_state_dict(ckpt["meta_actor_optim"])
        if "meta_critic_optim" in ckpt:
            self.meta_critic_optim.load_state_dict(ckpt["meta_critic_optim"])
        self.train_step = ckpt.get("train_step", 0)
        self.total_env_steps = ckpt.get("total_env_steps", 0)
        self.restore_meta_weights()

    @classmethod
    def from_config(
        cls,
        obs_size: int,
        action_size: int,
        num_agents: int,
        cfg: dict,
    ) -> "MetaAgent":
        meta_cfg = cfg.get("meta", {})
        mappo_cfg = cfg.get("mappo", {})
        return cls(
            obs_size=obs_size,
            action_size=action_size,
            num_agents=num_agents,
            hidden_size=mappo_cfg.get("hidden_size", 128),
            critic_hidden_size=mappo_cfg.get("critic_hidden_size", 256),
            inner_lr=meta_cfg.get("inner_lr", 0.01),
            inner_steps=meta_cfg.get("inner_steps", 5),
            meta_lr=meta_cfg.get("meta_lr", 3e-4),
            adapt_steps=meta_cfg.get("adapt_steps", 3),
            gamma=mappo_cfg.get("gamma", 0.99),
            gae_lambda=mappo_cfg.get("gae_lambda", 0.95),
            clip_eps=mappo_cfg.get("clip_eps", 0.2),
            entropy_coef=mappo_cfg.get("entropy_coef", 0.01),
            value_coef=mappo_cfg.get("value_coef", 0.5),
            max_grad_norm=mappo_cfg.get("max_grad_norm", 0.5),
            rollout_steps=mappo_cfg.get("rollout_steps", 2048),
            batch_size=mappo_cfg.get("batch_size", 64),
            device=cfg.get("training", {}).get("device", "cpu"),
        )
