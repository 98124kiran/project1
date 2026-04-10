"""
Edge Inference Engine — simulates deploying a lightweight policy on edge nodes.

In a real hybrid-compute deployment:
- Edge nodes run fast, low-latency inference (no training).
- Each edge node holds a frozen copy of the current policy.
- Inference latency is bounded by local compute + network round-trip.

This module simulates that behaviour:
- Applies a configurable latency to each inference call.
- Throttles the number of inferences per second based on bandwidth.
- Supports uploading experience to the cloud (bandwidth-limited).
- Provides metrics on inference latency and throughput.

Usage
-----
>>> from hybrid_compute.edge_inference import EdgeInferenceEngine
>>> engine = EdgeInferenceEngine(actor_network, bandwidth_mbps=10.0)
>>> actions, lp, v = engine.infer(observations)
>>> engine.upload_experience(experience_bundle)  # simulates cloud upload
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# --------------------------------------------------------------------------- #
# Bandwidth / latency simulation helpers                                        #
# --------------------------------------------------------------------------- #

def _simulate_latency(latency_ms: float) -> None:
    """Block for *latency_ms* milliseconds (simulates network/compute delay)."""
    if latency_ms > 0:
        time.sleep(latency_ms / 1000.0)


def _estimate_transfer_time_ms(
    data_size_bytes: int,
    bandwidth_mbps: float,
) -> float:
    """
    Estimate transfer time in milliseconds for a payload of *data_size_bytes*
    over a link of *bandwidth_mbps* megabits per second.
    """
    if bandwidth_mbps <= 0:
        return 0.0
    bits = data_size_bytes * 8
    bandwidth_bps = bandwidth_mbps * 1_000_000
    return (bits / bandwidth_bps) * 1000.0  # ms


# --------------------------------------------------------------------------- #
# Edge Inference Engine                                                         #
# --------------------------------------------------------------------------- #

class EdgeInferenceEngine:
    """
    Simulated edge-side inference engine.

    Wraps a PyTorch actor network and adds:
    - Simulated inference latency (fixed + network jitter).
    - Experience buffer for uploading to the cloud.
    - Bandwidth-limited upload simulation.
    - Per-call metrics tracking.

    Parameters
    ----------
    actor : nn.Module
        The actor network (must expose a ``get_distribution`` method or
        return logits from ``forward``).
    node_id : int
        Identifier of this edge node.
    base_latency_ms : float
        Fixed compute latency per inference call (milliseconds).
    network_latency_ms : float
        Additional network round-trip latency (milliseconds).
    bandwidth_mbps : float
        Simulated upload bandwidth in megabits per second.
    simulate_delays : bool
        If True, actually sleep to simulate latency. Set to False for
        unit-testing / fast evaluation.
    device : str
        PyTorch device for inference ("cpu" recommended for edge nodes).
    """

    def __init__(
        self,
        actor: nn.Module,
        node_id: int = 0,
        base_latency_ms: float = 2.0,
        network_latency_ms: float = 3.0,
        bandwidth_mbps: float = 10.0,
        simulate_delays: bool = False,
        device: str = "cpu",
    ) -> None:
        self.actor = actor
        self.actor.eval()
        self.node_id = node_id
        self.base_latency_ms = base_latency_ms
        self.network_latency_ms = network_latency_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.simulate_delays = simulate_delays
        self.device = torch.device(device)
        self.actor.to(self.device)

        # Experience buffer (for cloud upload)
        self._exp_buffer: List[Dict] = []

        # Metrics
        self._n_inferences: int = 0
        self._total_latency_ms: float = 0.0
        self._n_uploads: int = 0
        self._total_upload_ms: float = 0.0

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def infer(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[List[int], List[float], np.ndarray]:
        """
        Run inference for a list of agent observations.

        Parameters
        ----------
        observations : List[np.ndarray]
            One observation array per agent at this node.
        deterministic : bool
            If True, take greedy actions.

        Returns
        -------
        actions, log_probs, values (placeholder zeros)
        """
        start = time.perf_counter()

        obs_np = np.array(observations, dtype=np.float32)
        obs_t = torch.as_tensor(obs_np).to(self.device)

        with torch.no_grad():
            logits = self.actor(obs_t)
            dist = Categorical(logits=logits)
            if deterministic:
                actions_t = dist.probs.argmax(dim=-1)
            else:
                actions_t = dist.sample()
            log_probs_t = dist.log_prob(actions_t)

        values = np.zeros(len(observations), dtype=np.float32)
        actions = actions_t.cpu().tolist()
        log_probs = log_probs_t.cpu().tolist()

        # Simulate combined latency
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total_latency = self.base_latency_ms + self.network_latency_ms
        remaining_sleep = max(0.0, total_latency - elapsed_ms)
        if self.simulate_delays:
            _simulate_latency(remaining_sleep)

        self._n_inferences += 1
        self._total_latency_ms += total_latency

        return actions, log_probs, values

    # ------------------------------------------------------------------ #
    # Experience collection and upload                                     #
    # ------------------------------------------------------------------ #

    def store_experience(self, experience: Dict) -> None:
        """
        Store a transition dictionary in the local experience buffer.

        Parameters
        ----------
        experience : dict
            Should contain: obs, action, reward, next_obs, done, log_prob.
        """
        self._exp_buffer.append(experience)

    def upload_experience(self, max_size: Optional[int] = None) -> Tuple[List[Dict], float]:
        """
        Simulate uploading experience to the cloud.

        Computes transfer time based on buffer size and bandwidth.
        Returns the experience batch and the simulated transfer time (ms).

        Parameters
        ----------
        max_size : int, optional
            Maximum number of transitions to upload. If None, upload all.

        Returns
        -------
        experience_batch : List[Dict]
        transfer_time_ms : float
        """
        if max_size is not None:
            batch = self._exp_buffer[:max_size]
            self._exp_buffer = self._exp_buffer[max_size:]
        else:
            batch = self._exp_buffer.copy()
            self._exp_buffer.clear()

        # Estimate payload size: 4 bytes per float, rough estimate
        n_floats = sum(
            len(str(exp)) for exp in batch
        ) * 0.25  # rough byte count
        data_bytes = int(n_floats * 4)
        transfer_ms = _estimate_transfer_time_ms(data_bytes, self.bandwidth_mbps)

        if self.simulate_delays:
            _simulate_latency(transfer_ms)

        self._n_uploads += 1
        self._total_upload_ms += transfer_ms

        return batch, transfer_ms

    # ------------------------------------------------------------------ #
    # Weight synchronisation                                               #
    # ------------------------------------------------------------------ #

    def sync_weights(self, state_dict: dict, model_size_mb: float = 1.0) -> float:
        """
        Synchronise actor weights from the cloud (simulates download latency).

        Parameters
        ----------
        state_dict : dict
            PyTorch state dict of the updated actor.
        model_size_mb : float
            Approximate model size in megabytes (for latency simulation).

        Returns
        -------
        sync_latency_ms : float — simulated download time
        """
        data_bytes = int(model_size_mb * 1024 * 1024)
        latency_ms = _estimate_transfer_time_ms(data_bytes, self.bandwidth_mbps)
        latency_ms += self.network_latency_ms  # round-trip overhead

        if self.simulate_delays:
            _simulate_latency(latency_ms)

        self.actor.load_state_dict(state_dict)
        self.actor.eval()
        return latency_ms

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def get_metrics(self) -> Dict[str, float]:
        """Return performance metrics for this edge node."""
        avg_latency = (
            self._total_latency_ms / self._n_inferences
            if self._n_inferences > 0
            else 0.0
        )
        avg_upload = (
            self._total_upload_ms / self._n_uploads
            if self._n_uploads > 0
            else 0.0
        )
        return {
            "node_id": float(self.node_id),
            "n_inferences": float(self._n_inferences),
            "avg_inference_latency_ms": avg_latency,
            "n_uploads": float(self._n_uploads),
            "avg_upload_latency_ms": avg_upload,
            "exp_buffer_size": float(len(self._exp_buffer)),
        }

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self._n_inferences = 0
        self._total_latency_ms = 0.0
        self._n_uploads = 0
        self._total_upload_ms = 0.0
