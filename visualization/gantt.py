"""
Visualization utilities for the DRL adaptive scheduling project.

Provides
--------
plot_gantt           : Gantt chart of machine schedules.
plot_learning_curves : Reward / loss curves over training.
plot_metrics_comparison : Bar chart comparing agents on KPIs.
plot_disruption_timeline : Timeline showing disruption events + recovery.
save_figure          : Save a Matplotlib figure to disk.

All functions return Matplotlib Figure objects so that callers can either
display them interactively (plt.show()) or save them to disk.

Usage
-----
>>> from visualization.gantt import plot_gantt, plot_learning_curves
>>> fig = plot_learning_curves(rewards_dict={"MAPPO": [1.2, 2.3, ...],
...                                          "FIFO":  [0.5, 0.6, ...]})
>>> fig.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; switch to "TkAgg" locally
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# --------------------------------------------------------------------------- #
# Colour palette                                                               #
# --------------------------------------------------------------------------- #

_AGENT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

_STATUS_COLORS = {
    "BUSY": "#55A868",
    "IDLE": "#E8E8E8",
    "FAILED": "#C44E52",
}


# --------------------------------------------------------------------------- #
# 1. Gantt chart                                                               #
# --------------------------------------------------------------------------- #

def plot_gantt(
    schedule: List[Dict],
    num_machines: int,
    title: str = "Machine Schedule (Gantt Chart)",
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot a Gantt chart of the machine schedule.

    Parameters
    ----------
    schedule : List[Dict]
        Each dict represents one job-operation assignment:
            {
                "machine_id"  : int,
                "job_id"      : int,
                "start"       : float,
                "end"         : float,
                "node_id"     : int,      # optional
                "is_urgent"   : bool,     # optional
            }
    num_machines : int
        Total number of machines (for y-axis labelling).
    title : str
        Plot title.
    figsize : Tuple[int, int]

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    used_jobs: set = set()

    for entry in schedule:
        machine_id = entry["machine_id"]
        job_id = entry["job_id"]
        start = entry["start"]
        end = entry["end"]
        node_id = entry.get("node_id", 0)
        is_urgent = entry.get("is_urgent", False)

        color = _AGENT_COLORS[job_id % len(_AGENT_COLORS)]
        edgecolor = "red" if is_urgent else "black"
        linewidth = 2 if is_urgent else 0.5

        ax.barh(
            y=machine_id,
            width=end - start,
            left=start,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=0.85,
        )
        ax.text(
            x=start + (end - start) / 2,
            y=machine_id,
            s=f"J{job_id}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
        used_jobs.add(job_id)

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {i}" for i in range(num_machines)])
    ax.set_xlabel("Time (steps)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Legend: one patch per unique job
    patches = [
        mpatches.Patch(
            color=_AGENT_COLORS[jid % len(_AGENT_COLORS)],
            label=f"Job {jid}",
        )
        for jid in sorted(used_jobs)[:10]  # cap at 10 entries
    ]
    if patches:
        ax.legend(handles=patches, loc="lower right", fontsize=8, ncol=2)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 2. Learning curves                                                            #
# --------------------------------------------------------------------------- #

def plot_learning_curves(
    rewards_dict: Dict[str, List[float]],
    window: int = 10,
    title: str = "Training Reward Curves",
    xlabel: str = "Episode",
    ylabel: str = "Mean Reward",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot smoothed learning curves for multiple agents.

    Parameters
    ----------
    rewards_dict : Dict[str, List[float]]
        Mapping of agent name → list of per-episode total rewards.
    window : int
        Smoothing window for rolling mean.
    title, xlabel, ylabel : str
    figsize : Tuple[int, int]

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, rewards) in enumerate(rewards_dict.items()):
        rewards_arr = np.array(rewards, dtype=np.float32)
        episodes = np.arange(1, len(rewards_arr) + 1)
        color = _AGENT_COLORS[i % len(_AGENT_COLORS)]

        # Raw rewards (transparent)
        ax.plot(episodes, rewards_arr, color=color, alpha=0.2, linewidth=0.8)

        # Smoothed rewards
        if len(rewards_arr) >= window:
            smoothed = np.convolve(
                rewards_arr, np.ones(window) / window, mode="valid"
            )
            smooth_ep = episodes[window - 1:]
            ax.plot(smooth_ep, smoothed, color=color, linewidth=2.0, label=name)
        else:
            ax.plot(episodes, rewards_arr, color=color, linewidth=2.0, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 3. Metrics comparison bar chart                                               #
# --------------------------------------------------------------------------- #

def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
    title: str = "Agent Performance Comparison",
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing multiple agents across KPIs.

    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, float]]
        Outer key = agent name; inner dict = {metric_name: value}.
    metrics_to_plot : List[str], optional
        Subset of metrics to include.  Defaults to all keys in the first entry.
    title : str
    figsize : Tuple[int, int], optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    agents = list(metrics_dict.keys())
    if metrics_to_plot is None:
        metrics_to_plot = list(next(iter(metrics_dict.values())).keys())

    n_metrics = len(metrics_to_plot)
    n_agents = len(agents)
    bar_width = 0.8 / n_agents

    if figsize is None:
        figsize = (max(8, n_metrics * 2), 5)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_metrics)
    for i, agent in enumerate(agents):
        values = [metrics_dict[agent].get(m, 0.0) for m in metrics_to_plot]
        offset = (i - n_agents / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=agent,
            color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        # Value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(1, abs(bar.get_height())),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", " ").title() for m in metrics_to_plot],
        rotation=15,
        ha="right",
    )
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. Disruption timeline                                                        #
# --------------------------------------------------------------------------- #

def plot_disruption_timeline(
    reward_before: List[float],
    reward_after: List[float],
    disruption_step: int,
    agent_labels: Optional[Dict[str, List[float]]] = None,
    title: str = "Reward Before and After Disruption",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Visualise how reward changes before and after a disruption event.

    Parameters
    ----------
    reward_before : List[float]
        Per-step rewards before the disruption.
    reward_after  : List[float]
        Per-step rewards after the disruption (recovery phase).
    disruption_step : int
        The step at which the disruption occurred.
    agent_labels : Dict[str, List[float]], optional
        Additional curves for comparison (e.g., baselines).
    title : str
    figsize : Tuple[int, int]

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    all_rewards = list(reward_before) + list(reward_after)
    steps = np.arange(len(all_rewards))

    ax.plot(
        steps[: len(reward_before)],
        reward_before,
        color=_AGENT_COLORS[0],
        linewidth=2,
        label="Before disruption",
    )
    ax.plot(
        steps[len(reward_before):],
        reward_after,
        color=_AGENT_COLORS[2],
        linewidth=2,
        label="After disruption (recovery)",
    )

    if agent_labels:
        for i, (name, curve) in enumerate(agent_labels.items(), start=3):
            ax.plot(
                np.arange(len(curve)),
                curve,
                color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                linewidth=1.5,
                linestyle="--",
                label=name,
            )

    ax.axvline(
        x=disruption_step,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Disruption @ step {disruption_step}",
    )
    ax.fill_betweenx(
        [min(all_rewards) * 1.1, max(all_rewards) * 1.1],
        disruption_step,
        steps[-1],
        alpha=0.05,
        color="red",
    )

    ax.set_xlabel("Environment Step")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. Ablation study grid                                                        #
# --------------------------------------------------------------------------- #

def plot_ablation(
    ablation_results: Dict[str, Dict[str, float]],
    primary_metric: str = "mean_reward",
    title: str = "Ablation Study",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Horizontal bar chart for ablation study results.

    Parameters
    ----------
    ablation_results : Dict[str, Dict[str, float]]
        variant_name → {metric: value}
    primary_metric : str
        The metric to rank variants by.
    title : str
    figsize : Tuple[int, int]

    Returns
    -------
    matplotlib.figure.Figure
    """
    variants = list(ablation_results.keys())
    values = [ablation_results[v].get(primary_metric, 0.0) for v in variants]

    # Sort by value descending
    sorted_pairs = sorted(zip(values, variants), reverse=True)
    values_sorted = [v for v, _ in sorted_pairs]
    variants_sorted = [n for _, n in sorted_pairs]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(len(variants_sorted))]
    bars = ax.barh(variants_sorted, values_sorted, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values_sorted):
        ax.text(
            bar.get_width() + 0.01 * max(1, abs(bar.get_width())),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel(primary_metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Utility                                                                       #
# --------------------------------------------------------------------------- #

def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """
    Save *fig* to *path*, creating parent directories if needed.

    Parameters
    ----------
    fig  : matplotlib.figure.Figure
    path : str   — file path (e.g. "results/learning_curve.png")
    dpi  : int
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
