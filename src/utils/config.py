"""
Configuration management for MADRL edge computing scheduler.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EnvConfig:
    num_edge_nodes: int = 5
    num_task_types: int = 4
    max_tasks_per_step: int = 3
    max_queue_size: int = 20
    max_steps: int = 200
    task_arrival_rate: float = 0.7          # Poisson arrival rate
    node_failure_prob: float = 0.02         # probability per step
    node_recovery_prob: float = 0.1         # probability per step when failed
    dynamic_load_std: float = 0.1           # std for background load fluctuation
    time_slot_duration: float = 1.0         # seconds per time slot


@dataclass
class TaskConfig:
    # [min, max] for each task type: [cpu, memory, deadline, priority, size_mb]
    task_types: List[dict] = field(default_factory=lambda: [
        {"name": "welding_control",   "cpu": (2, 5),  "memory": (1, 3),  "deadline": (5, 15),  "priority": 3, "size_mb": (10, 50)},
        {"name": "vision_inspection", "cpu": (4, 8),  "memory": (4, 8),  "deadline": (3, 10),  "priority": 4, "size_mb": (50, 200)},
        {"name": "assembly_planning", "cpu": (6, 12), "memory": (8, 16), "deadline": (10, 30), "priority": 2, "size_mb": (20, 100)},
        {"name": "anomaly_detection", "cpu": (3, 6),  "memory": (2, 6),  "deadline": (2, 8),   "priority": 5, "size_mb": (5, 30)},
    ])


@dataclass
class NodeConfig:
    cpu_capacities: List[float] = field(default_factory=lambda: [16.0, 12.0, 20.0, 8.0, 16.0])
    memory_capacities: List[float] = field(default_factory=lambda: [32.0, 24.0, 48.0, 16.0, 32.0])
    # network bandwidth to cloud (Mbps)
    bandwidth: List[float] = field(default_factory=lambda: [100.0, 50.0, 200.0, 30.0, 100.0])


@dataclass
class AgentConfig:
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.01               # soft update coefficient
    buffer_size: int = 100_000
    batch_size: int = 256
    hidden_dim: int = 256
    num_layers: int = 3
    noise_std: float = 0.1          # exploration noise std
    noise_decay: float = 0.9995     # noise decay per step
    min_noise_std: float = 0.01
    update_every: int = 4           # steps between gradient updates
    warmup_steps: int = 1000        # random actions before training starts


@dataclass
class TrainingConfig:
    num_episodes: int = 3000
    eval_interval: int = 100
    eval_episodes: int = 10
    save_interval: int = 500
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int = 42


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    node: NodeConfig = field(default_factory=NodeConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        cfg = cls()
        if "env" in data:
            cfg.env = EnvConfig(**data["env"])
        if "agent" in data:
            cfg.agent = AgentConfig(**data["agent"])
        if "training" in data:
            cfg.training = TrainingConfig(**data["training"])
        return cfg

    def to_yaml(self, path: str) -> None:
        import dataclasses

        def _convert(obj):
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj

        with open(path, "w") as f:
            yaml.dump(_convert(dataclasses.asdict(self)), f, default_flow_style=False)
