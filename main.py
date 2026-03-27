"""
Main entry point for MADRL Edge Computing Scheduler.

Usage:
    python main.py train                    # train with default config
    python main.py train --config config.yaml
    python main.py eval  --checkpoint checkpoints/best
    python main.py eval  --checkpoint checkpoints/best --episodes 50
"""

import argparse
import sys
import os

# Ensure the repo root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Agent DRL Scheduler for Edge Computing in Smart Manufacturing"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train MADDPG agents")
    train_parser.add_argument("--config", type=str, default=None,
                              help="Path to YAML config file")
    train_parser.add_argument("--episodes", type=int, default=None,
                              help="Override number of training episodes")
    train_parser.add_argument("--device", type=str, default=None,
                              help="Device: 'cpu' or 'cuda'")
    train_parser.add_argument("--seed", type=int, default=None,
                              help="Random seed")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained agents")
    eval_parser.add_argument("--config", type=str, default=None,
                             help="Path to YAML config file")
    eval_parser.add_argument("--checkpoint", type=str, default=None,
                             help="Checkpoint directory to load")
    eval_parser.add_argument("--episodes", type=int, default=20,
                             help="Number of evaluation episodes")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load / build config
    if getattr(args, "config", None) and os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Apply CLI overrides
    if hasattr(args, "episodes") and args.episodes is not None:
        if args.command == "train":
            config.training.num_episodes = args.episodes
    if hasattr(args, "seed") and args.seed is not None:
        config.training.seed = args.seed

    if args.command == "train":
        trainer = Trainer(config, device=getattr(args, "device", None))
        trainer.train()

    elif args.command == "eval":
        evaluator = Evaluator(config, checkpoint_dir=getattr(args, "checkpoint", None))
        evaluator.evaluate_all(n_episodes=args.episodes)


if __name__ == "__main__":
    main()
