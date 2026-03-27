"""
Logging utilities.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
