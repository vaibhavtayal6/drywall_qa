"""
src/utils/logger.py
--------------------
Consistent logging across all scripts.

Industry practice:
- Use Python's built-in logging (not print) — allows log levels,
  file sinks, timestamps, and grep-ability.
- One logger per module: logger = get_logger(__name__)
- Log to both console (INFO) and a rotating file (DEBUG) so you can
  replay exactly what happened during a training run days later.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger that writes to console and optionally to a file.

    Args:
        name:    Module name, typically __name__
        log_dir: If provided, also write to log_dir/YYYYMMDD_HHMMSS.log
        level:   Logging level (INFO by default)

    Returns:
        Configured Logger instance

    Usage:
        logger = get_logger(__name__, log_dir="outputs/logs")
        logger.info("Training started")
        logger.warning("Low GPU memory")
        logger.error("Checkpoint not found")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler — DEBUG and above (full detail)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{timestamp}_{name.replace('.', '_')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class MetricLogger:
    """
    Lightweight CSV logger for training metrics.
    Writes one row per epoch to outputs/logs/metrics.csv

    Why CSV? It's the most portable format for later analysis in
    pandas, Excel, or any plotting library.
    """

    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        self.path = Path(log_dir) / filename
        self._initialized = False

    def log(self, metrics: dict) -> None:
        """Append one row. Creates header on first call."""
        if not self._initialized:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
            self._initialized = True

        with open(self.path, "a") as f:
            f.write(",".join(str(v) for v in metrics.values()) + "\n")

    def __repr__(self):
        return f"MetricLogger(path={self.path})"