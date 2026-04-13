"""
src/utils/seed.py
-----------------
Fix all sources of randomness so experiments are reproducible.

Why does this matter?
- Python's random module: used by data shuffling
- NumPy: used by Albumentations augmentation
- PyTorch: used by weight init, dropout, dataloaders
- CUDA: GPU ops have non-deterministic algorithms by default
  → torch.backends.cudnn.deterministic=True forces determinism
  (costs ~5% speed, worth it for research reproducibility)
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds. Call this before anything else in every script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Deterministic CUDA ops
    # WARNING: some CUDA ops don't have deterministic implementations.
    # If you hit a RuntimeError about non-determinism, set:
    #   torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disables auto-tuner (non-deterministic)

    # Ensure HuggingFace tokenizers don't parallelize (avoids fork issues)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Worker seed function for DataLoader
    # Each worker gets seed+worker_id so they don't produce identical data


def worker_init_fn(worker_id: int) -> None:
    """Pass to DataLoader(worker_init_fn=worker_init_fn)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)