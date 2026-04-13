"""
scripts/train.py
-----------------
Entry point for training DrywallCLIPSeg.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --phase 1
    python scripts/train.py --config configs/config.yaml --device cpu

This script:
1. Fixes all random seeds
2. Loads config
3. Builds datasets and dataloaders
4. Trains Phase 1 (decoder only)
5. Trains Phase 2 (partial unfreeze)
6. Saves the best checkpoint from each phase
"""

import argparse
from collections import Counter
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.utils.seed import set_seed, worker_init_fn
from src.utils.logger import get_logger
from src.data.dataset import (
    DrywallSegDataset,
    build_sample_list,
    make_splits,
    get_train_transforms,
    get_val_transforms,
)
from src.data.prompt_bank import PromptBank
from src.models.clipseg_model import DrywallCLIPSeg
from src.training.trainer import Trainer, make_collate_fn


def make_balanced_sampler(samples):
    """Oversample minority classes using inverse class-frequency weights."""
    class_counts = Counter(class_name for _, _, class_name in samples)
    sample_weights = [1.0 / class_counts[class_name] for _, _, class_name in samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(samples),
        replacement=True,
    )
    return sampler, class_counts


def parse_args():
    parser = argparse.ArgumentParser(description="Train DrywallCLIPSeg")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--phase", type=int, choices=[1, 2, 12], default=12,
                        help="1=phase1 only, 2=phase2 only, 12=both (default)")
    parser.add_argument("--phase1_ckpt", type=str, default=None,
                        help="Path to phase1 checkpoint (for --phase 2 only)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu. Auto-detected if not set.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Seed everything ────────────────────────────────────────
    # MUST be called before any torch/numpy/random usage
    set_seed(cfg["seed"])

    # ── Device ────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = get_logger(__name__, log_dir=cfg["paths"]["log_dir"])
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {cfg['seed']}")

    # ── Build datasets ─────────────────────────────────────────
    prompt_bank = PromptBank.from_config(args.config)
    aug_cfg = cfg["dataset"]["augmentation"]
    image_size = cfg["dataset"]["image_size"]

    # Gather samples from both datasets
    taping_samples = build_sample_list(
        data_dir=f"{cfg['paths']['raw_root']}/dataset1",
        class_name="taping",
    )
    crack_samples = build_sample_list(
        data_dir=f"{cfg['paths']['raw_root']}/dataset2",
        class_name="crack",
    )
    all_samples = taping_samples + crack_samples

    # Split
    train_samples, val_samples, test_samples = make_splits(
        samples=all_samples,
        train_ratio=cfg["dataset"]["train_split"],
        val_ratio=cfg["dataset"]["val_split"],
        seed=cfg["seed"],
    )

    # Datasets
    train_dataset = DrywallSegDataset(
        samples=train_samples,
        prompt_bank=prompt_bank,
        transforms=get_train_transforms(image_size, aug_cfg),
        split="train",
    )
    val_dataset = DrywallSegDataset(
        samples=val_samples,
        prompt_bank=prompt_bank,
        transforms=get_val_transforms(image_size),
        split="val",
    )

    # ── Model ──────────────────────────────────────────────────
    model = DrywallCLIPSeg(
        model_name=cfg["model"]["backbone"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
    )

    # ── DataLoaders ────────────────────────────────────────────
    collate_fn = make_collate_fn(model.processor, device=device)
    t_cfg = cfg["training"]

    train_sampler, class_counts = make_balanced_sampler(train_samples)
    logger.info(f"Train class counts: {dict(class_counts)}")
    logger.info("Using inverse-frequency weighted sampler to oversample taping")

    train_loader = DataLoader(
        train_dataset,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        sampler=train_sampler,
        num_workers=t_cfg["num_workers"],
        pin_memory=t_cfg["pin_memory"] and device == "cuda",
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        drop_last=True,  # avoid single-sample batches (batch norm issues)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=t_cfg["num_workers"],
        pin_memory=t_cfg["pin_memory"] and device == "cuda",
        collate_fn=collate_fn,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Train ──────────────────────────────────────────────────
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    phase1_ckpt = None
    if args.phase in (1, 12):
        phase1_ckpt = trainer.train_phase1()
        logger.info(f"Phase 1 complete. Best checkpoint: {phase1_ckpt}")

    if args.phase in (2, 12):
        ckpt_to_load = args.phase1_ckpt or phase1_ckpt
        phase2_ckpt = trainer.train_phase2(phase1_checkpoint=ckpt_to_load)
        logger.info(f"Phase 2 complete. Best checkpoint: {phase2_ckpt}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()