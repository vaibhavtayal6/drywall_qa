"""
This script:
1. Fixes all random seeds
2. Loads config
3. Builds datasets and dataloaders
4. Trains Phase 1 (decoder only)
5. Trains Phase 2 (partial unfreeze)
6. Saves the best checkpoint from each phase
"""

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
import cv2
import numpy as np

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
    """Build class-balanced sample weights so crack/taping are seen equally."""
    crack_count = sum(1 for _, _, class_name in samples if class_name == "crack")
    taping_count = sum(1 for _, _, class_name in samples if class_name == "taping")

    if crack_count == 0 or taping_count == 0:
        raise ValueError(
            f"Both classes must exist in train split (crack={crack_count}, taping={taping_count})"
        )

    # Boost minority class so expected draw probability per class is balanced.
    if crack_count >= taping_count:
        crack_weight = 1.0
        taping_weight = crack_count / taping_count
    else:
        crack_weight = taping_count / crack_count
        taping_weight = 1.0

    sample_weights = [
        taping_weight if class_name == "taping" else crack_weight
        for _, _, class_name in samples
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(samples),
        replacement=True,
    )
    class_counts = {"crack": crack_count, "taping": taping_count}
    class_weights = {"crack": crack_weight, "taping": taping_weight}
    return sampler, class_counts, class_weights


def log_annotation_structure(samples, logger, max_samples_per_class=3):
    """Quick sanity check for annotation files and mask value structure."""
    logger.info("Annotation structure check (images/ + masks/ + binary mask stats)")

    checked = {"crack": 0, "taping": 0}
    for image_path, mask_path, class_name in samples:
        if class_name not in checked or checked[class_name] >= max_samples_per_class:
            continue

        image_exists = Path(image_path).exists()
        mask_exists = Path(mask_path).exists()
        if not (image_exists and mask_exists):
            logger.warning(
                f"[{class_name}] missing file(s): image_exists={image_exists}, "
                f"mask_exists={mask_exists}, image={image_path}, mask={mask_path}"
            )
            checked[class_name] += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"[{class_name}] unreadable mask: {mask_path}")
            checked[class_name] += 1
            continue

        unique_vals = np.unique(mask)
        foreground_ratio = float((mask > 0).mean())
        logger.info(
            f"[{class_name}] {Path(mask_path).name}: shape={mask.shape}, "
            f"unique={unique_vals[:10].tolist()}, fg_ratio={foreground_ratio:.5f}"
        )
        checked[class_name] += 1


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

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

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

    # Quick annotation sanity check before building datasets/train loop.
    log_annotation_structure(all_samples, logger)

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


    model = DrywallCLIPSeg(
        model_name=cfg["model"]["backbone"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
    )


    collate_fn = make_collate_fn(model.processor, device=device)
    t_cfg = cfg["training"]

    train_sampler, class_counts, class_weights = make_balanced_sampler(train_samples)
    logger.info(f"Train class counts: {dict(class_counts)}")
    logger.info(f"Train sampler class weights: {class_weights}")
    logger.info("Using class-balanced weighted sampler (shuffle disabled)")

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