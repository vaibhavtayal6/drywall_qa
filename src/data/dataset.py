"""
src/data/dataset.py
-------------------
PyTorch Dataset for prompted drywall segmentation.

WHY THIS DESIGN:
- One unified Dataset handles both crack and taping classes.
  The class label is implicit in the prompt. This means the same
  model + same training loop handles both tasks simultaneously.

- Augmentations live inside the Dataset (via Albumentations).
  Industry standard: Albumentations is ~3x faster than torchvision
  transforms for segmentation tasks because it transforms image AND
  mask jointly with identical random parameters. Without joint
  transforms, your mask ends up misaligned with the flipped image.

- The Dataset is designed to be split into train/val/test externally.
  Never split inside the Dataset class — it makes testing harder.

DATA FORMAT EXPECTED (Roboflow export, "Semantic Segmentation" format):
  data/raw/dataset1/
    images/
      img_001.jpg
      img_002.jpg
    masks/                  ← PNG files, pixel values {0, 255} or {0, 1}
      img_001.png
      img_002.png

AUGMENTATION STRATEGY:
  Train: random flip + rotation + brightness + elastic (for cracks)
  Val/Test: only resize + normalize (no randomness)
  Rationale: val/test must be deterministic to compare across runs.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.prompt_bank import PromptBank
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Albumentations transform pipelines
# ──────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 352, cfg: dict = None) -> A.Compose:
    """
    Training augmentation pipeline.

    Albumentations applies every transform to image AND mask in sync.
    The 'additional_targets' argument tells it that 'mask' should be
    treated identically to 'image' for spatial transforms.

    Key augments explained:
    - HorizontalFlip: free data doubling, no domain assumptions broken
    - Rotate: drywall seams are usually vertical/horizontal, but slight
      rotation adds robustness to camera tilt
    - RandomBrightnessContrast: lighting varies massively in construction sites
    - ElasticTransform: distorts the image like a rubber sheet. This is
      specifically valuable for cracks — it generates new plausible crack
      shapes without needing more labelled data
    - Normalize: subtracts ImageNet mean/std because CLIPSeg's CLIP backbone
      was pre-trained on ImageNet-normalised images. Using different stats
      would shift all activations and slow down fine-tuning.
    """
    cfg = cfg or {}
    brightness = cfg.get("brightness_limit", 0.2)
    contrast = cfg.get("contrast_limit", 0.2)
    rot = cfg.get("rotation_limit", 15)
    blur = cfg.get("blur_limit", 3)
    e_alpha = cfg.get("elastic_alpha", 1)
    e_sigma = cfg.get("elastic_sigma", 50)

    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=rot, p=0.4, border_mode=cv2.BORDER_REFLECT),
        A.RandomBrightnessContrast(
            brightness_limit=brightness,
            contrast_limit=contrast,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(1, blur), p=0.3),
        A.ElasticTransform(
            alpha=e_alpha,
            sigma=e_sigma,
            p=0.2
        ),
        A.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),   # CLIP ImageNet mean
            std=(0.26862954, 0.26130258, 0.27577711),    # CLIP ImageNet std
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 352) -> A.Compose:
    """
    Validation/test pipeline — deterministic, no randomness.
    Only resize and normalize.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class DrywallSegDataset(Dataset):
    """
    Dataset for prompted drywall segmentation.

    Each item is a (image_tensor, mask_tensor, prompt_string, image_id) tuple.

    Args:
        samples:     List of (image_path, mask_path, class_name) tuples
        prompt_bank: PromptBank instance
        transforms:  Albumentations transform pipeline
        split:       "train" | "val" | "test"
                     - "train": sample random prompt each call
                     - "val"/"test": use primary (first) prompt for consistency

    Example:
        samples = build_sample_list("data/processed/train")
        dataset = DrywallSegDataset(
            samples=samples,
            prompt_bank=PromptBank.default(),
            transforms=get_train_transforms(),
            split="train"
        )
        image, mask, prompt, img_id = dataset[0]
        # image: torch.Tensor [3, 352, 352], float32
        # mask:  torch.Tensor [1, 352, 352], float32, values {0.0, 1.0}
        # prompt: str, e.g. "segment wall crack"
        # img_id: str, e.g. "img_042"
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, str]],
        prompt_bank: PromptBank,
        transforms: A.Compose,
        split: str = "train",
    ):
        self.samples = samples
        self.prompt_bank = prompt_bank
        self.transforms = transforms
        self.split = split

        logger.info(
            f"DrywallSegDataset [{split}]: {len(samples)} samples, "
            f"classes={prompt_bank.classes()}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        image_path, mask_path, class_name = self.samples[idx]
        image_id = Path(image_path).stem

        # ── Load image ──────────────────────────────────────────
        # Always load as RGB. OpenCV loads BGR by default → convert.
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # → RGB uint8 H×W×3

        # ── Load mask ───────────────────────────────────────────
        # Load as grayscale. Values may be {0,255} or {0,1}.
        # Normalise to {0.0, 1.0} for loss computation.
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Binarise: any non-zero value → foreground
        mask = (mask > 0).astype(np.float32)  # H×W, float32

        # ── Apply augmentations ──────────────────────────────────
        # Albumentations expects mask as H×W uint8 or float32.
        # ToTensorV2 converts image to C×H×W and mask to 1×H×W.
        augmented = self.transforms(image=image, mask=mask)
        image_tensor = augmented["image"]           # [3, H, W] float32
        mask_tensor = augmented["mask"].unsqueeze(0)  # [1, H, W] float32

        # ── Select prompt ────────────────────────────────────────
        if self.split == "train":
            prompt = self.prompt_bank.sample(class_name)
        else:
            # Deterministic: always use primary prompt for val/test
            prompt = self.prompt_bank.get_primary(class_name)

        return image_tensor, mask_tensor, prompt, image_id

    def __repr__(self):
        return (
            f"DrywallSegDataset(split={self.split}, "
            f"n={len(self.samples)}, "
            f"classes={self.prompt_bank.classes()})"
        )


# ──────────────────────────────────────────────────────────────
# Sample list builder
# ──────────────────────────────────────────────────────────────

def build_sample_list(
    data_dir: str,
    class_name: str,
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> List[Tuple[str, str, str]]:
    """
    Scan a directory for (image, mask) pairs and return sample list.

    Expected structure:
        data_dir/
            images/  ← source images
            masks/   ← binary mask PNGs (same stem as image)

    Args:
        data_dir:   Path to the dataset directory
        class_name: "crack" or "taping" — stored in each tuple
        image_extensions: accepted image file types

    Returns:
        List of (image_path, mask_path, class_name) tuples
        Only includes pairs where BOTH image and mask exist.
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    samples = []
    missing_masks = []

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue

        # Look for mask with same stem, any image extension
        mask_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = masks_dir / (img_path.stem + ext)
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            missing_masks.append(img_path.name)
            continue

        samples.append((str(img_path), str(mask_path), class_name))

    if missing_masks:
        logger.warning(
            f"[{class_name}] {len(missing_masks)} images have no matching mask: "
            f"{missing_masks[:5]}{'...' if len(missing_masks) > 5 else ''}"
        )

    logger.info(f"[{class_name}] Found {len(samples)} valid (image, mask) pairs in {data_dir}")
    return samples


def make_splits(
    samples: List[Tuple[str, str, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Split sample list into train/val/test.

    Uses deterministic shuffle (seed) so splits are identical across runs.
    test_ratio = 1 - train_ratio - val_ratio

    IMPORTANT: shuffle before split, not after — otherwise samples from the
    same physical scene may end up adjacent and split across train/val,
    causing data leakage.
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train: n_train + n_val]
    test = shuffled[n_train + n_val:]

    logger.info(
        f"Split: train={len(train)}, val={len(val)}, test={len(test)} "
        f"(total={n}, ratios={train_ratio}/{val_ratio}/{1-train_ratio-val_ratio:.2f})"
    )
    return train, val, test