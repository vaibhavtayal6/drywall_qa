"""
src/data/download.py
---------------------
Download both Roboflow datasets programmatically.

ROBOFLOW EXPORT FORMAT:
We use "Semantic Segmentation" format (mask PNG per image).
Each dataset downloads as:
    dataset_name/
        train/  ← or valid/ and test/ depending on Roboflow split
            images/
            masks/ (or _annotations.coco.json for COCO format)

We normalise both datasets into our project's expected structure:
    data/raw/dataset1/
        images/
        masks/
    data/raw/dataset2/
        images/
        masks/

SETUP REQUIRED:
    pip install roboflow
    Set ROBOFLOW_API_KEY environment variable, or pass api_key argument.
    Get your key from: https://app.roboflow.com/settings/api

NOTE ON ROBOFLOW EXPORT FORMATS:
- "png mask" format: one PNG per image, pixel values = class ID
- For binary segmentation (one class), pixel values are {0, 255}
- We verify this after download and warn if unexpected values found
"""

import json
import numpy as np
import cv2
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_dataset1(api_key: str, output_dir: str = "data/raw/dataset1") -> None:
    """
    Download Dataset 1: Drywall-Join-Detect (taping areas).
    https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect

    Args:
        api_key:    Roboflow API key
        output_dir: Where to save the normalised dataset
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("pip install roboflow")

    logger.info("Downloading Dataset 1: Drywall-Join-Detect...")
    rf = Roboflow(api_key=api_key)

    # Workspace: objectdetect-pu6rn
    # Project:   drywall-join-detect
    # Version:   use latest (check Roboflow UI for version number)
    project = rf.workspace("objectdetect-pu6rn").project("drywall-join-detect")
    version = project.version(1)  # update version number as needed

    # Export as "png-mask-semantic" = single-channel PNG masks
    dataset = version.download("png-mask-semantic", location=output_dir)

    logger.info(f"Dataset 1 downloaded to: {dataset.location}")
    _normalise_roboflow_structure(dataset.location, output_dir, class_name="taping")


def download_dataset2(api_key: str, output_dir: str = "data/raw/dataset2") -> None:
    """
    Download Dataset 2: Cracks-3ii36.
    https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36

    Args:
        api_key:    Roboflow API key
        output_dir: Where to save the normalised dataset
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("pip install roboflow")

    logger.info("Downloading Dataset 2: Cracks-3ii36...")
    rf = Roboflow(api_key=api_key)

    project = rf.workspace("fyp-ny1jt").project("cracks-3ii36")
    version = project.version(1)  # update version number as needed

    dataset = version.download("png-mask-semantic", location=output_dir)

    logger.info(f"Dataset 2 downloaded to: {dataset.location}")
    _normalise_roboflow_structure(dataset.location, output_dir, class_name="crack")


def _normalise_roboflow_structure(src_dir: str, dst_dir: str, class_name: str) -> None:
    """
    Convert COCO JSON annotations → binary mask PNGs.
    Saves them to dst_dir/masks/ so dataset.py works unchanged.
    """
    dst = Path(dst_dir)
    images_out = dst / "images"
    masks_out  = dst / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_masks  = 0

    for split in ("train", "valid", "test"):
        split_dir = Path(src_dir) / split
        json_path = split_dir / "_annotations.coco.json"

        if not json_path.exists():
            continue

        with open(json_path) as f:
            coco = json.load(f)

        # Build lookup: image_id → file_name and (height, width)
        id_to_info = {
            img["id"]: img for img in coco["images"]
        }

        # Build lookup: image_id → list of annotations
        from collections import defaultdict
        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in coco["images"]:
            img_id   = img_info["id"]
            filename = img_info["file_name"]
            h        = img_info["height"]
            w        = img_info["width"]

            # Copy image
            src_img = split_dir / "images" / filename
            if not src_img.exists():
                src_img = split_dir / filename   # some exports omit /images/
            if src_img.exists():
                import shutil
                shutil.copy2(src_img, images_out / filename)
                total_images += 1

            # Build binary mask from all polygons for this image
            mask = np.zeros((h, w), dtype=np.uint8)
            for ann in img_to_anns[img_id]:
                # segmentation is a list of polygon point lists
                for seg in ann.get("segmentation", []):
                    pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    pts = pts.astype(np.int32)
                    cv2.fillPoly(mask, [pts], color=255)

            # Save mask with same stem as image
            stem     = Path(filename).stem
            out_path = masks_out / (stem + ".png")
            cv2.imwrite(str(out_path), mask)
            total_masks += 1

    logger.info(
        f"[{class_name}] COCO→masks: {total_images} images, "
        f"{total_masks} masks written → {dst_dir}"
    )
    _verify_masks(str(masks_out), class_name)


def _verify_masks(masks_dir: str, class_name: str, sample_n: int = 10) -> None:
    """
    Sanity check: verify masks are binary {0, 255} or {0, 1}.
    Logs a warning if unexpected values found.

    In Roboflow semantic segmentation exports, multi-class datasets
    use pixel values = class index. For single-class datasets the
    values should be {0, 255}. If values are {0, 1}, our Dataset
    class handles that by treating any non-zero pixel as foreground.
    """
    import numpy as np
    import cv2

    masks = sorted(Path(masks_dir).glob("*.png"))[:sample_n]
    if not masks:
        logger.warning(f"[{class_name}] No mask PNGs found in {masks_dir}")
        return

    all_unique = set()
    for m in masks:
        arr = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            all_unique.update(np.unique(arr).tolist())

    logger.info(f"[{class_name}] Mask unique values (sample of {len(masks)}): {sorted(all_unique)}")

    expected = {0, 255}
    if not all_unique.issubset({0, 1, 255}):
        logger.warning(
            f"[{class_name}] Unexpected mask values: {all_unique}. "
            f"Expected {expected}. Check Roboflow export format."
        )
    else:
        logger.info(f"[{class_name}] Mask values OK.")


def dataset_stats(data_dir: str, class_name: str) -> dict:
    """
    Compute basic dataset statistics for the report.

    Returns dict with:
        n_images, n_masks, mean_foreground_ratio, image_sizes
    """
    import numpy as np
    import cv2

    masks_dir = Path(data_dir) / "masks"
    images_dir = Path(data_dir) / "images"

    masks = sorted(masks_dir.glob("*.png"))
    images = sorted(images_dir.iterdir())

    fg_ratios = []
    sizes = set()

    for m in masks:
        arr = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            fg = (arr > 0).sum() / arr.size
            fg_ratios.append(fg)
            sizes.add(arr.shape)

    stats = {
        "class": class_name,
        "n_images": len(list(images)),
        "n_masks": len(masks),
        "mean_fg_ratio": float(np.mean(fg_ratios)) if fg_ratios else 0.0,
        "min_fg_ratio": float(np.min(fg_ratios)) if fg_ratios else 0.0,
        "max_fg_ratio": float(np.max(fg_ratios)) if fg_ratios else 0.0,
        "image_sizes": list(sizes),
    }

    logger.info(f"[{class_name}] Stats: {stats}")
    return stats