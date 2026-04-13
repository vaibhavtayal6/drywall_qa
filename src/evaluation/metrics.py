"""
src/evaluation/metrics.py
--------------------------
Segmentation evaluation metrics for drywall QA.

METRICS EXPLAINED:

mIoU (mean Intersection over Union):
    IoU = TP / (TP + FP + FN) per class, then averaged.
    The most common segmentation metric. Ranges [0, 1].
    Weakness: penalises boundaries heavily. A prediction that
    covers the right area but is slightly off-boundary looks
    terrible in IoU but fine visually.

Dice / F1:
    Dice = 2*TP / (2*TP + FP + FN)
    Mathematically equivalent to F1 score.
    More generous than IoU: Dice(P, G) = 2*IoU/(1+IoU).
    Industry standard for medical/industrial segmentation.

Boundary IoU:
    Only considers pixels within D pixels of the boundary.
    (D = 2% of image diagonal, following the original paper)
    Better correlates with perceived mask quality.
    A mask with correct interior but fuzzy edges scores high
    on mIoU but low on Boundary IoU.

clDice (Centreline Dice):
    Specifically designed for tubular/linear structures like cracks.
    clDice = 2 * |S(P) ∩ G| * |S(G) ∩ P| / (|S(P)| + |S(G)|)
    Where S() is the skeletonisation (centreline extraction).
    Rewards topological correctness: detecting the full length
    of a crack matters, even if the width is wrong.
    Reference: Shit et al. (2021), "clDice - a Novel Topology-Preserving
    Loss Function for Tubular Structure Segmentation", CVPR 2021.

Precision / Recall:
    Precision = TP / (TP + FP) — of all predicted foreground, how
    many are actually foreground?
    Recall = TP / (TP + FN) — of all actual foreground, how many
    did we find?
    Important to report both: a model that predicts everything as
    foreground has Recall=1.0 but Precision≈0.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from scipy import ndimage

EPS = 1e-6  # numerical stability constant


# ──────────────────────────────────────────────────────────────
# Core pixel-level metrics
# ──────────────────────────────────────────────────────────────

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Intersection over Union for binary masks.

    Args:
        pred: [H, W] binary array {0, 1} — predicted mask
        gt:   [H, W] binary array {0, 1} — ground truth mask

    Returns:
        IoU scalar in [0, 1]
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return float(intersection) / (float(union) + EPS)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice coefficient (F1 score) for binary masks.

    Returns:
        Dice scalar in [0, 1]
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = (pred_b & gt_b).sum()
    return 2.0 * float(intersection) / (pred_b.sum() + gt_b.sum() + EPS)


def compute_precision_recall(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[float, float]:
    """
    Precision and Recall for binary masks.

    Returns:
        (precision, recall) scalars in [0, 1]
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = (pred_b & gt_b).sum()
    fp = (pred_b & ~gt_b).sum()
    fn = (~pred_b & gt_b).sum()
    precision = float(tp) / (float(tp + fp) + EPS)
    recall = float(tp) / (float(tp + fn) + EPS)
    return precision, recall


# ──────────────────────────────────────────────────────────────
# Boundary IoU
# ──────────────────────────────────────────────────────────────

def _get_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """
    Extract boundary pixels within D pixels of the mask edge.

    D = max(1, round(dilation_ratio * image_diagonal))
    At 352×352: diagonal ≈ 498, D ≈ 10 pixels.

    Uses erosion: boundary = mask XOR eroded_mask.
    """
    h, w = mask.shape
    diag = np.sqrt(h**2 + w**2)
    d = max(1, int(round(dilation_ratio * diag)))
    kernel = np.ones((d, d), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = (mask.astype(np.uint8) ^ eroded).astype(bool)
    return boundary


def compute_boundary_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    dilation_ratio: float = 0.02,
) -> float:
    """
    Boundary IoU — IoU computed only on boundary pixels.

    Better measure of mask quality at edges.
    Reference: Cheng et al. (2021), "Boundary IoU: Improving Object-Centric
    Image Segmentation Evaluation", CVPR 2021.

    Returns:
        Boundary IoU in [0, 1]
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    if gt_b.sum() == 0:
        return 1.0 if pred_b.sum() == 0 else 0.0

    pred_boundary = _get_boundary(pred.astype(np.uint8), dilation_ratio)
    gt_boundary = _get_boundary(gt.astype(np.uint8), dilation_ratio)

    # IoU on the union of both boundary regions
    boundary_union = pred_boundary | gt_boundary
    if not boundary_union.any():
        return 1.0

    intersection = (pred_b & gt_b & boundary_union).sum()
    union = ((pred_b | gt_b) & boundary_union).sum()
    return float(intersection) / (float(union) + EPS)


# ──────────────────────────────────────────────────────────────
# clDice — topology metric for thin structures
# ──────────────────────────────────────────────────────────────

def _skeletonise(mask: np.ndarray) -> np.ndarray:
    """
    Extract skeleton (centreline) of a binary mask.

    Uses Zhang-Suen thinning via OpenCV's ximgproc thinning,
    with fallback to scipy's distance-transform based approach.

    Returns:
        Binary skeleton array, same shape as input
    """
    # Try OpenCV ximgproc first (faster)
    try:
        import cv2.ximgproc
        skeleton = cv2.ximgproc.thinning(
            mask.astype(np.uint8) * 255,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        return (skeleton > 0).astype(np.uint8)
    except (AttributeError, ImportError):
        pass

    # Fallback: morphological skeleton via scipy
    # Repeatedly erode and record the difference
    skeleton = np.zeros_like(mask, dtype=np.uint8)
    element = ndimage.generate_binary_structure(2, 2)
    temp = mask.astype(bool).copy()

    for _ in range(50):  # max iterations
        eroded = ndimage.binary_erosion(temp, element)
        opened = ndimage.binary_dilation(eroded, element)
        skeleton |= (temp & ~opened).astype(np.uint8)
        temp = eroded
        if not temp.any():
            break

    return skeleton


def compute_cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Centreline Dice (clDice) for topology-preserving evaluation.

    Especially important for cracks: a prediction that follows the
    crack's centreline but is thicker or thinner than GT scores
    much better in clDice than in standard IoU.

    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    Tprec = |S_pred ∩ GT| / |S_pred|  (skeleton of pred covered by GT)
    Tsens = |S_gt ∩ Pred| / |S_gt|    (skeleton of GT covered by pred)

    Returns:
        clDice in [0, 1]
    """
    pred_b = pred.astype(bool).astype(np.uint8)
    gt_b = gt.astype(bool).astype(np.uint8)

    if gt_b.sum() == 0:
        return 1.0 if pred_b.sum() == 0 else 0.0
    if pred_b.sum() == 0:
        return 0.0

    skel_pred = _skeletonise(pred_b)
    skel_gt = _skeletonise(gt_b)

    # Topological precision: how much of pred skeleton is inside GT?
    tprec_num = (skel_pred.astype(bool) & gt_b.astype(bool)).sum()
    tprec = float(tprec_num) / (skel_pred.sum() + EPS)

    # Topological sensitivity: how much of GT skeleton is inside pred?
    tsens_num = (skel_gt.astype(bool) & pred_b.astype(bool)).sum()
    tsens = float(tsens_num) / (skel_gt.sum() + EPS)

    cldice = 2.0 * tprec * tsens / (tprec + tsens + EPS)
    return float(cldice)


# ──────────────────────────────────────────────────────────────
# Full metric suite
# ──────────────────────────────────────────────────────────────

def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    class_name: str = "unknown",
    compute_cldice_flag: bool = True,
) -> Dict[str, float]:
    """
    Compute all segmentation metrics for one (pred, gt) pair.

    Args:
        pred:               [H, W] predicted mask, values {0,255} or {0,1}
        gt:                 [H, W] ground truth mask, values {0,255} or {0,1}
        class_name:         "crack" or "taping" (used for logging)
        compute_cldice_flag: clDice is slow — set False for quick val loop

    Returns:
        Dict with keys: iou, dice, precision, recall, boundary_iou, cldice
    """
    # Normalise to {0, 1}
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    iou = compute_iou(pred_bin, gt_bin)
    dice = compute_dice(pred_bin, gt_bin)
    precision, recall = compute_precision_recall(pred_bin, gt_bin)
    b_iou = compute_boundary_iou(pred_bin, gt_bin)
    cldice = compute_cldice(pred_bin, gt_bin) if compute_cldice_flag else float("nan")

    return {
        "class": class_name,
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "boundary_iou": round(b_iou, 4),
        "cldice": round(cldice, 4),
    }


def aggregate_metrics(metric_list: list) -> Dict[str, float]:
    """
    Average a list of per-image metric dicts into dataset-level numbers.

    Args:
        metric_list: List of dicts from compute_all_metrics()

    Returns:
        Dict with mean of each numeric metric
    """
    if not metric_list:
        return {}

    def _is_number(value) -> bool:
        return isinstance(value, (int, float, np.integer, np.floating))

    keys = [k for k, v in metric_list[0].items() if _is_number(v)]
    result = {}
    for key in keys:
        values = [float(m[key]) for m in metric_list if key in m and _is_number(m[key])]
        result[f"mean_{key}"] = round(float(np.mean(values)), 4) if values else float("nan")
        result[f"std_{key}"] = round(float(np.std(values)), 4) if values else float("nan")

    return result