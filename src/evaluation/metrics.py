
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from scipy import ndimage

EPS = 1e-6  # numerical stability constant


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return float(intersection) / (float(union) + EPS)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = (pred_b & gt_b).sum()
    return 2.0 * float(intersection) / (pred_b.sum() + gt_b.sum() + EPS)


def compute_precision_recall(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[float, float]:

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = (pred_b & gt_b).sum()
    fp = (pred_b & ~gt_b).sum()
    fn = (~pred_b & gt_b).sum()
    precision = float(tp) / (float(tp + fp) + EPS)
    recall = float(tp) / (float(tp + fn) + EPS)
    return precision, recall


def _get_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:

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

def _skeletonise(mask: np.ndarray) -> np.ndarray:

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


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    class_name: str = "unknown",
    compute_cldice_flag: bool = True,
) -> Dict[str, float]:

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