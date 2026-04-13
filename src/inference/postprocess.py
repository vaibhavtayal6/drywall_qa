"""
src/inference/postprocess.py
-----------------------------
Postprocessing pipeline to convert model logit maps into clean binary masks.

WHY POSTPROCESSING MATTERS:
Raw CLIPSeg output is a probability map (after sigmoid). Simply thresholding
at 0.5 often gives:
- Noisy salt-and-pepper false positives (single isolated pixels)
- Broken crack lines (gaps where confidence dipped below threshold)
- Over-jagged boundaries on taping areas

Three-stage pipeline:
1. Threshold — convert probabilities to binary
2. Morphological operations — clean up the binary mask
3. Connected component filtering — remove tiny noise regions

Each stage is tunable via config.yaml, and each parameter is
tuned on the VALIDATION set (not test set!) to avoid leaking
test performance into hyperparameter choices.
"""

import cv2
import numpy as np
from typing import Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numpy sigmoid (for use outside PyTorch context)."""
    return 1.0 / (1.0 + np.exp(-x))


def postprocess_mask(
    logit_map: np.ndarray,
    class_name: str,
    threshold: float = 0.5,
    min_component_area: int = 50,
    morph_close_kernel: int = 3,
    morph_dilate_kernel: int = 0,
    output_size: Optional[tuple] = None,
) -> np.ndarray:
    """
    Convert a raw logit map to a clean binary mask.

    Args:
        logit_map:           [H, W] float32 numpy array of raw model logits
        class_name:          "crack" or "taping" (for logging context)
        threshold:           Sigmoid probability threshold for foreground
                             Crack: 0.35 (lower = more sensitive to thin lines)
                             Taping: 0.50
        min_component_area:  Minimum connected component size in pixels.
                             Components smaller than this are removed as noise.
        morph_close_kernel:  Morphological closing kernel size (pixels).
                             Closing = dilation then erosion.
                             Fills small gaps in detected regions.
                             0 = skip closing.
        morph_dilate_kernel: Additional dilation after closing.
                             Expands mask outward. 0 = skip dilation.
        output_size:         (H, W) to resize output mask.
                             If None, keep at logit_map resolution.

    Returns:
        Binary mask [H, W] uint8, values {0, 255}
        Ready to save as PNG.

    Pipeline visualised:
        logit_map → sigmoid → threshold → morph_close → morph_dilate
                 → CCA filter → resize → uint8 {0,255}
    """
    # ── Step 1: sigmoid + threshold ─────────────────────────────
    # Convert logits to probabilities, then binarise
    probs = sigmoid(logit_map.astype(np.float32))
    binary = (probs >= threshold).astype(np.uint8)  # {0, 1}

    # ── Step 2: morphological closing ───────────────────────────
    # Closing = dilation followed by erosion
    # Effect: fills small holes/gaps within detected regions
    # For cracks: fills gaps where a thin crack dipped below threshold
    # Kernel size of 3 is subtle; use 5 for taping areas
    if morph_close_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,  # ellipse is smoother than rectangle
            (morph_close_kernel, morph_close_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ── Step 3: additional dilation ─────────────────────────────
    # Expands the mask slightly outward
    # Useful for taping areas where we want to include the full tape width
    # Do NOT dilate cracks — it makes thin lines fat and hurts clDice
    if morph_dilate_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_dilate_kernel, morph_dilate_kernel)
        )
        binary = cv2.dilate(binary, kernel, iterations=1)

    # ── Step 4: connected component filtering ───────────────────
    # Remove small isolated blobs (noise from model uncertainty)
    # cv2.connectedComponentsWithStats returns:
    #   n_labels: total number of connected components (incl. background=0)
    #   labels:   H×W label map, each pixel = its component ID
    #   stats:    [n_labels, 5] array — x, y, w, h, area per component
    #   centroids: not used
    if min_component_area > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8  # 8-connected: diagonals count
        )
        cleaned = np.zeros_like(binary)
        for i in range(1, n_labels):  # skip 0 = background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_component_area:
                cleaned[labels == i] = 1
        binary = cleaned

    # ── Step 5: resize to output resolution ─────────────────────
    if output_size is not None and output_size != binary.shape[:2]:
        binary = cv2.resize(
            binary.astype(np.float32),
            (output_size[1], output_size[0]),  # cv2 takes (W, H)
            interpolation=cv2.INTER_NEAREST  # preserve binary values
        ).astype(np.uint8)

    # ── Step 6: convert to {0, 255} ─────────────────────────────
    # Rubric requires values {0, 255} in output PNGs
    return (binary * 255).astype(np.uint8)


def postprocess_from_config(
    logit_map: np.ndarray,
    class_name: str,
    cfg: dict,
    output_size: Optional[tuple] = None,
) -> np.ndarray:
    """
    Postprocess using parameters from config.yaml.

    Args:
        logit_map:   [H, W] float32 logit array
        class_name:  "crack" or "taping"
        cfg:         Full config dict (loaded from config.yaml)
        output_size: Optional (H, W) for output mask

    Returns:
        Binary mask [H, W] uint8, values {0, 255}
    """
    pp_cfg = cfg["postprocessing"][class_name]
    return postprocess_mask(
        logit_map=logit_map,
        class_name=class_name,
        threshold=pp_cfg["threshold"],
        min_component_area=pp_cfg["min_component_area"],
        morph_close_kernel=pp_cfg["morph_close_kernel"],
        morph_dilate_kernel=pp_cfg["morph_dilate_kernel"],
        output_size=output_size,
    )