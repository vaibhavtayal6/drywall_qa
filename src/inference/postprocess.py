
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
  
   
    probs = sigmoid(logit_map.astype(np.float32))
    binary = (probs >= threshold).astype(np.uint8)  # {0, 1}


    if morph_close_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,  # ellipse is smoother than rectangle
            (morph_close_kernel, morph_close_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

 
    if morph_dilate_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_dilate_kernel, morph_dilate_kernel)
        )
        binary = cv2.dilate(binary, kernel, iterations=1)

 
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

    if output_size is not None and output_size != binary.shape[:2]:
        binary = cv2.resize(
            binary.astype(np.float32),
            (output_size[1], output_size[0]),  # cv2 takes (W, H)
            interpolation=cv2.INTER_NEAREST  # preserve binary values
        ).astype(np.uint8)

    return (binary * 255).astype(np.uint8)


def postprocess_from_config(
    logit_map: np.ndarray,
    class_name: str,
    cfg: dict,
    output_size: Optional[tuple] = None,
) -> np.ndarray:
    
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