"""
src/inference/predict.py
-------------------------
Inference engine for drywall segmentation.

KEY FEATURES:
1. Prompt ensembling: run all prompts, average logit maps
2. Runtime benchmarking: measure per-image inference time
3. Batch inference over a directory
4. Output filenames match rubric: {image_id}__{prompt_slug}.png

PROMPT ENSEMBLING — THE IDEA IN DETAIL:
CLIPSeg's text encoder maps each prompt to a point in CLIP's
embedding space. "segment crack" and "detect wall damage" map
to nearby but not identical points. The model's response to
each slightly different point also slightly differs.

Averaging the resulting logit maps before thresholding smooths
out this per-prompt noise. It's equivalent to ensemble learning
applied to the input text modality.

Implementation:
    for prompt in ["segment crack", "detect wall damage", ...]:
        logits = model(image, prompt)
        all_logits.append(logits)
    avg_logits = mean(all_logits, axis=0)
    mask = threshold(sigmoid(avg_logits))

This costs N×inference_time but eliminates prompt-sensitivity,
which is the biggest failure mode of CLIPSeg in production.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import statistics

import cv2
import numpy as np
import torch
from PIL import Image

from src.models.clipseg_model import DrywallCLIPSeg
from src.data.prompt_bank import PromptBank, make_output_filename
from src.inference.postprocess import postprocess_from_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
  

    def __init__(
        self,
        model: DrywallCLIPSeg,
        prompt_bank: PromptBank,
        cfg: dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.prompt_bank = prompt_bank
        self.cfg = cfg
        self.device = device
        self.use_ensemble = cfg["inference"]["use_prompt_ensemble"]
        self._inference_times: List[float] = []

    @torch.no_grad()
    def predict_single(
        self,
        image_path: str,
        class_name: str,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, float]:
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        if original_size is None:
            original_size = (image.height, image.width)

        # Get prompts
        if self.use_ensemble:
            prompts = self.prompt_bank.get_all(class_name)
        else:
            prompts = [self.prompt_bank.get_primary(class_name)]

        # ── Inference with timing ──────────────────────────────
        t_start = time.perf_counter()

        all_logits = []
        for prompt in prompts:
            # Process image + text with CLIPSegProcessor
            inputs = self.model.processor(
                text=[prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            logits = self.model(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )  # [1, 1, H, W]

            all_logits.append(logits.squeeze().cpu().numpy())  # [H, W]

        # Average logits across prompts (ensemble)
        avg_logits = np.mean(all_logits, axis=0)  # [H, W]

        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000
        self._inference_times.append(inference_ms)

        # ── Postprocessing ────────────────────────────────────
        mask = postprocess_from_config(
            logit_map=avg_logits,
            class_name=class_name,
            cfg=self.cfg,
            output_size=original_size,
        )

        return mask, inference_ms

    def predict_directory(
        self,
        image_dir: str,
        class_name: str,
        output_dir: str,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> Dict[str, str]:
        
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        primary_prompt = self.prompt_bank.get_primary(class_name)
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        )

        if not image_paths:
            logger.warning(f"No images found in {image_dir}")
            return {}

        logger.info(
            f"Predicting {len(image_paths)} images for class='{class_name}' "
            f"({'ensemble' if self.use_ensemble else 'single prompt'})"
        )

        results = {}
        for i, img_path in enumerate(image_paths):
            try:
                # Get original size for output mask
                img = Image.open(img_path)
                original_size = (img.height, img.width)
                img.close()

                mask, inf_ms = self.predict_single(
                    image_path=str(img_path),
                    class_name=class_name,
                    original_size=original_size,
                )

                # Save with required naming convention
                image_id = img_path.stem
                filename = make_output_filename(image_id, primary_prompt)
                out_path = output_dir / filename
                cv2.imwrite(str(out_path), mask)

                results[image_id] = str(out_path)

                if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
                    logger.info(
                        f"  [{i+1}/{len(image_paths)}] "
                        f"avg_time={self.mean_inference_time_ms:.1f}ms"
                    )

            except Exception as e:
                logger.error(f"Failed on {img_path.name}: {e}")
                continue

        logger.info(
            f"Saved {len(results)} masks to {output_dir}. "
            f"Avg inference: {self.mean_inference_time_ms:.1f}ms/image"
        )
        return results

    def benchmark(
        self,
        image_path: str,
        class_name: str,
        n_runs: int = 20,
        warmup: int = 3,
    ) -> dict:
        
        logger.info(f"Benchmarking inference: {warmup} warmup + {n_runs} timed runs...")

        # Warmup
        for _ in range(warmup):
            self.predict_single(image_path, class_name)

        # Timed runs
        times = []
        self._inference_times = []
        for _ in range(n_runs):
            _, t = self.predict_single(image_path, class_name)
            times.append(t)

        result = {
            "n_runs": n_runs,
            "n_prompts": len(self.prompt_bank.get_all(class_name)),
            "ensemble": self.use_ensemble,
            "mean_ms": round(statistics.mean(times), 2),
            "median_ms": round(statistics.median(times), 2),
            "std_ms": round(statistics.stdev(times) if len(times) > 1 else 0, 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
        }
        logger.info(f"Benchmark result: {result}")
        return result

    @property
    def mean_inference_time_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return statistics.mean(self._inference_times)