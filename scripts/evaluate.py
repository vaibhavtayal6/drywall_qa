

import argparse
import json
import time
from pathlib import Path

import yaml
import torch
import cv2
import numpy as np

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.data.dataset import build_sample_list, make_splits
from src.data.prompt_bank import PromptBank
from src.models.clipseg_model import DrywallCLIPSeg
from src.inference.predict import InferenceEngine
from src.evaluation.metrics import compute_all_metrics, aggregate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DrywallCLIPSeg on test set")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_ensemble", action="store_true",
                        help="Disable prompt ensembling (faster but less accurate)")
    return parser.parse_args()


def make_side_by_side(image_bgr: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Create report panel in strict format: Original Image | Ground Truth | Prediction."""
    h, w = image_bgr.shape[:2]

    # Ensure masks are displayed as clean binary grayscale panels.
    gt_bin = (gt_mask > 127).astype(np.uint8) * 255
    pred_bin = (pred_mask > 127).astype(np.uint8) * 255
    gt_rgb = cv2.cvtColor(gt_bin, cv2.COLOR_GRAY2BGR)
    pred_rgb = cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2BGR)

    panel = np.hstack([image_bgr, gt_rgb, pred_rgb])

    # Add a thin header row with labels for report readability.
    header_h = 42
    header = np.full((header_h, 3 * w, 3), 245, dtype=np.uint8)
    labels = ["Original Image", "Ground Truth", "Prediction"]
    for i, label in enumerate(labels):
        x0 = i * w
        x1 = (i + 1) * w
        cv2.rectangle(header, (x0, 0), (x1 - 1, header_h - 1), (210, 210, 210), 1)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        text_x = x0 + max((w - text_size[0]) // 2, 8)
        text_y = (header_h + text_size[1]) // 2 - 4
        cv2.putText(
            header,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

    return np.vstack([header, panel])


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(__name__, log_dir=cfg["paths"]["log_dir"])
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")

    if args.no_ensemble:
        cfg["inference"]["use_prompt_ensemble"] = False

    # ── Load model ─────────────────────────────────────────────
    model = DrywallCLIPSeg.load(args.checkpoint)
    logger.info(f"Model size: {model.model_size_mb():.1f} MB")

    prompt_bank = PromptBank.from_config(args.config)

    # ── Build test split ───────────────────────────────────────
    taping_samples = build_sample_list(
        data_dir=f"{cfg['paths']['raw_root']}/dataset1",
        class_name="taping",
    )
    crack_samples = build_sample_list(
        data_dir=f"{cfg['paths']['raw_root']}/dataset2",
        class_name="crack",
    )
    _, val_samples, test_samples = make_splits(
        samples=taping_samples + crack_samples,
        train_ratio=cfg["dataset"]["train_split"],
        val_ratio=cfg["dataset"]["val_split"],
        seed=cfg["seed"],
    )

    eval_samples = test_samples if args.split == "test" else val_samples
    logger.info(f"Evaluating on {len(eval_samples)} {args.split} samples")

    # ── Inference engine ───────────────────────────────────────
    engine = InferenceEngine(model, prompt_bank, cfg, device=device)
    pred_dir = Path(cfg["paths"]["prediction_dir"]) / args.split
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(cfg["paths"]["report_dir"]) / "qualitative" / args.split
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Run evaluation ─────────────────────────────────────────
    all_metrics = []
    metrics_by_class = {"crack": [], "taping": []}

    for img_path, mask_path, class_name in eval_samples:
        # Predict
        mask_pred, inf_ms = engine.predict_single(
            image_path=img_path,
            class_name=class_name,
        )

        image_id = Path(img_path).stem
        pred_out = pred_dir / f"{image_id}__pred.png"
        cv2.imwrite(str(pred_out), mask_pred)

        # Load GT
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            logger.warning(f"Cannot load GT: {mask_path}")
            continue

        # Resize GT to match prediction if needed
        if gt.shape != mask_pred.shape:
            gt = cv2.resize(
                gt, (mask_pred.shape[1], mask_pred.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Compute metrics
        m = compute_all_metrics(
            pred=mask_pred,
            gt=gt,
            class_name=class_name,
            compute_cldice_flag=True,
        )
        m["inference_ms"] = round(inf_ms, 2)
        m["image_id"] = image_id

        all_metrics.append(m)
        metrics_by_class[class_name].append(m)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            if img_bgr.shape[:2] != mask_pred.shape:
                img_bgr = cv2.resize(
                    img_bgr,
                    (mask_pred.shape[1], mask_pred.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            panel = make_side_by_side(img_bgr, gt, mask_pred)
            panel_path = vis_dir / f"{image_id}__{class_name}__panel.png"
            cv2.imwrite(str(panel_path), panel)

    # ── Aggregate and report ───────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)

    report = {}
    for class_name, metrics_list in metrics_by_class.items():
        if not metrics_list:
            continue
        agg = aggregate_metrics(metrics_list)
        report[class_name] = agg
        logger.info(f"\nClass: {class_name} (n={len(metrics_list)})")
        for k, v in agg.items():
            if "mean" in k:
                logger.info(f"  {k}: {v:.4f}")

    # Overall
    overall = aggregate_metrics(all_metrics)
    report["overall"] = overall
    logger.info(f"\nOverall (n={len(all_metrics)})")
    for k, v in overall.items():
        if "mean" in k:
            logger.info(f"  {k}: {v:.4f}")

    # Runtime summary
    report["runtime"] = {
        "device": device,
        "mean_inference_ms": round(engine.mean_inference_time_ms, 2),
        "ensemble": cfg["inference"]["use_prompt_ensemble"],
        "n_prompts": {c: len(prompt_bank.get_all(c)) for c in prompt_bank.classes()},
        "model_size_mb": round(model.model_size_mb(), 1),
    }

    # ── Save report ────────────────────────────────────────────
    report_path = Path(cfg["paths"]["log_dir"]) / f"metrics_{args.split}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nMetrics saved to: {report_path}")

    # ── Benchmark ──────────────────────────────────────────────
    if eval_samples:
        sample_img = eval_samples[0][0]
        sample_class = eval_samples[0][2]
        bench = engine.benchmark(sample_img, sample_class, n_runs=10, warmup=2)
        report["benchmark"] = bench
        logger.info(f"\nBenchmark: {bench}")

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()