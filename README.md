# Drywall QA — Prompted Segmentation

A quality-assurance system that detects drywall surface defects using text-prompted segmentation. The model fine-tunes [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) to locate two defect classes in drywall images:

| Class | Description |
|-------|-------------|
| **crack** | Hairline and structural surface cracks |
| **taping** | Joint tape / mudded seam areas |

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Metrics](#metrics)
- [Outputs](#outputs)

---

## How It Works

The system is built on **CLIPSeg** (Lüddecke & Ecker, CVPR 2022), which combines CLIP's vision-language encoder with a FiLM-conditioned decoder to produce pixel-level segmentation masks from natural-language prompts.

**Training is split into two phases:**

1. **Phase 1 — Decoder only:** The CLIP backbone is frozen. Only the CLIPSeg decoder (~6M parameters) is trained. This fast convergence step leverages CLIP's pre-trained visual-semantic alignments without risk of catastrophic forgetting.

2. **Phase 2 — Partial unfreeze:** The top 4 transformer blocks of the CLIP vision encoder are unfrozen with a much lower learning rate, allowing the model to adapt its high-level features to the drywall domain while keeping low-level features intact.

**Prompt ensembling:** At inference time, logits from all prompts in the prompt bank (e.g. `"segment crack"`, `"segment wall crack"`, …) are averaged before thresholding, giving more robust predictions than any single prompt.

---

## Project Structure

```
drywall_qa/
├── configs/
│   ├── config.yaml          # Main config (all hyperparameters)
│   └── config_smoke.yaml    # Smoke-test config (1 epoch)
│
├── scripts/
│   ├── train.py             # Training entry-point
│   └── evaluate.py          # Evaluation / metrics entry-point
│
├── src/
│   ├── data/
│   │   ├── dataset.py       # DrywallSegDataset, splits, augmentation
│   │   ├── download.py      # Roboflow dataset downloader + COCO→mask conversion
│   │   └── prompt_bank.py   # Per-class text prompt management
│   ├── models/
│   │   └── clipseg_model.py # DrywallCLIPSeg wrapper (freeze/unfreeze, save/load)
│   ├── training/
│   │   ├── losses.py        # Combined Dice + weighted BCE loss
│   │   └── trainer.py       # Trainer class (phase 1 & 2), early stopping
│   ├── inference/
│   │   ├── predict.py       # InferenceEngine (single image, benchmarking)
│   │   └── postprocess.py   # Threshold, CCA filtering, morphological ops
│   ├── evaluation/
│   │   └── metrics.py       # IoU, Dice, Precision, Recall, Boundary IoU, clDice
│   └── utils/
│       ├── logger.py        # Structured logging setup
│       └── seed.py          # Reproducible seed helper
│
├── outputs/
│   ├── logs/                # Training and evaluation logs
│   └── predictions/         # Saved prediction masks
│
└── requirements.txt
```

---

## Requirements

- Python 3.9+
- CUDA 11.8+ (recommended; CPU also supported)

---

## Installation

```bash
git clone https://github.com/vaibhavtayal6/drywall_qa.git
cd drywall_qa
pip install -r requirements.txt
```

---

## Dataset Setup

Two datasets are downloaded from [Roboflow](https://roboflow.com/) and converted to binary PNG masks automatically.

```python
from src.data.download import download_dataset1, download_dataset2

api_key = "YOUR_ROBOFLOW_API_KEY"

# Dataset 1 — Taping / drywall joint detection
download_dataset1(api_key, output_dir="data/raw/dataset1")

# Dataset 2 — Surface crack detection
download_dataset2(api_key, output_dir="data/raw/dataset2")
```

After download the data directory will look like:

```
data/raw/
├── dataset1/
│   ├── images/   # .jpg / .png images
│   └── masks/    # binary PNG masks (0 = background, 255 = taping)
└── dataset2/
    ├── images/
    └── masks/    # binary PNG masks (0 = background, 255 = crack)
```

Splits (70 / 15 / 15 train / val / test) are created deterministically at runtime from `seed: 42` in the config.

---

## Configuration

All hyperparameters live in `configs/config.yaml`. Nothing is hardcoded in `.py` files.

Key sections:

| Section | Notable settings |
|---------|-----------------|
| `model` | `backbone: "CIDAS/clipseg-rd64-refined"`, `freeze_backbone`, `unfreeze_top_n_layers` |
| `training.phase1` | 15 epochs, lr 1e-4, decoder only |
| `training.phase2` | 15 epochs, lr 5e-5, partial unfreeze |
| `training.loss` | `dice_weight: 0.6`, `bce_weight: 0.4`, `pos_weight: 10.0` |
| `prompts` | Per-class text prompt lists for training sampling and inference ensembling |
| `postprocessing` | Per-class threshold, CCA area filter, morphological kernel sizes |
| `inference` | `use_prompt_ensemble: true`, optional SAM refinement |

Use `configs/config_smoke.yaml` for a quick sanity-check run (phase 1 set to 1 epoch).

---

## Training

```bash
# Run both phases (default)
python scripts/train.py --config configs/config.yaml

# Run phase 1 only
python scripts/train.py --config configs/config.yaml --phase 1

# Run phase 2 only, loading a saved phase 1 checkpoint
python scripts/train.py --config configs/config.yaml --phase 2 \
    --phase1_ckpt outputs/checkpoints/phase1_best

# Override device
python scripts/train.py --device cpu

# Smoke test (fast, 1 epoch)
python scripts/train.py --config configs/config_smoke.yaml
```

Checkpoints are saved to `outputs/checkpoints/`. The best validation-Dice checkpoint is kept for each phase.

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/phase2_best \
    --split test

# Evaluate on validation split
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/phase2_best \
    --split val

# Disable prompt ensembling (faster)
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/phase2_best \
    --no_ensemble
```

Results are written to:
- `outputs/logs/metrics_test.json` — per-class and overall aggregate metrics
- `outputs/report/qualitative/<split>/` — side-by-side panels: *Original | Ground Truth | Prediction*

---

## Inference

The `InferenceEngine` in `src/inference/predict.py` handles single-image prediction and benchmarking. It can be used programmatically:

```python
from src.models.clipseg_model import DrywallCLIPSeg
from src.data.prompt_bank import PromptBank
from src.inference.predict import InferenceEngine
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

model = DrywallCLIPSeg.load("outputs/checkpoints/phase2_best")
prompt_bank = PromptBank.from_config("configs/config.yaml")
engine = InferenceEngine(model, prompt_bank, cfg, device="cuda")

# Returns a binary mask (uint8, values 0/255) and inference time in ms
mask, inference_ms = engine.predict_single("path/to/image.jpg", class_name="crack")
```

**Postprocessing** is applied per-class and includes:
- Sigmoid → threshold (0.45 for crack, 0.50 for taping)
- Connected component analysis — small blobs below `min_component_area` are removed
- Morphological close / dilate with class-specific kernel sizes

---

## Metrics

All metrics are implemented from scratch in `src/evaluation/metrics.py`.

| Metric | Description |
|--------|-------------|
| **IoU** (mIoU) | Intersection over Union |
| **Dice** | F1-score at pixel level |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **Boundary IoU** | IoU restricted to boundary regions (shape accuracy) |
| **clDice** | Topology-aware Dice on skeletons — critical for thin cracks |

Per-class results (crack / taping) and an overall aggregate are reported with mean ± std.

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/checkpoints/` | Saved model checkpoints (HuggingFace format) |
| `outputs/logs/` | Training logs, `metrics_<split>.json` |
| `outputs/predictions/` | Predicted binary masks per split |
| `outputs/report/qualitative/` | Side-by-side visualisation panels |
