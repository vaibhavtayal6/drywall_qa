

import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.models.clipseg_model import DrywallCLIPSeg
from src.training.losses import CombinedSegLoss
from src.utils.logger import get_logger, MetricLogger

logger = get_logger(__name__)


def make_collate_fn(processor, device: str = "cpu"):

    def collate_fn(batch):
   
        images, masks, prompts, image_ids = zip(*batch)

    
        images = torch.stack(images)   # [B, 3, H, W]
        masks = torch.stack(masks)     # [B, 1, H, W]

        text_inputs = processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        return {
            "pixel_values": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "masks": masks,
            "image_ids": list(image_ids),
            "prompts": list(prompts),
        }

    return collate_fn

class EarlyStopping:


    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("-inf") if mode == "max" else float("inf")
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop



class Trainer:


    def __init__(
        self,
        cfg: dict,
        model: DrywallCLIPSeg,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Paths
        self.ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = cfg["paths"]["log_dir"]

        # Metric logger
        self.metric_logger = MetricLogger(self.log_dir)

        # Training config
        t_cfg = cfg["training"]
        self.batch_size = t_cfg["batch_size"]
        self.grad_clip = t_cfg.get("grad_clip", 1.0)

        # Loss function
        loss_cfg = t_cfg["loss"]
        self.loss_fn = CombinedSegLoss(
            dice_weight=loss_cfg["dice_weight"],
            bce_weight=loss_cfg["bce_weight"],
            pos_weight=loss_cfg.get("pos_weight", 10.0),
        )
        logger.info(f"Loss: {self.loss_fn}")


    def train_phase1(self) -> str:
        """
        Phase 1: train CLIPSeg decoder with frozen backbone.

        Returns:
            Path to best checkpoint saved during phase 1
        """
        p1_cfg = self.cfg["training"]["phase1"]
        logger.info(
            f"=== Phase 1: decoder only, {p1_cfg['epochs']} epochs, "
            f"lr={p1_cfg['lr']} ==="
        )

        param_groups = self.model.get_parameter_groups(
            decoder_lr=p1_cfg["lr"],
            encoder_lr=None,  # encoder is frozen
            weight_decay=p1_cfg.get("weight_decay", 1e-4),
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = self._make_scheduler(optimizer, p1_cfg)
        early_stop = EarlyStopping(
            patience=self.cfg["training"]["early_stopping"]["patience"],
            mode=self.cfg["training"]["early_stopping"]["mode"],
        )

        best_path = self._run_epochs(
            optimizer=optimizer,
            scheduler=scheduler,
            early_stop=early_stop,
            n_epochs=p1_cfg["epochs"],
            phase_name="phase1",
        )
        return best_path


    def train_phase2(self, phase1_checkpoint: Optional[str] = None) -> str:
        """
        Phase 2: fine-tune with top N CLIP vision blocks unfrozen.

        Args:
            phase1_checkpoint: Path to best Phase 1 checkpoint to load first.
                               If None, continues from current model state.

        Returns:
            Path to best checkpoint saved during phase 2
        """
        p2_cfg = self.cfg["training"]["phase2"]
        n_unfreeze = self.cfg["model"]["unfreeze_top_n_layers"]

        # Load best Phase 1 weights
        if phase1_checkpoint:
            logger.info(f"Loading Phase 1 checkpoint: {phase1_checkpoint}")
            self.model = DrywallCLIPSeg.load(phase1_checkpoint).to(self.device)

        # Unfreeze top N blocks
        self.model.unfreeze_top_n_layers(n=n_unfreeze)

        logger.info(
            f"=== Phase 2: top {n_unfreeze} blocks unfrozen, "
            f"{p2_cfg['epochs']} epochs, lr={p2_cfg['lr']} ==="
        )

        # Differential LRs: encoder gets 10x lower LR
        param_groups = self.model.get_parameter_groups(
            decoder_lr=p2_cfg["lr"],
            encoder_lr=p2_cfg["lr"] / 10,
            weight_decay=p2_cfg.get("weight_decay", 1e-4),
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = self._make_scheduler(optimizer, p2_cfg)
        early_stop = EarlyStopping(
            patience=self.cfg["training"]["early_stopping"]["patience"],
            mode=self.cfg["training"]["early_stopping"]["mode"],
        )

        best_path = self._run_epochs(
            optimizer=optimizer,
            scheduler=scheduler,
            early_stop=early_stop,
            n_epochs=p2_cfg["epochs"],
            phase_name="phase2",
        )
        return best_path

    def _run_epochs(
        self,
        optimizer,
        scheduler,
        early_stop: EarlyStopping,
        n_epochs: int,
        phase_name: str,
    ) -> str:
        """Run training + validation for n_epochs. Return best checkpoint path."""
        best_val_dice = 0.0
        best_ckpt_path = ""

        for epoch in range(1, n_epochs + 1):
            epoch_start = time.time()

            train_loss = self._train_one_epoch(optimizer)

            val_loss, val_dice, val_iou = self._validate()
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start

            logger.info(
                f"[{phase_name}] Epoch {epoch:03d}/{n_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_dice={val_dice:.4f} | "
                f"val_iou={val_iou:.4f} | "
                f"lr={current_lr:.2e} | "
                f"time={epoch_time:.1f}s"
            )

            self.metric_logger.log({
                "phase": phase_name,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_dice": round(val_dice, 6),
                "val_iou": round(val_iou, 6),
                "lr": current_lr,
                "epoch_time_s": round(epoch_time, 2),
            })

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_ckpt_path = str(self.ckpt_dir / f"best_{phase_name}")
                self.model.save(best_ckpt_path)
                logger.info(f"  ✓ New best val_dice={val_dice:.4f} — saved to {best_ckpt_path}")

         
            if early_stop(val_dice):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        logger.info(
            f"[{phase_name}] Done. Best val_dice={best_val_dice:.4f}, "
            f"checkpoint: {best_ckpt_path}"
        )
        return best_ckpt_path

    def _train_one_epoch(self, optimizer) -> float:
        """One training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            masks = batch["masks"].to(self.device, non_blocking=True)

            logits = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = self.loss_fn(logits, masks)
            loss.backward()

           
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float, float]:
        """One validation pass. Returns (loss, dice, iou)."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        n_batches = 0

        for batch in self.val_loader:
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            masks = batch["masks"].to(self.device, non_blocking=True)

            logits = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = self.loss_fn(logits, masks)

            # Quick inline Dice and IoU for monitoring
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            targets = masks

            intersection = (preds * targets).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            iou = (intersection + 1e-6) / (union - intersection + 1e-6)

            total_loss += loss.item()
            total_dice += dice.mean().item()
            total_iou += iou.mean().item()
            n_batches += 1

        n = max(n_batches, 1)
        return total_loss / n, total_dice / n, total_iou / n

    def _make_scheduler(self, optimizer, phase_cfg: dict):
        """Create cosine annealing scheduler."""
        sch_cfg = self.cfg["training"]["scheduler"]
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=phase_cfg["epochs"],
            eta_min=sch_cfg["eta_min"],
        )