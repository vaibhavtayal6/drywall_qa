"""
src/training/losses.py
-----------------------
Custom loss functions for imbalanced binary segmentation.

THE CORE PROBLEM — CLASS IMBALANCE:
In a crack image, cracks might cover only 1–5% of pixels.
If the model always predicts "background", accuracy = 97%.
But mIoU = 0 — the model is useless.

Standard cross-entropy treats every pixel equally → model ignores cracks.

SOLUTION — Combined loss:
    L = w_dice * DiceLoss + w_bce * WeightedBCE

DICE LOSS — Overlap-based metric turned into a loss:
    Dice = 2 * |P ∩ G| / (|P| + |G|)
    DiceLoss = 1 - Dice
    - Directly optimises the metric we care about (F1/Dice score)
    - Naturally handles imbalance because it's a ratio, not a sum
    - Downside: unstable when |G| = 0 (no foreground in image)
      → solved by smooth term in denominator

WEIGHTED BCE — Binary cross entropy with class weight:
    WeightedBCE = -[w_pos * y*log(p) + (1-y)*log(1-p)]
    - w_pos = foreground_weight (30 for cracks, 8 for taping)
    - Multiplying foreground loss by w_pos forces model to care about
      the rare positive pixels
    - Combined with Dice: the BCE catches individual pixel errors while
      Dice ensures global shape coverage

FOCAL LOSS (bonus, not used by default):
    FL = -(1-p_t)^γ * log(p_t)
    - Down-weights easy background examples automatically
    - γ=2 is standard
    - Alternative to weighted BCE, sometimes better for very sparse masks
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Dice = 2*TP / (2*TP + FP + FN)
    DiceLoss = 1 - Dice

    "Soft" means we use sigmoid probabilities rather than hard binary
    predictions in the numerator. This makes the loss differentiable.

    Args:
        smooth: Small constant to avoid division by zero when mask is empty.
                Typical values: 1.0 (classic) or 1e-6 (strict).
                Using 1.0 is safer for small batches with rare positives.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  [B, 1, H, W] — raw model outputs (before sigmoid)
            targets: [B, 1, H, W] — ground truth masks, values {0.0, 1.0}

        Returns:
            Scalar loss value
        """
        # Apply sigmoid to get probabilities in [0, 1]
        probs = torch.sigmoid(logits)

        # Flatten spatial dims for each batch element
        # [B, 1, H, W] → [B, H*W]
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Dice numerator: 2 * sum(pred * gt)
        intersection = (probs_flat * targets_flat).sum(dim=1)

        # Dice denominator: sum(pred) + sum(gt) + smooth
        denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth

        dice = (2.0 * intersection + self.smooth) / denominator

        # Average over batch, return 1 - Dice
        return 1.0 - dice.mean()


class WeightedBCELoss(nn.Module):
    """
    Binary cross-entropy with per-class pixel weight.

    Multiplies the foreground (positive) loss by pos_weight,
    effectively penalising false negatives more than false positives.

    Args:
        pos_weight: Scalar weight applied to positive (foreground) pixels.
                    Rule of thumb: pos_weight ≈ (n_negative / n_positive)
                    For cracks (1% FG): pos_weight ≈ 99 → use 20–30 in practice
                    Too high → model becomes over-sensitive, many false positives
    """

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:  [B, 1, H, W] — raw model outputs
            targets: [B, 1, H, W] — ground truth masks {0.0, 1.0}
        """
        # torch.tensor wrapping so weight is on the same device as logits
        weight = torch.tensor(
            [self.pos_weight],
            dtype=logits.dtype,
            device=logits.device
        )
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=weight,
        )


class CombinedSegLoss(nn.Module):
    """
    Combined Dice + Weighted BCE loss.

        L = dice_weight * DiceLoss + bce_weight * WeightedBCE

    This is the default loss for both crack and taping segmentation.
    The weights are tuned via the config.yaml.

    Args:
        dice_weight:  Weight of Dice component (default: 0.6)
        bce_weight:   Weight of BCE component  (default: 0.4)
        pos_weight:   Foreground weight for BCE (from config, per class)
        dice_smooth:  Smoothing constant for Dice

    Usage:
        loss_fn = CombinedSegLoss(
            dice_weight=0.6, bce_weight=0.4, pos_weight=30.0
        )
        loss = loss_fn(logits, masks)
        loss.backward()
    """

    def __init__(
        self,
        dice_weight: float = 0.6,
        bce_weight: float = 0.4,
        pos_weight: float = 10.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=dice_smooth)
        self.bce = WeightedBCELoss(pos_weight=pos_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

    def __repr__(self):
        return (
            f"CombinedSegLoss("
            f"dice_weight={self.dice_weight}, "
            f"bce_weight={self.bce_weight}, "
            f"pos_weight={self.bce.pos_weight})"
        )


class FocalLoss(nn.Module):
    """
    Focal loss for extremely imbalanced segmentation.
    Alternative to WeightedBCE — use if BCE over-penalises easy backgrounds.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    - gamma: focusing parameter (2.0 is standard)
    - alpha: balance factor for positive class (0.25 standard)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()