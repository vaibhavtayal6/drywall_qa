
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
  

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
    
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