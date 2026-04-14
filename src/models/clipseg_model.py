"""
1. CLIP image encoder (ViT-B/16):
   - Divides image into 16×16 patches → sequence of patch tokens
   - Each patch gets a learned position embedding
   - Processes through 12 transformer blocks → global image embedding
   - Also produces intermediate activations from blocks 3, 6, 9, 12
     (these are the "skip connections" used by the decoder)

2. CLIP text encoder:
   - Tokenises the prompt text
   - Encodes through transformer → single [CLS] token embedding
   - This text embedding is what "conditions" the decoder

3. CLIPSeg decoder (FiLM-conditioned transformer):
   - FiLM = Feature-wise Linear Modulation
   - For each CLIP skip connection, the text embedding is used to
     compute a scale γ and shift β:
       output = γ(text) * image_features + β(text)
   - This gates which image features are activated based on the text
   - After FiLM modulation: upsampling transposed convolutions → logit map
   - Output: 1×H×W logit map (NOT a probability yet — apply sigmoid)

WHY THIS ARCHITECTURE FOR OUR TASK:
- We don't need to train from scratch (expensive, needs massive data)
- CLIP already understands "crack" and "taping" in its text-image space
- We just need to teach the decoder WHERE those concepts are in the image
- Fine-tuning only the decoder first (freeze backbone) is conservative:
  it changes only 10% of parameters while leveraging all of CLIP's
  learned visual-semantic alignments

REFERENCE: Lüddecke & Ecker (2022), "Image Segmentation Using Text and
Image Prompts", CVPR 2022. HuggingFace model: CIDAS/clipseg-rd64-refined
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DrywallCLIPSeg(nn.Module):
    """
    CLIPSeg wrapper for drywall QA.

    Wraps HuggingFace CLIPSegForImageSegmentation with:
    - Controlled freeze/unfreeze strategy
    - Clean forward pass returning (logits, loss_optional)
    - Parameter group creation for differential learning rates

    Args:
        model_name:  HuggingFace model identifier
        freeze_backbone: If True, freeze CLIP encoder weights (Phase 1)
    """

    def __init__(
        self,
        model_name: str = "CIDAS/clipseg-rd64-refined",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        logger.info(f"Loading CLIPSeg from: {model_name}")
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)

        self.model_name = model_name
        self._freeze_backbone = freeze_backbone

        if freeze_backbone:
            self._freeze_clip_encoder()

        self._log_parameter_counts()

    # ──────────────────────────────────────────────────────────
    # Freeze / unfreeze controls
    # ──────────────────────────────────────────────────────────

    def _freeze_clip_encoder(self) -> None:
        """
        Freeze ALL CLIP encoder parameters (both vision and text).
        Only the CLIPSeg decoder remains trainable.

        Phase 1 rationale:
        - CLIP encoder has 86M parameters, decoder ~6M
        - Freezing means we only optimise 7% of total params
        - Fast convergence, no risk of destroying CLIP's alignments
        - Acts as a strong feature extractor out of the box
        """
        # Freeze vision encoder
        for param in self.model.clip.vision_model.parameters():
            param.requires_grad = False

        # Freeze text encoder
        for param in self.model.clip.text_model.parameters():
            param.requires_grad = False

        # Freeze CLIP projection layers
        for param in self.model.clip.visual_projection.parameters():
            param.requires_grad = False
        for param in self.model.clip.text_projection.parameters():
            param.requires_grad = False

        logger.info("CLIP encoder frozen. Only CLIPSeg decoder is trainable.")

    def unfreeze_top_n_layers(self, n: int = 4) -> None:
        """
        Unfreeze the top N transformer blocks of the CLIP vision encoder.
        Used for Phase 2 fine-tuning.

        WHY ONLY TOP N:
        Lower layers of a ViT learn universal low-level features
        (edges, textures, colour gradients). These are stable across
        domains and should stay frozen.
        Upper layers learn high-level semantic features (object parts,
        scene context). These benefit from domain adaptation — the model
        learns what "crack texture" and "mudded joint" look like.

        n=4 is a common industry default (ViT-B has 12 blocks total,
        so we unfreeze the top 1/3). Use n=2 for very small datasets.

        Args:
            n: Number of top transformer blocks to unfreeze
        """
        vision_blocks = self.model.clip.vision_model.encoder.layers
        total_blocks = len(vision_blocks)  # 12 for ViT-B/16
        unfreeze_from = total_blocks - n

        for i, block in enumerate(vision_blocks):
            if i >= unfreeze_from:
                for param in block.parameters():
                    param.requires_grad = True
                logger.info(f"  Unfroze vision block {i}/{total_blocks-1}")

        # Also unfreeze the final layer norm
        for param in self.model.clip.vision_model.post_layernorm.parameters():
            param.requires_grad = True

        self._freeze_backbone = False
        logger.info(
            f"Unfroze top {n} vision transformer blocks "
            f"({unfreeze_from}–{total_blocks-1})."
        )
        self._log_parameter_counts()

    # ──────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pixel_values:   [B, 3, 352, 352] — normalised image tensors
            input_ids:      [B, seq_len]      — tokenised prompt
            attention_mask: [B, seq_len]      — 1 for real tokens, 0 for padding

        Returns:
            logits: [B, 1, 352, 352] — raw scores before sigmoid
                    Apply sigmoid to get probabilities [0, 1]
                    Apply threshold to get binary mask {0, 1}

        Note on CLIPSeg output:
            The HF model returns CLIPSegImageSegmentationOutput with
            .logits of shape [B, H, W] (no channel dim). We add the
            channel dim for compatibility with our loss functions.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # outputs.logits: [B, H, W]
        logits = outputs.logits.unsqueeze(1)  # → [B, 1, H, W]
        return logits

    # ──────────────────────────────────────────────────────────
    # Optimizer parameter groups
    # ──────────────────────────────────────────────────────────

    def get_parameter_groups(
        self,
        decoder_lr: float,
        encoder_lr: Optional[float] = None,
        weight_decay: float = 1e-4,
    ) -> List[dict]:
        """
        Create parameter groups for differential learning rates.

        Why differential LRs?
        - The decoder is randomly (re-)initialised for our task → needs
          a higher learning rate to converge quickly.
        - The unfrozen CLIP blocks carry valuable pretrained knowledge →
          need a MUCH lower LR so we don't overwrite them with noise.
          Industry convention: encoder_lr = decoder_lr / 10 to decoder_lr / 100.

        Args:
            decoder_lr: Learning rate for CLIPSeg decoder (e.g. 1e-4)
            encoder_lr: Learning rate for unfrozen CLIP blocks (e.g. 5e-6)
                        If None, only decoder group is returned (Phase 1).
            weight_decay: L2 regularisation coefficient

        Returns:
            List of parameter group dicts for torch.optim.AdamW
        """
        groups = []

        # ── Decoder group (always trainable) ──────────────────
        decoder_params = [
            p for p in self.model.decoder.parameters()
            if p.requires_grad
        ]
        if decoder_params:
            groups.append({
                "params": decoder_params,
                "lr": decoder_lr,
                "weight_decay": weight_decay,
                "name": "decoder",
            })
            logger.info(
                f"Decoder param group: {sum(p.numel() for p in decoder_params):,} params, "
                f"lr={decoder_lr}"
            )

        # ── Encoder group (unfrozen blocks only) ──────────────
        if encoder_lr is not None:
            encoder_params = [
                p for p in self.model.clip.parameters()
                if p.requires_grad
            ]
            if encoder_params:
                groups.append({
                    "params": encoder_params,
                    "lr": encoder_lr,
                    "weight_decay": weight_decay,
                    "name": "encoder_top_blocks",
                })
                logger.info(
                    f"Encoder param group: {sum(p.numel() for p in encoder_params):,} params, "
                    f"lr={encoder_lr}"
                )

        if not groups:
            raise RuntimeError("No trainable parameters found. Check freeze settings.")

        return groups

    # ──────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────

    def _log_parameter_counts(self) -> None:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Parameters: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/total:.1f}%)"
        )

    def model_size_mb(self) -> float:
        """Estimate model size in MB (float32 params only)."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 4 / (1024 ** 2)

    def save(self, path: str) -> None:
        """Save model weights + processor."""
        import os
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "DrywallCLIPSeg":
        """Load a saved model."""
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.processor = CLIPSegProcessor.from_pretrained(path)
        instance.model = CLIPSegForImageSegmentation.from_pretrained(path)
        instance.model_name = path
        instance._freeze_backbone = False
        logger.info(f"Model loaded from: {path}")
        return instance