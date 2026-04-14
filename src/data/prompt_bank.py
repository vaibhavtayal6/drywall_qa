"""
src/data/prompt_bank.py
-----------------------
Prompt bank for text-conditioned segmentation.

CORE IDEA:
CLIPSeg conditions its decoder on a CLIP text embedding of the prompt.
CLIP encodes "segment crack" and "find wall damage" as different vectors
in embedding space — even though they describe the same task.

If you train with only "segment crack", the model learns to respond to
THAT specific vector. At test time, any synonym degrades performance.

The fix: during training, randomly sample one prompt from a bank of
semantically equivalent phrasings. This forces the model to respond
to a *region* of embedding space (the semantic concept "crack") rather
than a single point (one specific sentence).

At inference we go further: run ALL prompts, average the logit maps,
then threshold once. This is "prompt ensembling" and it's the single
highest-value post-training improvement for CLIPSeg.

Industry analogy: test-time augmentation (TTA) — same idea but for
the text modality instead of the image modality.
"""

import random
from typing import List, Optional
import yaml
from pathlib import Path


# Default prompt banks (loaded from config.yaml at runtime,
# but also usable standalone via these defaults)
DEFAULT_PROMPTS = {
    "taping": [
        "segment taping area",
        "segment joint tape",
        "segment drywall seam",
        "white tape on wall",
        "segment wall repair",
        "segment plastered joint",
        "light colored seam on drywall",
        "segment white strip on wall",
    ],
    "crack": [
        "segment crack",
        "segment wall crack",
        "detect surface crack",
        "find structural crack",
        "segment hairline crack",
        "detect wall damage",
    ],
}


class PromptBank:
    """
    Manages prompt sampling and ensembling for a segmentation task.

    Attributes:
        prompts: dict mapping class_name → list of prompt strings
        seed:    random seed (for reproducibility in val/test sampling)

    Example:
        bank = PromptBank.from_config("configs/config.yaml")

        # Training: one random prompt per image
        prompt = bank.sample("crack")
        # → "segment hairline crack"

        # Inference: all prompts for ensembling
        prompts = bank.get_all("crack")
        # → ["segment crack", "segment wall crack", ...]
    """

    def __init__(self, prompts: dict, seed: int = 42):
        self.prompts = prompts
        self._rng = random.Random(seed)

    @classmethod
    def from_config(cls, config_path: str) -> "PromptBank":
        """Load prompt bank from config.yaml."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            prompts=cfg["prompts"],
            seed=cfg.get("seed", 42)
        )

    @classmethod
    def default(cls) -> "PromptBank":
        """Use default built-in prompts (no config file needed)."""
        return cls(prompts=DEFAULT_PROMPTS)

    def sample(self, class_name: str) -> str:
        """
        Sample one prompt randomly from the bank.
        Used during training — different prompt each epoch.

        Args:
            class_name: "crack" or "taping"
        Returns:
            A single prompt string
        Raises:
            KeyError if class_name not in bank
        """
        if class_name not in self.prompts:
            raise KeyError(
                f"Class '{class_name}' not in prompt bank. "
                f"Available: {list(self.prompts.keys())}"
            )
        return self._rng.choice(self.prompts[class_name])

    def get_all(self, class_name: str) -> List[str]:
        """
        Return all prompts for a class.
        Used during inference for prompt ensembling.
        """
        return list(self.prompts[class_name])

    def get_primary(self, class_name: str) -> str:
        """Return the first (canonical) prompt — used for naming outputs."""
        return self.prompts[class_name][0]

    def classes(self) -> List[str]:
        """List all class names in the bank."""
        return list(self.prompts.keys())

    def __repr__(self):
        summary = {k: len(v) for k, v in self.prompts.items()}
        return f"PromptBank({summary})"


def make_output_filename(image_id: str, prompt: str) -> str:
    """
    Generate the required output mask filename.

    Format required by rubric: {image_id}__{prompt_with_underscores}.png
    Example: "123__segment_crack.png"

    Args:
        image_id: stem of the source image file (e.g. "123", "img_042")
        prompt:   the canonical prompt string (e.g. "segment crack")
    Returns:
        Filename string
    """
    prompt_slug = prompt.replace(" ", "_")
    return f"{image_id}__{prompt_slug}.png"