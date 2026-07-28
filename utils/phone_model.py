"""Checkpoint-compatible model helpers for the phone DNG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.ELD_models import UNetSeeInDark
from models.natnet_arch import NAFNet


def build_phone_model(model_args: dict[str, Any]) -> tuple[torch.nn.Module, str]:
    """Construct a phone denoiser from the architecture stored in a checkpoint."""
    model_name = str(model_args.get("model", "nafnet")).lower()
    if model_name == "unet":
        return UNetSeeInDark(in_nc=4, out_nc=4, nf=int(model_args.get("model_width", 16))), model_name
    if model_name in {"nafnet", "natnet"}:
        return (
            NAFNet(
                img_channel=4,
                width=int(model_args.get("model_width", 16)),
                enc_blk_nums=tuple(model_args.get("encoder_blocks", [1, 1, 1, 1])),
                middle_blk_num=int(model_args.get("middle_blocks", 2)),
                dec_blk_nums=tuple(model_args.get("decoder_blocks", [1, 1, 1, 1])),
            ),
            model_name,
        )
    raise ValueError(f"Unsupported phone checkpoint model: {model_name!r}")


def load_phone_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    """Load a ``train_phone.py`` checkpoint and return its eval-mode model.

    A plain state dict is accepted for compatibility, but then the caller must
    use the default NAFNet-Tiny architecture or provide a checkpoint produced
    by ``train_phone.py`` with the original ``args`` metadata.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint_path}: checkpoint does not contain a state dict")
    model, model_name = build_phone_model(checkpoint_args)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, checkpoint_args, model_name
