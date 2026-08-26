"""Checkpoint-compatible model helpers for the phone DNG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.ELD_models import UNetSeeInDark
from models.mrlfn_arch import MRLFN
from models.natnet_arch import NAFNet


def build_phone_model(model_args: dict[str, Any], *, deploy: bool = False) -> tuple[torch.nn.Module, str]:
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
    if model_name == "mrlfn":
        return (
            MRLFN(
                in_channels=4,
                out_channels=4,
                feature_channels=int(model_args.get("feature_channels", 16)),
                num_blocks=int(model_args.get("num_blocks", 4)),
                bias=bool(model_args.get("model_bias", True)),
                deploy=deploy,
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
    model_name = str(checkpoint_args.get("model", "nafnet")).lower()
    has_deploy_weights = (
        model_name == "mrlfn" and isinstance(checkpoint, dict) and "model_deploy" in checkpoint
    )
    state_dict = (
        checkpoint["model_deploy"]
        if has_deploy_weights
        else checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint_path}: checkpoint does not contain a state dict")
    bare_deploy_weights = (
        model_name == "mrlfn"
        and isinstance(state_dict, dict)
        and any(".reparam_conv." in key for key in state_dict)
    )
    model, model_name = build_phone_model(checkpoint_args, deploy=has_deploy_weights or bare_deploy_weights)
    model.load_state_dict(state_dict, strict=True)
    if model_name == "mrlfn" and not (has_deploy_weights or bare_deploy_weights):
        model = model.deploy()
    model = model.to(device)
    model.eval()
    return model, checkpoint_args, model_name
