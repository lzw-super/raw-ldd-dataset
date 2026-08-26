#!/usr/bin/env python3
"""Fine-tune a packed-RAW denoiser on one phone sensor's DNG collection.

This entry point is intentionally separate from ``train_sid_sony.py``.  It
uses DNG BlackLevel/WhiteLevel/NoiseProfile metadata, condition-specific dark
residuals, and the MEY-AN00 canonical CFA order rather than Sony constants.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import yaml

from datasets.phone_synthetic_train import PhoneSyntheticTrainDataset
from losses.mrlfn_loss import RawReconstructionChromaticLoss
from models.ELD_models import UNetSeeInDark
from models.mrlfn_arch import MRLFN
from models.natnet_arch import NAFNet
from noise.dng_noise_synthesis import synthesize_phone_noise
from noise.phone_dark_shading import DEFAULT_MODEL_DIRECTORY
from tools.calculate_model_info import calculate_model_info, format_model_info
from utils.model_deployment import deploy_state_dict, prepare_model_for_inference
from utils.phone_evaluation import evaluate_phone_synthetic


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    preliminary, _ = pre_parser.parse_known_args()
    config: dict[str, Any] = {}
    if preliminary.config:
        config_path = Path(preliminary.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise ValueError("Training config must be a flat YAML mapping")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=preliminary.config)
    parser.add_argument("--clean-manifest", default="data/MEY_AN00/manifests/clean_train.jsonl")
    parser.add_argument("--dark-manifest", default="data/MEY_AN00/manifests/dark_train.jsonl")
    parser.add_argument("--calibration-root", default="data/MEY_AN00/calibration")
    parser.add_argument("--val-clean-manifest", default=None)
    parser.add_argument("--val-dark-manifest", default=None)
    parser.add_argument("--output-dir", default="experiments/mey_an00_pseudoclean")
    parser.add_argument("--resume", default=None, help="Checkpoint produced by this script")
    parser.add_argument("--init-checkpoint", default=None, help="Load model weights only; do not use --resume for SID")
    parser.add_argument("--model", choices=["unet", "nafnet", "natnet", "mrlfn"], default="nafnet")
    parser.add_argument("--model-width", type=int, default=16)
    parser.add_argument("--encoder-blocks", type=int, nargs="+", default=[1, 1, 1, 1])
    parser.add_argument("--middle-blocks", type=int, default=2)
    parser.add_argument("--decoder-blocks", type=int, nargs="+", default=[1, 1, 1, 1])
    parser.add_argument("--feature-channels", type=int, default=16, help="MRLFN feature depth d")
    parser.add_argument("--num-blocks", type=int, default=4, help="MRLFN mRLFB count N")
    parser.add_argument("--model-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-loss-weight", type=float, default=0.6)
    parser.add_argument("--chromatic-loss-weight", type=float, default=0.4)
    parser.add_argument(
        "--chromatic-channel-order",
        type=int,
        nargs=4,
        default=[0, 1, 3, 2],
        metavar=("R", "G1", "B", "G2"),
        help="Indices for semantic [R,G1,B,G2]; packed tensors here are [R,G1,G2,B]",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None, help="Hard total-step cap; recommended for pseudo-clean smoke runs")
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--crops-per-image", type=int, default=2)
    parser.add_argument("--ratios", type=float, nargs="+", default=[300.0], help="Synthetic exposure ratios sampled per item")
    parser.add_argument("--ratio", type=float, default=None, help="Deprecated single-ratio override; takes precedence over --ratios")
    parser.add_argument("--clean-cache-size", type=int, default=4)
    parser.add_argument("--dark-cache-size", type=int, default=2)
    parser.add_argument(
        "--dark-shading-model",
        choices=["continuous_iso_fit", "per_condition_mean"],
        default="continuous_iso_fit",
        help="Continuous PMN/SID-style k*ISO+b model, or legacy per-ISO mean-map ablation",
    )
    parser.add_argument("--max-target-saturation-fraction", type=float, default=0.01)
    parser.add_argument("--max-crop-attempts", type=int, default=32)
    parser.add_argument("--exposure-ratio-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--dark-exposure-policy",
        choices=["strict_match", "approximate_reuse_1_30s"],
        default="strict_match",
        help="Whether a 1/30 s residual may be reused for simulated 1/10 s and 1/25 s inputs",
    )
    parser.add_argument("--synthesis", choices=["hybrid", "analytic"], default="hybrid")

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--validate-steps", type=int, default=0, help="0 disables held-out synthetic validation")
    parser.add_argument("--val-ratios", type=float, nargs="+", default=[300.0], help="Fixed ratios for held-out synthetic validation")
    parser.add_argument("--val-crops-per-image", type=int, default=1, help="Deterministic held-out crops per clean validation DNG")
    parser.add_argument("--val-seed", type=int, default=10_001, help="Fixed crop/dark/Poisson seed for held-out synthetic validation")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.set_defaults(**config)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: PhoneSyntheticTrainDataset, args: argparse.Namespace, *, shuffle: bool, workers: int) -> DataLoader:
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = (args.seed + worker_id + 1) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "worker_init_fn": worker_init_fn if workers else None,
    }
    if workers:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 1})
    return DataLoader(dataset, **kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = min(max(0, int(warmup_steps)), total_steps - 1)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps + 1:
            return 1.0
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, multiplier)


def make_grad_scaler(enabled: bool) -> Any:
    """Use the non-deprecated AMP API while retaining older Torch support."""
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        return amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool) -> Any:
    if device.type != "cuda":
        return nullcontext()
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "autocast"):
        return amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def build_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    if args.model == "unet":
        return UNetSeeInDark(in_nc=4, out_nc=4, nf=args.model_width), {
            "model": "unet",
            "in_channels": 4,
            "out_channels": 4,
            "width": int(args.model_width),
        }
    if args.model == "mrlfn":
        architecture = {
            "model": "mrlfn",
            "in_channels": 4,
            "out_channels": 4,
            "feature_channels": int(args.feature_channels),
            "num_blocks": int(args.num_blocks),
            "bias": bool(args.model_bias),
            "deploy": False,
        }
        return (
            MRLFN(
                in_channels=4,
                out_channels=4,
                feature_channels=args.feature_channels,
                num_blocks=args.num_blocks,
                bias=args.model_bias,
                deploy=False,
            ),
            architecture,
        )
    architecture = {
        "model": "nafnet",
        "img_channel": 4,
        "width": int(args.model_width),
        "enc_blk_nums": [int(value) for value in args.encoder_blocks],
        "middle_blk_num": int(args.middle_blocks),
        "dec_blk_nums": [int(value) for value in args.decoder_blocks],
    }
    return (
        NAFNet(
            img_channel=4,
            width=args.model_width,
            enc_blk_nums=tuple(args.encoder_blocks),
            middle_blk_num=args.middle_blocks,
            dec_blk_nums=tuple(args.decoder_blocks),
        ),
        architecture,
    )


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "clean",
        "dark_residual_dn",
        "noise_profile_s4",
        "noise_profile_o4",
        "dark_dynamic_range",
        "ratio",
        "iso_exif",
        "simulated_short_exposure_s",
        "dark_source_exposure_s",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def synthesize(batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    return synthesize_phone_noise(
        batch["clean"],
        batch["dark_residual_dn"],
        batch["noise_profile_s4"],
        batch["dark_dynamic_range"],
        batch["ratio"],
        mode=args.synthesis,
        noise_profile_o4=batch["noise_profile_o4"],
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    fused_state = deploy_state_dict(model)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
        "checkpoint_format": 2,
        "training_graph": "reparameterizable" if fused_state is not None else "native",
    }
    if fused_state is not None:
        state["model_deploy"] = fused_state
        state["inference_graph"] = "deploy"
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    scaler: Any | None = None,
    expected_args: argparse.Namespace | None = None,
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    if "model" in state:
        if optimizer is not None and expected_args is not None:
            saved_args = state.get("args", {})
            saved_ds_model = saved_args.get("dark_shading_model", "per_condition_mean")
            current_ds_model = expected_args.dark_shading_model
            comparisons = {
                "model": (saved_args.get("model", "nafnet"), expected_args.model),
                "model_width": (saved_args.get("model_width", 16), expected_args.model_width),
                "encoder_blocks": (saved_args.get("encoder_blocks", [1, 1, 1, 1]), expected_args.encoder_blocks),
                "middle_blocks": (saved_args.get("middle_blocks", 2), expected_args.middle_blocks),
                "decoder_blocks": (saved_args.get("decoder_blocks", [1, 1, 1, 1]), expected_args.decoder_blocks),
                "feature_channels": (saved_args.get("feature_channels", 16), expected_args.feature_channels),
                "num_blocks": (saved_args.get("num_blocks", 4), expected_args.num_blocks),
                "model_bias": (saved_args.get("model_bias", True), expected_args.model_bias),
                "raw_loss_weight": (saved_args.get("raw_loss_weight", 0.6), expected_args.raw_loss_weight),
                "chromatic_loss_weight": (
                    saved_args.get("chromatic_loss_weight", 0.4),
                    expected_args.chromatic_loss_weight,
                ),
                "chromatic_channel_order": (
                    saved_args.get("chromatic_channel_order", [0, 1, 3, 2]),
                    expected_args.chromatic_channel_order,
                ),
                "synthesis": (saved_args.get("synthesis", "hybrid"), expected_args.synthesis),
                "dark_shading_model": (saved_ds_model, current_ds_model),
                "dark_exposure_policy": (
                    saved_args.get("dark_exposure_policy", "strict_match"),
                    expected_args.dark_exposure_policy,
                ),
                "ratios": (saved_args.get("ratios", [300.0]), expected_args.ratios),
                "patch_size": (saved_args.get("patch_size", 512), expected_args.patch_size),
                "crops_per_image": (saved_args.get("crops_per_image", 2), expected_args.crops_per_image),
                "epochs": (saved_args.get("epochs", 100), expected_args.epochs),
                "steps_per_epoch": (saved_args.get("steps_per_epoch"), expected_args.steps_per_epoch),
                "max_steps": (saved_args.get("max_steps"), expected_args.max_steps),
            }
            mismatches = {
                key: {"checkpoint": saved, "current": current}
                for key, (saved, current) in comparisons.items()
                if saved != current
            }
            if mismatches:
                raise ValueError(
                    "Refusing to resume with a changed training protocol: "
                    + json.dumps(mismatches, ensure_ascii=False)
                    + ". Start a new output directory and use --init-checkpoint for model-only transfer."
                )
        model.load_state_dict(state["model"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state.get("scaler", {}))
        return int(state.get("epoch", 0)), int(state.get("global_step", 0))
    model.load_state_dict(state, strict=True)
    return 0, 0


def prune_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    if keep < 0:
        return
    paths = sorted(checkpoint_dir.glob("epoch_*.pth"), key=lambda path: path.stat().st_mtime)
    for stale in paths[: max(0, len(paths) - keep)]:
        stale.unlink()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.epochs <= 0 or args.batch_size <= 0 or args.crops_per_image <= 0:
        raise ValueError("epochs, batch-size and crops-per-image must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if bool(args.val_clean_manifest) != bool(args.val_dark_manifest):
        raise ValueError("Set both val-clean-manifest and val-dark-manifest, or neither")
    ratios = (float(args.ratio),) if args.ratio is not None else tuple(float(value) for value in args.ratios)
    if not ratios or any(value <= 0 for value in ratios):
        raise ValueError("ratios must contain one or more positive values")
    val_ratios = tuple(float(value) for value in args.val_ratios)
    if not val_ratios or any(value <= 0 for value in val_ratios):
        raise ValueError("val-ratios must contain one or more positive values")
    if args.val_crops_per_image <= 0 or args.validate_every <= 0:
        raise ValueError("val-crops-per-image and validate-every must be positive")
    if args.feature_channels <= 0 or args.num_blocks <= 0:
        raise ValueError("feature-channels and num-blocks must be positive")
    if args.model == "mrlfn" and list(args.chromatic_channel_order) != [0, 1, 3, 2]:
        print("Warning: current packed RAW order is [R,G1,G2,B]; the aligned semantic order is [0,1,3,2]")

    device = torch.device(args.device)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    train_dataset = PhoneSyntheticTrainDataset(
        args.clean_manifest,
        args.dark_manifest,
        args.calibration_root,
        patch_size=args.patch_size,
        crops_per_image=args.crops_per_image,
        ratios=ratios,
        cache_size=args.clean_cache_size,
        dark_cache_size=args.dark_cache_size,
        max_target_saturation_fraction=args.max_target_saturation_fraction,
        max_crop_attempts=args.max_crop_attempts,
        seed=args.seed,
        exposure_ratio_tolerance=args.exposure_ratio_tolerance,
        dark_exposure_policy=args.dark_exposure_policy,
        dark_shading_model=args.dark_shading_model,
    )
    train_loader = make_loader(train_dataset, args, shuffle=True, workers=args.num_workers)
    available_steps = len(train_loader)
    steps_per_epoch = int(args.steps_per_epoch or available_steps)
    if steps_per_epoch <= 0 or steps_per_epoch > available_steps:
        raise ValueError(f"steps-per-epoch must be in [1,{available_steps}], got {steps_per_epoch}")
    val_dataset = None
    if args.val_clean_manifest:
        val_dataset = PhoneSyntheticTrainDataset(
            args.val_clean_manifest,
            args.val_dark_manifest,
            args.calibration_root,
            patch_size=args.patch_size,
            crops_per_image=args.val_crops_per_image,
            ratios=val_ratios,
            cache_size=2,
            dark_cache_size=1,
            max_target_saturation_fraction=args.max_target_saturation_fraction,
            max_crop_attempts=args.max_crop_attempts,
            seed=args.seed + 10_000,
            exposure_ratio_tolerance=args.exposure_ratio_tolerance,
            dark_exposure_policy=args.dark_exposure_policy,
            dark_shading_model=args.dark_shading_model,
        )
    elif args.validate_steps:
        raise ValueError("Held-out synthetic validation requires both val clean and val dark manifests")

    total_steps = int(args.max_steps) if args.max_steps else args.epochs * steps_per_epoch
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "stage": "pseudo_clean_prototype",
        "target_protocol": "10s pseudo-clean; no 1/30s DS subtraction",
        "synthesis": args.synthesis,
        "noise_condition": "same EXIF ISO as clean source",
        "ratios": list(ratios),
        "dark_exposure_policy": args.dark_exposure_policy,
        "dark_shading_model": args.dark_shading_model,
        "loss": {
            "name": "raw_reconstruction_chromatic" if args.model == "mrlfn" else "l1",
            "raw_weight": args.raw_loss_weight if args.model == "mrlfn" else None,
            "chromatic_weight": args.chromatic_loss_weight if args.model == "mrlfn" else None,
            "channel_order_for_R_G1_B_G2": (
                list(args.chromatic_channel_order) if args.model == "mrlfn" else None
            ),
        },
        "clean_manifest": str(Path(args.clean_manifest).resolve()),
        "clean_manifest_sha256": file_sha256(args.clean_manifest),
        "dark_manifest": str(Path(args.dark_manifest).resolve()),
        "dark_manifest_sha256": file_sha256(args.dark_manifest),
        "val_clean_manifest": str(Path(args.val_clean_manifest).resolve()) if args.val_clean_manifest else None,
        "val_clean_manifest_sha256": file_sha256(args.val_clean_manifest) if args.val_clean_manifest else None,
        "val_dark_manifest": str(Path(args.val_dark_manifest).resolve()) if args.val_dark_manifest else None,
        "val_dark_manifest_sha256": file_sha256(args.val_dark_manifest) if args.val_dark_manifest else None,
        "validation_protocol": {
            "metric": "synthetic_heldout_pseudoclean_not_real_pair_psnr",
            "ratios": list(val_ratios),
            "crops_per_image": int(args.val_crops_per_image),
            "max_samples": int(args.validate_steps),
            "seed": int(args.val_seed),
        },
        "calibration_root": str(Path(args.calibration_root).resolve()),
        "dark_shading_model_metadata": (
            str((Path(args.calibration_root) / DEFAULT_MODEL_DIRECTORY / "metadata.json").resolve())
            if args.dark_shading_model == "continuous_iso_fit"
            else None
        ),
        "dark_shading_model_metadata_sha256": (
            file_sha256(Path(args.calibration_root) / DEFAULT_MODEL_DIRECTORY / "metadata.json")
            if args.dark_shading_model == "continuous_iso_fit"
            else None
        ),
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
        "init_checkpoint_sha256": file_sha256(args.init_checkpoint) if args.init_checkpoint else None,
    }
    (output_dir / "run_protocol.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model, architecture = build_model(args)
    model = model.to(device)
    profile_model, profile_graph = prepare_model_for_inference(model)
    model_info = calculate_model_info(profile_model, (1, 4, args.patch_size, args.patch_size), device)
    model_info["architecture_config"] = {**architecture, "deploy": profile_graph == "deploy"}
    model_info["graph_state"] = profile_graph
    model_info["loss_config"] = (
        {
            "name": "RawReconstructionChromaticLoss",
            "raw_weight": args.raw_loss_weight,
            "chromatic_weight": args.chromatic_loss_weight,
            "channel_order_for_R_G1_B_G2": list(args.chromatic_channel_order),
        }
        if args.model == "mrlfn"
        else {"name": "L1Loss"}
    )
    (output_dir / "model_info.json").write_text(json.dumps(model_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(format_model_info(model_info))
    del profile_model
    optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_epochs * steps_per_epoch)
    scaler = make_grad_scaler(bool(args.amp))
    start_epoch, global_step = 1, 0
    if args.resume:
        previous_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, expected_args=args
        )
        start_epoch = previous_epoch + 1
        print(f"Resumed {args.resume} at epoch {start_epoch}, global step {global_step}")
    elif args.init_checkpoint:
        load_checkpoint(args.init_checkpoint, model)
        print(f"Initialised model weights from {args.init_checkpoint}")

    criterion = (
        RawReconstructionChromaticLoss(
            raw_weight=args.raw_loss_weight,
            chromatic_weight=args.chromatic_loss_weight,
            channel_order=tuple(args.chromatic_channel_order),
        )
        if args.model == "mrlfn"
        else None
    )

    print(
        f"Training {type(model).__name__} on {len(train_dataset)} pseudo-clean samples; "
        f"{steps_per_epoch} steps/epoch, total cap={total_steps}, device={device}, synthesis={args.synthesis}, "
        f"ratios={list(ratios)}, dark_policy={args.dark_exposure_policy}, DS={args.dark_shading_model}"
    )
    metrics_path = output_dir / "metrics.jsonl"
    finished = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss, epoch_raw_l1, epoch_chromatic_l1 = 0.0, 0.0, 0.0
        epoch_samples, epoch_start = 0, time.perf_counter()
        epoch_ratio_counts: Counter[float] = Counter()
        epoch_simulated_exposures: list[float] = []
        epoch_dark_exposures: list[float] = []
        for step, raw_batch in enumerate(train_loader, start=1):
            if step > steps_per_epoch or global_step >= total_steps:
                break
            batch = to_device(raw_batch, device)
            noisy, target = synthesize(batch, args)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, bool(args.amp)):
                prediction = model(noisy)
                if criterion is not None:
                    per_sample_loss, per_sample_raw, per_sample_chromatic = criterion.per_sample_components(
                        prediction, target
                    )
                    loss = per_sample_loss.mean()
                    raw_l1 = per_sample_raw.mean()
                    chromatic_l1 = per_sample_chromatic.mean()
                else:
                    loss = F.l1_loss(torch.clamp(prediction, 0.0, 1.0), target)
                    raw_l1 = loss
                    chromatic_l1 = loss.new_zeros(())
            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            epoch_loss += float(loss.detach()) * target.shape[0]
            epoch_raw_l1 += float(raw_l1.detach()) * target.shape[0]
            epoch_chromatic_l1 += float(chromatic_l1.detach()) * target.shape[0]
            epoch_samples += target.shape[0]
            epoch_ratio_counts.update(float(value) for value in batch["ratio"].detach().cpu().tolist())
            epoch_simulated_exposures.extend(
                float(value) for value in batch["simulated_short_exposure_s"].detach().cpu().tolist()
            )
            epoch_dark_exposures.extend(
                float(value) for value in batch["dark_source_exposure_s"].detach().cpu().tolist()
            )
            if step == 1 or step % args.log_every == 0 or global_step == total_steps:
                print(
                    f"epoch {epoch:04d} step {step:04d}/{steps_per_epoch} global={global_step}/{total_steps} "
                    f"loss={float(loss.detach()):.6f} raw_l1={float(raw_l1.detach()):.6f} "
                    f"chromatic_l1={float(chromatic_l1.detach()):.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"iso={int(batch['iso_exif'][0])} ratio={float(batch['ratio'][0]):g} "
                    f"sim_t={float(batch['simulated_short_exposure_s'][0]):.5f}s "
                    f"dark_t={float(batch['dark_source_exposure_s'][0]):.5f}s "
                    f"negative_input={float((noisy < 0).float().mean()):.4f}"
                )
        metrics: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / max(1, epoch_samples),
            "train_l1": epoch_raw_l1 / max(1, epoch_samples),
            "train_chromatic_l1": epoch_chromatic_l1 / max(1, epoch_samples),
            "loss_name": "raw_reconstruction_chromatic" if criterion is not None else "l1",
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - epoch_start,
            "stage": "pseudo_clean_prototype",
            "dark_exposure_policy": args.dark_exposure_policy,
            "dark_shading_model": args.dark_shading_model,
            "ratio_counts": {f"{ratio:g}": count for ratio, count in sorted(epoch_ratio_counts.items())},
            "simulated_short_exposure_s": {
                "min": min(epoch_simulated_exposures) if epoch_simulated_exposures else None,
                "max": max(epoch_simulated_exposures) if epoch_simulated_exposures else None,
            },
            "dark_source_exposure_s": {
                "min": min(epoch_dark_exposures) if epoch_dark_exposures else None,
                "max": max(epoch_dark_exposures) if epoch_dark_exposures else None,
            },
        }
        if val_dataset is not None and args.validate_steps and (epoch % args.validate_every == 0 or global_step >= total_steps):
            inference_model, validation_graph = prepare_model_for_inference(model)
            metrics.update(
                evaluate_phone_synthetic(
                    inference_model,
                    val_dataset,
                    device=device,
                    ratios=val_ratios,
                    max_samples=args.validate_steps,
                    seed=args.val_seed,
                    synthesis=args.synthesis,
                )
            )
            metrics["validation_graph"] = validation_graph
            del inference_model
            model.train()
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print("epoch summary", json.dumps(metrics, ensure_ascii=False))
        save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, scheduler, scaler, epoch, global_step, args)
        if epoch % args.save_every == 0 or global_step >= total_steps:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pth", model, optimizer, scheduler, scaler, epoch, global_step, args)
            prune_checkpoints(checkpoint_dir, args.keep_checkpoints)
        if global_step >= total_steps:
            finished = True
            break
    if not finished and args.max_steps:
        raise RuntimeError("max-steps was not reached; inspect epochs/steps-per-epoch")


if __name__ == "__main__":
    main()
