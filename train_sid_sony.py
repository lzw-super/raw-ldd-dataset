#!/usr/bin/env python3
"""Train a RAW denoiser with SID clean RAW plus LLD dark-frame synthesis.

This is a reproducible implementation of the paper-fair path described in
``ref-pdf/Noise Modeling in One Hour - SID Sony 训练复现指南.md``.  It never
uses SID short images for loss supervision; only prefix-0 SID long-exposure
patches provide targets.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import yaml

from datasets.sid_synthetic_train import (
    SIDSyntheticTrainDataset,
    build_sid_patch_manifest,
    build_sid_raw_manifest,
)
from datasets.sid_real_validation import SIDRealValidationDataset
from utils.utils import ELDIlluminanceCorrect, PMN_metric, tensor_dim5to4
from losses.mrlfn_loss import RawReconstructionChromaticLoss
from losses.wavelet_ll_loss import WaveletLLLoss
from losses.detail_loss import GradientLoss, WaveletHFLoss
from models.ELD_models import UNetSeeInDark
from models.mrlfn_arch import MRLFN
from models.natnet_arch import NAFNet
from learning_wt.learning_dwt import LearningDWT, DWT_DEFAULTS, learning_dwt_kwargs
from noise.sid_noise_synthesis import synthesize_sid_noise
from tools.calculate_model_info import calculate_model_info, format_model_info
from utils.argparse_compat import add_boolean_optional_argument
from utils.model_deployment import deploy_state_dict, prepare_model_for_inference


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    preliminary, _ = pre_parser.parse_known_args()
    config: Dict[str, object] = {}
    if preliminary.config:
        config_path = Path(preliminary.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Training config must be a flat YAML mapping")
        config = loaded

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=preliminary.config)
    parser.add_argument("--patch-dir", default="/home/shared_files/dataset/SID/Sony_train_long_patches")
    parser.add_argument("--pair-list", default="/home/shared_files/dataset/SID/Sony_train_list.txt")
    parser.add_argument("--sid-long-dir", default="/home/shared_files/dataset/SID/Sony/long")
    parser.add_argument("--clean-source", choices=["raw", "packed"], default="raw")
    parser.add_argument("--manifest", default="infos/SID_train_clean_raw.json")
    parser.add_argument("--val-manifest", default=None, help="Legacy option; real validation uses val-pair-list")
    parser.add_argument("--val-pair-list", default=None, help="Defaults to Sony_val_list.txt beside pair-list")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--dark-root", default="biasframe_et_1_30")
    parser.add_argument("--pmn-resource-dir", default="resources/SonyA7S2")
    parser.add_argument("--output-dir", default="experiments/sid_sony_paper_fair")
    parser.add_argument("--resume", default=None, help="Checkpoint produced by this script")
    parser.add_argument("--init-checkpoint", default=None, help="Model-only checkpoint; do not use the official test checkpoint")
    parser.add_argument("--model", choices=["unet", "nafnet", "natnet", "mrlfn", "learning_dwt", "learning_dwt_repncb"], default="unet")
    from models.learning_dwt_repncb import REFINER_DEFAULTS
    for key, default in {**DWT_DEFAULTS, **REFINER_DEFAULTS}.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            add_boolean_optional_argument(parser, flag, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--encoder-blocks", type=int, nargs="+", default=[2, 2, 2, 2])
    parser.add_argument("--middle-blocks", type=int, default=2)
    parser.add_argument("--decoder-blocks", type=int, nargs="+", default=[2, 2, 2, 2])
    parser.add_argument("--feature-channels", type=int, default=16, help="MRLFN feature depth d")
    parser.add_argument("--num-blocks", type=int, default=4, help="MRLFN mRLFB count N")
    add_boolean_optional_argument(parser, "--model-bias", default=True)
    parser.add_argument(
        "--space-to-depth-factor",
        type=int,
        default=1,
        help="MRLFN mosaic-domain S2D/D2S factor; paper Model A/B use k=4",
    )
    parser.add_argument("--gradient-loss-weight", type=float, default=0.0)
    parser.add_argument("--wavelet-hf-loss-weight", type=float, default=0.0)
    parser.add_argument("--wavelet-hf-basis", choices=WaveletLLLoss.SUPPORTED, default="haar")
    parser.add_argument("--wavelet-hf-levels", type=int, default=2)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--distill-loss-weight", type=float, default=0.0)
    parser.add_argument("--wavelet-loss-weight", type=float, default=0.0, help="Auxiliary LL loss weight; zero disables it")
    parser.add_argument("--wavelet-basis", choices=WaveletLLLoss.SUPPORTED, default="haar")
    parser.add_argument("--wavelet-levels", type=int, default=3)
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

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional exact total optimization-step budget")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Defaults to all clean patches in the manifest")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument(
        "--crops-per-image",
        type=int,
        default=8,
        help="Dynamic crops per full RAW; use 1 only for the legacy packed source",
    )
    parser.add_argument("--ratios", type=int, nargs="+", default=[100, 250, 300])
    parser.add_argument("--clean-cache-size", type=int, default=32)
    parser.add_argument("--dark-cache-size", type=int, default=2)

    parser.add_argument("--synthesis", choices=["ratio_aware", "paper_literal"], default="ratio_aware")
    parser.add_argument("--k-scale", type=float, default=0.1, help="K=ISO/100*k_scale; paper setting is 0.1")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=0.0, help="Final LR of cosine annealing")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--amp", action="store_true", help="Off by default for the initial faithful reproduction")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--validate-every", type=int, default=10, help="Epoch interval for real-pair validation")
    parser.add_argument("--validate-steps", type=int, default=-1, help="Real validation pairs: -1 for all, 0 disables, positive randomly samples pairs using seed")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=20)
    # Apply YAML values after arguments are declared. Otherwise each
    # add_argument(default=...) call overwrites the value loaded from YAML.
    # Explicit command-line options still take precedence during parse_args().
    parser.set_defaults(**config)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: SIDSyntheticTrainDataset, args: argparse.Namespace, shuffle: bool, workers: int) -> DataLoader:
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = (args.seed + worker_id + 1) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    kwargs: Dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "worker_init_fn": worker_init_fn if workers else None,
    }
    if workers:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate: float = 0.0,
) -> LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = min(max(0, int(warmup_steps)), total_steps - 1)
    initial_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if not initial_learning_rates or any(rate <= 0 for rate in initial_learning_rates):
        raise ValueError("Optimizer learning rates must be positive")
    if min_learning_rate < 0 or any(min_learning_rate > rate for rate in initial_learning_rates):
        raise ValueError("min-learning-rate must be between zero and every initial learning rate")
    min_multiplier = float(min_learning_rate) / initial_learning_rates[0]
    if any(not math.isclose(rate, initial_learning_rates[0]) for rate in initial_learning_rates):
        raise ValueError("This scheduler requires the same initial learning rate for all parameter groups")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps + 1:
            return 1.0
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_multiplier + (1.0 - min_multiplier) * cosine

    return LambdaLR(optimizer, multiplier)


def to_device(batch: Dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clean = batch["clean"].to(device, non_blocking=True)
    dark = batch["dark_residual_dn"].to(device, non_blocking=True)
    iso = batch["iso"].to(device, non_blocking=True)
    ratio = batch["ratio"].to(device, non_blocking=True)
    return clean, dark, iso, ratio


def build_model(args: argparse.Namespace) -> tuple[torch.nn.Module, Dict[str, object]]:
    """Construct the selected 4-channel packed-RAW denoiser."""
    if args.model == "learning_dwt_repncb":
        from models.learning_dwt_repncb import LearningDWTRepNCB, refiner_kwargs
        config = learning_dwt_kwargs(args)
        refine_config = refiner_kwargs(args)
        return LearningDWTRepNCB(config, refine_config=refine_config), {"model": args.model, **config, "refinement": refine_config}
    if args.model == "learning_dwt":
        config = learning_dwt_kwargs(args)
        return LearningDWT(**config), {"model": "learning_dwt", **config}
    if args.model == "unet":
        config: Dict[str, object] = {
            "model": "unet",
            "in_channels": 4,
            "out_channels": 4,
            "width": int(args.model_width),
        }
        return UNetSeeInDark(in_nc=4, out_nc=4, nf=args.model_width), config

    if args.model == "mrlfn":
        config = {
            "model": "mrlfn",
            "in_channels": 4,
            "out_channels": 4,
            "feature_channels": int(args.feature_channels),
            "num_blocks": int(args.num_blocks),
            "bias": bool(args.model_bias),
            "deploy": False,
            "space_to_depth_factor": int(args.space_to_depth_factor),
        }
        return (
            MRLFN(
                in_channels=4,
                out_channels=4,
                feature_channels=args.feature_channels,
                num_blocks=args.num_blocks,
                bias=args.model_bias,
                deploy=False,
                space_to_depth_factor=args.space_to_depth_factor,
            ),
            config,
        )

    config = {
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
        config,
    )


@torch.inference_mode()
def real_validate(model, loader, args, device):
    was_training = model.training
    inference_model, graph_state = prepare_model_for_inference(model)
    scores, ssims = [], []
    try:
        for batch in loader:
            target = tensor_dim5to4(batch["hr"]).to(device)
            noisy = tensor_dim5to4(batch["lr"]).to(device)
            prediction = ELDIlluminanceCorrect().correct(inference_model(noisy), target)
            metric = PMN_metric(prediction.clamp(0, 1), target.clamp(0, 1))
            scores.append(float(metric["psnr"]))
            ssims.append(float(metric["ssim"]))
            if len(scores) == 1 or len(scores) % 20 == 0:
                print(f"Real validation: {len(scores)} pairs, mean PSNR={np.mean(scores):.4f}", flush=True)
    finally:
        model.train(was_training)
    return {"real_psnr": float(np.mean(scores)), "real_ssim": float(np.mean(ssims)),
            "validation_pairs": len(scores), "validation_graph": graph_state}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    best_psnr: float = -math.inf,
    best_epoch: int = 0,
    best_train_l1: float = math.inf,
    best_train_l1_epoch: int = 0,
) -> None:
    fused_state = deploy_state_dict(model)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "best_epoch": best_epoch,
        "best_train_l1": best_train_l1,
        "best_train_l1_epoch": best_train_l1_epoch,
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


def prune_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    if keep < 0:
        return
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pth"), key=lambda path: path.stat().st_mtime)
    for stale in checkpoints[: max(0, len(checkpoints) - keep)]:
        stale.unlink()


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    initialize_haar: bool = False,
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    if initialize_haar:
        weights = state["model"] if "model" in state else state
        weights = dict(weights)
        keys = {"wavelet.transform.analysis", "wavelet.transform.synthesis"}
        # Only the two NEW Haar tensors may be absent; all CNN keys remain strict.
        if not keys.intersection(weights):
            weights.update({k: v for k, v in model.state_dict().items() if k in keys})
        model.load_state_dict(weights, strict=True)
        return int(state.get("epoch", 0)), int(state.get("global_step", 0))
    if "model" in state:
        model.load_state_dict(state["model"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state.get("scaler", {}))
        return int(state.get("epoch", 0)), int(state.get("global_step", 0))
    model.load_state_dict(state, strict=True)
    return 0, 0


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.crops_per_image <= 0:
        raise ValueError("epochs, batch-size, and crops-per-image must be positive")
    if args.validate_every <= 0 or args.save_every <= 0 or args.validate_steps < -1:
        raise ValueError("validate-every/save-every must be positive; validate-steps must be >= -1")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.feature_channels <= 0 or args.num_blocks <= 0:
        raise ValueError("feature-channels and num-blocks must be positive")
    if args.adam_epsilon <= 0:
        raise ValueError("adam-epsilon must be positive")
    if args.model == "mrlfn" and args.space_to_depth_factor != 1:
        if args.space_to_depth_factor < 2 or args.space_to_depth_factor % 2:
            raise ValueError("space-to-depth-factor must be 1 or an even integer >= 2")
        if (2 * args.patch_size) % args.space_to_depth_factor:
            raise ValueError("Twice the packed patch size must be divisible by space-to-depth-factor")
    if args.model == "mrlfn" and list(args.chromatic_channel_order) != [0, 1, 3, 2]:
        print("Warning: current packed RAW order is [R,G1,G2,B]; the aligned semantic order is [0,1,3,2]")
    if not math.isfinite(args.wavelet_loss_weight) or args.wavelet_loss_weight < 0:
        raise ValueError("wavelet-loss-weight must be finite and non-negative")
    if args.wavelet_levels < 1 or args.wavelet_basis not in WaveletLLLoss.SUPPORTED:
        raise ValueError("Invalid wavelet-levels or wavelet-basis")
    device = torch.device(args.device)
    wavelet_criterion = (WaveletLLLoss(args.wavelet_basis, args.wavelet_levels).to(device)
                         if args.wavelet_loss_weight > 0 else None)
    if wavelet_criterion is not None:
        with torch.no_grad():
            probe = torch.zeros(1, 1, args.patch_size, args.patch_size, device=device)
            wavelet_criterion(probe, probe)
        del probe
    for weight in (args.gradient_loss_weight, args.wavelet_hf_loss_weight):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Detail loss weights must be finite and non-negative")
    if args.wavelet_hf_basis not in WaveletLLLoss.SUPPORTED or args.wavelet_hf_levels < 1:
        raise ValueError("Invalid wavelet-hf-basis or wavelet-hf-levels")
    gradient_criterion = GradientLoss().to(device) if args.gradient_loss_weight > 0 else None
    hf_criterion = (WaveletHFLoss(args.wavelet_hf_basis, args.wavelet_hf_levels).to(device)
                    if args.wavelet_hf_loss_weight > 0 else None)
    if hf_criterion is not None:
        with torch.no_grad():
            probe = torch.zeros(1, 1, args.patch_size, args.patch_size, device=device)
            hf_criterion(probe, probe)
        del probe
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    manifest_path = Path(args.manifest)
    if args.rebuild_manifest or not manifest_path.is_file():
        if args.clean_source == "raw":
            manifest = build_sid_raw_manifest(args.sid_long_dir, manifest_path, scene_prefixes=("0",))
            print(f"Built raw training manifest with {manifest['summary']['records']} scenes")
        else:
            manifest = build_sid_patch_manifest(args.patch_dir, args.pair_list, manifest_path, args.sid_long_dir)
            print(f"Built packed training manifest with {manifest['summary']['records']} patches")

    train_dataset = SIDSyntheticTrainDataset(
        manifest_path,
        args.dark_root,
        args.pmn_resource_dir,
        patch_size=args.patch_size,
        crops_per_image=args.crops_per_image,
        ratios=tuple(args.ratios),
        cache_size=args.clean_cache_size,
        dark_cache_size=args.dark_cache_size,
        augment=True,
        seed=args.seed,
        clean_source=args.clean_source,
        allowed_scene_prefixes=("0",),
    )
    train_loader = make_loader(train_dataset, args, shuffle=True, workers=args.num_workers)
    val_loader = None
    if args.validate_steps:
        val_dataset = SIDRealValidationDataset(
            args.val_pair_list or Path(args.pair_list).with_name("Sony_val_list.txt"),
            args.sid_long_dir, args.pmn_resource_dir, ratios=args.ratios,
            max_items=args.validate_steps if args.validate_steps > 0 else None,
            seed=args.seed,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"Real SID validation: {len(val_dataset)} full-resolution pairs")
    available_steps = len(train_loader)
    steps_per_epoch = args.steps_per_epoch or available_steps
    if steps_per_epoch > available_steps:
        raise ValueError(f"steps-per-epoch ({steps_per_epoch}) exceeds loader length ({available_steps})")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model, architecture_config = build_model(args)
    if args.dwt_trainable_haar:
        if args.model != "learning_dwt_repncb" or not (args.init_checkpoint or args.resume):
            raise ValueError("Haar-kernel fine-tuning requires learning_dwt_repncb and init_checkpoint or resume")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in {"wavelet.transform.analysis", "wavelet.transform.synthesis"})
        print(f"Fine-tuning only Haar analysis/synthesis: {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    model = model.to(device)
    if not math.isfinite(args.distill_loss_weight) or args.distill_loss_weight < 0:
        raise ValueError("distill-loss-weight must be finite and non-negative")
    teacher = None
    teacher_meta = None
    if args.distill_loss_weight > 0:
        if not args.teacher_checkpoint:
            raise ValueError("Distillation requires teacher-checkpoint")
        from utils.model_factory import build_denoiser_from_checkpoint
        teacher, teacher_meta = build_denoiser_from_checkpoint(args.teacher_checkpoint, device)
        teacher.requires_grad_(False)
        teacher.eval()
    profile_model, profile_graph = prepare_model_for_inference(model)
    model_info = calculate_model_info(profile_model, (1, 4, args.patch_size, args.patch_size), device)
    model_info["architecture_config"] = {**architecture_config, "deploy": profile_graph == "deploy"}
    model_info["graph_state"] = profile_graph
    if args.model in ("learning_dwt", "learning_dwt_repncb"):
        model_info["profiling_note"] = "CNN MACs only; functional DWT/IWT convolutions and atlas/shrink operations excluded"
    model_info["loss_config"] = (
        {
            "name": "RawReconstructionChromaticLoss",
            "raw_weight": args.raw_loss_weight,
            "chromatic_weight": args.chromatic_loss_weight,
            "channel_order_for_R_G1_B_G2": list(args.chromatic_channel_order),
        }
        if args.model in ("mrlfn", "learning_dwt", "learning_dwt_repncb")
        else {"name": "L1Loss"}
    )
    model_info["loss_config"]["wavelet_auxiliary"] = {
        "weight": args.wavelet_loss_weight, "basis": args.wavelet_basis,
        "levels": args.wavelet_levels, "aggregation": "mean_of_all_levels",
        "boundary": "symmetric", "coefficient_normalization": "none",
    }
    model_info["loss_config"]["gradient_auxiliary"] = {
        "weight": args.gradient_loss_weight, "operator": "sobel/8",
        "padding": "replicate", "aggregation": "mean_xy_channels_pixels",
    }
    model_info["loss_config"]["wavelet_hf_auxiliary"] = {
        "weight": args.wavelet_hf_loss_weight, "basis": args.wavelet_hf_basis,
        "levels": args.wavelet_hf_levels, "boundary": "symmetric",
        "aggregation": "equal_mean_levels_and_three_bands", "coefficient_normalization": "none",
    }
    model_info["loss_config"]["distillation"] = {
        "weight": args.distill_loss_weight, "teacher_checkpoint": args.teacher_checkpoint,
        "teacher": teacher_meta, "loss": "output_l1_unclipped",
    }
    (output_dir / "model_info.json").write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(format_model_info(model_info))
    del profile_model
    optimizer = Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    total_steps = int(args.max_steps) if args.max_steps is not None else args.epochs * steps_per_epoch
    scheduler = build_scheduler(
        optimizer,
        total_steps,
        args.warmup_epochs * steps_per_epoch,
        args.min_learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    start_epoch, global_step = 1, 0
    best_psnr, best_epoch = -math.inf, 0
    best_train_l1, best_train_l1_epoch = math.inf, 0
    if args.resume:
        epoch, global_step = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        resume_state = torch.load(args.resume, map_location="cpu")
        best_psnr = float(resume_state.get("best_psnr", -math.inf))
        best_epoch = int(resume_state.get("best_epoch", 0))
        best_train_l1 = float(resume_state.get("best_train_l1", math.inf))
        best_train_l1_epoch = int(resume_state.get("best_train_l1_epoch", 0))
        del resume_state
        start_epoch = epoch + 1
        print(f"Resumed {args.resume} at epoch {start_epoch}, global step {global_step}")
    elif args.init_checkpoint:
        load_checkpoint(args.init_checkpoint, model, initialize_haar=args.dwt_trainable_haar)
        print(f"Initialised model weights from {args.init_checkpoint}")

    criterion = (
        RawReconstructionChromaticLoss(
            raw_weight=args.raw_loss_weight,
            chromatic_weight=args.chromatic_loss_weight,
            channel_order=tuple(args.chromatic_channel_order),
        )
        if args.model in ("mrlfn", "learning_dwt", "learning_dwt_repncb")
        else None
    )

    log_path = output_dir / "metrics.jsonl"
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
    print(
        f"Training {type(model).__name__} on {len(train_dataset)} samples "
        f"({steps_per_epoch} steps/epoch) on {device}; "
        f"synthesis={args.synthesis}, ratios={args.ratios}, AMP={args.amp}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss, epoch_raw_l1, epoch_chromatic_l1, epoch_samples = 0.0, 0.0, 0.0, 0
        epoch_distill_l1 = 0.0
        epoch_wavelet_l1 = 0.0
        epoch_gradient_l1, epoch_hf_l1 = 0.0, 0.0
        epoch_start = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            if step > steps_per_epoch:
                break
            if global_step >= total_steps:
                break
            clean, dark, iso, ratio = to_device(batch, device)
            noisy, target = synthesize_sid_noise(clean, dark, iso, ratio, k_scale=args.k_scale, mode=args.synthesis)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(args.amp)) if device.type == "cuda" else autocast():
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
                wavelet_l1 = (wavelet_criterion(prediction, target)
                              if wavelet_criterion is not None else loss.new_zeros(()))
                if wavelet_criterion is not None:
                    loss = loss + args.wavelet_loss_weight * wavelet_l1
                gradient_l1 = (gradient_criterion(prediction, target)
                               if gradient_criterion is not None else loss.new_zeros(()))
                hf_l1 = (hf_criterion(prediction, target)
                         if hf_criterion is not None else loss.new_zeros(()))
                if gradient_criterion is not None:
                    loss = loss + args.gradient_loss_weight * gradient_l1
                if hf_criterion is not None:
                    loss = loss + args.wavelet_hf_loss_weight * hf_l1
                distill_l1 = loss.new_zeros(())
                if teacher is not None:
                    with torch.no_grad():
                        teacher_prediction = teacher(noisy)
                    if teacher_prediction.shape != prediction.shape:
                        raise ValueError("Teacher/student output shape mismatch")
                    distill_l1 = F.l1_loss(prediction.float(), teacher_prediction.float())
                    loss = loss + args.distill_loss_weight * distill_l1
            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            epoch_gradient_l1 += float(gradient_l1.detach()) * clean.shape[0]
            epoch_hf_l1 += float(hf_l1.detach()) * clean.shape[0]
            epoch_wavelet_l1 += float(wavelet_l1.detach()) * clean.shape[0]
            epoch_distill_l1 += float(distill_l1.detach()) * clean.shape[0]
            epoch_loss += float(loss.detach()) * clean.shape[0]
            epoch_raw_l1 += float(raw_l1.detach()) * clean.shape[0]
            epoch_chromatic_l1 += float(chromatic_l1.detach()) * clean.shape[0]
            epoch_samples += clean.shape[0]
            if step == 1 or step % args.log_every == 0 or step == steps_per_epoch:
                print(
                    f"epoch {epoch:04d} step {step:04d}/{steps_per_epoch} "
                    f"distill_l1={float(distill_l1.detach()):.6f} "
                    f"loss={float(loss.detach()):.6f} raw_l1={float(raw_l1.detach()):.6f} "
                    f"gradient_l1={float(gradient_l1.detach()):.6f} wavelet_hf_l1={float(hf_l1.detach()):.6f} "
                    f"wavelet_ll_l1={float(wavelet_l1.detach()):.6f} "
                    f"chromatic_l1={float(chromatic_l1.detach()):.6f} lr={scheduler.get_last_lr()[0]:.3e} "
                    f"iso={int(iso[0])} ratio={int(ratio[0])}"
                )

        metrics: Dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_gradient_l1": epoch_gradient_l1 / max(1, epoch_samples),
            "train_gradient_weighted": args.gradient_loss_weight * epoch_gradient_l1 / max(1, epoch_samples),
            "train_wavelet_hf_l1": epoch_hf_l1 / max(1, epoch_samples),
            "train_wavelet_hf_weighted": args.wavelet_hf_loss_weight * epoch_hf_l1 / max(1, epoch_samples),
            "train_wavelet_ll_l1": epoch_wavelet_l1 / max(1, epoch_samples),
            "train_wavelet_weighted": args.wavelet_loss_weight * epoch_wavelet_l1 / max(1, epoch_samples),
            "train_distill_l1": epoch_distill_l1 / max(1, epoch_samples),
            "train_distill_weighted": args.distill_loss_weight * epoch_distill_l1 / max(1, epoch_samples),
            "train_loss": epoch_loss / max(1, epoch_samples),
            "train_l1": epoch_raw_l1 / max(1, epoch_samples),
            "train_chromatic_l1": epoch_chromatic_l1 / max(1, epoch_samples),
            "loss_name": ("raw_reconstruction_chromatic" if criterion is not None else "l1")
                         + ("+wavelet_ll" if wavelet_criterion is not None else "")
                         + ("+gradient" if gradient_criterion is not None else "")
                         + ("+wavelet_hf" if hf_criterion is not None else ""),
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - epoch_start,
            # 模型信息(model_info)仅在训练前保存到 model_info.json 并打印一次，
            # 此处不再重复写入每个 epoch 的 metrics，避免日志与 metrics.jsonl 冗余。
        }
        improved_train_l1 = (epoch_samples > 0 and math.isfinite(metrics["train_l1"])
                             and metrics["train_l1"] < best_train_l1)
        if improved_train_l1:
            best_train_l1, best_train_l1_epoch = metrics["train_l1"], epoch
        if args.validate_steps and (epoch % args.validate_every == 0 or epoch == args.epochs or global_step >= total_steps):
            metrics.update(real_validate(model, val_loader, args, device))
            if metrics["real_psnr"] > best_psnr:
                best_psnr, best_epoch = metrics["real_psnr"], epoch
                save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, scheduler, scaler,
                                epoch, global_step, args, best_psnr, best_epoch, best_train_l1, best_train_l1_epoch)
                print(f"Saved best.pth: PSNR={best_psnr:.4f}, epoch={best_epoch}")
        if improved_train_l1:
            save_checkpoint(checkpoint_dir / "best_train_l1.pth", model, optimizer, scheduler, scaler,
                            epoch, global_step, args, best_psnr, best_epoch, best_train_l1, best_train_l1_epoch)
            print(f"Saved best_train_l1.pth: train_l1={best_train_l1:.6f}, epoch={best_train_l1_epoch}")
        metrics.update(best_psnr=best_psnr if best_epoch else None, best_epoch=best_epoch,
                       best_train_l1=best_train_l1 if best_train_l1_epoch else None,
                       best_train_l1_epoch=best_train_l1_epoch)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print("epoch summary", json.dumps(metrics, ensure_ascii=False))

        save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, scheduler, scaler, epoch, global_step, args, best_psnr, best_epoch, best_train_l1, best_train_l1_epoch)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pth", model, optimizer, scheduler, scaler, epoch, global_step, args, best_psnr, best_epoch, best_train_l1, best_train_l1_epoch)
            prune_checkpoints(checkpoint_dir, args.keep_checkpoints)
        if global_step >= total_steps:
            print(f"Reached total optimization-step budget: {total_steps}")
            break


if __name__ == "__main__":
    main()
