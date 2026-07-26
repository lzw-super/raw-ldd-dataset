#!/usr/bin/env python3
"""Train UNetSeeInDark with SID clean RAW plus LLD dark-frame synthesis.

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
from typing import Dict, Iterable

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
from models.ELD_models import UNetSeeInDark
from noise.sid_noise_synthesis import synthesize_sid_noise


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
    parser.set_defaults(**config)
    parser.add_argument("--config", default=preliminary.config)
    parser.add_argument("--patch-dir", default="/home/shared_files/dataset/SID/Sony_train_long_patches")
    parser.add_argument("--pair-list", default="/home/shared_files/dataset/SID/Sony_train_list.txt")
    parser.add_argument("--sid-long-dir", default="/home/shared_files/dataset/SID/Sony/long")
    parser.add_argument("--clean-source", choices=["raw", "packed"], default="raw")
    parser.add_argument("--manifest", default="infos/SID_train_clean_raw.json")
    parser.add_argument("--val-manifest", default="infos/SID_validation_clean_raw.json")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--dark-root", default="biasframe_et_1_30")
    parser.add_argument("--pmn-resource-dir", default="resources/SonyA7S2")
    parser.add_argument("--output-dir", default="experiments/sid_sony_paper_fair")
    parser.add_argument("--resume", default=None, help="Checkpoint produced by this script")
    parser.add_argument("--init-checkpoint", default=None, help="Model-only checkpoint; do not use the official test checkpoint")

    parser.add_argument("--epochs", type=int, default=1000)
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
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--amp", action="store_true", help="Off by default for the initial faithful reproduction")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--validate-every", type=int, default=10, help="Epoch interval for synthetic sanity validation")
    parser.add_argument("--validate-steps", type=int, default=4, help="0 disables synthetic sanity validation")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=20)
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


def to_device(batch: Dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clean = batch["clean"].to(device, non_blocking=True)
    dark = batch["dark_residual_dn"].to(device, non_blocking=True)
    iso = batch["iso"].to(device, non_blocking=True)
    ratio = batch["ratio"].to(device, non_blocking=True)
    return clean, dark, iso, ratio


@torch.no_grad()
def synthetic_validate(
    model: torch.nn.Module, loader: Iterable[Dict[str, object]], args: argparse.Namespace, device: torch.device
) -> Dict[str, float]:
    if args.validate_steps <= 0:
        return {}
    model.eval()
    l1_sum, mse_sum, image_count = 0.0, 0.0, 0
    for step, batch in enumerate(loader):
        if step >= args.validate_steps:
            break
        clean, dark, iso, ratio = to_device(batch, device)
        noisy, target = synthesize_sid_noise(clean, dark, iso, ratio, k_scale=args.k_scale, mode=args.synthesis)
        prediction = torch.clamp(model(noisy), 0.0, 1.0)
        l1_sum += float(F.l1_loss(prediction, target, reduction="mean"))
        mse_sum += float(F.mse_loss(prediction, target, reduction="mean"))
        image_count += 1
    model.train()
    if not image_count:
        return {}
    mse = mse_sum / image_count
    return {"synthetic_l1": l1_sum / image_count, "synthetic_psnr": -10.0 * math.log10(max(mse, 1e-12))}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }
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
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
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
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
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

    val_manifest_path = Path(args.val_manifest)
    if args.clean_source == "raw" and (args.rebuild_manifest or not val_manifest_path.is_file()):
        val_manifest = build_sid_raw_manifest(args.sid_long_dir, val_manifest_path, scene_prefixes=("2",))
        print(f"Built held-out raw validation manifest with {val_manifest['summary']['records']} scenes")
    elif args.clean_source == "packed" and not val_manifest_path.is_file():
        # Packed mode is kept only for reproducing/diagnosing the old fixed
        # patch run. It has no independent synthetic validation split.
        val_manifest_path = manifest_path
        print("Warning: packed clean source reuses its training manifest for synthetic validation")

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
    val_dataset = SIDSyntheticTrainDataset(
        val_manifest_path,
        args.dark_root,
        args.pmn_resource_dir,
        patch_size=args.patch_size,
        crops_per_image=1,
        ratios=tuple(args.ratios),
        cache_size=2,
        dark_cache_size=1,
        augment=False,
        seed=args.seed + 10000,
        clean_source=args.clean_source,
        allowed_scene_prefixes=("2",) if args.clean_source == "raw" else ("0",),
    )
    train_loader = make_loader(train_dataset, args, shuffle=True, workers=args.num_workers)
    val_loader = make_loader(val_dataset, args, shuffle=False, workers=0)
    available_steps = len(train_loader)
    steps_per_epoch = args.steps_per_epoch or available_steps
    if steps_per_epoch > available_steps:
        raise ValueError(f"steps-per-epoch ({steps_per_epoch}) exceeds loader length ({available_steps})")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model = UNetSeeInDark(in_nc=4, out_nc=4, nf=32).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    start_epoch, global_step = 1, 0
    if args.resume:
        epoch, global_step = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = epoch + 1
        print(f"Resumed {args.resume} at epoch {start_epoch}, global step {global_step}")
    elif args.init_checkpoint:
        load_checkpoint(args.init_checkpoint, model)
        print(f"Initialised model weights from {args.init_checkpoint}")

    log_path = output_dir / "metrics.jsonl"
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
    print(
        f"Training {len(train_dataset)} samples ({steps_per_epoch} steps/epoch) on {device}; "
        f"synthesis={args.synthesis}, ratios={args.ratios}, AMP={args.amp}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_l1, epoch_samples = 0.0, 0
        epoch_start = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            if step > steps_per_epoch:
                break
            clean, dark, iso, ratio = to_device(batch, device)
            noisy, target = synthesize_sid_noise(clean, dark, iso, ratio, k_scale=args.k_scale, mode=args.synthesis)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(args.amp)) if device.type == "cuda" else autocast():
                prediction = model(noisy)
                loss = F.l1_loss(torch.clamp(prediction, 0.0, 1.0), target)
            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            epoch_l1 += float(loss.detach()) * clean.shape[0]
            epoch_samples += clean.shape[0]
            if step == 1 or step % args.log_every == 0 or step == steps_per_epoch:
                print(
                    f"epoch {epoch:04d} step {step:04d}/{steps_per_epoch} "
                    f"l1={float(loss.detach()):.6f} lr={scheduler.get_last_lr()[0]:.3e} "
                    f"iso={int(iso[0])} ratio={int(ratio[0])}"
                )

        metrics: Dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_l1": epoch_l1 / max(1, epoch_samples),
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - epoch_start,
        }
        if args.validate_steps and (epoch % args.validate_every == 0 or epoch == args.epochs):
            metrics.update(synthetic_validate(model, val_loader, args, device))
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print("epoch summary", json.dumps(metrics, ensure_ascii=False))

        save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, scheduler, scaler, epoch, global_step, args)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pth", model, optimizer, scheduler, scaler, epoch, global_step, args)
            prune_checkpoints(checkpoint_dir, args.keep_checkpoints)


if __name__ == "__main__":
    main()
