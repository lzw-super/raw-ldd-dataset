#!/usr/bin/env python3
"""Report reproducible held-out *synthetic* RAW PSNR for a phone checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.phone_synthetic_train import PhoneSyntheticTrainDataset
from utils.phone_evaluation import evaluate_phone_synthetic
from utils.phone_model import load_phone_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by train_phone.py")
    parser.add_argument("--clean-manifest", default="data/MEY_AN00/manifests/clean_synthetic_val.jsonl")
    parser.add_argument("--dark-manifest", default="data/MEY_AN00/manifests/dark_val.jsonl")
    parser.add_argument("--calibration-root", default="data/MEY_AN00/calibration")
    parser.add_argument(
        "--dark-shading-model",
        choices=["continuous_iso_fit", "per_condition_mean"],
        default=None,
        help="Defaults to checkpoint setting; legacy checkpoints use per_condition_mean",
    )
    parser.add_argument("--ratios", type=float, nargs="+", default=[100.0, 250.0, 300.0])
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--crops-per-image", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates every deterministic validation crop")
    parser.add_argument("--max-target-saturation-fraction", type=float, default=0.01)
    parser.add_argument("--max-crop-attempts", type=int, default=32)
    parser.add_argument(
        "--dark-exposure-policy",
        choices=["strict_match", "approximate_reuse_1_30s"],
        default="approximate_reuse_1_30s",
        help="Required for the default ratio 100/250/300 evaluation; ratio 300 itself remains exact",
    )
    parser.add_argument("--synthesis", choices=["hybrid", "analytic"], default=None, help="Defaults to the checkpoint setting")
    parser.add_argument("--seed", type=int, default=10_001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--result-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    model, checkpoint_args, model_name = load_phone_checkpoint(args.checkpoint, device)
    synthesis = args.synthesis or str(checkpoint_args.get("synthesis", "hybrid"))
    dark_shading_model = args.dark_shading_model or str(
        checkpoint_args.get("dark_shading_model", "per_condition_mean")
    )
    dataset = PhoneSyntheticTrainDataset(
        args.clean_manifest,
        args.dark_manifest,
        args.calibration_root,
        patch_size=args.patch_size,
        crops_per_image=args.crops_per_image,
        ratios=tuple(args.ratios),
        cache_size=2,
        dark_cache_size=1,
        max_target_saturation_fraction=args.max_target_saturation_fraction,
        max_crop_attempts=args.max_crop_attempts,
        seed=args.seed,
        dark_exposure_policy=args.dark_exposure_policy,
        dark_shading_model=dark_shading_model,
    )
    result = {
        "metric_protocol": "synthetic_heldout_pseudoclean_not_real_pair_psnr",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model": model_name,
        "clean_manifest": str(Path(args.clean_manifest).resolve()),
        "dark_manifest": str(Path(args.dark_manifest).resolve()),
        "calibration_root": str(Path(args.calibration_root).resolve()),
        "patch_size": args.patch_size,
        "crops_per_image": args.crops_per_image,
        "max_samples": args.max_samples,
        "synthesis": synthesis,
        "dark_exposure_policy": args.dark_exposure_policy,
        "dark_shading_model": dark_shading_model,
        **evaluate_phone_synthetic(
            model,
            dataset,
            device=device,
            ratios=args.ratios,
            max_samples=args.max_samples,
            seed=args.seed,
            synthesis=synthesis,
        ),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
