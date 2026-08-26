#!/usr/bin/env python3
"""Evaluate a checkpoint on held-out full-RAW synthetic SID patches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.sid_synthetic_train import SIDSyntheticTrainDataset
from noise.sid_noise_synthesis import synthesize_sid_noise
from utils.model_factory import build_denoiser_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="infos/SID_validation_clean_raw.json")
    parser.add_argument("--dark-root", default="biasframe_et_1_30")
    parser.add_argument("--pmn-resource-dir", default="resources/SonyA7S2")
    parser.add_argument("--ratios", type=int, nargs="+", default=[100, 250, 300])
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--result-json", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    synthesis = checkpoint_args.get("synthesis", "ratio_aware")
    k_scale = float(checkpoint_args.get("k_scale", 0.1))

    dataset = SIDSyntheticTrainDataset(
        args.manifest,
        args.dark_root,
        args.pmn_resource_dir,
        patch_size=512,
        crops_per_image=1,
        ratios=tuple(args.ratios),
        cache_size=2,
        dark_cache_size=1,
        augment=False,
        seed=args.seed,
        clean_source="raw",
        allowed_scene_prefixes=("2",),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model, model_meta = build_denoiser_from_checkpoint(args.checkpoint, device=device)
    model_name = model_meta["model"]

    accumulators = {ratio: {"l1": 0.0, "mse": 0.0, "count": 0} for ratio in args.ratios}
    with torch.inference_mode():
        for scene_index, batch in enumerate(loader):
            if scene_index >= args.max_scenes:
                break
            clean = batch["clean"].to(device)
            dark = batch["dark_residual_dn"].to(device)
            iso = batch["iso"].to(device)
            for ratio in args.ratios:
                ratio_tensor = torch.full_like(iso, float(ratio))
                noisy, target = synthesize_sid_noise(
                    clean, dark, iso, ratio_tensor, k_scale=k_scale, mode=synthesis
                )
                prediction = torch.clamp(model(noisy), 0.0, 1.0)
                stats = accumulators[ratio]
                stats["l1"] += float(F.l1_loss(prediction, target))
                stats["mse"] += float(F.mse_loss(prediction, target))
                stats["count"] += 1

    result = {
        "checkpoint": args.checkpoint,
        "model": model_name,
        "graph_state": model_meta["graph_state"],
        "manifest": args.manifest,
        "held_out_scene_prefix": "2",
        "synthesis": synthesis,
        "k_scale": k_scale,
        "ratios": {},
    }
    all_psnr = []
    for ratio, stats in accumulators.items():
        count = max(1, stats["count"])
        mse = stats["mse"] / count
        psnr = -10.0 * math.log10(max(mse, 1e-12))
        result["ratios"][str(ratio)] = {
            "scenes": stats["count"],
            "l1": stats["l1"] / count,
            "psnr": psnr,
        }
        all_psnr.append(psnr)
    result["mean_psnr"] = sum(all_psnr) / len(all_psnr)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
