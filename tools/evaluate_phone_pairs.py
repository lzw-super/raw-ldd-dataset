#!/usr/bin/env python3
"""Evaluate registered phone noisy-clean DNG pairs in canonical linear RAW."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.phone_dng import read_phone_dng_packed
from utils.phone_inference import tiled_phone_inference
from utils.phone_model import load_phone_checkpoint
from noise.phone_dark_shading import PhoneContinuousDarkShading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pair-manifest", required=True, help="JSONL records with noisy_path, clean_path and registration metadata")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--calibration-root", default="data/MEY_AN00/calibration")
    parser.add_argument(
        "--dark-shading-model",
        choices=["continuous_iso_fit", "per_condition_mean"],
        default=None,
        help="Defaults to checkpoint setting; legacy checkpoints use per_condition_mean",
    )
    parser.add_argument("--no-dark-shading", action="store_true", help="Ablation only: do not subtract calibrated short-exposure DS")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 evaluates every pair")
    parser.add_argument(
        "--allow-unverified-registration",
        action="store_true",
        help="Permit records without registration.status='verified'; unsafe for a reportable PSNR",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"Pair manifest is empty: {path}")
    return records


def resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def load_valid_mask(record: dict[str, Any], shape: tuple[int, int, int], manifest_path: Path) -> np.ndarray:
    """Return [4,H,W] valid pixels: finite, unsaturated target plus optional mask."""
    mask_value = record.get("valid_mask_path")
    if mask_value is None:
        return np.ones(shape, dtype=bool)
    mask = np.load(resolve_path(str(mask_value), manifest_path), allow_pickle=False)
    if mask.shape == shape[1:]:
        return np.broadcast_to(mask.astype(bool, copy=False), shape)
    if mask.shape == shape:
        return mask.astype(bool, copy=False)
    raise ValueError(f"{record.get('pair_id', '<unknown>')}: valid mask {mask.shape} must be {shape[1:]} or {shape}")


def load_dark_shading(calibration_root: Path, condition_key: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    directory = calibration_root / condition_key
    ds_path, metadata_path = directory / "dark_shading_dn.npy", directory / "metadata.json"
    if not ds_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing dark-shading artifact for {condition_key}; calibrate it first or use --no-dark-shading"
        )
    artifact_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if artifact_metadata.get("condition_key") != condition_key:
        raise ValueError(f"{metadata_path}: condition key mismatch")
    if artifact_metadata.get("units") != "linearized_DN" or not artifact_metadata.get("black_subtracted"):
        raise ValueError(f"{metadata_path}: unsupported DS domain")
    dark_shading = np.load(ds_path, mmap_mode="r")
    if tuple(dark_shading.shape) != expected_shape:
        raise ValueError(f"{ds_path}: shape {dark_shading.shape} != noisy packed shape {expected_shape}")
    return np.asarray(dark_shading, dtype=np.float32)


def psnr_from_sse(squared_error: float, count: int) -> float:
    return -10.0 * math.log10(max(squared_error / max(1, count), 1e-12))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    manifest_path = Path(args.pair_manifest).resolve()
    records = read_jsonl(manifest_path)
    if args.max_pairs > 0:
        records = records[: args.max_pairs]
    device = torch.device(args.device)
    model, checkpoint_args, model_name = load_phone_checkpoint(args.checkpoint, device)
    calibration_root = Path(args.calibration_root)
    dark_shading_model = args.dark_shading_model or str(
        checkpoint_args.get("dark_shading_model", "per_condition_mean")
    )
    continuous_ds = (
        PhoneContinuousDarkShading(calibration_root)
        if dark_shading_model == "continuous_iso_fit" and not args.no_dark_shading
        else None
    )

    total_absolute_error, total_squared_error, total_pixels = 0.0, 0.0, 0
    per_pair: list[dict[str, Any]] = []
    condition_stats: dict[str, dict[str, float | int]] = {}
    for index, record in enumerate(records, start=1):
        required = {"pair_id", "noisy_path", "clean_path"}
        missing = required - set(record)
        if missing:
            raise ValueError(f"Pair record {index} is missing {sorted(missing)}")
        registration = record.get("registration")
        is_verified = isinstance(registration, dict) and registration.get("status") == "verified"
        if not is_verified and not args.allow_unverified_registration:
            raise ValueError(
                f"{record['pair_id']}: registration.status='verified' is required for reportable PSNR; "
                "use --allow-unverified-registration only for diagnostics"
            )
        noisy_path = resolve_path(str(record["noisy_path"]), manifest_path)
        clean_path = resolve_path(str(record["clean_path"]), manifest_path)
        noisy_packed, noisy_metadata = read_phone_dng_packed(noisy_path)
        clean_packed, clean_metadata = read_phone_dng_packed(clean_path)
        if noisy_packed.shape != clean_packed.shape:
            raise ValueError(f"{record['pair_id']}: packed RAW shapes differ: {noisy_packed.shape} vs {clean_packed.shape}")
        if noisy_metadata.canonical_position_indices != clean_metadata.canonical_position_indices:
            raise ValueError(f"{record['pair_id']}: noisy and clean CFA packing differs")
        ratio = float(record.get("exposure_ratio", clean_metadata.exposure_s / noisy_metadata.exposure_s))
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"{record['pair_id']}: exposure_ratio must be finite and positive")
        noisy_black = np.asarray(noisy_metadata.black_level_canonical, dtype=np.float32).reshape(4, 1, 1)
        clean_black = np.asarray(clean_metadata.black_level_canonical, dtype=np.float32).reshape(4, 1, 1)
        if args.no_dark_shading:
            dark_shading = np.zeros_like(noisy_packed, dtype=np.float32)
        elif continuous_ds is not None:
            dark_shading = continuous_ds.full(noisy_metadata.iso_exif, noisy_packed.shape)
        else:
            dark_shading = load_dark_shading(
                calibration_root, noisy_metadata.condition_key, noisy_packed.shape
            )
        model_input = np.minimum(
            (noisy_packed.astype(np.float32, copy=False) - noisy_black - dark_shading)
            / float(noisy_metadata.dynamic_range)
            * ratio,
            1.0,
        ).astype(np.float32, copy=False)
        target = (clean_packed.astype(np.float32, copy=False) - clean_black) / float(clean_metadata.dynamic_range)
        target = np.clip(target, 0.0, 1.0)
        prediction = torch.clamp(
            tiled_phone_inference(
                model, torch.from_numpy(model_input).unsqueeze(0).to(device), tile_size=args.tile_size, tile_overlap=args.tile_overlap
            ),
            0.0,
            1.0,
        )[0].cpu().numpy()
        valid = np.isfinite(target) & (target < 1.0) & load_valid_mask(record, target.shape, manifest_path)
        valid_pixels = int(valid.sum())
        if valid_pixels == 0:
            raise ValueError(f"{record['pair_id']}: no valid, non-saturated pixels remain")
        error = prediction[valid] - target[valid]
        absolute_error, squared_error = float(np.abs(error).sum()), float(np.square(error).sum())
        total_absolute_error += absolute_error
        total_squared_error += squared_error
        total_pixels += valid_pixels
        condition_key = f"iso-{noisy_metadata.iso_exif}_exp-{noisy_metadata.exposure_s:.9g}s"
        stats = condition_stats.setdefault(
            condition_key,
            {"pairs": 0, "absolute_error": 0.0, "squared_error": 0.0, "pixels": 0},
        )
        stats["pairs"] = int(stats["pairs"]) + 1
        stats["absolute_error"] = float(stats["absolute_error"]) + absolute_error
        stats["squared_error"] = float(stats["squared_error"]) + squared_error
        stats["pixels"] = int(stats["pixels"]) + valid_pixels
        pair_result = {
            "pair_id": record["pair_id"],
            "scene_id": record.get("scene_id"),
            "split": record.get("split"),
            "iso_exif": {"noisy": noisy_metadata.iso_exif, "clean": clean_metadata.iso_exif},
            "exposure_s": {"noisy": noisy_metadata.exposure_s, "clean": clean_metadata.exposure_s},
            "exposure_ratio": ratio,
            "dark_shading_subtracted": not args.no_dark_shading,
            "dark_shading_model": dark_shading_model if not args.no_dark_shading else "disabled",
            "dark_shading_condition_key": (
                noisy_metadata.condition_key
                if not args.no_dark_shading and dark_shading_model == "per_condition_mean"
                else None
            ),
            "registration_verified": is_verified,
            "valid_pixels": valid_pixels,
            "l1": absolute_error / valid_pixels,
            "psnr": psnr_from_sse(squared_error, valid_pixels),
        }
        per_pair.append(pair_result)
        print(f"[{index}/{len(records)}] {record['pair_id']}: PSNR={pair_result['psnr']:.3f} dB, valid={valid_pixels}")

    result = {
        "metric_protocol": "registered_real_noisy_clean_linear_canonical_raw",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pair_manifest": str(manifest_path),
        "calibration_root": str(calibration_root.resolve()),
        "dark_shading_subtracted": not args.no_dark_shading,
        "dark_shading_model": dark_shading_model if not args.no_dark_shading else "disabled",
        "model": model_name,
        "pairs": len(per_pair),
        "global_l1": total_absolute_error / max(1, total_pixels),
        "global_psnr": psnr_from_sse(total_squared_error, total_pixels),
        "macro_pair_psnr": float(np.mean([item["psnr"] for item in per_pair])),
        "total_valid_pixels": total_pixels,
        "by_noisy_condition": {
            condition: {
                "pairs": int(stats["pairs"]),
                "valid_pixels": int(stats["pixels"]),
                "l1": float(stats["absolute_error"]) / max(1, int(stats["pixels"])),
                "psnr": psnr_from_sse(float(stats["squared_error"]), int(stats["pixels"])),
            }
            for condition, stats in sorted(condition_stats.items())
        },
        "per_pair": per_pair,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("pairs", "global_psnr", "macro_pair_psnr")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
