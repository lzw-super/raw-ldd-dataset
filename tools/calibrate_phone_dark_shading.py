#!/usr/bin/env python3
"""Estimate condition-specific dark shading from calibration-only DNG frames.

Artifacts are in black-subtracted, stored-pre-opcode DN and canonical packed
[R,G1,G2,B] coordinates.  Residual frames used by training are deliberately
not part of this estimate, avoiding the finite-sample residual shrinkage that
occurs when the same dark frames define both DS and residuals.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.phone_dng import read_phone_dng_packed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", default="data/MEY_AN00/manifests/dark_calibration.jsonl")
    parser.add_argument("--output-root", default="data/MEY_AN00/calibration")
    parser.add_argument("--clip-sigma", type=float, default=4.0)
    parser.add_argument("--minimum-std-dn", type=float, default=0.25)
    parser.add_argument("--defect-threshold-dn", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def _save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def _frame_dn(record: dict[str, Any]) -> tuple[np.ndarray, Any]:
    packed, metadata = read_phone_dng_packed(record["path"])
    black = np.asarray(metadata.black_level_canonical, dtype=np.float32).reshape(4, 1, 1)
    return packed.astype(np.float32, copy=False) - black, metadata


def calibrate_condition(records: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    condition = str(records[0]["condition_key"])
    if any(str(record["condition_key"]) != condition for record in records):
        raise ValueError("calibrate_condition received mixed conditions")
    if len(records) < 2:
        raise ValueError(f"{condition}: at least two calibration frames are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / "dark_shading_dn.npy", output_dir / "defect_mask.npy", output_dir / "metadata.json"]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{condition}: artifacts already exist ({existing}); pass --overwrite to replace them")

    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    reference_metadata = None
    for index, record in enumerate(records, start=1):
        frame, metadata = _frame_dn(record)
        if mean is None:
            mean = np.zeros_like(frame, dtype=np.float32)
            m2 = np.zeros_like(frame, dtype=np.float32)
            reference_metadata = metadata
        elif frame.shape != mean.shape:
            raise ValueError(f"{condition}: inconsistent packed shape {frame.shape} != {mean.shape}")
        if metadata.condition_key != condition:
            raise ValueError(f"{record['path']}: DNG condition changed since manifest generation")
        delta = frame - mean
        mean += delta / float(index)
        m2 += delta * (frame - mean)
        print(f"{condition}: first-pass {index}/{len(records)}", flush=True)
    assert mean is not None and m2 is not None and reference_metadata is not None
    std = np.sqrt(np.maximum(m2 / float(len(records) - 1), 0.0), dtype=np.float32)
    threshold = float(args.clip_sigma) * np.maximum(std, float(args.minimum_std_dn))

    clipped_sum = np.zeros_like(mean, dtype=np.float32)
    clipped_count = np.zeros(mean.shape, dtype=np.uint16)
    for index, record in enumerate(records, start=1):
        frame, _ = _frame_dn(record)
        valid = np.abs(frame - mean) <= threshold
        clipped_sum += np.where(valid, frame, 0.0)
        clipped_count += valid.astype(np.uint16)
        print(f"{condition}: clip-pass {index}/{len(records)}", flush=True)
    dark_shading = clipped_sum / np.maximum(clipped_count, 1).astype(np.float32)
    # Isolated persistent pixels belong to the DS artifact, but are retained in
    # a separate mask for future pair evaluation / target masking.
    local_median = median_filter(dark_shading, size=(1, 5, 5), mode="nearest")
    defect_mask = np.abs(dark_shading - local_median) >= float(args.defect_threshold_dn)

    _save_npy(output_dir / "dark_shading_dn.npy", dark_shading.astype(np.float32, copy=False))
    _save_npy(output_dir / "defect_mask.npy", defect_mask.astype(bool, copy=False))
    (output_dir / "calibration_frames.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "format_version": 1,
        "algorithm": "two_pass_sigma_clipped_mean",
        "clip_sigma": float(args.clip_sigma),
        "minimum_std_dn": float(args.minimum_std_dn),
        "condition_key": condition,
        "units": "linearized_DN",
        "black_subtracted": True,
        "ds_includes_black": False,
        "domain": "stored_pre_opcode_mosaic",
        "opcode_policy": "skip",
        "canonical_channel_order": ["R", "G1", "G2", "B"],
        "packed_shape_chw": list(dark_shading.shape),
        "source_frame_count": len(records),
        "source_paths": [record["path"] for record in records],
        "source_hashes": [record.get("sha256") for record in records],
        "dng_metadata": reference_metadata.as_dict(),
        "dark_shading_stats_dn": {
            "mean": [float(value) for value in dark_shading.mean(axis=(1, 2))],
            "std": [float(value) for value in dark_shading.std(axis=(1, 2))],
            "minimum": [float(value) for value in dark_shading.min(axis=(1, 2))],
            "maximum": [float(value) for value in dark_shading.max(axis=(1, 2))],
        },
        "temporal_std_stats_dn": {
            "mean": [float(value) for value in std.mean(axis=(1, 2))],
            "median": [float(value) for value in np.median(std, axis=(1, 2))],
        },
        "clip_rejection_fraction": float(1.0 - clipped_count.astype(np.float32).mean() / float(len(records))),
        "defect_pixel_count": int(defect_mask.sum()),
        "defect_threshold_dn": float(args.defect_threshold_dn),
    }
    temporary = output_dir / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "metadata.json")
    del mean, m2, std, threshold, clipped_sum, clipped_count, local_median
    return metadata


def main() -> None:
    args = parse_args()
    if args.clip_sigma <= 0 or args.minimum_std_dn <= 0 or args.defect_threshold_dn <= 0:
        raise ValueError("clip-sigma, minimum-std-dn and defect-threshold-dn must be positive")
    records = read_jsonl(Path(args.manifest))
    if any(record.get("role") != "dark" or record.get("split") != "calibration" for record in records):
        raise ValueError("Calibration manifest must contain only role=dark, split=calibration records")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["condition_key"])].append(record)
    report = []
    for condition, condition_records in sorted(groups.items()):
        print(f"Calibrating {condition} from {len(condition_records)} frames", flush=True)
        report.append(calibrate_condition(condition_records, Path(args.output_root) / condition, args))
    summary = {
        "conditions": len(report),
        "frames": len(records),
        "output_root": str(Path(args.output_root).resolve()),
        "artifacts": [item["condition_key"] for item in report],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
