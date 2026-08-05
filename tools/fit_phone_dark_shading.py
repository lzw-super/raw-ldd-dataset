#!/usr/bin/env python3
"""Fit a PMN/SID-style continuous ISO dark-shading model for phone DNGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noise.phone_dark_shading import DEFAULT_MODEL_DIRECTORY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--calibration-root", default="data/MEY_AN00/calibration")
    parser.add_argument("--model-directory", default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument(
        "--iso-breakpoint",
        type=float,
        default=1600.0,
        help="Low branch uses ISO <= breakpoint; high branch uses ISO > breakpoint",
    )
    parser.add_argument(
        "--uniform-iso-weight",
        action="store_true",
        help="Give each calibrated ISO equal weight instead of weighting by calibration frame count",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_artifacts(calibration_root: Path, model_directory: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for metadata_path in sorted(calibration_root.glob("*/metadata.json")):
        if metadata_path.parent.name == model_directory:
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "dng_metadata" not in metadata or "condition_key" not in metadata:
            continue
        ds_path = metadata_path.parent / "dark_shading_dn.npy"
        if not ds_path.is_file():
            raise FileNotFoundError(f"{metadata_path}: missing {ds_path.name}")
        dng = metadata["dng_metadata"]
        artifacts.append(
            {
                "iso": int(dng["iso_exif"]),
                "exposure_s": float(dng["exposure_s"]),
                "condition_key": str(metadata["condition_key"]),
                "shape": tuple(int(value) for value in metadata["packed_shape_chw"]),
                "frames": int(metadata["source_frame_count"]),
                "ds_path": ds_path,
                "metadata_path": metadata_path,
                "camera": str(dng["unique_camera_model"]),
                "domain": str(metadata["domain"]),
                "opcode_policy": str(metadata["opcode_policy"]),
                "channel_order": tuple(metadata["canonical_channel_order"]),
            }
        )
    if not artifacts:
        raise FileNotFoundError(f"No per-condition dark-shading artifacts found below {calibration_root}")
    artifacts.sort(key=lambda item: (item["iso"], item["condition_key"]))
    isos = [item["iso"] for item in artifacts]
    if len(isos) != len(set(isos)):
        raise ValueError("Continuous fitting currently requires exactly one condition per EXIF ISO")
    reference = artifacts[0]
    invariant_keys = ("shape", "camera", "domain", "opcode_policy", "channel_order")
    for item in artifacts[1:]:
        for key in invariant_keys:
            if item[key] != reference[key]:
                raise ValueError(f"Cannot combine DS artifacts with different {key}: {reference[key]} vs {item[key]}")
        relative_exposure_error = abs(item["exposure_s"] - reference["exposure_s"]) / reference["exposure_s"]
        if relative_exposure_error > 0.02:
            raise ValueError("Cannot fit one ISO model from dark frames with different exposures")
    if reference["domain"] != "stored_pre_opcode_mosaic" or reference["opcode_policy"] != "skip":
        raise ValueError("Continuous model requires stored pre-opcode mosaic artifacts")
    if reference["channel_order"] != ("R", "G1", "G2", "B"):
        raise ValueError("Continuous model requires canonical [R,G1,G2,B] artifacts")
    return artifacts


def fit_branch(artifacts: list[dict[str, Any]], *, uniform_weight: bool) -> tuple[np.ndarray, np.ndarray]:
    if len(artifacts) < 2:
        raise ValueError("Each ISO branch needs at least two calibrated ISO points")
    weights = np.asarray([1.0 if uniform_weight else item["frames"] for item in artifacts], dtype=np.float64)
    isos = np.asarray([item["iso"] for item in artifacts], dtype=np.float64)
    mean_iso = float(np.sum(weights * isos) / np.sum(weights))
    centered = isos - mean_iso
    denominator = float(np.sum(weights * np.square(centered)))
    if denominator <= 0:
        raise ValueError("ISO branch has no ISO variation")

    shape = artifacts[0]["shape"]
    weighted_y = np.zeros(shape, dtype=np.float64)
    weighted_centered_y = np.zeros(shape, dtype=np.float64)
    for item, weight, centered_iso in zip(artifacts, weights, centered):
        dark_shading = np.load(item["ds_path"], mmap_mode="r")
        if tuple(dark_shading.shape) != shape or dark_shading.dtype != np.float32:
            raise ValueError(f"{item['ds_path']}: expected float32 {shape}, got {dark_shading.shape} {dark_shading.dtype}")
        weighted_y += weight * dark_shading
        weighted_centered_y += weight * centered_iso * dark_shading
    slope = weighted_centered_y / denominator
    intercept = weighted_y / float(np.sum(weights)) - slope * mean_iso
    return slope.astype(np.float32), intercept.astype(np.float32)


def save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, value.astype(np.float32, copy=False))
    os.replace(temporary, path)


def residual_statistics(
    artifacts: list[dict[str, Any]],
    coefficients: dict[str, tuple[np.ndarray, np.ndarray]],
    breakpoint: float,
    ble_by_iso: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for item in artifacts:
        branch = "low" if item["iso"] <= breakpoint else "high"
        slope, intercept = coefficients[branch]
        observed = np.load(item["ds_path"], mmap_mode="r")
        prediction = slope * float(item["iso"]) + intercept + ble_by_iso[item["iso"]].reshape(4, 1, 1)
        error = np.asarray(observed, dtype=np.float32) - prediction
        report.append(
            {
                "iso_exif": item["iso"],
                "branch": branch,
                "source_frame_count": item["frames"],
                "mean_error_dn": [float(value) for value in error.mean(axis=(1, 2))],
                "mae_dn": [float(value) for value in np.abs(error).mean(axis=(1, 2))],
                "rmse_dn": [float(value) for value in np.sqrt(np.square(error).mean(axis=(1, 2)))],
            }
        )
    return report


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.iso_breakpoint) or args.iso_breakpoint <= 0:
        raise ValueError("iso-breakpoint must be finite and positive")
    calibration_root = Path(args.calibration_root)
    artifacts = discover_artifacts(calibration_root, args.model_directory)
    low = [item for item in artifacts if item["iso"] <= args.iso_breakpoint]
    high = [item for item in artifacts if item["iso"] > args.iso_breakpoint]
    if len(low) < 2 or len(high) < 2:
        raise ValueError(
            f"Need at least two ISO points in each branch; low={len(low)}, high={len(high)}, "
            f"breakpoint={args.iso_breakpoint:g}"
        )

    output_dir = calibration_root / args.model_directory
    outputs = [
        output_dir / "darkshading_lowISO_k.npy",
        output_dir / "darkshading_lowISO_b.npy",
        output_dir / "darkshading_highISO_k.npy",
        output_dir / "darkshading_highISO_b.npy",
        output_dir / "metadata.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Continuous DS model already exists ({existing}); pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficients = {
        "low": fit_branch(low, uniform_weight=args.uniform_iso_weight),
        "high": fit_branch(high, uniform_weight=args.uniform_iso_weight),
    }
    ble_by_iso: dict[int, np.ndarray] = {}
    for item in artifacts:
        branch = "low" if item["iso"] <= args.iso_breakpoint else "high"
        slope, intercept = coefficients[branch]
        observed = np.load(item["ds_path"], mmap_mode="r")
        spatial_prediction = slope * float(item["iso"]) + intercept
        ble_by_iso[item["iso"]] = np.asarray(observed - spatial_prediction).mean(axis=(1, 2)).astype(np.float32)
    save_array(outputs[0], coefficients["low"][0])
    save_array(outputs[1], coefficients["low"][1])
    save_array(outputs[2], coefficients["high"][0])
    save_array(outputs[3], coefficients["high"][1])
    fit_report = residual_statistics(artifacts, coefficients, args.iso_breakpoint, ble_by_iso)
    metadata = {
        "format_version": 1,
        "model_type": "piecewise_linear_iso",
        "equation": "dark_shading_dn(iso) = k_branch * iso + b_branch + BLE(iso)",
        "iso_breakpoint": float(args.iso_breakpoint),
        "branch_policy": "low_if_iso_le_breakpoint_else_high",
        "fit_weighting": "uniform_iso" if args.uniform_iso_weight else "calibration_frame_count",
        "units": "linearized_DN",
        "black_subtracted": True,
        "ds_includes_black": False,
        "domain": artifacts[0]["domain"],
        "opcode_policy": artifacts[0]["opcode_policy"],
        "canonical_channel_order": list(artifacts[0]["channel_order"]),
        "packed_shape_chw": list(artifacts[0]["shape"]),
        "camera": artifacts[0]["camera"],
        "source_exposure_s": artifacts[0]["exposure_s"],
        "calibrated_isos": [item["iso"] for item in artifacts],
        "calibrated_iso_range": [min(item["iso"] for item in artifacts), max(item["iso"] for item in artifacts)],
        "low_branch_isos": [item["iso"] for item in low],
        "high_branch_isos": [item["iso"] for item in high],
        "ble_policy": "per_channel_log2_iso_linear_interpolation_with_endpoint_clamp",
        "ble_by_iso_dn": {
            str(iso): [float(value) for value in ble]
            for iso, ble in sorted(ble_by_iso.items())
        },
        "coefficient_files": {
            path.name: {"sha256": file_sha256(path), "dtype": "float32", "shape": list(artifacts[0]["shape"])}
            for path in outputs[:4]
        },
        "source_artifacts": [
            {
                "iso_exif": item["iso"],
                "condition_key": item["condition_key"],
                "source_frame_count": item["frames"],
                "dark_shading_path": str(item["ds_path"].resolve()),
                "dark_shading_sha256": file_sha256(item["ds_path"]),
                "metadata_path": str(item["metadata_path"].resolve()),
                "metadata_sha256": file_sha256(item["metadata_path"]),
            }
            for item in artifacts
        ],
        "fit_residuals": fit_report,
    }
    temporary = output_dir / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "metadata.json")
    print(
        json.dumps(
            {
                "output": str(output_dir.resolve()),
                "iso_breakpoint": args.iso_breakpoint,
                "low_branch_isos": metadata["low_branch_isos"],
                "high_branch_isos": metadata["high_branch_isos"],
                "fit_residuals": fit_report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
