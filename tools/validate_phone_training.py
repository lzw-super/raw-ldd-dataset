#!/usr/bin/env python3
"""Smoke-test MEY-AN00 manifests, CFA packing, DS residuals and synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.phone_synthetic_train import PhoneSyntheticTrainDataset
from noise.dng_noise_synthesis import synthesize_phone_noise
from noise.phone_dark_frame_bank import PhoneDarkFrameBank
from utils.phone_dng import read_phone_dng_packed, unpack_to_dng_cfa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", default="data/MEY_AN00/manifests")
    parser.add_argument("--calibration-root", default="data/MEY_AN00/calibration")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--output", default="worklog/mey_an00_training_validation.json")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    summary = json.loads((manifest_dir / "manifest_summary.json").read_text(encoding="utf-8"))
    if summary.get("clean_train_source_records") != 90 or summary["dark_records"] != 134:
        raise AssertionError(
            "Unexpected current collection counts: "
            f"{summary.get('clean_train_source_records')}, {summary['dark_records']}"
        )
    if summary.get("clean_synthetic_val_records") != 10 or summary.get("qualitative_noisy_records") != 10:
        raise AssertionError(
            "Expected 10 independent synthetic-val and 10 unpaired qualitative noisy DNGs; got "
            f"{summary.get('clean_synthetic_val_records')}, {summary.get('qualitative_noisy_records')}"
        )
    if len(summary["folder_iso_outliers"]) != 4:
        raise AssertionError("Expected four clean folder-ISO outliers")
    clean_records = read_jsonl(manifest_dir / "clean_train.jsonl")
    synthetic_val_records = read_jsonl(manifest_dir / "clean_synthetic_val.jsonl")
    qualitative_noisy_records = read_jsonl(manifest_dir / "noisy_qualitative.jsonl")
    dark_train = read_jsonl(manifest_dir / "dark_train.jsonl")
    dark_val = read_jsonl(manifest_dir / "dark_val.jsonl")
    if not clean_records or not synthetic_val_records or not qualitative_noisy_records or not dark_train or not dark_val:
        raise AssertionError("One of the training manifests is empty")
    if any(record.get("role") != "clean_source" or record.get("split") != "synthetic_val" for record in synthetic_val_records):
        raise AssertionError("Synthetic validation manifest has an invalid role or split")
    if any(record.get("role") != "qualitative_noisy" or record.get("pair_status") != "no_ground_truth" for record in qualitative_noisy_records):
        raise AssertionError("Qualitative noisy manifest must be explicitly unpaired")
    val_iso = {int(record["iso_exif"]) for record in synthetic_val_records}
    held_out_dark_iso = {int(record["iso_exif"]) for record in dark_val}
    if not val_iso <= held_out_dark_iso:
        raise AssertionError(f"Synthetic validation ISO(s) lack held-out dark residuals: {sorted(val_iso - held_out_dark_iso)}")

    packed, metadata = read_phone_dng_packed(clean_records[0]["path"])
    expected_crop = tifffile.imread(clean_records[0]["path"])[8:3040, 8:4056]
    roundtrip_equal = bool(np.array_equal(unpack_to_dng_cfa(packed, metadata), expected_crop))
    if not roundtrip_equal:
        raise AssertionError("Canonical pack/unpack is not a pixel-exact DefaultCrop round-trip")
    if metadata.cfa_name != "GBRG" or metadata.canonical_position_indices != (2, 3, 0, 1):
        raise AssertionError(f"Unexpected MEY-AN00 CFA mapping: {metadata.cfa_name}, {metadata.canonical_position_indices}")

    iso200 = next(record for record in clean_records if int(record["iso_exif"]) == 200)
    k_dn = np.asarray(iso200["noise_profile_s4"], dtype=np.float64) * float(iso200["dynamic_range"])
    expected_k_dn = np.asarray([0.06221, 0.10115, 0.10115, 0.07396])
    if not np.allclose(k_dn, expected_k_dn, rtol=0.0, atol=3e-4):
        raise AssertionError(f"NoiseProfile S -> K_DN mismatch: {k_dn}")

    dataset = PhoneSyntheticTrainDataset(
        manifest_dir / "clean_train.jsonl",
        manifest_dir / "dark_train.jsonl",
        args.calibration_root,
        patch_size=args.patch_size,
        crops_per_image=1,
        ratios=(100.0, 250.0, 300.0),
        dark_exposure_policy="approximate_reuse_1_30s",
        cache_size=1,
        dark_cache_size=1,
        seed=7,
    )
    item = dataset[0]
    clean = item["clean"].unsqueeze(0)
    dark = item["dark_residual_dn"].unsqueeze(0)
    s4 = item["noise_profile_s4"].unsqueeze(0)
    o4 = item["noise_profile_o4"].unsqueeze(0)
    noisy, target = synthesize_phone_noise(
        clean, dark, s4, item["dark_dynamic_range"].view(1), item["ratio"].view(1), mode="hybrid", noise_profile_o4=o4
    )
    if noisy.shape != target.shape != (1, 4, args.patch_size, args.patch_size):
        raise AssertionError("Unexpected training batch shape")
    if float(noisy.max()) > 1.0 + 1e-6 or float(target.min()) < -1e-6 or float(target.max()) > 1.0 + 1e-6:
        raise AssertionError("Synthetic input/target clamp policy is wrong")
    if not (float(dark.min()) < 0.0 < float(dark.max())):
        raise AssertionError("Dark residual lost its signed distribution")

    # The documented interim protocol must accept ratio 100/250 while retaining
    # the same 1/30 s dark source and expose both exposure values to logs.
    approximate_ratios = {float(item["ratio"])}
    approximate_items = [item]
    for _ in range(32):
        approximate_item = dataset[0]
        approximate_items.append(approximate_item)
        approximate_ratios.add(float(approximate_item["ratio"]))
    if approximate_ratios != {100.0, 250.0, 300.0}:
        raise AssertionError(f"Approximate dark-reuse sampler missed ratios: {approximate_ratios}")
    for approximate_item in approximate_items:
        if approximate_item["dark_exposure_policy"] != "approximate_reuse_1_30s":
            raise AssertionError("Approximate dark-exposure policy was not propagated")
        expected_short = float(approximate_item["clean_source_exposure_s"]) / float(approximate_item["ratio"])
        if not np.isclose(float(approximate_item["simulated_short_exposure_s"]), expected_short, rtol=0.0, atol=1e-6):
            raise AssertionError("Simulated short-exposure metadata is inconsistent with ratio")
        if not np.isclose(float(approximate_item["dark_source_exposure_s"]), 1.0 / 30.0, rtol=0.0, atol=1e-6):
            raise AssertionError("Approximate policy did not reuse a 1/30 s dark source")
    strict_dataset = PhoneSyntheticTrainDataset(
        manifest_dir / "clean_train.jsonl",
        manifest_dir / "dark_train.jsonl",
        args.calibration_root,
        patch_size=args.patch_size,
        crops_per_image=1,
        ratios=(100.0,),
        dark_exposure_policy="strict_match",
        cache_size=1,
        dark_cache_size=1,
        seed=13,
    )
    try:
        strict_dataset[0]
    except ValueError as error:
        if "incompatible with matched dark exposure" not in str(error):
            raise
    else:
        raise AssertionError("strict_match unexpectedly accepted ratio=100 with only 1/30 s dark frames")

    # Hold-out dark frames use the calibration artifact without participating in
    # it.  This is only a distribution smoke check, not a calibration claim.
    val_bank = PhoneDarkFrameBank(manifest_dir / "dark_val.jsonl", args.calibration_root, cache_size=1)
    held_out = val_bank.sample_residual_crop(200, 0, 0, args.patch_size, np.random.default_rng(9))
    if not (float(held_out.residual_dn.min()) < 0.0 < float(held_out.residual_dn.max())):
        raise AssertionError("Held-out residual is not signed")

    # Moment check with zero dark residual.  For ratio-aware DNG Poisson noise:
    # Var(Y | X) = ratio * S * X in the normalized output domain.
    flat = torch.full((1, 4, 64, 64), 0.2)
    zeros = torch.zeros_like(flat)
    constant_s = torch.full((1, 4), 1e-4)
    samples = []
    for _ in range(64):
        synthetic, _ = synthesize_phone_noise(flat, zeros, constant_s, 1.0, 300.0, mode="hybrid")
        samples.append(synthetic.reshape(-1))
    values = torch.cat(samples)
    expected_variance = 300.0 * 1e-4 * 0.2
    relative_variance_error = abs(float(values.var(unbiased=True)) - expected_variance) / expected_variance
    if relative_variance_error > 0.10:
        raise AssertionError(f"Poisson variance check failed: relative error={relative_variance_error:.3f}")

    report = {
        "manifest_summary": summary,
        "roundtrip_equal": roundtrip_equal,
        "metadata": {"cfa": metadata.cfa_name, "canonical_position_indices": list(metadata.canonical_position_indices)},
        "iso200_k_dn": k_dn.tolist(),
        "train_sample": {
            "clean_path": item["clean_path"],
            "dark_path": item["dark_path"],
            "condition": item["dark_condition_key"],
            "negative_input_fraction": float((noisy < 0).float().mean()),
            "dark_residual_mean_dn": float(dark.mean()),
            "dark_residual_std_dn": float(dark.std()),
            "target_saturation_fraction": float(item["target_saturation_fraction"]),
            "ratios_seen": sorted(approximate_ratios),
            "dark_exposure_policy": item["dark_exposure_policy"],
            "strict_match_rejects_ratio_100": True,
        },
        "held_out_dark": {"mean_dn": float(held_out.residual_dn.mean()), "std_dn": float(held_out.residual_dn.std())},
        "synthetic_validation": {
            "records": len(synthetic_val_records),
            "iso_exif": sorted(val_iso),
            "held_out_dark_iso_coverage": True,
        },
        "qualitative_noisy": {
            "records": len(qualitative_noisy_records),
            "has_ground_truth": False,
        },
        "poisson_variance": {
            "empirical": float(values.var(unbiased=True)),
            "expected": expected_variance,
            "relative_error": relative_variance_error,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
