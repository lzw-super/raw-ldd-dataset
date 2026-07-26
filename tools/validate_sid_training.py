#!/usr/bin/env python3
"""Validate the local SID/LLD assets and core synthesis invariants."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.sid_synthetic_train import SIDSyntheticTrainDataset, build_sid_patch_manifest
from noise.dark_frame_bank import DarkFrameBank
from noise.sid_noise_synthesis import synthesize_sid_noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="infos/SID_train_clean_patches.json")
    parser.add_argument("--patch-dir", default="/home/shared_files/dataset/SID/Sony_train_long_patches")
    parser.add_argument("--pair-list", default="/home/shared_files/dataset/SID/Sony_train_list.txt")
    parser.add_argument("--sid-long-dir", default="/home/shared_files/dataset/SID/Sony/long")
    parser.add_argument("--dark-root", default="biasframe_et_1_30")
    parser.add_argument("--pmn-resource-dir", default="resources/SonyA7S2")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--output", default="worklog/sid_training_asset_validation.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        build_sid_patch_manifest(args.patch_dir, args.pair_list, manifest_path, args.sid_long_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    assert records and all(str(record["scene_id"]).startswith("0") for record in records)

    bank = DarkFrameBank(args.dark_root, args.pmn_resource_dir)
    dark_metadata = bank.validate_metadata()
    if dark_metadata["failures"]:
        raise AssertionError(f"Dark-frame metadata verification failed: {dark_metadata['failures'][:3]}")
    clean_isos = sorted({int(record["iso"]) for record in records})
    mapped = {iso: bank.map_iso(iso) for iso in clean_isos}
    sample_iso = clean_isos[len(clean_isos) // 2]
    residual, matched_iso, dark_path = bank.sample_residual_patch(sample_iso, args.patch_size, np.random.default_rng(7))
    assert residual.shape == (4, args.patch_size, args.patch_size)
    assert np.isfinite(residual).all()

    dataset = SIDSyntheticTrainDataset(
        manifest_path, args.dark_root, args.pmn_resource_dir, patch_size=args.patch_size, crops_per_image=1, seed=7
    )
    item = dataset[0]
    clean = item["clean"].unsqueeze(0)
    dark = item["dark_residual_dn"].unsqueeze(0)
    noisy, target = synthesize_sid_noise(clean, dark, item["iso"].view(1), item["ratio"].view(1))
    assert noisy.shape == target.shape == (1, 4, args.patch_size, args.patch_size)
    assert float(noisy.max()) <= 1.0 + 1e-6

    # Poisson moment check in DN.  Use a flat patch and no dark noise, so it
    # catches the common error of forgetting the SID ratio in the variance.
    pixels = 128 * 128
    flat = torch.full((1, 4, 128, 128), 0.2)
    zeros = torch.zeros_like(flat)
    samples = []
    for _ in range(48):
        generated, _ = synthesize_sid_noise(flat, zeros, 800, 100, mode="ratio_aware")
        samples.append((generated * (16383 - 512)).reshape(-1))
    values = torch.stack(samples)
    empirical_mean = float(values.mean())
    empirical_variance = float(values.var(unbiased=True))
    expected_mean = 0.2 * (16383 - 512)
    expected_variance = 100 * (800 / 1000.0) * expected_mean
    result = {
        "manifest_summary": manifest["summary"],
        "dark_available_isos": bank.available_isos,
        "dark_metadata": dark_metadata,
        "clean_to_dark_iso": mapped,
        "residual_sample": {
            "requested_iso": sample_iso,
            "matched_iso": matched_iso,
            "path": dark_path,
            "mean_dn": float(residual.mean()),
            "std_dn": float(residual.std()),
            "min_dn": float(residual.min()),
            "max_dn": float(residual.max()),
        },
        "synthetic_sample": {
            "scene_id": item["scene_id"],
            "iso": float(item["iso"]),
            "ratio": float(item["ratio"]),
            "negative_input_fraction": float((noisy < 0).float().mean()),
        },
        "poisson_moments_dn": {
            "empirical_mean": empirical_mean,
            "expected_mean": expected_mean,
            "relative_mean_error": abs(empirical_mean - expected_mean) / expected_mean,
            "empirical_variance": empirical_variance,
            "expected_variance": expected_variance,
            "relative_variance_error": abs(empirical_variance - expected_variance) / expected_variance,
        },
    }
    if result["poisson_moments_dn"]["relative_mean_error"] > 0.03:
        raise AssertionError("Poisson mean check failed")
    if result["poisson_moments_dn"]["relative_variance_error"] > 0.10:
        raise AssertionError("Poisson variance check failed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote validation report: {output}")


if __name__ == "__main__":
    main()
