#!/usr/bin/env python3
"""Run unpaired qualitative denoising on phone DNGs; intentionally no PSNR."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.phone_dng import read_phone_dng_packed
from utils.phone_inference import make_side_by_side, packed_raw_preview, tiled_phone_inference
from utils.phone_model import load_phone_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by train_phone.py")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-manifest", default=None, help="JSONL manifest from build_phone_manifest.py")
    inputs.add_argument("--input-root", default=None, help="Directory recursively containing DNGs")
    parser.add_argument("--output-dir", default="experiments/mey_an00_pseudoclean/qualitative_noisy")
    parser.add_argument("--calibration-root", default="data/MEY_AN00/calibration", help="DS artifacts used during hybrid training")
    parser.add_argument("--no-dark-shading", action="store_true", help="Ablation only: do not subtract calibrated 1/30 s DS")
    parser.add_argument(
        "--missing-ds-policy",
        choices=["error", "nearest_log2", "none"],
        default="nearest_log2",
        help="What to do when this unpaired noisy DNG has no exact ISO/exposure DS artifact",
    )
    parser.add_argument(
        "--reference-exposure-s",
        type=float,
        default=10.0,
        help="Long-exposure target domain; used when --ratio is not supplied",
    )
    parser.add_argument("--ratio", type=float, default=None, help="Override target/input exposure ratio for every DNG")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=2.2, help="Display-only packed-RGB preview gamma")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=0, help="0 processes all inputs")
    parser.add_argument("--write-summary", action="store_true", help="Also write run_summary.json; default output contains only comparison PNGs")
    return parser.parse_args()


def _dng_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".dng")


def load_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_manifest:
        manifest = Path(args.input_manifest)
        with manifest.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not records:
            raise ValueError(f"Input manifest is empty: {manifest}")
        for record in records:
            if "path" not in record:
                raise ValueError(f"{manifest}: each record needs a path")
        return records
    root = Path(args.input_root) if args.input_root else Path("raw-test/noisy")
    paths = _dng_paths(root)
    if not paths:
        raise FileNotFoundError(f"No DNG files found below {root}")
    return [{"path": str(path.resolve()), "relative_path": str(path.relative_to(root))} for path in paths]


def _load_dark_shading_artifact(directory: Path, expected_shape: tuple[int, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    ds_path, metadata_path = directory / "dark_shading_dn.npy", directory / "metadata.json"
    if not ds_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing dark-shading artifact below {directory}")
    artifact_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if artifact_metadata.get("units") != "linearized_DN" or not artifact_metadata.get("black_subtracted"):
        raise ValueError(f"{metadata_path}: unsupported DS domain")
    dark_shading = np.load(ds_path, mmap_mode="r")
    if tuple(dark_shading.shape) != expected_shape:
        raise ValueError(f"{ds_path}: shape {dark_shading.shape} != noisy packed shape {expected_shape}")
    return np.asarray(dark_shading, dtype=np.float32), artifact_metadata


def load_dark_shading(
    calibration_root: Path,
    *,
    condition_key: str,
    iso_exif: int,
    exposure_s: float,
    expected_shape: tuple[int, ...],
    missing_policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load exact DS or a clearly labelled nearest-log-ISO qualitative fallback."""
    try:
        dark_shading, artifact_metadata = _load_dark_shading_artifact(calibration_root / condition_key, expected_shape)
        if artifact_metadata.get("condition_key") != condition_key:
            raise ValueError(f"{calibration_root / condition_key / 'metadata.json'}: condition key mismatch")
        return dark_shading, {
            "mode": "exact",
            "requested_condition_key": condition_key,
            "used_condition_key": condition_key,
            "requested_iso_exif": int(iso_exif),
            "used_iso_exif": int(iso_exif),
        }
    except FileNotFoundError:
        if missing_policy == "error":
            raise FileNotFoundError(
                f"Missing dark-shading artifact for {condition_key}; run calibrate_phone_dark_shading.py, "
                "use --missing-ds-policy nearest_log2, or use --no-dark-shading"
            ) from None
        if missing_policy == "none":
            return np.zeros(expected_shape, dtype=np.float32), {
                "mode": "none_missing_exact",
                "requested_condition_key": condition_key,
                "used_condition_key": None,
                "requested_iso_exif": int(iso_exif),
                "used_iso_exif": None,
            }

    candidates: list[tuple[float, int, str, Path]] = []
    for metadata_path in calibration_root.glob("*/metadata.json"):
        try:
            artifact_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate_dng_metadata = artifact_metadata["dng_metadata"]
            candidate_iso = int(candidate_dng_metadata["iso_exif"])
            candidate_exposure_s = float(candidate_dng_metadata["exposure_s"])
            candidate_key = str(artifact_metadata["condition_key"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if tuple(artifact_metadata.get("packed_shape_chw", ())) != expected_shape:
            continue
        if not math.isclose(candidate_exposure_s, exposure_s, rel_tol=0.02, abs_tol=1e-9):
            continue
        candidates.append((abs(math.log2(candidate_iso / float(iso_exif))), candidate_iso, candidate_key, metadata_path.parent))
    if not candidates:
        raise FileNotFoundError(
            f"No readable same-exposure, same-shape DS artifacts found below {calibration_root} for {condition_key}"
        )
    _, candidate_iso, candidate_key, candidate_directory = min(candidates)
    dark_shading, _ = _load_dark_shading_artifact(candidate_directory, expected_shape)
    return dark_shading, {
        "mode": "nearest_log2",
        "requested_condition_key": condition_key,
        "used_condition_key": candidate_key,
        "requested_iso_exif": int(iso_exif),
        "used_iso_exif": candidate_iso,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.reference_exposure_s <= 0 or (args.ratio is not None and args.ratio <= 0):
        raise ValueError("reference-exposure-s and ratio must be positive")
    device = torch.device(args.device)
    model, checkpoint_args, model_name = load_phone_checkpoint(args.checkpoint, device)
    records = load_inputs(args)
    if args.max_images > 0:
        records = records[: args.max_images]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_root = Path(args.calibration_root)

    run_summary: dict[str, Any] | None = None
    if args.write_summary:
        run_summary = {
            "protocol": "qualitative_unpaired_no_reference_metrics",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model": model_name,
            "checkpoint_synthesis": checkpoint_args.get("synthesis"),
            "reference_exposure_s": args.reference_exposure_s,
            "ratio_override": args.ratio,
            "missing_ds_policy": args.missing_ds_policy,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "outputs": [],
        }
    for index, record in enumerate(records, start=1):
        source_path = Path(record["path"])
        packed, metadata = read_phone_dng_packed(source_path)
        black = np.asarray(metadata.black_level_canonical, dtype=np.float32).reshape(4, 1, 1)
        ratio = float(args.ratio) if args.ratio is not None else args.reference_exposure_s / metadata.exposure_s
        if args.no_dark_shading:
            dark_shading = np.zeros_like(packed, dtype=np.float32)
            ds_info = {
                "mode": "disabled",
                "requested_condition_key": metadata.condition_key,
                "used_condition_key": None,
                "requested_iso_exif": metadata.iso_exif,
                "used_iso_exif": None,
            }
        else:
            dark_shading, ds_info = load_dark_shading(
                calibration_root,
                condition_key=metadata.condition_key,
                iso_exif=metadata.iso_exif,
                exposure_s=metadata.exposure_s,
                expected_shape=packed.shape,
                missing_policy=args.missing_ds_policy,
            )
        normalized_input = (packed.astype(np.float32, copy=False) - black - dark_shading) / float(metadata.dynamic_range)
        model_input = np.minimum(normalized_input * ratio, 1.0).astype(np.float32, copy=False)
        input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(device)
        prediction = torch.clamp(
            tiled_phone_inference(model, input_tensor, tile_size=args.tile_size, tile_overlap=args.tile_overlap), 0.0, 1.0
        )[0].cpu().numpy()
        identifier = f"{index:03d}_{source_path.stem}"
        # Input and output use the exact same display scale, so their apparent
        # brightness is comparable.  This preview is not an ISP/color render.
        input_preview, display_upper = packed_raw_preview(model_input, gamma=args.gamma)
        denoised_preview, _ = packed_raw_preview(prediction, upper=display_upper, gamma=args.gamma)
        comparison_png = output_dir / f"{identifier}_comparison.png"
        imageio.imwrite(comparison_png, make_side_by_side(input_preview, denoised_preview, ratio=ratio))
        item_summary = {
            "source_dng": str(source_path.resolve()),
            "relative_path": record.get("relative_path"),
            "iso_exif": metadata.iso_exif,
            "input_exposure_s": metadata.exposure_s,
            "reference_exposure_s": args.reference_exposure_s,
            "exposure_ratio": ratio,
            "canonical_channel_order": ["R", "G1", "G2", "B"],
            "domain": metadata.domain,
            "dark_shading": ds_info,
            "comparison_png": comparison_png.name,
            "display_only_preview_upper": display_upper,
            "metrics": None,
        }
        if run_summary is not None:
            run_summary["outputs"].append(item_summary)
        ds_note = "exact DS" if ds_info["mode"] == "exact" else (
            f"DS={ds_info['mode']} (ISO {ds_info['used_iso_exif']})"
        )
        print(
            f"[{index}/{len(records)}] {source_path.name}: ISO {metadata.iso_exif}, "
            f"{metadata.exposure_s:.7f}s -> x{ratio:.5g}; {ds_note}; comparison saved"
        )
    if run_summary is not None:
        (output_dir / "run_summary.json").write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"[done] {len(records)} unpaired comparison PNGs written to {output_dir.resolve()}; PSNR/SSIM were not computed.")


if __name__ == "__main__":
    main()
