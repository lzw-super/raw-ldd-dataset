#!/usr/bin/env python3
"""Build auditable manifests for the current phone DNG collection.

The tool never moves or renames raw files.  It reads final DNG/EXIF metadata,
so a folder such as ``iso-160`` remains a provenance label rather than the
noise-condition key.  The latter is always ``iso_exif`` from the DNG.

Without ``--scene-groups`` all files under ``cleanframe`` are intentionally
put in the prototype training manifest.  ``val`` is different: it is a
separately captured, held-out scene set and is always emitted as
``clean_synthetic_val.jsonl``.  It is suitable for *synthetic* validation,
not for a noisy-clean paired result.  Supply a complete scene map before
claiming a held-out result from files inside ``cleanframe``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.phone_dng import PhoneDNGMetadata, read_phone_dng_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-root", default="raw-test", help="Root containing cleanframe and biasframe-1-30")
    parser.add_argument("--output-dir", default="data/MEY_AN00/manifests")
    parser.add_argument(
        "--scene-groups",
        default=None,
        help="Optional JSON object keyed by raw-root-relative clean path; each value needs scene_id and split",
    )
    parser.add_argument("--sha256", action="store_true", help="Hash every DNG; slow but useful for a frozen experiment")
    parser.add_argument("--calibration-fraction", type=float, default=0.60)
    parser.add_argument("--dark-train-fraction", type=float, default=0.20)
    return parser.parse_args()


def jsonl_write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def dng_paths(root: Path) -> list[Path]:
    """Find DNGs without relying on the file-system's case behaviour."""
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".dng")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def folder_iso_label(path: Path) -> int | None:
    for parent in path.parents:
        match = re.search(r"(?i)iso[-_ ]?(\d+)", parent.name)
        if match:
            return int(match.group(1))
    return None


def session_from_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").strftime("%Y%m%d")
    except ValueError:
        return value[:10].replace(":", "")


def record_from_metadata(path: Path, raw_root: Path, role: str, metadata: PhoneDNGMetadata, include_hash: bool) -> dict[str, Any]:
    record = metadata.as_dict()
    record.update(
        {
            "path": str(path.resolve()),
            "relative_path": str(path.relative_to(raw_root)),
            "role": role,
            "camera_id": "_".join(
                part for part in (metadata.make, metadata.model) if part
            ) or metadata.unique_camera_model or "unknown",
            "sensor_mode_id": f"{metadata.raw_shape_hw[0]}x{metadata.raw_shape_hw[1]}_{metadata.cfa_name}_mode_unknown",
            "iso_folder_label": folder_iso_label(path),
            "iso_requested": None,
            "capture_session": session_from_datetime(metadata.datetime),
            "thermal_bin": None,
            "sha256": sha256(path) if include_hash else None,
        }
    )
    return record


def load_scene_groups(path: str | None, raw_root: Path, clean_paths: list[Path]) -> dict[str, dict[str, str]] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scene-groups must be a JSON object")
    expected = {str(item.relative_to(raw_root)) for item in clean_paths}
    provided = set(payload)
    missing, extra = sorted(expected - provided), sorted(provided - expected)
    if missing or extra:
        raise ValueError(
            "scene-groups must annotate every clean DNG exactly once; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    result: dict[str, dict[str, str]] = {}
    for relative_path, value in payload.items():
        if not isinstance(value, dict) or set(value) < {"scene_id", "split"}:
            raise ValueError(f"{relative_path}: scene map value must contain scene_id and split")
        split = str(value["split"])
        if split not in {"train", "val", "test"}:
            raise ValueError(f"{relative_path}: invalid split {split!r}")
        result[relative_path] = {"scene_id": str(value["scene_id"]), "split": split}
    return result


def split_dark_time_blocks(records: list[dict[str, Any]], calibration_fraction: float, train_fraction: float) -> dict[str, list[dict[str, Any]]]:
    if not (0.0 < calibration_fraction < 1.0 and 0.0 < train_fraction < 1.0 and calibration_fraction + train_fraction < 1.0):
        raise ValueError("dark split fractions must be positive and leave a validation remainder")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_condition[str(record["condition_key"])].append(record)
    outputs = {"calibration": [], "train": [], "val": []}
    for condition, items in sorted(by_condition.items()):
        items.sort(key=lambda item: (str(item.get("datetime") or ""), item["relative_path"]))
        count = len(items)
        if count < 5:
            raise ValueError(f"{condition}: need at least five dark frames for prototype time-block split, got {count}")
        calibration_end = max(1, int(round(count * calibration_fraction)))
        train_end = calibration_end + max(1, int(round(count * train_fraction)))
        # Preserve at least one held-out frame, rather than silently emitting an empty val manifest.
        train_end = min(train_end, count - 1)
        if train_end <= calibration_end:
            raise ValueError(f"{condition}: insufficient dark frames after calibration split")
        for index, item in enumerate(items):
            split = "calibration" if index < calibration_end else "train" if index < train_end else "val"
            copied = dict(item)
            copied["split"] = split
            copied["dark_split_protocol"] = "chronological_blocks_prototype"
            copied["dark_split_index"] = index
            outputs[split].append(copied)
    return outputs


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    clean_root = raw_root / "cleanframe"
    synthetic_val_root = raw_root / "val"
    qualitative_noisy_root = raw_root / "noisy"
    dark_root = raw_root / "biasframe-1-30"
    if not clean_root.is_dir() or not dark_root.is_dir():
        raise FileNotFoundError(f"Expected cleanframe and biasframe-1-30 below {raw_root}")
    clean_paths = dng_paths(clean_root)
    synthetic_val_paths = dng_paths(synthetic_val_root)
    qualitative_noisy_paths = dng_paths(qualitative_noisy_root)
    dark_paths = dng_paths(dark_root)
    if not clean_paths or not dark_paths:
        raise RuntimeError("No clean or dark DNG files found")
    scene_groups = load_scene_groups(args.scene_groups, raw_root, clean_paths)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    input_groups = (
        ("clean_source", "cleanframe", clean_paths),
        ("clean_source", "synthetic_val", synthetic_val_paths),
        ("qualitative_noisy", "qualitative_noisy", qualitative_noisy_paths),
        ("dark", "dark", dark_paths),
    )
    for role, source_set, paths in input_groups:
        for index, path in enumerate(paths, start=1):
            try:
                metadata = read_phone_dng_metadata(path)
                record = record_from_metadata(path, raw_root, role, metadata, args.sha256)
                record["source_set"] = source_set
                if role == "clean_source":
                    if source_set == "synthetic_val":
                        record["scene_id"] = f"synthetic_val_{path.stem}"
                        record["split"] = "synthetic_val"
                        record["scene_group_status"] = "independent_synthetic_val"
                    else:
                        annotation = scene_groups.get(record["relative_path"]) if scene_groups else None
                        record["scene_id"] = annotation["scene_id"] if annotation else f"unlabeled_{path.stem}"
                        record["split"] = annotation["split"] if annotation else "train"
                        record["scene_group_status"] = "annotated" if annotation else "unannotated_prototype"
                elif role == "qualitative_noisy":
                    record["split"] = "qualitative_unpaired"
                    record["pair_status"] = "no_ground_truth"
                records.append(record)
            except Exception as error:  # report every malformed DNG together
                failures.append({"path": str(path), "error": str(error)})
            if index % 25 == 0 or index == len(paths):
                print(f"read {role} metadata: {index}/{len(paths)}")
    if failures:
        raise RuntimeError(json.dumps({"invalid_dng_count": len(failures), "examples": failures[:10]}, ensure_ascii=False))

    clean_records = [record for record in records if record["role"] == "clean_source" and record["source_set"] == "cleanframe"]
    synthetic_val_records = [
        record for record in records if record["role"] == "clean_source" and record["source_set"] == "synthetic_val"
    ]
    qualitative_noisy_records = [record for record in records if record["role"] == "qualitative_noisy"]
    dark_records = [record for record in records if record["role"] == "dark"]
    camera_modes = {(record["camera_id"], record["sensor_mode_id"], record["cfa"], tuple(record["raw_shape_hw"])) for record in records}
    if len(camera_modes) != 1:
        raise ValueError(f"This single-sensor pipeline requires one camera/mode/CFA/shape, got {camera_modes}")
    clean_iso = {int(record["iso_exif"]) for record in clean_records}
    synthetic_val_iso = {int(record["iso_exif"]) for record in synthetic_val_records}
    dark_iso = {int(record["iso_exif"]) for record in dark_records}
    missing_dark_conditions = sorted((clean_iso | synthetic_val_iso) - dark_iso)
    if missing_dark_conditions:
        raise ValueError(f"No 1/30 s dark condition for clean/synthetic-val ISO(s) {missing_dark_conditions}")

    dark_splits = split_dark_time_blocks(dark_records, args.calibration_fraction, args.dark_train_fraction)
    missing_held_out_dark_conditions = sorted(
        synthetic_val_iso - {int(record["iso_exif"]) for record in dark_splits["val"]}
    )
    if missing_held_out_dark_conditions:
        raise ValueError(
            "No held-out dark residual condition for synthetic validation ISO(s) "
            f"{missing_held_out_dark_conditions}"
        )
    output_dir = Path(args.output_dir)
    jsonl_write(output_dir / "all_dng.jsonl", records)
    jsonl_write(output_dir / "clean_train.jsonl", [record for record in clean_records if record["split"] == "train"])
    if synthetic_val_records:
        jsonl_write(output_dir / "clean_synthetic_val.jsonl", synthetic_val_records)
    if qualitative_noisy_records:
        jsonl_write(output_dir / "noisy_qualitative.jsonl", qualitative_noisy_records)
    if scene_groups:
        jsonl_write(output_dir / "clean_val.jsonl", [record for record in clean_records if record["split"] == "val"])
        jsonl_write(output_dir / "clean_test.jsonl", [record for record in clean_records if record["split"] == "test"])
    for split, split_records in dark_splits.items():
        jsonl_write(output_dir / f"dark_{split}.jsonl", split_records)

    # Folder labels are UI/requested values and are expected to differ from the
    # final EXIF sensitivity on this phone.  Flag only files that disagree with
    # the dominant final ISO in their own role/folder bucket (the four known
    # ISO-500 clean files inside the ``iso 100-10`` folder are one example).
    folder_buckets: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["iso_folder_label"] is None:
            continue
        folder_buckets[(str(record["role"]), record["iso_folder_label"])].append(record)
    folder_majority_iso = {
        bucket: Counter(int(item["iso_exif"]) for item in items).most_common(1)[0][0]
        for bucket, items in folder_buckets.items()
    }
    folder_iso_outliers = [
        record
        for record in records
        if record["iso_folder_label"] is not None
        and int(record["iso_exif"]) != folder_majority_iso[(str(record["role"]), record["iso_folder_label"])]
    ]
    summary = {
        "format_version": 2,
        "raw_root": str(raw_root),
        "opcode_policy": "skip",
        "domain": "stored_pre_opcode_mosaic",
        "canonical_channel_order": ["R", "G1", "G2", "B"],
        "clean_train_source_records": len(clean_records),
        "clean_synthetic_val_records": len(synthetic_val_records),
        "qualitative_noisy_records": len(qualitative_noisy_records),
        "dark_records": len(dark_records),
        "clean_iso_exif_counts": dict(sorted(Counter(int(record["iso_exif"]) for record in clean_records).items())),
        "clean_synthetic_val_iso_exif_counts": dict(
            sorted(Counter(int(record["iso_exif"]) for record in synthetic_val_records).items())
        ),
        "qualitative_noisy_iso_exif_counts": dict(
            sorted(Counter(int(record["iso_exif"]) for record in qualitative_noisy_records).items())
        ),
        "dark_iso_exif_counts": dict(sorted(Counter(int(record["iso_exif"]) for record in dark_records).items())),
        "dark_split_counts": {key: len(value) for key, value in dark_splits.items()},
        "folder_iso_dominant_exif": {
            f"{role}:{folder}": iso for (role, folder), iso in sorted(folder_majority_iso.items())
        },
        "folder_iso_outliers": [record["relative_path"] for record in folder_iso_outliers],
        "scene_groups": "annotated" if scene_groups else "unannotated_prototype_train_only",
        "sha256": bool(args.sha256),
        "files": {
            "all": "all_dng.jsonl",
            "clean_train": "clean_train.jsonl",
            "clean_synthetic_val": "clean_synthetic_val.jsonl" if synthetic_val_records else None,
            "noisy_qualitative": "noisy_qualitative.jsonl" if qualitative_noisy_records else None,
            "dark_calibration": "dark_calibration.jsonl",
            "dark_train": "dark_train.jsonl",
            "dark_val": "dark_val.jsonl",
        },
    }
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
