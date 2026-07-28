"""Condition-matched DNG dark residual sampling for phone RAW synthesis."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils.phone_dng import PhoneDNGMetadata, read_phone_dng_packed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"Dark manifest is empty: {path}")
    return records


@dataclass(frozen=True)
class PhoneDarkSample:
    residual_dn: np.ndarray
    noise_profile_s4: np.ndarray
    noise_profile_o4: np.ndarray
    dynamic_range: float
    exposure_s: float
    condition_key: str
    path: str


class PhoneDarkFrameBank:
    """Lazy dark DNG sampler aligned to the clean crop's sensor coordinates.

    Every sampled residual is ``dark - black - DS`` in signed DN.  The crop
    position is supplied by the clean dataset rather than sampled independently
    so residual FPN keeps its physical sensor coordinate.
    """

    def __init__(self, manifest_path: str | Path, calibration_root: str | Path, *, cache_size: int = 2):
        self.manifest_path = Path(manifest_path)
        self.calibration_root = Path(calibration_root)
        self.records = _read_jsonl(self.manifest_path)
        if any(record.get("role") != "dark" for record in self.records):
            raise ValueError("Dark manifest contains a non-dark record")
        self.by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.conditions_by_iso: dict[int, list[str]] = defaultdict(list)
        for record in self.records:
            condition = str(record["condition_key"])
            self.by_condition[condition].append(record)
            iso = int(record["iso_exif"])
            if condition not in self.conditions_by_iso[iso]:
                self.conditions_by_iso[iso].append(condition)
        self.cache_size = max(0, int(cache_size))
        self._frame_cache: "OrderedDict[str, tuple[np.ndarray, PhoneDNGMetadata]]" = OrderedDict()
        self._ds_cache: dict[str, np.ndarray] = {}
        self._artifact_metadata: dict[str, dict[str, Any]] = {}

    @property
    def available_isos(self) -> list[int]:
        return sorted(self.conditions_by_iso)

    def condition_for_iso(self, iso_exif: int) -> str:
        conditions = self.conditions_by_iso.get(int(iso_exif), [])
        if not conditions:
            raise KeyError(f"No dark residual condition for EXIF ISO {iso_exif}; available {self.available_isos}")
        if len(conditions) != 1:
            raise ValueError(
                f"EXIF ISO {iso_exif} has multiple exposure/mode conditions {conditions}; "
                "select a complete condition key explicitly before training"
            )
        return conditions[0]

    def _load_artifact(self, condition_key: str) -> tuple[np.ndarray, dict[str, Any]]:
        if condition_key not in self._ds_cache:
            directory = self.calibration_root / condition_key
            ds_path = directory / "dark_shading_dn.npy"
            metadata_path = directory / "metadata.json"
            if not ds_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Missing DS artifact for {condition_key}; run tools/calibrate_phone_dark_shading.py first"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("condition_key") != condition_key:
                raise ValueError(f"{metadata_path}: condition key does not match directory")
            if metadata.get("units") != "linearized_DN" or not metadata.get("black_subtracted"):
                raise ValueError(f"{metadata_path}: unsupported DS domain")
            if metadata.get("domain") != "stored_pre_opcode_mosaic" or metadata.get("opcode_policy") != "skip":
                raise ValueError(f"{metadata_path}: DNG domain/policy mismatch")
            ds = np.load(ds_path, mmap_mode="r")
            if ds.ndim != 3 or ds.shape[0] != 4 or ds.dtype != np.float32:
                raise ValueError(f"{ds_path}: expected float32 [4,H,W], got {ds.shape} {ds.dtype}")
            self._ds_cache[condition_key] = ds
            self._artifact_metadata[condition_key] = metadata
        return self._ds_cache[condition_key], self._artifact_metadata[condition_key]

    def _load_frame(self, path: str) -> tuple[np.ndarray, PhoneDNGMetadata]:
        cached = self._frame_cache.get(path)
        if cached is not None:
            self._frame_cache.move_to_end(path)
            return cached
        packed, metadata = read_phone_dng_packed(path)
        if self.cache_size:
            self._frame_cache[path] = (packed, metadata)
            self._frame_cache.move_to_end(path)
            while len(self._frame_cache) > self.cache_size:
                self._frame_cache.popitem(last=False)
        return packed, metadata

    def sample_residual_crop(
        self,
        iso_exif: int,
        packed_top: int,
        packed_left: int,
        patch_size: int,
        rng: np.random.Generator,
    ) -> PhoneDarkSample:
        condition = self.condition_for_iso(int(iso_exif))
        records = self.by_condition[condition]
        record = records[int(rng.integers(len(records)))]
        packed, metadata = self._load_frame(str(record["path"]))
        ds, artifact_metadata = self._load_artifact(condition)
        if metadata.condition_key != condition:
            raise ValueError(f"{record['path']}: DNG metadata no longer matches its manifest condition")
        if packed.shape != ds.shape:
            raise ValueError(f"{record['path']}: packed dark shape {packed.shape} != DS shape {ds.shape}")
        if tuple(artifact_metadata.get("canonical_channel_order", [])) != ("R", "G1", "G2", "B"):
            raise ValueError(f"{condition}: unsupported canonical channel order in DS artifact")
        bottom, right = int(packed_top) + int(patch_size), int(packed_left) + int(patch_size)
        if packed_top < 0 or packed_left < 0 or bottom > packed.shape[1] or right > packed.shape[2]:
            raise ValueError("Requested dark residual crop lies outside DefaultCrop packed coordinates")
        black = np.asarray(metadata.black_level_canonical, dtype=np.float32).reshape(4, 1, 1)
        raw_crop = packed[:, packed_top:bottom, packed_left:right].astype(np.float32, copy=False)
        ds_crop = np.asarray(ds[:, packed_top:bottom, packed_left:right], dtype=np.float32)
        residual = raw_crop - black - ds_crop
        return PhoneDarkSample(
            residual_dn=residual.astype(np.float32, copy=False),
            noise_profile_s4=np.asarray(metadata.noise_profile_s4, dtype=np.float32),
            noise_profile_o4=np.asarray(metadata.noise_profile_o4, dtype=np.float32),
            dynamic_range=float(metadata.dynamic_range),
            exposure_s=float(metadata.exposure_s),
            condition_key=condition,
            path=str(record["path"]),
        )
