"""Synthetic training dataset for the current sensor-specific phone DNGs."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from noise.phone_dark_frame_bank import PhoneDarkFrameBank
from utils.phone_dng import PhoneDNGMetadata, read_phone_dng_packed


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"Clean manifest is empty: {path}")
    return records


class PhoneSyntheticTrainDataset(Dataset):
    """Return pseudo-clean targets plus aligned, condition-matched dark noise.

    The 10 s clean DNGs are not clean masters.  ``target_ds_policy='none'`` is
    therefore intentional: 1/30 s DS must not be subtracted from a 10 s frame.
    This class is for the documented pseudo-clean/smoke-training stage only;
    a future burst-master pipeline should expose an explicit clean DS artifact.
    """

    def __init__(
        self,
        clean_manifest: str | Path,
        dark_manifest: str | Path,
        calibration_root: str | Path,
        *,
        patch_size: int = 512,
        crops_per_image: int = 2,
        ratios: Sequence[float] | None = None,
        ratio: float | None = None,
        cache_size: int = 4,
        dark_cache_size: int = 2,
        max_target_saturation_fraction: float = 0.01,
        max_crop_attempts: int = 32,
        spatial_augmentation: bool = False,
        seed: int = 1,
        exposure_ratio_tolerance: float = 0.02,
        dark_exposure_policy: str = "strict_match",
    ):
        self.records = _read_jsonl(clean_manifest)
        if any(record.get("role") != "clean_source" for record in self.records):
            raise ValueError("Clean manifest contains a non-clean_source record")
        self.patch_size = int(patch_size)
        self.crops_per_image = int(crops_per_image)
        if ratios is not None and ratio is not None:
            raise ValueError("Pass either ratios or the legacy single ratio, not both")
        selected_ratios = ratios if ratios is not None else (300.0 if ratio is None else ratio,)
        self.ratios = tuple(float(value) for value in selected_ratios)
        if self.patch_size <= 0 or self.crops_per_image <= 0 or not self.ratios or any(value <= 0 for value in self.ratios):
            raise ValueError("patch_size, crops_per_image and every ratio must be positive")
        self.cache_size = max(0, int(cache_size))
        self.dark_cache_size = max(0, int(dark_cache_size))
        self._dark_manifest = dark_manifest
        self._calibration_root = calibration_root
        self.max_target_saturation_fraction = float(max_target_saturation_fraction)
        self.max_crop_attempts = max(1, int(max_crop_attempts))
        self.spatial_augmentation = bool(spatial_augmentation)
        if self.spatial_augmentation:
            raise ValueError(
                "Spatial augmentation is disabled for this sensor-coordinate residual model; "
                "flipping moves FPN relative to its physical pixel position"
            )
        self.seed = int(seed)
        self.exposure_ratio_tolerance = float(exposure_ratio_tolerance)
        if dark_exposure_policy not in {"strict_match", "approximate_reuse_1_30s"}:
            raise ValueError("dark_exposure_policy must be 'strict_match' or 'approximate_reuse_1_30s'")
        self.dark_exposure_policy = str(dark_exposure_policy)
        self._clean_cache: "OrderedDict[str, tuple[np.ndarray, PhoneDNGMetadata]]" = OrderedDict()
        self._dark_bank: PhoneDarkFrameBank | None = None
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_image

    def _rng_for_worker(self) -> np.random.Generator:
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            seed = worker.seed if worker is not None else self.seed
            self._rng = np.random.default_rng(int(seed) % (2**63 - 1))
        return self._rng

    def _dark(self) -> PhoneDarkFrameBank:
        if self._dark_bank is None:
            self._dark_bank = PhoneDarkFrameBank(
                self._dark_manifest, self._calibration_root, cache_size=self.dark_cache_size
            )
        return self._dark_bank

    @property
    def _dark_manifest(self) -> str:
        return self.__dark_manifest

    @_dark_manifest.setter
    def _dark_manifest(self, value: str | Path) -> None:
        self.__dark_manifest = str(value)

    @property
    def _calibration_root(self) -> str:
        return self.__calibration_root

    @_calibration_root.setter
    def _calibration_root(self, value: str | Path) -> None:
        self.__calibration_root = str(value)

    def _load_clean(self, path: str) -> tuple[np.ndarray, PhoneDNGMetadata]:
        cached = self._clean_cache.get(path)
        if cached is not None:
            self._clean_cache.move_to_end(path)
            return cached
        packed, metadata = read_phone_dng_packed(path)
        if self.cache_size:
            self._clean_cache[path] = (packed, metadata)
            self._clean_cache.move_to_end(path)
            while len(self._clean_cache) > self.cache_size:
                self._clean_cache.popitem(last=False)
        return packed, metadata

    def _sample_clean_crop(
        self, packed: np.ndarray, metadata: PhoneDNGMetadata, rng: np.random.Generator
    ) -> tuple[np.ndarray, int, int, float]:
        _, height, width = packed.shape
        if self.patch_size > height or self.patch_size > width:
            raise ValueError(f"Patch {self.patch_size} exceeds clean packed shape {(height, width)}")
        last_saturated_fraction = 1.0
        for _ in range(self.max_crop_attempts):
            top = int(rng.integers(height - self.patch_size + 1))
            left = int(rng.integers(width - self.patch_size + 1))
            crop = packed[:, top : top + self.patch_size, left : left + self.patch_size]
            last_saturated_fraction = float((crop >= metadata.white_level).mean())
            if last_saturated_fraction <= self.max_target_saturation_fraction:
                black = np.asarray(metadata.black_level_canonical, dtype=np.float32).reshape(4, 1, 1)
                clean = (crop.astype(np.float32, copy=False) - black) / float(metadata.dynamic_range)
                return clean.astype(np.float32, copy=False), top, left, last_saturated_fraction
        raise RuntimeError(
            f"{metadata.path}: could not sample a patch below saturation threshold "
            f"{self.max_target_saturation_fraction}; last fraction={last_saturated_fraction:.4f}"
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index // self.crops_per_image]
        rng = self._rng_for_worker()
        packed, metadata = self._load_clean(str(record["path"]))
        if int(record["iso_exif"]) != metadata.iso_exif:
            raise ValueError(f"{record['path']}: EXIF ISO changed since manifest generation")
        clean, top, left, saturated_fraction = self._sample_clean_crop(packed, metadata, rng)
        dark = self._dark().sample_residual_crop(metadata.iso_exif, top, left, self.patch_size, rng)
        ratio = float(rng.choice(self.ratios))
        measured_ratio = metadata.exposure_s / dark.exposure_s
        relative_ratio_error = abs(measured_ratio - ratio) / ratio
        if self.dark_exposure_policy == "strict_match" and relative_ratio_error > self.exposure_ratio_tolerance:
            raise ValueError(
                f"{metadata.path}: requested ratio {ratio:g} is incompatible with matched dark exposure; "
                f"metadata ratio={measured_ratio:.6f}"
            )
        if self.dark_exposure_policy == "approximate_reuse_1_30s":
            expected_dark_exposure = 1.0 / 30.0
            relative_dark_exposure_error = abs(dark.exposure_s - expected_dark_exposure) / expected_dark_exposure
            if relative_dark_exposure_error > self.exposure_ratio_tolerance:
                raise ValueError(
                    f"{dark.path}: approximate_reuse_1_30s requires a 1/30 s dark source; "
                    f"got {dark.exposure_s:.9f} s"
                )
        simulated_short_exposure_s = metadata.exposure_s / ratio
        return {
            "clean": torch.from_numpy(clean),
            "dark_residual_dn": torch.from_numpy(dark.residual_dn),
            "noise_profile_s4": torch.from_numpy(dark.noise_profile_s4),
            "noise_profile_o4": torch.from_numpy(dark.noise_profile_o4),
            "dark_dynamic_range": torch.tensor(dark.dynamic_range, dtype=torch.float32),
            "ratio": torch.tensor(ratio, dtype=torch.float32),
            "iso_exif": torch.tensor(metadata.iso_exif, dtype=torch.int32),
            "clean_source_exposure_s": torch.tensor(metadata.exposure_s, dtype=torch.float32),
            "simulated_short_exposure_s": torch.tensor(simulated_short_exposure_s, dtype=torch.float32),
            "dark_source_exposure_s": torch.tensor(dark.exposure_s, dtype=torch.float32),
            "clean_path": metadata.path,
            "dark_path": dark.path,
            "dark_condition_key": dark.condition_key,
            "packed_top": torch.tensor(top, dtype=torch.int32),
            "packed_left": torch.tensor(left, dtype=torch.int32),
            "target_saturation_fraction": torch.tensor(saturated_fraction, dtype=torch.float32),
            "target_protocol": "pseudo_clean_no_1_30s_ds_subtraction",
            "dark_exposure_policy": self.dark_exposure_policy,
        }
