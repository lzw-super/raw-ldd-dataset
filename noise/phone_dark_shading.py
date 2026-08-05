"""Continuous ISO-conditioned dark-shading model for phone packed RAW.

The model mirrors the PMN/SID representation: every sensor position has a
linear ``k * ISO + b`` response, with independent low- and high-ISO fits.  The
input/output domain is black-subtracted linear DN in canonical packed
``[R,G1,G2,B]`` coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_DIRECTORY = "continuous_dark_shading"
_ARRAY_NAMES = {
    "low_k": "darkshading_lowISO_k.npy",
    "low_b": "darkshading_lowISO_b.npy",
    "high_k": "darkshading_highISO_k.npy",
    "high_b": "darkshading_highISO_b.npy",
}


class PhoneContinuousDarkShading:
    """Lazy evaluator for a calibrated piecewise-linear phone DS model."""

    def __init__(
        self,
        calibration_root: str | Path,
        *,
        model_directory: str = DEFAULT_MODEL_DIRECTORY,
    ) -> None:
        self.directory = Path(calibration_root) / model_directory
        metadata_path = self.directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing continuous dark-shading model at {metadata_path}; "
                "run tools/fit_phone_dark_shading.py first"
            )
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("model_type") != "piecewise_linear_iso":
            raise ValueError(f"{metadata_path}: unsupported dark-shading model type")
        if self.metadata.get("units") != "linearized_DN" or not self.metadata.get("black_subtracted"):
            raise ValueError(f"{metadata_path}: unsupported dark-shading units/domain")
        if self.metadata.get("domain") != "stored_pre_opcode_mosaic" or self.metadata.get("opcode_policy") != "skip":
            raise ValueError(f"{metadata_path}: DNG domain/policy mismatch")
        if tuple(self.metadata.get("canonical_channel_order", ())) != ("R", "G1", "G2", "B"):
            raise ValueError(f"{metadata_path}: unsupported canonical channel order")

        self.iso_breakpoint = float(self.metadata["iso_breakpoint"])
        self.minimum_iso = float(self.metadata["calibrated_iso_range"][0])
        self.maximum_iso = float(self.metadata["calibrated_iso_range"][1])
        ble_by_iso = self.metadata.get("ble_by_iso_dn")
        if not isinstance(ble_by_iso, dict) or len(ble_by_iso) < 2:
            raise ValueError(f"{metadata_path}: missing per-ISO BLE calibration")
        self.ble_by_iso = {
            int(key): np.asarray(value, dtype=np.float32)
            for key, value in ble_by_iso.items()
        }
        if any(value.shape != (4,) for value in self.ble_by_iso.values()):
            raise ValueError(f"{metadata_path}: every BLE entry must contain four canonical channels")
        self.shape = tuple(int(value) for value in self.metadata["packed_shape_chw"])
        if len(self.shape) != 3 or self.shape[0] != 4:
            raise ValueError(f"{metadata_path}: expected packed [4,H,W] shape, got {self.shape}")

        self.arrays: dict[str, np.ndarray] = {}
        for key, filename in _ARRAY_NAMES.items():
            path = self.directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing continuous dark-shading coefficient: {path}")
            array = np.load(path, mmap_mode="r")
            if tuple(array.shape) != self.shape or array.dtype != np.float32:
                raise ValueError(f"{path}: expected float32 {self.shape}, got {array.shape} {array.dtype}")
            self.arrays[key] = array

    def _coefficients(self, iso_exif: float) -> tuple[np.ndarray, np.ndarray, str]:
        iso = float(iso_exif)
        if not np.isfinite(iso) or iso <= 0:
            raise ValueError(f"ISO must be finite and positive, got {iso_exif}")
        branch = "low" if iso <= self.iso_breakpoint else "high"
        return self.arrays[f"{branch}_k"], self.arrays[f"{branch}_b"], branch

    def _ble(self, iso_exif: float) -> tuple[np.ndarray, list[int]]:
        """Continuously interpolate the fitted global offset in log2 ISO."""
        iso = float(iso_exif)
        keys = np.asarray(sorted(self.ble_by_iso), dtype=np.float64)
        if iso <= keys[0]:
            key = int(keys[0])
            return self.ble_by_iso[key], [key, key]
        if iso >= keys[-1]:
            key = int(keys[-1])
            return self.ble_by_iso[key], [key, key]
        upper_index = int(np.searchsorted(keys, iso, side="right"))
        lower, upper = int(keys[upper_index - 1]), int(keys[upper_index])
        alpha = (np.log2(iso) - np.log2(lower)) / (np.log2(upper) - np.log2(lower))
        value = (1.0 - alpha) * self.ble_by_iso[lower] + alpha * self.ble_by_iso[upper]
        return value.astype(np.float32, copy=False), [lower, upper]

    def crop(
        self,
        iso_exif: float,
        packed_top: int,
        packed_left: int,
        patch_size: int,
    ) -> np.ndarray:
        """Evaluate ``DS(ISO)`` for one square crop without materialising a full map."""
        top, left, size = int(packed_top), int(packed_left), int(patch_size)
        bottom, right = top + size, left + size
        if top < 0 or left < 0 or size <= 0 or bottom > self.shape[1] or right > self.shape[2]:
            raise ValueError("Requested continuous dark-shading crop lies outside packed coordinates")
        k, b, _ = self._coefficients(iso_exif)
        ble, _ = self._ble(iso_exif)
        return (
            np.asarray(k[:, top:bottom, left:right], dtype=np.float32) * float(iso_exif)
            + np.asarray(b[:, top:bottom, left:right], dtype=np.float32)
            + ble.reshape(4, 1, 1)
        ).astype(np.float32, copy=False)

    def full(self, iso_exif: float, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
        """Evaluate a full packed DS map for inference/evaluation."""
        if expected_shape is not None and tuple(expected_shape) != self.shape:
            raise ValueError(f"Continuous DS shape {self.shape} != requested packed shape {expected_shape}")
        k, b, _ = self._coefficients(iso_exif)
        ble, _ = self._ble(iso_exif)
        return (
            np.asarray(k, dtype=np.float32) * float(iso_exif)
            + np.asarray(b, dtype=np.float32)
            + ble.reshape(4, 1, 1)
        ).astype(np.float32, copy=False)

    def describe(self, iso_exif: float) -> dict[str, Any]:
        _, _, branch = self._coefficients(iso_exif)
        _, ble_interpolation_isos = self._ble(iso_exif)
        return {
            "mode": "continuous_iso_fit",
            "branch": branch,
            "requested_iso_exif": int(iso_exif),
            "iso_breakpoint": self.iso_breakpoint,
            "calibrated_iso_range": [self.minimum_iso, self.maximum_iso],
            "model_directory": str(self.directory.resolve()),
            "ble_interpolation_isos": ble_interpolation_isos,
        }


__all__ = ["DEFAULT_MODEL_DIRECTORY", "PhoneContinuousDarkShading"]
