"""Sampling of LLD Sony A7S II dark-frame residuals.

The training data in ``biasframe_et_1_30`` contains Bayer mosaics in MATLAB
files.  This module deliberately keeps the mosaics in DN units until a random
packed-RAW crop is requested: clipping a dark residual before synthesis would
discard the negative part of the real sensor-noise distribution.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import pickle
import re
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat


BLACK_LEVEL = 512.0


def pack_bayer(mosaic: np.ndarray) -> np.ndarray:
    """Pack an RGGB mosaic to ``[4, H/2, W/2]`` without normalising it."""
    if mosaic.ndim != 2 or mosaic.shape[0] % 2 or mosaic.shape[1] % 2:
        raise ValueError("Expected an even-sized two-dimensional Bayer mosaic")
    return np.stack(
        [mosaic[0::2, 0::2], mosaic[0::2, 1::2], mosaic[1::2, 0::2], mosaic[1::2, 1::2]], axis=0
    )


class PMNDarkShading:
    """PMN dark-shading model used by the official SID evaluator.

    The arrays are memory-mapped and only the requested crop is materialised.
    The returned shading excludes the camera black level, exactly as in
    ``datasets.real_dataset.SIDEvalDataset``.
    """

    def __init__(self, resource_dir: str | Path):
        root = Path(resource_dir)
        required = [
            "darkshading_BLE.pkl",
            "darkshading_highISO_k.npy",
            "darkshading_highISO_b.npy",
            "darkshading_lowISO_k.npy",
            "darkshading_lowISO_b.npy",
        ]
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing PMN dark-shading resources in {root}: {missing}")
        with (root / "darkshading_BLE.pkl").open("rb") as f:
            self.ble = pickle.load(f)
        self.high_k = np.load(root / "darkshading_highISO_k.npy", mmap_mode="r")
        self.high_b = np.load(root / "darkshading_highISO_b.npy", mmap_mode="r")
        self.low_k = np.load(root / "darkshading_lowISO_k.npy", mmap_mode="r")
        self.low_b = np.load(root / "darkshading_lowISO_b.npy", mmap_mode="r")

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.low_k.shape)

    def _ble_for_iso(self, iso: int) -> float:
        if iso in self.ble:
            return float(np.asarray(self.ble[iso]).squeeze())
        keys = np.asarray(list(self.ble.keys()), dtype=np.float64)
        nearest = int(keys[np.argmin(np.abs(np.log2(keys) - np.log2(float(iso))))])
        return float(np.asarray(self.ble[nearest]).squeeze())

    def mosaic_crop(self, iso: int, packed_top: int, packed_left: int, patch_size: int) -> np.ndarray:
        """Return a dark-shading crop in mosaic DN units."""
        mt, ml = packed_top * 2, packed_left * 2
        mh, mw = patch_size * 2, patch_size * 2
        if mt < 0 or ml < 0 or mt + mh > self.shape[0] or ml + mw > self.shape[1]:
            raise ValueError("Dark-shading crop lies outside the sensor area")
        if iso <= 1600:
            k, b = self.low_k, self.low_b
        else:
            k, b = self.high_k, self.high_b
        return (np.asarray(k[mt : mt + mh, ml : ml + mw], dtype=np.float32) * float(iso)
                + np.asarray(b[mt : mt + mh, ml : ml + mw], dtype=np.float32)
                + self._ble_for_iso(int(iso))).astype(np.float32, copy=False)


class DarkFrameBank:
    """Lazy, bounded-memory LLD MAT dark-frame sampler.

    ``dark_shading_mode='pmn'`` is the paper-fair mode: it subtracts the PMN
    shading and black level before adding the sampled residual to clean data.
    ``mean`` is intentionally not exposed as a main training option because a
    per-ISO mean must be prepared separately and is not equivalent to PMN's
    evaluation protocol.
    """

    def __init__(
        self,
        dark_root: str | Path,
        pmn_resource_dir: str | Path,
        cache_size: int = 2,
        dark_shading_mode: str = "pmn",
    ):
        self.dark_root = Path(dark_root)
        if not self.dark_root.is_dir():
            raise FileNotFoundError(f"Dark-frame root does not exist: {self.dark_root}")
        if dark_shading_mode != "pmn":
            raise ValueError("Only 'pmn' is implemented for the paper-fair training pipeline")
        self.dark_shading_mode = dark_shading_mode
        self.cache_size = max(0, int(cache_size))
        self.pmn = PMNDarkShading(pmn_resource_dir)
        self.frames: Dict[int, List[Path]] = self._discover_frames()
        if not self.frames:
            raise RuntimeError(f"No files matching ISO*/**.mat found below {self.dark_root}")
        self._cache: "OrderedDict[Path, np.ndarray]" = OrderedDict()

    def _discover_frames(self) -> Dict[int, List[Path]]:
        frames: Dict[int, List[Path]] = {}
        for directory in sorted(self.dark_root.glob("ISO*")):
            if not directory.is_dir():
                continue
            match = re.fullmatch(r"ISO(\d+)", directory.name)
            if not match:
                continue
            paths = sorted(directory.glob("*.mat"))
            if paths:
                frames[int(match.group(1))] = paths
        return frames

    @property
    def available_isos(self) -> List[int]:
        return sorted(self.frames)

    def map_iso(self, requested_iso: int) -> int:
        requested_iso = int(requested_iso)
        if requested_iso in self.frames:
            return requested_iso
        candidates = np.asarray(self.available_isos, dtype=np.float64)
        return int(candidates[np.argmin(np.abs(np.log2(candidates) - np.log2(float(requested_iso))))])

    def validate_metadata(self) -> Dict[str, object]:
        """Read MAT metadata only and verify it agrees with each ISO folder.

        ``variable_names`` prevents SciPy from materialising ``Inoisy_crop``;
        this checks all locally downloaded dark frames without creating a second
        multi-gigabyte cache.
        """
        checked, failures, exposures = 0, [], []
        for folder_iso, paths in self.frames.items():
            for path in paths:
                mat = loadmat(path, variable_names=["ISO", "expo"])
                if "ISO" not in mat or "expo" not in mat:
                    failures.append({"path": str(path), "reason": "missing ISO or expo variable"})
                    continue
                actual_iso = int(np.asarray(mat["ISO"]).squeeze())
                exposure = float(np.asarray(mat["expo"]).squeeze())
                checked += 1
                exposures.append(exposure)
                if actual_iso != folder_iso:
                    failures.append(
                        {"path": str(path), "reason": "ISO mismatch", "folder_iso": folder_iso, "mat_iso": actual_iso}
                    )
        return {
            "checked": checked,
            "failures": failures,
            "exposure_seconds": {
                "min": min(exposures) if exposures else None,
                "max": max(exposures) if exposures else None,
                "mean": float(np.mean(exposures)) if exposures else None,
            },
        }

    def _load_mosaic(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        mat = loadmat(path, variable_names=["Inoisy_crop"])
        if "Inoisy_crop" not in mat:
            raise KeyError(f"{path} has no Inoisy_crop array")
        mosaic = np.asarray(mat["Inoisy_crop"])
        if mosaic.ndim != 2:
            raise ValueError(f"Unexpected dark-frame shape in {path}: {mosaic.shape}")
        if self.cache_size:
            self._cache[path] = mosaic
            self._cache.move_to_end(path)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return mosaic

    def sample_residual_patch(
        self, requested_iso: int, patch_size: int, rng: np.random.Generator | None = None
    ) -> Tuple[np.ndarray, int, str]:
        """Sample signed packed residual ``[4, patch, patch]`` in DN units."""
        rng = rng if rng is not None else np.random.default_rng()
        matched_iso = self.map_iso(int(requested_iso))
        paths = self.frames[matched_iso]
        path = paths[int(rng.integers(len(paths)))]
        mosaic = self._load_mosaic(path)
        packed_h, packed_w = mosaic.shape[0] // 2, mosaic.shape[1] // 2
        if patch_size > packed_h or patch_size > packed_w:
            raise ValueError(
                f"Requested packed patch {patch_size} exceeds dark-frame size {(packed_h, packed_w)}"
            )
        top = int(rng.integers(packed_h - patch_size + 1))
        left = int(rng.integers(packed_w - patch_size + 1))
        mt, ml = top * 2, left * 2
        span = patch_size * 2
        dark_crop = mosaic[mt : mt + span, ml : ml + span].astype(np.float32, copy=False)
        shading_crop = self.pmn.mosaic_crop(matched_iso, top, left, patch_size)
        residual = pack_bayer(dark_crop - BLACK_LEVEL - shading_crop).astype(np.float32, copy=False)
        return residual, matched_iso, str(path)
