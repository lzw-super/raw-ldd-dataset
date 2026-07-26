"""Leakage-safe SID clean-patch dataset for synthetic-noise training."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import re
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from noise.dark_frame_bank import DarkFrameBank
from utils.utils import get_ISO_ExposureTime


BLACK_LEVEL = 512.0
WHITE_LEVEL = 16383.0


def _long_name_from_patch_name(name: str) -> str:
    match = re.fullmatch(r"(.+)_s\d+\.ARW\.npz", name)
    if match is None:
        raise ValueError(f"Unexpected SID packed-patch name: {name}")
    return f"{match.group(1)}.ARW"


def _read_sid_long_names(pair_list: str | Path) -> set[str]:
    long_names: set[str] = set()
    for line_number, line in enumerate(Path(pair_list).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 3 or not fields[2].startswith("ISO"):
            raise ValueError(f"Malformed pair-list row {line_number}: {line}")
        long_names.add(Path(fields[1]).name)
    return long_names


def build_sid_patch_manifest(
    patch_dir: str | Path,
    pair_list: str | Path,
    output_path: str | Path | None = None,
    sid_long_dir: str | Path | None = None,
) -> Dict[str, object]:
    """Create a JSON manifest from prepacked SID long-exposure patches.

    The official local preprocessing contains patches for train/test/validation.
    This function intentionally keeps only prefix ``0`` files, and fails if
    their metadata cannot be resolved through ``Sony_train_list.txt``.
    """
    patch_dir = Path(patch_dir)
    pair_list = Path(pair_list)
    sid_long_dir = Path(sid_long_dir) if sid_long_dir is not None else pair_list.parent / "Sony" / "long"
    paired_long_names = _read_sid_long_names(pair_list)
    records: List[Dict[str, object]] = []
    ignored_nontrain = 0
    missing_pair: List[str] = []
    training_long_names: set[str] = set()
    for patch in sorted(patch_dir.glob("*.npz")):
        long_name = _long_name_from_patch_name(patch.name)
        scene_id = long_name[:5]
        if not scene_id.startswith("0"):
            ignored_nontrain += 1
            continue
        if long_name not in paired_long_names:
            missing_pair.append(long_name)
            continue
        training_long_names.add(long_name)
    if missing_pair:
        raise RuntimeError(f"{len(missing_pair)} training patches lack a SID pair-list entry, e.g. {missing_pair[:3]}")

    # Some SID short captures of one scene use a different ISO than the long
    # capture.  The long RAW's EXIF is the unambiguous clean-scene ISO and is
    # therefore used for K and dark-frame matching.
    iso_by_long: Dict[str, int] = {}
    missing_long: List[str] = []
    for long_name in sorted(training_long_names):
        long_path = sid_long_dir / long_name
        if not long_path.is_file():
            missing_long.append(str(long_path))
            continue
        iso_by_long[long_name] = int(get_ISO_ExposureTime(str(long_path))["ISO"])
    if missing_long:
        raise RuntimeError(f"Missing {len(missing_long)} SID long RAW files, e.g. {missing_long[:3]}")

    for patch in sorted(patch_dir.glob("*.npz")):
        long_name = _long_name_from_patch_name(patch.name)
        scene_id = long_name[:5]
        if not scene_id.startswith("0"):
            continue
        records.append(
            {
                "scene_id": scene_id,
                "long_name": long_name,
                "iso": iso_by_long[long_name],
                "patch": str(patch.resolve()),
            }
        )
    if not records:
        raise RuntimeError(f"No prefix-0 training patches found in {patch_dir}")
    manifest: Dict[str, object] = {
        "format_version": 1,
        "source_patch_dir": str(patch_dir.resolve()),
        "source_pair_list": str(pair_list.resolve()),
        "source_long_dir": str(sid_long_dir.resolve()),
        "iso_source": "EXIF ISO of each SID long RAW (not the potentially heterogeneous short-pair list)",
        "selection": "SID Sony long-exposure packed patches whose scene ID begins with '0' only",
        "records": records,
        "summary": {
            "records": len(records),
            "scenes": len({record["scene_id"] for record in records}),
            "ignored_nontrain_patches": ignored_nontrain,
            "iso_values": sorted({int(record["iso"]) for record in records}),
        },
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


class SIDSyntheticTrainDataset(Dataset):
    """Returns clean packed RAW and an independent real dark-residual crop."""

    def __init__(
        self,
        manifest_path: str | Path,
        dark_root: str | Path,
        pmn_resource_dir: str | Path,
        *,
        patch_size: int = 512,
        crops_per_image: int = 8,
        ratios: tuple[int, ...] = (100, 250, 300),
        cache_size: int = 32,
        dark_cache_size: int = 2,
        augment: bool = True,
        seed: int = 1,
    ):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.records = list(manifest["records"])
        if any(not str(record["scene_id"]).startswith("0") for record in self.records):
            raise ValueError("Training manifest contains non-train SID scenes")
        self.patch_size = int(patch_size)
        self.crops_per_image = int(crops_per_image)
        self.ratios = tuple(int(value) for value in ratios)
        self.cache_size = max(0, int(cache_size))
        self.augment = bool(augment)
        self.seed = int(seed)
        self._clean_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._dark_root = str(dark_root)
        self._pmn_resource_dir = str(pmn_resource_dir)
        self._dark_cache_size = int(dark_cache_size)
        self._dark_bank: DarkFrameBank | None = None
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_image

    def _get_rng(self) -> np.random.Generator:
        # PyTorch gives every worker a different initial torch seed.  Creating
        # this generator lazily keeps NumPy augmentation independent per worker.
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            offset = worker_info.seed if worker_info is not None else self.seed
            self._rng = np.random.default_rng(int(offset) % (2**63 - 1))
        return self._rng

    def _get_dark_bank(self) -> DarkFrameBank:
        if self._dark_bank is None:
            self._dark_bank = DarkFrameBank(
                self._dark_root, self._pmn_resource_dir, cache_size=self._dark_cache_size, dark_shading_mode="pmn"
            )
        return self._dark_bank

    def _load_clean(self, path: str) -> np.ndarray:
        cached = self._clean_cache.get(path)
        if cached is not None:
            self._clean_cache.move_to_end(path)
            return cached
        with np.load(path, allow_pickle=False) as archive:
            if "im" not in archive:
                raise KeyError(f"{path} contains no 'im' array")
            packed = np.asarray(archive["im"], dtype=np.float32)
        if packed.ndim != 3 or packed.shape[0] != 4:
            raise ValueError(f"Expected [4,H,W] packed RAW in {path}, got {packed.shape}")
        clean = np.clip((packed - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL), 0.0, 1.0)
        if self.cache_size:
            self._clean_cache[path] = clean
            self._clean_cache.move_to_end(path)
            while len(self._clean_cache) > self.cache_size:
                self._clean_cache.popitem(last=False)
        return clean

    def _random_clean_crop(self, clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        _, height, width = clean.shape
        if self.patch_size > height or self.patch_size > width:
            raise ValueError(f"Clean patch {self.patch_size} exceeds stored patch size {(height, width)}")
        top = int(rng.integers(height - self.patch_size + 1))
        left = int(rng.integers(width - self.patch_size + 1))
        return clean[:, top : top + self.patch_size, left : left + self.patch_size].copy()

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index // self.crops_per_image]
        rng = self._get_rng()
        clean = self._random_clean_crop(self._load_clean(str(record["patch"])), rng)
        iso = int(record["iso"])
        ratio = int(rng.choice(self.ratios))
        dark, matched_dark_iso, dark_path = self._get_dark_bank().sample_residual_patch(iso, self.patch_size, rng)
        if self.augment:
            if bool(rng.integers(2)):
                clean = clean[:, :, ::-1].copy()
                dark = dark[:, :, ::-1].copy()
            if bool(rng.integers(2)):
                clean = clean[:, ::-1, :].copy()
                dark = dark[:, ::-1, :].copy()
        return {
            "clean": torch.from_numpy(clean),
            "dark_residual_dn": torch.from_numpy(dark),
            "iso": torch.tensor(iso, dtype=torch.float32),
            "ratio": torch.tensor(ratio, dtype=torch.float32),
            "matched_dark_iso": torch.tensor(matched_dark_iso, dtype=torch.int32),
            "scene_id": str(record["scene_id"]),
            "clean_path": str(record["patch"]),
            "dark_path": dark_path,
        }
