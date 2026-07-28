"""DNG reader used by the sensor-specific phone training pipeline.

The SID code in this repository relies on camera-specific ``rawpy`` defaults.
That is not sufficient for a DNG training target: DNG's BlackLevel,
DefaultCrop and NoiseProfile tags are part of the training-domain definition.
This module deliberately reads the stored, pre-opcode CFA mosaic with
``tifffile`` and exposes a fixed canonical packed order:

    [R, G1 (same row as R), G2, B]

For the current HONOR MEY-AN00 GBRG files this is [R_BL, G_BR, G_TL, B_TR],
which is compatible with the [R, G1, G2, B] order used by the SID checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile


_CFA_PHOTOMETRIC = 32803
_TAG_BLACK_LEVEL_DELTA_H = 50715
_TAG_BLACK_LEVEL_DELTA_V = 50716
_TAG_LINEARIZATION_TABLE = 50712


def _tag(page: tifffile.TiffPage, name: str, *, required: bool = True) -> tifffile.TiffTag | None:
    value = page.tags.get(name)
    if value is None and required:
        raise ValueError(f"DNG RAW IFD is missing required tag {name}")
    return value


def _tag_value(page: tifffile.TiffPage, name: str, *, required: bool = True) -> Any:
    tag = _tag(page, name, required=required)
    return None if tag is None else tag.value


def _numbers(tag: tifffile.TiffTag | None) -> list[float]:
    """Return TIFF numeric values, expanding RATIONAL pairs to floats."""
    if tag is None:
        return []
    raw = tag.value
    if isinstance(raw, (bytes, bytearray)):
        return [float(value) for value in raw]
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, (tuple, list)):
        return [float(raw)]
    values = list(raw)
    # tifffile exposes RATIONAL values as numerator/denominator pairs.
    if int(tag.dtype) in {5, 10} and len(values) == 2 * int(tag.count):
        result = []
        for numerator, denominator in zip(values[0::2], values[1::2]):
            if float(denominator) == 0:
                raise ValueError(f"Invalid zero denominator in TIFF tag {tag.name}")
            result.append(float(numerator) / float(denominator))
        return result
    return [float(value) for value in values]


def _integer_values(value: Any) -> list[int]:
    if isinstance(value, (bytes, bytearray)):
        return [int(item) for item in value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return [int(item) for item in value]
    return [int(value)]


def _ascii_tag(page: tifffile.TiffPage, name: str) -> str | None:
    value = _tag_value(page, name, required=False)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def _find_raw_page(tiff: tifffile.TiffFile) -> tifffile.TiffPage:
    candidates: list[tifffile.TiffPage] = []
    for page in tiff.pages:
        photometric = page.photometric
        photometric_code = int(photometric) if photometric is not None else None
        if page.ndim == 2 and photometric_code == _CFA_PHOTOMETRIC:
            candidates.append(page)
    if not candidates:
        raise ValueError("No two-dimensional CFA RAW IFD was found in DNG")
    # A full-resolution page wins over a preview if a non-standard file has two.
    return max(candidates, key=lambda item: int(item.shape[0]) * int(item.shape[1]))


def _as_xy(value: Iterable[int | float], name: str) -> tuple[int, int]:
    values = list(value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values, got {values}")
    return int(values[0]), int(values[1])


def _canonical_position_indices(color_pattern: np.ndarray) -> tuple[int, int, int, int]:
    """Derive [R, G1, G2, B] positions from a 2x2 RGB Bayer pattern.

    G1 is selected as the green in R's row.  This yields the expected
    MEY-AN00 GBRG permutation [2, 3, 0, 1] and normal RGGB [0, 1, 2, 3].
    """
    if color_pattern.shape != (2, 2):
        raise ValueError(f"Only 2x2 Bayer CFA is supported, got {color_pattern.shape}")
    locations = {color: np.argwhere(color_pattern == color) for color in (0, 1, 2)}
    if len(locations[0]) != 1 or len(locations[1]) != 2 or len(locations[2]) != 1:
        raise ValueError(f"Expected one R, two G, one B in CFA; got {color_pattern.tolist()}")
    red_row, red_col = (int(value) for value in locations[0][0])
    green_locations = [(int(row), int(col)) for row, col in locations[1]]
    same_row = [location for location in green_locations if location[0] == red_row and location[1] != red_col]
    if len(same_row) != 1:
        raise ValueError(f"Could not choose canonical G1 from CFA {color_pattern.tolist()}")
    g1 = same_row[0]
    g2 = next(location for location in green_locations if location != g1)
    blue = tuple(int(value) for value in locations[2][0])
    return tuple(row * 2 + col for row, col in ((red_row, red_col), g1, g2, blue))


@dataclass(frozen=True)
class PhoneDNGMetadata:
    """Metadata defining the stored pre-opcode RAW training domain."""

    path: str
    make: str | None
    model: str | None
    unique_camera_model: str | None
    datetime: str | None
    iso_exif: int
    exposure_s: float
    raw_shape_hw: tuple[int, int]
    active_area_tlbr: tuple[int, int, int, int]
    default_crop_xywh: tuple[int, int, int, int]
    cfa_pattern_active: tuple[int, int, int, int]
    cfa_after_crop: tuple[int, int, int, int]
    cfa_name: str
    canonical_position_indices: tuple[int, int, int, int]
    black_level_position: tuple[float, float, float, float]
    black_level_canonical: tuple[float, float, float, float]
    white_level: float
    dynamic_range: float
    noise_profile_rgb: tuple[tuple[float, float], ...]
    noise_profile_s4: tuple[float, float, float, float]
    noise_profile_o4: tuple[float, float, float, float]
    opcode_policy: str = "skip"
    domain: str = "stored_pre_opcode_mosaic"

    @property
    def packed_shape_hw(self) -> tuple[int, int]:
        return self.default_crop_xywh[3] // 2, self.default_crop_xywh[2] // 2

    @property
    def condition_key(self) -> str:
        exposure_ns = int(round(self.exposure_s * 1_000_000_000.0))
        camera = (self.unique_camera_model or self.model or "unknown").replace("/", "_").replace(" ", "_")
        return (
            f"{camera}_{self.raw_shape_hw[0]}x{self.raw_shape_hw[1]}_{self.cfa_name}_"
            f"preopcode_iso-{self.iso_exif}_expns-{exposure_ns}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "make": self.make,
            "model": self.model,
            "unique_camera_model": self.unique_camera_model,
            "datetime": self.datetime,
            "iso_exif": self.iso_exif,
            "exposure_s": self.exposure_s,
            "raw_shape_hw": list(self.raw_shape_hw),
            "active_area_tlbr": list(self.active_area_tlbr),
            "default_crop_xywh": list(self.default_crop_xywh),
            "cfa_pattern_active": list(self.cfa_pattern_active),
            "cfa_after_crop": list(self.cfa_after_crop),
            "cfa": self.cfa_name,
            "canonical_channel_order": ["R", "G1", "G2", "B"],
            "canonical_position_indices": list(self.canonical_position_indices),
            "black_level_2x2_position": list(self.black_level_position),
            "black_level_4": list(self.black_level_canonical),
            "white_level": self.white_level,
            "dynamic_range": self.dynamic_range,
            "noise_profile_rgb": [list(pair) for pair in self.noise_profile_rgb],
            "noise_profile_s4": list(self.noise_profile_s4),
            "noise_profile_o4": list(self.noise_profile_o4),
            "condition_key": self.condition_key,
            "opcode_policy": self.opcode_policy,
            "domain": self.domain,
        }


def _metadata_from_page(path: Path, page: tifffile.TiffPage) -> PhoneDNGMetadata:
    if _tag(page, "LinearizationTable", required=False) is not None:
        raise ValueError(f"{path}: LinearizationTable is not supported by this stored-domain reader")
    if page.tags.get(_TAG_BLACK_LEVEL_DELTA_H) is not None or page.tags.get(_TAG_BLACK_LEVEL_DELTA_V) is not None:
        raise ValueError(f"{path}: BlackLevelDeltaH/V is not supported by this reader")

    raw_height, raw_width = (int(value) for value in page.shape)
    active_values = _integer_values(_tag_value(page, "ActiveArea", required=False) or (0, 0, raw_height, raw_width))
    if len(active_values) != 4:
        raise ValueError(f"{path}: ActiveArea must be [top,left,bottom,right]")
    active_top, active_left, active_bottom, active_right = active_values
    if not (0 <= active_top < active_bottom <= raw_height and 0 <= active_left < active_right <= raw_width):
        raise ValueError(f"{path}: invalid ActiveArea {active_values} for raw shape {page.shape}")

    crop_x, crop_y = _as_xy(_integer_values(_tag_value(page, "DefaultCropOrigin", required=False) or (0, 0)), "DefaultCropOrigin")
    crop_width, crop_height = _as_xy(
        _integer_values(_tag_value(page, "DefaultCropSize", required=False) or (active_right - active_left, active_bottom - active_top)),
        "DefaultCropSize",
    )
    if crop_width <= 0 or crop_height <= 0 or crop_width % 2 or crop_height % 2:
        raise ValueError(f"{path}: DefaultCrop must have positive even size, got {(crop_x, crop_y, crop_width, crop_height)}")
    crop_top, crop_left = active_top + crop_y, active_left + crop_x
    if crop_top < active_top or crop_left < active_left or crop_top + crop_height > active_bottom or crop_left + crop_width > active_right:
        raise ValueError(f"{path}: DefaultCrop lies outside ActiveArea")

    repeat_h, repeat_w = _as_xy(_integer_values(_tag_value(page, "CFARepeatPatternDim")), "CFARepeatPatternDim")
    if (repeat_h, repeat_w) != (2, 2):
        raise ValueError(f"{path}: only 2x2 CFA is supported, got {(repeat_h, repeat_w)}")
    cfa_indices = _integer_values(_tag_value(page, "CFAPattern"))
    if len(cfa_indices) != 4:
        raise ValueError(f"{path}: CFAPattern must contain four samples")
    plane_colors = _integer_values(_tag_value(page, "CFAPlaneColor", required=False) or (0, 1, 2))
    if max(cfa_indices) >= len(plane_colors):
        raise ValueError(f"{path}: CFAPattern refers to missing CFAPlaneColor")
    active_pattern = np.asarray([plane_colors[index] for index in cfa_indices], dtype=np.int8).reshape(2, 2)
    # CFAPattern/BlackLevel repeat coordinates are based on ActiveArea.  The
    # local crop phase changes when DefaultCropOrigin is odd.
    cfa_after_crop = np.roll(active_pattern, shift=(-crop_y % 2, -crop_x % 2), axis=(0, 1))
    canonical_indices = _canonical_position_indices(cfa_after_crop)
    color_name = {0: "R", 1: "G", 2: "B"}
    cfa_name = "".join(color_name.get(int(value), str(int(value))) for value in cfa_after_crop.reshape(-1))

    black_repeat_h, black_repeat_w = _as_xy(
        _integer_values(_tag_value(page, "BlackLevelRepeatDim", required=False) or (1, 1)), "BlackLevelRepeatDim"
    )
    black_values = _numbers(_tag(page, "BlackLevel"))
    expected_black_values = black_repeat_h * black_repeat_w
    if len(black_values) != expected_black_values:
        raise ValueError(
            f"{path}: BlackLevel has {len(black_values)} values; expected {expected_black_values} for one-sample CFA"
        )
    black_pattern = np.asarray(black_values, dtype=np.float32).reshape(black_repeat_h, black_repeat_w)
    black_after_crop = np.empty((2, 2), dtype=np.float32)
    for row in range(2):
        for col in range(2):
            black_after_crop[row, col] = black_pattern[(crop_y + row) % black_repeat_h, (crop_x + col) % black_repeat_w]
    black_position = tuple(float(value) for value in black_after_crop.reshape(-1))
    black_canonical = tuple(black_position[index] for index in canonical_indices)

    white_values = _numbers(_tag(page, "WhiteLevel"))
    if len(white_values) != 1:
        raise ValueError(f"{path}: one-sample CFA requires one WhiteLevel, got {white_values}")
    white_level = float(white_values[0])
    dynamic_range = white_level - max(black_position)
    if not np.isfinite(dynamic_range) or dynamic_range <= 0:
        raise ValueError(f"{path}: WhiteLevel must exceed every BlackLevel")

    profile_values = _numbers(_tag(page, "NoiseProfile"))
    if len(profile_values) == 2:
        profile_rgb = tuple((float(profile_values[0]), float(profile_values[1])) for _ in range(3))
    elif len(profile_values) == 6:
        profile_rgb = tuple((float(profile_values[index]), float(profile_values[index + 1])) for index in range(0, 6, 2))
    else:
        raise ValueError(f"{path}: NoiseProfile must contain 2 or 6 values, got {len(profile_values)}")
    profile_array = np.asarray(profile_rgb, dtype=np.float64)
    if not np.isfinite(profile_array).all() or np.any(profile_array[:, 0] <= 0) or np.any(profile_array[:, 1] < 0):
        raise ValueError(f"{path}: invalid NoiseProfile values {profile_rgb}")
    # profile rows are R/G/B; canonical packed order is R/G1/G2/B.
    s4 = (profile_rgb[0][0], profile_rgb[1][0], profile_rgb[1][0], profile_rgb[2][0])
    o4 = (profile_rgb[0][1], profile_rgb[1][1], profile_rgb[1][1], profile_rgb[2][1])

    exposure_values = _numbers(_tag(page, "ExposureTime"))
    if len(exposure_values) != 1 or exposure_values[0] <= 0:
        raise ValueError(f"{path}: invalid ExposureTime {exposure_values}")
    iso_values = _numbers(_tag(page, "ISOSpeedRatings"))
    if len(iso_values) != 1 or iso_values[0] <= 0:
        raise ValueError(f"{path}: invalid ISOSpeedRatings {iso_values}")

    return PhoneDNGMetadata(
        path=str(path.resolve()),
        make=_ascii_tag(page, "Make"),
        model=_ascii_tag(page, "Model"),
        unique_camera_model=_ascii_tag(page, "UniqueCameraModel"),
        datetime=_ascii_tag(page, "DateTimeOriginal") or _ascii_tag(page, "DateTime"),
        iso_exif=int(round(iso_values[0])),
        exposure_s=float(exposure_values[0]),
        raw_shape_hw=(raw_height, raw_width),
        active_area_tlbr=(active_top, active_left, active_bottom, active_right),
        default_crop_xywh=(crop_x, crop_y, crop_width, crop_height),
        cfa_pattern_active=tuple(int(value) for value in active_pattern.reshape(-1)),
        cfa_after_crop=tuple(int(value) for value in cfa_after_crop.reshape(-1)),
        cfa_name=cfa_name,
        canonical_position_indices=canonical_indices,
        black_level_position=black_position,
        black_level_canonical=black_canonical,
        white_level=white_level,
        dynamic_range=dynamic_range,
        noise_profile_rgb=profile_rgb,
        noise_profile_s4=tuple(float(value) for value in s4),
        noise_profile_o4=tuple(float(value) for value in o4),
    )


def read_phone_dng_metadata(path: str | Path) -> PhoneDNGMetadata:
    """Read DNG metadata without decoding the full CFA pixel array."""
    path = Path(path)
    with tifffile.TiffFile(path) as tiff:
        return _metadata_from_page(path, _find_raw_page(tiff))


def pack_canonical(mosaic: np.ndarray, metadata: PhoneDNGMetadata) -> np.ndarray:
    """Crop a stored CFA mosaic and pack it into canonical [R,G1,G2,B]."""
    if mosaic.ndim != 2:
        raise ValueError(f"Expected a two-dimensional CFA mosaic, got {mosaic.shape}")
    if tuple(mosaic.shape) != metadata.raw_shape_hw:
        raise ValueError(f"Mosaic shape {mosaic.shape} does not match metadata {metadata.raw_shape_hw}")
    crop_x, crop_y, crop_width, crop_height = metadata.default_crop_xywh
    active_top, active_left, _, _ = metadata.active_area_tlbr
    top, left = active_top + crop_y, active_left + crop_x
    cropped = mosaic[top : top + crop_height, left : left + crop_width]
    positions = np.stack(
        [cropped[0::2, 0::2], cropped[0::2, 1::2], cropped[1::2, 0::2], cropped[1::2, 1::2]], axis=0
    )
    return positions[np.asarray(metadata.canonical_position_indices)]


def unpack_to_dng_cfa(packed: np.ndarray, metadata: PhoneDNGMetadata) -> np.ndarray:
    """Inverse of :func:`pack_canonical` in DefaultCrop coordinates."""
    expected_shape = (4, *metadata.packed_shape_hw)
    if tuple(packed.shape) != expected_shape:
        raise ValueError(f"Packed shape {packed.shape} does not match expected {expected_shape}")
    positions = np.empty_like(packed)
    positions[np.asarray(metadata.canonical_position_indices)] = packed
    mosaic = np.empty((metadata.default_crop_xywh[3], metadata.default_crop_xywh[2]), dtype=packed.dtype)
    mosaic[0::2, 0::2] = positions[0]
    mosaic[0::2, 1::2] = positions[1]
    mosaic[1::2, 0::2] = positions[2]
    mosaic[1::2, 1::2] = positions[3]
    return mosaic


def read_phone_dng_packed(path: str | Path, metadata: PhoneDNGMetadata | None = None) -> tuple[np.ndarray, PhoneDNGMetadata]:
    """Decode stored pre-opcode CFA samples and pack the DefaultCrop region."""
    path = Path(path)
    with tifffile.TiffFile(path) as tiff:
        page = _find_raw_page(tiff)
        file_metadata = _metadata_from_page(path, page)
        if metadata is not None and metadata != file_metadata:
            raise ValueError(f"{path}: supplied metadata does not match the DNG tags")
        mosaic = page.asarray()
    return pack_canonical(np.asarray(mosaic), file_metadata), file_metadata
