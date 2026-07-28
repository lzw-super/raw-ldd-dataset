"""Shared tiled inference and display-only previews for phone packed RAW."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch


def tiled_phone_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    tile_size: int = 512,
    tile_overlap: int = 64,
) -> torch.Tensor:
    """Run a packed RAW tensor through a model, stitching overlapping tiles.

    ``image`` is a normalized ``[1, 4, H, W]`` tensor.  A full frame is used
    when it fits in one tile; otherwise cosine-weighted overlapping tiles keep
    the result usable on GPUs that cannot hold a full phone RAW frame.
    """
    if image.ndim != 4 or image.shape[:2] != (1, 4):
        raise ValueError(f"Expected [1,4,H,W] packed RAW, got {tuple(image.shape)}")
    if tile_size <= 0 or tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= tile_overlap < tile_size")
    _, _, height, width = image.shape
    if height <= tile_size and width <= tile_size:
        return model(image)

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = tile_size - tile_overlap
        values = list(range(0, length - tile_size + 1, stride))
        if values[-1] != length - tile_size:
            values.append(length - tile_size)
        return values

    rows, columns = starts(height), starts(width)
    device, dtype = image.device, image.dtype
    # Hann is zero at the endpoints.  Keep a tiny positive edge weight so the
    # outer image border is covered and no division by zero is possible.
    window_y = torch.hann_window(tile_size, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    window_x = torch.hann_window(tile_size, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    window = (window_y[:, None] * window_x[None, :]).view(1, 1, tile_size, tile_size)
    accumulated = torch.zeros_like(image)
    weights = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    for top in rows:
        for left in columns:
            prediction = model(image[:, :, top : top + tile_size, left : left + tile_size])
            accumulated[:, :, top : top + tile_size, left : left + tile_size] += prediction * window
            weights[:, :, top : top + tile_size, left : left + tile_size] += window
    return accumulated / weights


def packed_raw_preview(
    packed: np.ndarray, *, upper: float | None = None, gamma: float = 2.2
) -> tuple[np.ndarray, float]:
    """Create an explicitly display-only packed-RGB preview.

    This is not a camera-color-managed DNG render: it simply combines canonical
    R/G/G/B channels at packed resolution and applies a shared percentile
    scale.  It is intentionally used only for qualitative before/after PNGs.
    """
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected [4,H,W] packed RAW, got {packed.shape}")
    rgb = np.stack((packed[0], 0.5 * (packed[1] + packed[2]), packed[3]), axis=-1).astype(np.float32, copy=False)
    if upper is None:
        positive = rgb[np.isfinite(rgb) & (rgb > 0)]
        upper = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    upper = max(float(upper), 1e-6)
    preview = np.clip(rgb / upper, 0.0, 1.0)
    preview = np.power(preview, 1.0 / max(float(gamma), 1e-6))
    return np.rint(preview * 255.0).astype(np.uint8), upper


def make_side_by_side(left: np.ndarray, right: np.ndarray, *, ratio: float | None = None) -> np.ndarray:
    """Join two previews and label the actual brightening ratio when supplied."""
    if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 3:
        raise ValueError("Expected equal HxWx3 preview images")
    separator = np.full((left.shape[0], max(4, math.ceil(left.shape[1] * 0.01)), 3), 235, dtype=np.uint8)
    comparison = np.concatenate((left, separator, right), axis=1)
    if ratio is None:
        return comparison
    header_height = 34
    header = np.full((header_height, comparison.shape[1], 3), 255, dtype=np.uint8)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1
    cv2.putText(header, f"Noisy input (x{ratio:g})", (10, 23), font, scale, (25, 25, 25), thickness, cv2.LINE_AA)
    right_x = left.shape[1] + separator.shape[1] + 10
    cv2.putText(header, "Denoised", (right_x, 23), font, scale, (25, 25, 25), thickness, cv2.LINE_AA)
    return np.concatenate((header, comparison), axis=0)
