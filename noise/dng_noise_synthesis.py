"""DNG NoiseProfile-aware synthetic RAW noise for the phone pipeline."""

from __future__ import annotations

import torch


def _batch_scalar(value: torch.Tensor | float | int, reference: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1).expand(reference.shape[0])
    if tensor.ndim != 1 or tensor.shape[0] != reference.shape[0]:
        raise ValueError(f"{name} must be a scalar or [B], got {tuple(tensor.shape)}")
    return tensor.view(-1, 1, 1, 1)


def _batch_channels(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 1:
        tensor = tensor.view(1, -1).expand(reference.shape[0], -1)
    if tensor.ndim != 2 or tensor.shape != reference.shape[:2]:
        raise ValueError(f"{name} must be [4] or [B,4], got {tuple(tensor.shape)}")
    return tensor.view(reference.shape[0], 4, 1, 1)


def synthesize_phone_noise(
    clean_norm: torch.Tensor,
    dark_residual_dn: torch.Tensor,
    noise_profile_s4: torch.Tensor,
    dark_dynamic_range: torch.Tensor | float,
    ratio: torch.Tensor | float,
    *,
    mode: str = "hybrid",
    noise_profile_o4: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a ratio-aware noisy input from a pseudo/clean normalized target.

    ``S4`` and ``O4`` are DNG NoiseProfile parameters in DNG's normalized
    domain.  Hybrid mode adds Poisson shot noise and an empirical signed dark
    residual, deliberately *not* the analytic ``O`` term (otherwise read noise
    would be counted twice).  Analytic mode is provided only as a fallback for
    ablations without real dark frames.
    """
    if clean_norm.ndim != 4 or clean_norm.shape[1] != 4:
        raise ValueError("clean_norm must have shape [B,4,H,W]")
    if dark_residual_dn.shape != clean_norm.shape:
        raise ValueError(f"dark residual shape {dark_residual_dn.shape} != clean shape {clean_norm.shape}")
    s4 = _batch_channels(noise_profile_s4, clean_norm, "noise_profile_s4")
    if torch.any(s4 <= 0) or not torch.isfinite(s4).all():
        raise ValueError("NoiseProfile S values must be finite and positive")
    dynamic_range = _batch_scalar(dark_dynamic_range, clean_norm, "dark_dynamic_range")
    ratio_batch = _batch_scalar(ratio, clean_norm, "ratio")
    if torch.any(dynamic_range <= 0) or torch.any(ratio_batch <= 0):
        raise ValueError("dark_dynamic_range and ratio must be positive")

    target = torch.clamp(clean_norm, 0.0, 1.0)
    # Sample in short-exposure normalized units, then apply the same digital
    # brightening ratio to both photon and read/residual terms.
    mu_short = torch.clamp(clean_norm / ratio_batch, min=0.0)
    shot_short = s4 * torch.poisson(mu_short / s4)
    if mode == "hybrid":
        residual_short = dark_residual_dn / dynamic_range
    elif mode == "analytic":
        if noise_profile_o4 is None:
            raise ValueError("analytic synthesis requires noise_profile_o4")
        o4 = _batch_channels(noise_profile_o4, clean_norm, "noise_profile_o4")
        if torch.any(o4 < 0) or not torch.isfinite(o4).all():
            raise ValueError("NoiseProfile O values must be finite and non-negative")
        residual_short = torch.sqrt(o4) * torch.randn_like(clean_norm)
    else:
        raise ValueError("mode must be 'hybrid' or 'analytic'")
    noisy = ratio_batch * (shot_short + residual_short)
    # Preserve the negative half of empirical dark residuals.  Target clipping
    # is separate because the target represents a valid normalized RAW signal.
    return torch.clamp_max(noisy, 1.0), target
