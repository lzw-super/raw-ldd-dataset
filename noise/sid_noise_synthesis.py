"""DN-domain shot-noise and dark-residual synthesis for SID."""

from __future__ import annotations

import torch


def _batch_scalar(value: torch.Tensor | float | int, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        value = value.reshape(1).expand(reference.shape[0])
    if value.ndim != 1 or value.shape[0] != reference.shape[0]:
        raise ValueError(f"Expected a scalar or {reference.shape[0]} per-sample values, got {tuple(value.shape)}")
    return value.view(-1, 1, 1, 1)


def synthesize_sid_noise(
    clean_norm: torch.Tensor,
    dark_residual_dn: torch.Tensor,
    iso: torch.Tensor | float | int,
    ratio: torch.Tensor | float | int,
    *,
    black_level: float = 512.0,
    white_level: float = 16383.0,
    k_scale: float = 0.1,
    mode: str = "ratio_aware",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic SID input and its clean target.

    ``clean_norm`` is black-level-subtracted packed RAW. ``dark_residual_dn``
    is a signed PMN-corrected dark-frame residual in DN.  The default follows
    the ratio-aware interpretation in the reproduction guide: sample photon
    noise in the short exposure, then amplify both noise sources by the SID
    digital gain.  Only the input upper bound is clipped, matching evaluation.
    """
    if clean_norm.shape != dark_residual_dn.shape:
        raise ValueError(f"Clean and dark residual shapes differ: {clean_norm.shape} vs {dark_residual_dn.shape}")
    if clean_norm.ndim != 4 or clean_norm.shape[1] != 4:
        raise ValueError("Expected packed RAW tensors with shape [B, 4, H, W]")
    dynamic_range = float(white_level) - float(black_level)
    if dynamic_range <= 0:
        raise ValueError("white_level must exceed black_level")
    iso_batch = _batch_scalar(iso, clean_norm)
    ratio_batch = _batch_scalar(ratio, clean_norm)
    if torch.any(iso_batch <= 0) or torch.any(ratio_batch <= 0):
        raise ValueError("ISO and ratio must be positive")

    target = torch.clamp(clean_norm, 0.0, 1.0)
    clean_dn = target * dynamic_range
    # k = ISO / 100 * 0.1 = ISO / 1000 DN/e-, as specified for SID/ELD.
    k_dn = iso_batch / 100.0 * float(k_scale)

    if mode == "ratio_aware":
        shot_short_dn = k_dn * torch.poisson(torch.clamp(clean_dn / (ratio_batch * k_dn), min=0.0))
        noisy_dn = ratio_batch * (shot_short_dn + dark_residual_dn)
    elif mode == "paper_literal":
        noisy_dn = k_dn * torch.poisson(torch.clamp(clean_dn / k_dn, min=0.0)) + dark_residual_dn
    else:
        raise ValueError("mode must be 'ratio_aware' or 'paper_literal'")

    return torch.clamp_max(noisy_dn / dynamic_range, 1.0), target
