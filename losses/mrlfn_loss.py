"""Composite RAW reconstruction/chromatic objective used by MRLFN."""

from __future__ import annotations

import torch
from torch import nn


class RawReconstructionChromaticLoss(nn.Module):
    """Weighted RAW reconstruction and chromatic-aberration objective.

    ``channel_order`` contains indices for the semantic order
    ``[R, G1, B, G2]``.  This repository packs RGGB as ``[R,G1,G2,B]``, so
    its correct setting is ``(0,1,3,2)``.
    """

    def __init__(
        self,
        raw_weight: float = 0.6,
        chromatic_weight: float = 0.4,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        channel_order: tuple[int, int, int, int] = (0, 1, 3, 2),
    ):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        if len(channel_order) != 4 or len(set(channel_order)) != 4:
            raise ValueError("channel_order must contain four distinct indices for [R,G1,B,G2]")
        if min(channel_order) < 0:
            raise ValueError("channel_order indices must be non-negative")
        self.raw_weight = float(raw_weight)
        self.chromatic_weight = float(chromatic_weight)
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.channel_order = tuple(int(index) for index in channel_order)

    def per_sample_components(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pred.shape != target.shape or pred.ndim != 4:
            raise ValueError(
                f"pred and target must have the same NCHW shape, got {pred.shape} and {target.shape}"
            )
        if max(self.channel_order) >= pred.shape[1]:
            raise ValueError(f"channel_order {self.channel_order} is invalid for {pred.shape[1]} channels")

        raw = (pred - target).abs().flatten(1).mean(1)
        red, green1, blue, green2 = self.channel_order
        pred_green = 0.5 * (pred[:, green1] + pred[:, green2])
        target_green = 0.5 * (target[:, green1] + target[:, green2])
        chromatic = (
            ((pred[:, blue] - pred_green) - (target[:, blue] - target_green))
            .abs()
            .flatten(1)
            .mean(1)
            + ((pred[:, red] - pred_green) - (target[:, red] - target_green))
            .abs()
            .flatten(1)
            .mean(1)
        )
        total = self.loss_weight * (self.raw_weight * raw + self.chromatic_weight * chromatic)
        return total, raw, chromatic

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        if weight is not None:
            raise ValueError("Element-wise weights are not defined for the composite chromatic loss")
        loss, _, _ = self.per_sample_components(pred, target)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
