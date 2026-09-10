"""Differentiable multilevel LL L1 on independent packed-RAW channels."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WaveletLLLoss(nn.Module):
    """Mean of L1 distances on LL_1,...,LL_levels (standard DWT scale).

    Uses PyWavelets decomposition filters and symmetric boundary extension.
    Only the LL branch is computed; filters are fixed and gradients flow to
    the input through grouped convolutions. No thresholding or clipping.
    """

    SUPPORTED = ('haar', 'db2', 'sym4', 'coif1', 'bior2.2')

    def __init__(self, wavelet: str = 'haar', levels: int = 3):
        super().__init__()
        if wavelet not in self.SUPPORTED:
            raise ValueError(f'wavelet must be one of {self.SUPPORTED}')
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise ValueError('levels must be a positive integer')
        import pywt
        self.wavelet, self.levels = wavelet, levels
        # conv2d performs correlation, hence reverse the decomposition filter.
        low = torch.tensor(pywt.Wavelet(wavelet).dec_lo[::-1], dtype=torch.float64)
        self.register_buffer('low', low)

    @staticmethod
    def _extend(x: torch.Tensor, axis: int, left: int, right: int) -> torch.Tensor:
        size = x.shape[axis]
        indices = torch.arange(-left, size + right, device=x.device) % (2 * size)
        indices = torch.where(indices < size, indices, 2 * size - 1 - indices)
        return x.index_select(axis, indices)

    def ll(self, x: torch.Tensor) -> torch.Tensor:
        """One level, numerically matching pywt.dwt2(..., mode='symmetric')[0]."""
        length = self.low.numel()
        channels = x.shape[1]
        low = self.low.to(dtype=x.dtype)
        x = self._extend(x, -1, length - 2, length - 2 + x.shape[-1] % 2)
        x = F.conv2d(x, low.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
                     stride=(1, 2), groups=channels)
        x = self._extend(x, -2, length - 2, length - 2 + x.shape[-2] % 2)
        return F.conv2d(x, low.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
                        stride=(2, 1), groups=channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4 or pred.shape != target.shape or min(pred.shape) < 1:
            raise ValueError('pred and target must have the same nonempty NCHW shape')
        # Reject depths whose entire approximation is affected by boundaries.
        max_levels = max(0, (min(pred.shape[-2:]) // (self.low.numel() - 1)).bit_length() - 1)
        if self.levels > max_levels:
            raise ValueError(f'{self.wavelet}: levels={self.levels} exceeds maximum {max_levels} for {pred.shape[-2:]}')
        # Keep loss filtering in float32 under AMP, retaining float64 for gradcheck.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float() if pred.dtype in (torch.float16, torch.bfloat16) else pred
            target = target.to(dtype=pred.dtype)
            losses = []
            for _ in range(self.levels):
                pred, target = self.ll(pred), self.ll(target)
                losses.append(F.l1_loss(pred, target))
            return torch.stack(losses).mean()
