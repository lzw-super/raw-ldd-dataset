"""Equivalent fixed-Haar soft-shrinkage deployment graphs for NPU ablations.

This wrapper consumes a deployed checkpoint; it does not change training defaults.
Threshold rows are deepest-first and band-major, exactly as in LearningDWT.
"""
import torch
from torch import nn
from torch.nn import functional as F
from models.fixed_raw_ops import FixedHaarConv


class HaarSoftExport(nn.Module):
    def __init__(self, wavelet, layout="band", formula="relu"):
        super().__init__()
        if layout not in ("band", "level") or formula not in ("relu", "minmax"):
            raise ValueError("Expected band/level layout and relu/minmax formula")
        if not (isinstance(wavelet.transform, FixedHaarConv)
                and wavelet.shrink_mode == "soft" and wavelet.leak == 0
                and wavelet.ll_mode == "ll_only" and not wavelet.ll_normalize
                and wavelet.direct_shrink is None):
            raise ValueError("Requires deployed fixed Haar, soft, leak=0, LL-only without normalization")
        thresholds = wavelet.fixed_hf_thresholds.detach().clone()
        if not bool(torch.isfinite(thresholds).all() and (thresholds >= 0).all()):
            raise ValueError("Soft thresholds must be finite and nonnegative")
        self.register_buffer("weight", wavelet.transform.weight.detach().clone())
        self.register_buffer("thresholds", thresholds)
        self.ll_restorer = wavelet.ll_restorer
        self.levels = wavelet.levels
        self.layout = layout
        self.formula = formula

    def shrink(self, z, t):
        if self.formula == "relu":
            return F.relu(z - t) - F.relu(-z - t)
        return z - torch.minimum(torch.maximum(z, -t), t)

    def forward(self, x):
        # This exporter deliberately requires a fixed, divisible input shape.
        if not torch.jit.is_tracing() and (x.shape[-2] % (2**self.levels)
                                         or x.shape[-1] % (2**self.levels)):
            raise ValueError("Export ablations require dimensions divisible by 2**levels")

        def visit(current, level):
            coefficients = F.conv2d(current, self.weight, stride=2)
            offset = 3 * (self.levels - level)
            if self.layout == "band":
                ll, *details = coefficients.split(4, dim=1)
                filtered = [self.shrink(z, self.thresholds[offset+i].view(1, 4, 1, 1))
                            for i, z in enumerate(details)]
            else:
                # One contiguous 12-channel view, never split then re-concatenate HF.
                ll, hf = coefficients.split([4, 12], dim=1)
                filtered = [self.shrink(hf, self.thresholds[offset:offset+3].reshape(1, 12, 1, 1))]
            restored = visit(ll, level+1) if level < self.levels else self.ll_restorer(ll)
            return F.conv_transpose2d(torch.cat([restored, *filtered], dim=1), self.weight, stride=2)

        return visit(x, 1)
