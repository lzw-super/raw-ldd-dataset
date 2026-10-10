"""Fixed-shape BayesShrink export preserving the nonzero median and FP64 statistics.

TopK is an ONNX representation, not a claim that a particular NPU supports it.
No data-dependent threshold is frozen or replaced with a variance approximation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from learning_wt.bayes_shrink import bayes_soft_shrink
from models.fixed_raw_ops import FixedHaarConv


def nonzero_noise_variance(hh):
    values = hh.detach().double().abs().flatten(2)
    # Nonzero absolute values sort ahead of zeros. The two median ranks can
    # never exceed N//2, so the largest N//2+1 values suffice for any zero mask.
    n = math.prod(hh.shape[-2:])
    ordered = torch.topk(values, n // 2 + 1, dim=-1, largest=True, sorted=True).values
    count = (values != 0).sum(dim=-1, keepdim=True).clamp_min(1)
    lower = torch.div(count - 1, 2, rounding_mode='trunc')
    upper = torch.div(count, 2, rounding_mode='trunc')
    median = (ordered.gather(-1, lower) + ordered.gather(-1, upper)) * 0.5
    # An all-zero channel selects zero twice and therefore returns zero variance.
    return ((median / 0.6744897501960817) ** 2).unsqueeze(-1)


class HaarBayesExport(nn.Module):
    def __init__(self, wavelet):
        super().__init__()
        if not (isinstance(wavelet.transform, FixedHaarConv)
                and wavelet.threshold_mode == 'bayes_shrink'
                and wavelet.ll_mode == 'bypass' and wavelet.leak == 0):
            raise ValueError('Requires fixed Haar BayesShrink with LL bypass')
        self.register_buffer('weight', wavelet.transform.weight.detach().clone())
        self.levels = wavelet.levels

    def forward(self, x):
        if not torch.jit.is_tracing() and any(s % (2 ** self.levels) for s in x.shape[-2:]):
            raise ValueError('Fixed export input must be divisible by 2**levels')

        def visit(current, level, noise_variance=None):
            ll, lh, hl, hh = F.conv2d(current, self.weight, stride=2).split(4, dim=1)
            if noise_variance is None:
                noise_variance = nonzero_noise_variance(hh)
            details = [bayes_soft_shrink(band, noise_variance) for band in (lh, hl, hh)]
            restored = visit(ll, level + 1, noise_variance) if level < self.levels else ll
            return F.conv_transpose2d(torch.cat([restored, *details], dim=1), self.weight, stride=2)

        return visit(x, 1)
