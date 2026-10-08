"""Per-level HF processors shared by training and deployment speed experiments."""
import torch
from torch import nn
from torch.nn import functional as F
from models.haar_soft_export import HaarSoftExport


class HighFrequencyCNN(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in ('dw1', 'dw3', 'dw3_pw1', 'dw1_residual', 'dw3_residual'):
            raise ValueError(variant)
        kernel = 3 if variant in ('dw3', 'dw3_pw1', 'dw3_residual') else 1
        self.depthwise = nn.Conv2d(12, 12, kernel, padding=kernel//2, groups=12, bias=True)
        self.relu = nn.ReLU()
        self.pointwise = nn.Conv2d(12, 12, 1, bias=True) if variant == 'dw3_pw1' else None
        self.residual = variant in ('dw1_residual', 'dw3_residual')

    def forward(self, x):
        y = self.relu(self.depthwise(x))
        if self.pointwise is not None:
            y = self.pointwise(y)
        return x + y if self.residual else y


class HaarHFCNNExport(HaarSoftExport):
    def __init__(self, wavelet, variant, seed=2026):
        super().__init__(wavelet, layout='level', formula='relu')
        # These replace shrinkage; do not keep unused learned threshold values.
        del self.thresholds
        # Same reproducible, nonzero initialization independent of model loading.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.processors = nn.ModuleList([HighFrequencyCNN(variant) for _ in range(self.levels)])

    def forward(self, x):
        if not torch.jit.is_tracing() and (x.shape[-2] % (2**self.levels)
                                         or x.shape[-1] % (2**self.levels)):
            raise ValueError('Dimensions must be divisible by 2**levels')

        def visit(current, level):
            coefficients = F.conv2d(current, self.weight, stride=2)
            ll, hf = coefficients.split([4, 12], dim=1)
            # Independent modules: index 0 is finest (level 1), index 2 coarsest.
            hf = self.processors[level-1](hf)
            restored = visit(ll, level+1) if level < self.levels else self.ll_restorer(ll)
            return F.conv_transpose2d(torch.cat([restored, hf], dim=1), self.weight, stride=2)

        return visit(x, 1)
