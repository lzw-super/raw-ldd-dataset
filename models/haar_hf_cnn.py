"""Per-level HF processors shared by training and deployment speed experiments."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.haar_soft_export import HaarSoftExport


class HighFrequencyCNN(nn.Module):
    def __init__(self, variant, init_bias=0.0):
        super().__init__()
        variants = ('dw1', 'dw3', 'dw3_pw1', 'dw1_residual', 'dw3_residual',
                    'dw3_prelu_residual', 'dw3_skipdw1_residual',
                    'dw3_prelu_skipdw1_residual', 'repncb_prelu_residual',
                    'dw1_dual_relu', 'dw3_dual_relu', 'dw1_dual_prelu')
        if variant not in variants:
            raise ValueError(variant)
        self.residual = variant.endswith('_residual')
        self.dual = '_dual_' in variant
        if self.dual:
            if not math.isfinite(init_bias):
                raise ValueError('HF CNN initial bias must be finite')
            kernel = 3 if variant.startswith('dw3') else 1
            self.branches = nn.ModuleList()
            for sign in (1.0, -1.0):
                conv = nn.Conv2d(12, 12, kernel, padding=kernel//2, groups=12, bias=True)
                with torch.no_grad():
                    conv.weight.zero_()
                    conv.weight[:, 0, kernel//2, kernel//2] = sign
                    conv.bias.fill_(init_bias)
                # PReLU starts as ReLU to retain the same initial shrinkage function.
                activation = nn.PReLU(12, init=0.0) if variant.endswith('prelu') else nn.ReLU()
                self.branches.append(nn.Sequential(conv, activation))
            return
        if variant == 'repncb_prelu_residual':
            from models.learning_dwt_repncb import RepNCB
            # Dense 12->12 Rep-NCB already contains one channel-wise PReLU.
            self.block = RepNCB(12)
        else:
            kernel = 3 if variant.startswith('dw3') else 1
            self.depthwise = nn.Conv2d(12, 12, kernel, padding=kernel//2, groups=12, bias=True)
            self.relu = nn.PReLU(12, init=0.25) if 'prelu' in variant else nn.ReLU()
        self.pointwise = nn.Conv2d(12, 12, 1, bias=True) if variant == 'dw3_pw1' else None
        self.skip_projection = nn.Identity()
        if 'skipdw1' in variant:
            self.skip_projection = nn.Conv2d(12, 12, 1, groups=12, bias=True)
            # Start as the original identity bypass, without any activation.
            nn.init.ones_(self.skip_projection.weight)
            nn.init.zeros_(self.skip_projection.bias)

    def forward(self, x):
        if self.dual:
            return self.branches[0](x) - self.branches[1](x)
        y = self.block(x) if hasattr(self, 'block') else self.relu(self.depthwise(x))
        if self.pointwise is not None:
            y = self.pointwise(y)
        return self.skip_projection(x) + y if self.residual else y


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
