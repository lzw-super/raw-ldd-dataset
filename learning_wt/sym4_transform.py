"""Differentiable orthogonal DWT with critically sampled periodization boundaries."""
import pywt
import torch
from torch import nn
from torch.nn import functional as F


class PeriodizedWavelet(nn.Module):
    def __init__(self, wavelet='sym4'):
        super().__init__()
        w = pywt.Wavelet(wavelet)
        if not w.orthogonal:
            raise ValueError('An orthogonal wavelet is required')
        self.register_buffer('filters', torch.tensor([w.dec_lo[::-1], w.dec_hi[::-1]], dtype=torch.float64))

    def _analysis(self, x):
        n, c = x.shape[-1], x.shape[1]
        if n % 2:
            raise ValueError('Periodized transform requires even dimensions')
        p = self.filters.shape[-1] // 2 - 1
        indices = torch.arange(-p, n+p, device=x.device) % n
        padded = x.index_select(-1, indices)
        weights = self.filters.to(x).view(2, 1, 1, -1).repeat(c, 1, 1, 1)
        y = F.conv2d(padded, weights, stride=(1, 2), groups=c)
        y = y.reshape(x.shape[0], c, 2, x.shape[-2], n//2)
        return y[:, :, 0], y[:, :, 1]

    def _synthesis(self, lo, hi):
        b, c, h, half = lo.shape
        n = 2*half
        p = self.filters.shape[-1] // 2 - 1
        y = torch.stack((lo, hi), 2).reshape(b, 2*c, h, half)
        weights = self.filters.to(lo).view(2, 1, 1, -1).repeat(c, 1, 1, 1)
        padded = F.conv_transpose2d(y, weights, stride=(1, 2), groups=c)
        indices = torch.arange(-p, n+p, device=lo.device) % n
        return lo.new_zeros(b, c, h, n).index_add(-1, indices, padded)

    def dwt2(self, x):
        lo, hi = self._analysis(x)
        ll, hl = self._analysis(lo.transpose(-2, -1))
        lh, hh = self._analysis(hi.transpose(-2, -1))
        return tuple(t.transpose(-2, -1) for t in (ll, lh, hl, hh))

    def iwt2(self, ll, lh, hl, hh):
        lo = self._synthesis(ll.transpose(-2, -1), hl.transpose(-2, -1)).transpose(-2, -1)
        hi = self._synthesis(lh.transpose(-2, -1), hh.transpose(-2, -1)).transpose(-2, -1)
        return self._synthesis(lo, hi)

    @staticmethod
    def _validate(x, levels):
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise ValueError('levels must be a positive integer')
        if x.ndim != 4 or not x.is_floating_point() or min(x.shape) < 1:
            raise ValueError('Expected a nonempty floating NCHW tensor')
        factor = 2 ** levels
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(f'Height and width must be multiples of {factor}')

    def encode(self, x, levels=3):
        self._validate(x, levels)
        ll, lh, hl, hh = self.dwt2(x)
        if levels > 1:
            ll = self.encode(ll, levels-1)
        return torch.cat((torch.cat((ll, lh), -1), torch.cat((hl, hh), -1)), -2)

    def decode(self, x, levels=3):
        self._validate(x, levels)
        h, w = x.shape[-2]//2, x.shape[-1]//2
        ll = self.decode(x[..., :h, :w], levels-1) if levels > 1 else x[..., :h, :w]
        return self.iwt2(ll, x[..., :h, w:], x[..., h:, :w], x[..., h:, w:])
