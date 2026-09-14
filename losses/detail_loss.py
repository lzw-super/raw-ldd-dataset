"""Fixed-filter detail matching losses for independent packed RAW channels."""
import torch
from torch import nn
from torch.nn import functional as F
from losses.wavelet_ll_loss import WaveletLLLoss


def check_pair(pred, target):
    if pred.ndim != 4 or pred.shape != target.shape or min(pred.shape) < 1:
        raise ValueError('pred and target must have the same nonempty NCHW shape')


class GradientLoss(nn.Module):
    """Mean signed x/y Sobel L1, kernels normalized by 8, replicate padding."""
    def __init__(self):
        super().__init__()
        dx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float64) / 8
        self.register_buffer('kernels', torch.stack((dx, dx.T)).unsqueeze(1))

    def gradients(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'),
                        self.kernels.to(dtype=x.dtype).repeat(x.shape[1], 1, 1, 1), groups=x.shape[1])

    def forward(self, pred, target):
        check_pair(pred, target)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float() if pred.dtype in (torch.float16, torch.bfloat16) else pred
            return F.l1_loss(self.gradients(pred), self.gradients(target.to(dtype=pred.dtype)))


class WaveletHFLoss(WaveletLLLoss):
    """Equal mean across levels and three detail bands; recurse through LL.

    Bands returned by dwt: LL, high-y/low-x (pywt cH), low-y/high-x
    (pywt cV), high-y/high-x (pywt cD). Symmetric extension, standard scale.
    """
    def __init__(self, wavelet='haar', levels=2):
        super().__init__(wavelet, levels)
        import pywt
        high = torch.tensor(pywt.Wavelet(wavelet).dec_hi[::-1], dtype=torch.float64)
        self.register_buffer('filters', torch.stack([
            torch.outer(y, x) for y, x in
            ((self.low, self.low), (high, self.low), (self.low, high), (high, high))
        ]).unsqueeze(1))

    def dwt(self, x):
        length = self.low.numel()
        channels = x.shape[1]
        x = self._extend(x, -1, length-2, length-2 + x.shape[-1] % 2)
        x = self._extend(x, -2, length-2, length-2 + x.shape[-2] % 2)
        out = F.conv2d(x, self.filters.to(dtype=x.dtype).repeat(channels, 1, 1, 1),
                       stride=2, groups=channels)
        out = out.reshape(out.shape[0], channels, 4, *out.shape[-2:])
        return out[:, :, 0], out[:, :, 1:]

    def forward(self, pred, target):
        check_pair(pred, target)
        max_levels = max(0, (min(pred.shape[-2:]) // (self.low.numel()-1)).bit_length()-1)
        if self.levels > max_levels:
            raise ValueError(f'{self.wavelet}: levels={self.levels} exceeds maximum {max_levels}')
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float() if pred.dtype in (torch.float16, torch.bfloat16) else pred
            target = target.to(dtype=pred.dtype)
            losses = []
            for _ in range(self.levels):
                pred, pred_hf = self.dwt(pred)
                target, target_hf = self.dwt(target)
                losses.append(F.l1_loss(pred_hf, target_hf))
            return torch.stack(losses).mean()
