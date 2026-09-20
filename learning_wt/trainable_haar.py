"""Haar-initialized banks with optional channel and decomposition-level sharing."""
import torch
from torch import nn
from torch.nn import functional as F
from .sym4_transform import PeriodizedWavelet


class TrainableHaar(nn.Module):
    _validate = staticmethod(PeriodizedWavelet._validate)

    def __init__(self, share_channels=True, share_levels=True, levels=3):
        super().__init__()
        self.share_channels, self.share_levels = share_channels, share_levels
        self.levels = levels
        filters = torch.tensor([[[1,1],[1,1]], [[1,-1],[1,-1]],
                                [[1,1],[-1,-1]], [[1,-1],[-1,1]]], dtype=torch.float32)[:,None] / 2
        # Preserve the original fully-shared checkpoint tensor shapes.
        if not (share_channels and share_levels):
            filters = filters[None,None].repeat(1 if share_levels else levels,
                                               1 if share_channels else 4,1,1,1,1)
        self.analysis = nn.Parameter(filters.clone())
        self.synthesis = nn.Parameter(filters.clone())

    def _kernel(self, bank, level):
        if self.share_channels and self.share_levels:
            return bank
        bank = bank[0 if self.share_levels else level]
        return bank.reshape(-1,1,2,2)

    def dwt2(self, x, level=0):
        b,c,h,w = x.shape
        kernel = self._kernel(self.analysis, level)
        if self.share_channels:
            y = F.conv2d(x.reshape(b*c,1,h,w), kernel, stride=2)
        else:
            if c != 4:
                raise ValueError('Unshared RAW kernels require four channels')
            y = F.conv2d(x, kernel, stride=2, groups=4)
        y = y.reshape(b,c,4,h//2,w//2)
        return tuple(y[:,:,i] for i in range(4))

    def iwt2(self, ll, lh, hl, hh, level=0):
        b,c,h,w = ll.shape
        kernel = self._kernel(self.synthesis, level)
        y = torch.stack((ll,lh,hl,hh),2)
        if self.share_channels:
            out = F.conv_transpose2d(y.reshape(b*c,4,h,w),kernel,stride=2)
        else:
            out = F.conv_transpose2d(y.reshape(b,4*c,h,w),kernel,stride=2,groups=4)
        return out.reshape(b,c,2*h,2*w)

    def encode(self, x, levels=3):
        self._validate(x, levels)
        if levels > self.levels and not self.share_levels:
            raise ValueError('Requested more levels than configured banks')
        def visit(x, level):
            ll,lh,hl,hh = self.dwt2(x,level)
            if level+1 < levels:
                ll = visit(ll,level+1)
            return torch.cat((torch.cat((ll,lh),-1),torch.cat((hl,hh),-1)),-2)
        return visit(x,0)

    def decode(self, x, levels=3):
        self._validate(x, levels)
        if levels > self.levels and not self.share_levels:
            raise ValueError('Requested more levels than configured banks')
        def visit(x, level):
            h,w = x.shape[-2]//2,x.shape[-1]//2
            ll = x[...,:h,:w]
            if level+1 < levels:
                ll = visit(ll,level+1)
            return self.iwt2(ll,x[...,:h,w:],x[...,h:,:w],x[...,h:,w:],level)
        return visit(x,0)
