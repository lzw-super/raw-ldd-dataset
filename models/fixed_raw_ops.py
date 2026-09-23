"""Fixed dense 2x2 convolutions for packed-RAW deployment; no channel shuffles."""
import torch
from torch import nn
from torch.nn import functional as F
from learning_wt.sym4_transform import PeriodizedWavelet


class FixedSpaceDepth(nn.Module):
    def __init__(self, inverse=False):
        super().__init__()
        weight=torch.zeros(16,4,2,2)
        for c in range(4):
            for k in range(4):weight[c*4+k,c,k//2,k%2]=1
        self.register_buffer('weight',weight,persistent=False)
        self.inverse=inverse

    def forward(self,x):
        if self.inverse:
            return F.conv_transpose2d(x,self.weight,stride=2)
        return F.conv2d(x,self.weight,stride=2)


class FixedHaarConv(nn.Module):
    _validate=staticmethod(PeriodizedWavelet._validate)
    encode=PeriodizedWavelet.encode
    decode=PeriodizedWavelet.decode

    def __init__(self):
        super().__init__()
        kernels=torch.tensor([[[1,1],[1,1]],[[1,-1],[1,-1]],[[1,1],[-1,-1]],[[1,-1],[-1,1]]],dtype=torch.float32)/2
        # Band-major channels: each output band is four contiguous RAW channels.
        weight=torch.zeros(16,4,2,2)
        for band in range(4):
            for c in range(4):weight[band*4+c,c]=kernels[band]
        self.register_buffer('weight',weight,persistent=False)

    def dwt2(self,x):
        return F.conv2d(x,self.weight,stride=2).split(4,dim=1)

    def iwt2(self,ll,lh,hl,hh):
        return F.conv_transpose2d(torch.cat((ll,lh,hl,hh),1),self.weight,stride=2)
