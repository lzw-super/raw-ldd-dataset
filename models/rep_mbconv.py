"""PlainUSR MBConv/ASR adaptation (MIT; see licenses/PlainUSR-MIT.txt).
Source: icandle/PlainUSR, 2024_PlainUSR_ACCV/archs/PlainUSR_train_arch.py.
Preserves official ratio=2 and ASR initialization; fusion is device/dtype safe.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RepMBConv(nn.Module):
    def __init__(self, channels, deploy=False):
        super().__init__()
        if deploy:
            self.reparam_conv = nn.Conv2d(channels, channels, 3, padding=1)
            return
        hidden = channels * 2
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.spatial = nn.Conv2d(hidden, hidden, 3)
        self.reduce = nn.Conv2d(hidden, channels, 1)
        self.attention_tensor = nn.Parameter(torch.full((1, hidden), 0.1))
        self.attention = nn.Sequential(nn.Linear(hidden, hidden//4, bias=False), nn.SiLU(),
                                       nn.Linear(hidden//4, hidden, bias=False), nn.Sigmoid())
        nn.init.ones_(self.attention[0].weight)
        nn.init.ones_(self.attention[2].weight)

    def forward(self, x):
        if hasattr(self, 'reparam_conv'):
            return self.reparam_conv(x)
        expanded = self.expand(x)
        # Padding the input before the 1x1 yields bias-valued borders.
        spatial = self.spatial(self.expand(F.pad(x, (1,1,1,1))))
        scale = self.attention(self.attention_tensor)[...,None,None]
        return self.reduce(spatial * scale + expanded) + x

    @torch.no_grad()
    def switch_to_deploy(self):
        if hasattr(self, 'reparam_conv'):
            return self
        scale = self.attention(self.attention_tensor).reshape(-1)
        spatial = self.spatial.weight * scale[:,None,None,None]
        idx = torch.arange(spatial.shape[0], device=spatial.device)
        spatial[idx,idx,1,1] += 1
        bias = self.spatial.bias * scale
        weight = torch.einsum('omhw,mi->oihw', spatial, self.expand.weight[:,:,0,0])
        bias = bias + torch.einsum('omhw,m->o', spatial, self.expand.bias)
        projection = self.reduce.weight[:,:,0,0]
        weight = torch.einsum('om,mihw->oihw', projection, weight)
        bias = projection @ bias + self.reduce.bias
        idx = torch.arange(weight.shape[0],device=weight.device)
        weight[idx,idx,1,1] += 1
        self.reparam_conv = nn.Conv2d(weight.shape[1],weight.shape[0],3,padding=1).to(weight)
        self.reparam_conv.weight.copy_(weight)
        self.reparam_conv.bias.copy_(bias)
        for name in ('expand','spatial','reduce','attention_tensor','attention'):
            delattr(self,name)
        return self
