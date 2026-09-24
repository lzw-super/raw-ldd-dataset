"""Band/channel firm and PWL shrinkage following tmp/div-sub-code.txt.
Odd ReLU extension avoids explicit Abs/Sign; all parameter transforms freeze at deploy.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DirectShrinkage(nn.Module):
    def __init__(self, mode, levels=3, normalize=True):
        super().__init__()
        self.mode=mode
        self.register_buffer('scales',torch.tensor([2**j if normalize else 1 for j in range(levels,0,-1) for _ in range(3)])[:,None,None],persistent=False)
        if mode=='firm':
            self.p1=nn.Parameter(torch.zeros(3*levels,4))
            self.p2=nn.Parameter(torch.ones(3*levels,4))
        elif mode=='pwl':
            self.p_tau=nn.Parameter(torch.zeros(3*levels,4,3))
            self.p_slope=nn.Parameter(torch.zeros(3*levels,4,3))
        else:
            raise ValueError('Expected firm or pwl')

    def coefficients(self):
        if hasattr(self,'fixed_tau'):
            return self.fixed_tau,self.fixed_weights
        if self.mode=='firm':
            t1=F.softplus(self.p1);delta=F.softplus(self.p2)+1e-4
            t2=t1+delta;k=t2/delta
            tau=torch.stack((t1,t2),-1)*self.scales
            weights=torch.stack((torch.zeros_like(k),k,1-k),-1)
        else:
            tau=(F.softplus(self.p_tau)+1e-4).cumsum(-1)*self.scales
            m=torch.sigmoid(self.p_slope)
            weights=torch.stack((m[...,0],m[...,1]-m[...,0],m[...,2]-m[...,1],1-m[...,2]),-1)
        return tau,weights

    def forward(self,z,band):
        tau,w=self.coefficients()
        out=w[band,:,0].view(1,4,1,1)*z
        for j in range(tau.shape[-1]):
            t=tau[band,:,j].view(1,4,1,1)
            out=out+w[band,:,j+1].view(1,4,1,1)*(F.relu(z-t)-F.relu(-z-t))
        return out

    @torch.no_grad()
    def freeze(self):
        if hasattr(self,'fixed_tau'):return
        tau,w=self.coefficients()
        self.register_buffer('fixed_tau',tau.detach().clone())
        self.register_buffer('fixed_weights',w.detach().clone())
        for name in list(self._parameters):delattr(self,name)
