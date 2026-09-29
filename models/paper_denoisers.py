"""SID adapters for official BRVE and a PyTorch port of SplitterNet.
See ref-doc/BRVE与SplitterNet_SID对照.md for provenance and task adaptations.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .brve_vendor.bnnvd import BNNVD


class BRVESingleFrame(nn.Module):
    """Repeat one observed RAW frame three times; return the centre output.

    No additional noisy observations are generated. The original video core remains
    available as .core for N,T,4,H,W input. This is not temporal-video evaluation.
    """
    def __init__(self):
        super().__init__()
        self.core=BNNVD(mid_channels=24,feat_extract_blocks=3,num_unets=1,
                        unet_n_feat=[24,48,96],unet_n_block=[1,3,3],
                        stage1_n_feat=[24,48,48,48],task='Raw2Raw')

    def forward(self,x):
        h,w=x.shape[-2:]
        # Spatial shifts reach 8 pixels at 1/8 resolution.
        ph=max(128,((h+7)//8)*8)-h;pw=max(128,((w+7)//8)*8)-w
        padded=F.pad(x,(0,pw,0,ph),mode='replicate') if ph or pw else x
        frames=padded[:,None].repeat(1,3,1,1,1)
        return self.core(frames)[:,1,:,:h,:w]


class ReflectConv(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.conv=nn.Conv2d(cin,cout,3,stride=stride)
    def forward(self,x):
        return self.conv(F.pad(x,(1,1,1,1),mode='reflect'))


class SplitterMiddle(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.conv1=ReflectConv(width,width)
        self.channel=nn.Conv2d(width,width,1)
        self.conv2=ReflectConv(width,width)
        self.spatial=ReflectConv(2,1)
    def forward(self,x):
        y=F.leaky_relu(self.conv1(x),negative_slope=.3)
        residual=x+y*self.channel(y.mean((2,3),keepdim=True))
        y=F.leaky_relu(self.conv2(residual),negative_slope=.3)
        attention=torch.sigmoid(self.spatial(torch.cat((y.mean(1,keepdim=True),y.amax(1,keepdim=True)),1)))
        return residual+y*attention


class SplitterNet(nn.Module):
    """Official four-level, width-32, no-LN SplitterNet port; RGB adapted to RAW.

    Keras ConvTranspose(k=3,s=2,padding=same) aligns to padding=0 followed
    by bottom/right cropping, NOT PyTorch padding=1/output_padding=1.
    """
    def __init__(self,channels=4,width=32):
        super().__init__()
        self.stem=nn.Conv2d(channels,width,3,padding=1)
        self.down=nn.ModuleList([nn.ModuleList([ReflectConv(width//2,width,2) for _ in range(2**(i+1))]) for i in range(4)])
        self.middle=nn.ModuleList([SplitterMiddle(width) for _ in range(16)])
        self.up=nn.ModuleList([nn.ModuleList([nn.ConvTranspose2d(2*width,width,3,stride=2) for _ in range(2**i)]) for i in reversed(range(4))])
        self.head=nn.Conv2d(width,channels,3,padding=1)
        # Keras Conv2D/Conv2DTranspose default Glorot uniform.
        for module in self.modules():
            if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self,x):
        h,w=x.shape[-2:]
        ph=max(32,((h+15)//16)*16)-h;pw=max(32,((w+15)//16)*16)-w
        raw=F.pad(x,(0,pw,0,ph),mode='replicate') if ph or pw else x
        branches=[self.stem(raw)];skips=[]
        for stage in self.down:
            skips.append(branches)
            halves=[part for b in branches for part in b.chunk(2,dim=1)]
            branches=[F.leaky_relu(layer(b),negative_slope=.3) for layer,b in zip(stage,halves)]
        branches=[layer(b) for layer,b in zip(self.middle,branches)]
        for stage,skip in zip(self.up,reversed(skips)):
            branches=[F.leaky_relu(layer(torch.cat((branches[2*i],branches[2*i+1]),1))[...,:s.shape[-2],:s.shape[-1]],negative_slope=.3)+s for i,(layer,s) in enumerate(zip(stage,skip))]
        return (raw+self.head(branches[0]))[...,:h,:w]


def build_paper_denoiser(name):
    if name=='brve_single_frame':return BRVESingleFrame()
    if name=='splitternet':return SplitterNet()
    raise ValueError(name)
