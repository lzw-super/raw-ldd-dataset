import torch
from models.fixed_raw_ops import FixedHaarConv,FixedSpaceDepth
from learning_wt.learning_dwt import haar_dwt2,haar_iwt2


def test_fixed_kernels_channel_order_and_inverse():
    x=torch.arange(4*8*10,dtype=torch.float64).reshape(1,4,8,10)/100
    s2d=FixedSpaceDepth().double();d2s=FixedSpaceDepth(True).double()
    torch.testing.assert_close(s2d(x),torch.nn.functional.pixel_unshuffle(x,2),rtol=0,atol=0)
    torch.testing.assert_close(d2s(s2d(x)),x,rtol=0,atol=0)
    haar=FixedHaarConv().double()
    for a,b in zip(haar.dwt2(x),haar_dwt2(x)):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    bands=tuple(torch.randn(1,4,4,5,dtype=torch.float64) for _ in range(4))
    torch.testing.assert_close(haar.iwt2(*bands),haar_iwt2(*bands),rtol=1e-12,atol=1e-12)
    assert not haar.state_dict() and not s2d.state_dict()
