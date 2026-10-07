import torch
from tools.profile_sid_models import Census


def test_conv_count_and_live_storage():
    model=torch.nn.Conv2d(4,8,3,padding=1,bias=False)
    x=torch.ones(1,4,8,8);c=Census(model)
    with torch.inference_mode(),c:y=model(x)
    assert c.macs==18432 and sum(c.ops.values())==36864
    assert c.peak==(x.numel()+y.numel())*4
    assert not c.unknown


def test_skip_lifetime_and_alias_deduplication():
    class Skip(torch.nn.Module):
        def forward(self,x):
            skip=x.clone();alias=skip.view_as(skip);v=x.clone()
            return alias+v
    model=Skip();x=torch.ones(1,4,8,8);c=Census(model)
    with torch.inference_mode(),c:y=model(x)
    # Input, retained skip, second branch and output coexist. View adds no storage.
    assert c.peak==4*x.numel()*4
    assert c.current==2*x.numel()*4
    assert not c.unknown


def test_reductions_and_pooling_count_comparisons():
    model=torch.nn.Identity();x=torch.ones(1,4,8,8);c=Census(model)
    with torch.inference_mode(),c:
        mean=x.mean(1)
        maximum=x.amax(1)
        pooled=torch.nn.functional.max_pool2d(x,2)
    assert c.ops['mean']==256  # 64 groups: 3 additions + 1 division.
    assert c.ops['amax']==192  # 64 groups: 3 comparisons.
    assert c.ops['max_pool2d']==192
    assert sum(c.categories.values())==sum(c.ops.values())==640
    assert not c.unknown


def test_soft_shrink_counts_six_ops_per_coefficient():
    z=torch.randn(1,4,8,8);threshold=torch.ones(1,4,1,1)*.1
    c=Census(torch.nn.Identity())
    with torch.inference_mode(),c:
        restored=torch.relu(z-threshold)-torch.relu(-z-threshold)
    assert sum(c.ops.values())==6*z.numel()
    assert c.ops['sub']==3*z.numel()
    assert c.ops['neg']==z.numel()
    assert c.ops['relu']==2*z.numel()
    assert c.ops['div']==0 and not c.unknown


def test_fixed_haar_dense_forward_inverse_accounting():
    from models.fixed_raw_ops import FixedHaarConv
    model=torch.nn.Module();model.wavelet=torch.nn.Module()
    model.wavelet.transform=FixedHaarConv()
    x=torch.randn(1,4,8,8);c=Census(model)
    with torch.inference_mode(),c:
        bands=model.wavelet.transform.dwt2(x)
        restored=model.wavelet.transform.iwt2(*bands)
    torch.testing.assert_close(restored,x)
    assert c.categories['fixed_haar']==64*x.numel()
    assert len(c.fixed_events)==2
    assert all(e['ops']==32*x.numel() for e in c.fixed_events)
    assert sum(c.ops.values())==c.categories['fixed_haar']
    assert not c.unknown
