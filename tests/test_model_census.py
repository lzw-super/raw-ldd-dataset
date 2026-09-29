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
