import torch
from learning_wt.trainable_haar import TrainableHaar
from learning_wt.learning_dwt import dwt_atlas
from models.learning_dwt_repncb import LearningDWTRepNCB
from train_sid_sony import load_checkpoint


def test_haar_initialization_and_frozen_cnn_step(tmp_path):
    torch.set_num_threads(2)
    transform = TrainableHaar()
    x = torch.rand(1,4,32,40)
    torch.testing.assert_close(transform.encode(x),dwt_atlas(x),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(transform.decode(transform.encode(x)),x,atol=1e-6,rtol=1e-6)
    original = LearningDWTRepNCB({'wavelet':'haar'})
    model = LearningDWTRepNCB({'wavelet':'haar','trainable_haar':True})
    path = tmp_path/'baseline.pth'
    torch.save({'model':original.state_dict()},path)
    load_checkpoint(path,model,initialize_haar=True)
    torch.testing.assert_close(model(x),original(x),atol=2e-6,rtol=2e-5)
    keys={'wavelet.transform.analysis','wavelet.transform.synthesis'}
    for name,p in model.named_parameters():p.requires_grad_(name in keys)
    before={k:v.clone() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    (model(x)-x).abs().mean().backward()
    opt.step()
    assert {k for k,v in model.state_dict().items() if not torch.equal(v,before[k])}==keys
    torch.testing.assert_close(model.deploy()(x),model(x),atol=2e-6,rtol=2e-5)


import pytest


@pytest.mark.parametrize('channels,levels,count', [(False,True,128),(True,False,96),(False,False,384)])
def test_unshared_banks(channels,levels,count,tmp_path):
    from utils.model_factory import build_denoiser_from_checkpoint
    from learning_wt.learning_dwt import DWT_DEFAULTS,learning_dwt_kwargs
    torch.set_num_threads(2)
    w=TrainableHaar(channels,levels,3)
    assert sum(p.numel() for p in w.parameters())==count
    x=torch.rand(1,4,32,40)
    torch.testing.assert_close(w.encode(x),dwt_atlas(x),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(w.decode(w.encode(x)),x,atol=1e-6,rtol=1e-6)
    # Each bank must act only on the requested RAW channel and level.
    with torch.no_grad():
        w.analysis[0,0,0,0,0,0]+=0.1
    a=w.dwt2(x,0);b=TrainableHaar().dwt2(x)
    if not channels:
        torch.testing.assert_close(a[0][:,1:],b[0][:,1:])
    if not levels:
        for actual,expected in zip(w.dwt2(x,1),b):torch.testing.assert_close(actual,expected)
    args={**DWT_DEFAULTS,'model':'learning_dwt_repncb','dwt_wavelet':'haar','dwt_trainable_haar':True,
          'dwt_haar_share_channels':channels,'dwt_haar_share_levels':levels}
    model=LearningDWTRepNCB(learning_dwt_kwargs(args))
    keys={'wavelet.transform.analysis','wavelet.transform.synthesis'}
    for name,p in model.named_parameters():p.requires_grad_(name in keys)
    before={k:v.clone() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    (model(x)-x).square().mean().backward();opt.step()
    assert {k for k,v in model.state_dict().items() if not torch.equal(v,before[k])}==keys
    for p in model.wavelet.transform.parameters():
        assert torch.isfinite(p.grad).all()
        assert (p.grad.abs().sum(dim=(2,3,4,5))>0).all()
    path=tmp_path/'banks.pth'
    torch.save({'model':model.state_dict(),'args':args},path)
    loaded,_=build_denoiser_from_checkpoint(str(path),'cpu')
    torch.testing.assert_close(loaded(x),model(x),atol=2e-6,rtol=2e-5)
