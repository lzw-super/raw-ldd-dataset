import copy
from pathlib import Path
import pytest
import torch
import yaml
from models.rep_mbconv import RepMBConv
from models.learning_dwt_repncb import LearningDWTRepNCB,refiner_kwargs
from learning_wt.learning_dwt import learning_dwt_kwargs
from utils.model_factory import build_denoiser_from_checkpoint


@pytest.mark.parametrize('shape',[(1,8,1,1),(2,8,7,9)])
def test_fusion(shape):
    torch.set_num_threads(2)
    m=RepMBConv(8).double()
    with torch.no_grad():
        for p in m.parameters():p.uniform_(-.2,.2)
    x=torch.randn(shape,dtype=torch.float64)
    fused=copy.deepcopy(m).switch_to_deploy()
    torch.testing.assert_close(fused(x),m(x),atol=1e-12,rtol=1e-12)
    assert hasattr(m,'expand')
    torch.testing.assert_close(fused.switch_to_deploy()(x),m(x),atol=1e-12,rtol=1e-12)


def test_config_and_checkpoint(tmp_path):
    args=yaml.safe_load(Path('configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repmbconv.yaml').read_text())
    m=LearningDWTRepNCB(learning_dwt_kwargs(args),refine_config=refiner_kwargs(args))
    layers=m.wavelet.ll_restorer.layers
    assert isinstance(layers[0],torch.nn.Conv2d) and layers[0].in_channels==4
    assert isinstance(layers[-1],torch.nn.Conv2d) and layers[-1].out_channels==4
    assert sum(isinstance(b,RepMBConv) for b in m.modules())==2
    x=torch.rand(1,4,33,41);y=m(x)
    (y-x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    for deploy in (False,True):
        cp={'args':args,'model':m.state_dict()}
        if deploy:cp['model_deploy']=m.deploy().state_dict()
        p=tmp_path/'model.pth';torch.save(cp,p)
        loaded,_=build_denoiser_from_checkpoint(str(p),'cpu')
        torch.testing.assert_close(loaded(x),y,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('suffix,kind,count', [('ll_repncb','repncb',2),('ll_repmbconv_depth5','repmbconv',3)])
def test_ll_variants(suffix,kind,count,tmp_path):
    from models.learning_dwt_repncb import RepNCB
    args=yaml.safe_load(Path('configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_'+suffix+'.yaml').read_text())
    model=LearningDWTRepNCB(learning_dwt_kwargs(args),refine_config=refiner_kwargs(args))
    layers=model.wavelet.ll_restorer.layers
    block_cls=RepNCB if kind=='repncb' else RepMBConv
    assert sum(isinstance(b,block_cls) for b in layers)==count
    assert isinstance(layers[0],torch.nn.Conv2d) and layers[0].weight.shape==(32,4,3,3)
    assert isinstance(layers[-1],torch.nn.Conv2d) and layers[-1].weight.shape==(4,32,3,3)
    x=torch.rand(1,4,33,41);y=model(x)
    (y-x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for deploy in (False,True):
        cp={'args':args,'model':model.state_dict()}
        if deploy:cp['model_deploy']=model.deploy().state_dict()
        path=tmp_path/'model.pth';torch.save(cp,path)
        loaded,_=build_denoiser_from_checkpoint(str(path),'cpu')
        torch.testing.assert_close(loaded(x),y,atol=2e-6,rtol=2e-5)
