from pathlib import Path
import numpy as np
import torch
import yaml
from learning_wt.learning_dwt import learning_dwt_kwargs, haar_dwt2
from learning_wt.bayes_shrink import estimate_noise_variance, bayes_soft_shrink
from models.learning_dwt_repncb import LearningDWTRepNCB, refiner_kwargs
from models.wavelet_denoiser import WaveletDenoiser
from utils.model_factory import build_denoiser_from_checkpoint


def test_classical_parity_and_refiner_training(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    root=Path(__file__).resolve().parents[1]
    path=next((root/'configs').glob('*ablation_ll_bypass_bayes_soft.yaml'))
    cfg=yaml.safe_load(path.read_text())
    base=yaml.safe_load(path.with_name(path.name.replace('bayes_soft','soft')).read_text())
    assert refiner_kwargs(cfg)==refiner_kwargs(base)
    model=LearningDWTRepNCB(learning_dwt_kwargs(cfg),refine_config=refiner_kwargs(cfg))
    assert not list(model.wavelet.parameters())
    assert model.wavelet.ll_restorer is None
    oracle=WaveletDenoiser(levels=3,wavelet='haar',boundary='periodization')
    x=torch.randn(2,4,32,40)*.2
    x[0,0]=0
    x[0,1]=.25
    # Different per-image noise levels must not share statistics.
    x[1]*=3
    actual=model.wavelet(x)
    expected=torch.from_numpy(np.stack([oracle(v.numpy()) for v in x]))
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=2e-5)
    a,b=x,actual
    for _ in range(3):a,b=haar_dwt2(a)[0],haar_dwt2(b)[0]
    torch.testing.assert_close(a,b,atol=1e-6,rtol=2e-5)
    out=model(x)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.refiner.parameters())
    deploy=model.deploy()
    for shape in [(2,4,32,40),(1,4,31,37)]:
        inp=torch.randn(shape)*.2
        torch.testing.assert_close(deploy(inp),model(inp),atol=3e-6,rtol=3e-5)
    for key,state in [('model',model.state_dict()),('model_deploy',deploy.state_dict())]:
        cp=tmp_path/f'{key}.pth'
        torch.save({'args':cfg,key:state},cp)
        restored,_=build_denoiser_from_checkpoint(str(cp),'cpu')
        torch.testing.assert_close(restored(x),deploy(x),atol=3e-6,rtol=3e-5)


def test_median_zero_noise_and_noise_only_bands():
    hh=torch.tensor([[[[0.,1.,2.,3.,4.]]]])
    var=estimate_noise_variance(hh)
    torch.testing.assert_close(var,torch.tensor([[[[(2.5/.6744897501960817)**2]]]],dtype=torch.float64))
    detail=torch.tensor([[[[-.1,.2]]]])
    torch.testing.assert_close(bayes_soft_shrink(detail,torch.zeros_like(var)),detail,rtol=0,atol=0)
    assert torch.equal(bayes_soft_shrink(detail,var),torch.zeros_like(detail))
