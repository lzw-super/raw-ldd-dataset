from pathlib import Path
import pytest
import torch
import yaml
from learning_wt.learning_dwt import learning_dwt_kwargs, haar_dwt2
from models.learning_dwt_repncb import LearningDWTRepNCB, RepNCB, refiner_kwargs
from utils.model_factory import build_denoiser_from_checkpoint

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('label,nrep', [('all_conv',0),('ll_bypass_soft',5),('ll_bypass',8),('hf_soft',8)])
def test_ablations(label,nrep,tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    path=next((ROOT/'configs').glob('*repncb_prelu_residual_ll_no_norm_ablation_'+label+'.yaml'))
    cfg=yaml.safe_load(path.read_text())
    model=LearningDWTRepNCB(learning_dwt_kwargs(cfg),refine_config=refiner_kwargs(cfg))
    assert sum(isinstance(m,RepNCB) for m in model.modules())==nrep
    wave=model.wavelet
    if label=='all_conv':
        assert sum(isinstance(m,torch.nn.PReLU) for m in model.modules())==11
        for processor in wave.processors:
            assert processor.block[0].groups==1
    bypass=label.startswith('ll_bypass')
    assert (wave.ll_restorer is None)==bypass
    if bypass:
        assert not any('ll_restorer' in k for k in model.state_dict())
        ll=torch.randn(1,4,4,5)
        assert wave.restore_ll(ll) is ll
    if label.endswith('soft'):
        assert wave.hf_logits.numel()==36 and wave.predictor is None
        assert not hasattr(wave,'processors')
        expected=torch.tensor([.08]*3+[.04]*3+[.02]*3)[:,None].expand(9,4)
        torch.testing.assert_close(wave.hf_threshold_values(),expected)
    x=torch.randn(1,4,32,40)*.2
    out=model(x)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if label.endswith('soft'):
        assert wave.hf_logits.grad.abs().sum()>0
    if bypass:
        before=x
        after=wave(x)
        for _ in range(3):
            before=haar_dwt2(before)[0];after=haar_dwt2(after)[0]
        torch.testing.assert_close(after,before,atol=2e-6,rtol=2e-5)
    deployed=model.deploy().eval()
    for shape in [(1,4,32,40),(1,4,31,37)]:
        inp=torch.randn(shape)*.2
        with torch.no_grad():
            torch.testing.assert_close(deployed(inp),model(inp),atol=3e-6,rtol=3e-5)
    for key,state in [('model',model.state_dict()),('model_deploy',deployed.state_dict())]:
        cp=tmp_path/f'{key}.pth'
        torch.save({'args':cfg,key:state},cp)
        restored,_=build_denoiser_from_checkpoint(str(cp),'cpu')
        with torch.no_grad():
            torch.testing.assert_close(restored(x),deployed(x),atol=3e-6,rtol=3e-5)
