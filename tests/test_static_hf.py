from pathlib import Path
import torch
import yaml
from learning_wt.learning_dwt import learning_dwt_kwargs
from models.learning_dwt_repncb import LearningDWTRepNCB,refiner_kwargs
from utils.model_factory import build_denoiser_from_checkpoint


def test_static_hf_and_deployment(tmp_path):
    torch.set_num_threads(2)
    args=yaml.safe_load(Path('configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf.yaml').read_text())
    m=LearningDWTRepNCB(learning_dwt_kwargs(args),refine_config=refiner_kwargs(args))
    assert m.wavelet.predictor is None and m.wavelet.hf_logits.numel()==36
    assert not hasattr(m.wavelet,'band_bias')
    x=torch.rand(2,4,33,41)
    _,aux=m.wavelet(x,True)
    ll=aux['bands'][0]
    assert aux['threshold'][...,ll.rows,ll.cols].count_nonzero()==0
    for band in aux['bands'][1:]:
        v=aux['threshold'][...,band.rows,band.cols]
        torch.testing.assert_close(v,torch.full_like(v,.01*2**band.level))
    with torch.no_grad():m.wavelet.hf_logits.add_(torch.randn_like(m.wavelet.hf_logits)*.1)
    y=m(x);(y-x).square().mean().backward()
    assert torch.isfinite(m.wavelet.hf_logits.grad).all() and (m.wavelet.hf_logits.grad!=0).all()
    deployed=m.deploy()
    assert not hasattr(deployed.wavelet,'hf_logits')
    assert deployed.wavelet.fixed_hf_thresholds.shape==(9,4)
    torch.testing.assert_close(deployed(x),y,atol=2e-6,rtol=2e-5)
    for deploy in (False,True):
        cp={'args':args,'model':m.state_dict()}
        if deploy:cp['model_deploy']=deployed.state_dict()
        path=tmp_path/'model.pth';torch.save(cp,path)
        loaded,_=build_denoiser_from_checkpoint(str(path),'cpu')
        torch.testing.assert_close(loaded(x),y,atol=2e-6,rtol=2e-5)
