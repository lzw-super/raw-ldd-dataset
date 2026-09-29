from pathlib import Path
import pytest
import torch
import yaml
from models.paper_denoisers import build_paper_denoiser,SplitterNet
from utils.model_factory import build_denoiser_from_checkpoint


@pytest.mark.parametrize('name',['splitternet','brve_single_frame'])
def test_training_and_checkpoint(name,tmp_path):
    torch.set_num_threads(2)
    args=yaml.safe_load(Path('configs/train_sid_sony_'+name+'.yaml').read_text())
    net=build_paper_denoiser(name)
    x=torch.rand(1,4,65,73);y=net(x)
    assert y.shape==x.shape and torch.isfinite(y).all()
    (y-x).abs().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    path=tmp_path/'checkpoint.pth';torch.save({'args':args,'model':net.state_dict()},path)
    loaded,meta=build_denoiser_from_checkpoint(str(path),'cpu')
    with torch.no_grad():torch.testing.assert_close(loaded(x),y)
    assert meta['model']==name


def test_paper_parameter_count_and_training_config():
    assert sum(p.numel() for p in SplitterNet(channels=3).parameters())==731059
    base=yaml.safe_load(Path('configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft.yaml').read_text())
    for name in ['splitternet','brve_single_frame']:
        args=yaml.safe_load(Path('configs/train_sid_sony_'+name+'.yaml').read_text())
        assert {k for k in args if args[k]!=base[k]}=={'model','output_dir'}
