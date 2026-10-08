from pathlib import Path

import torch
import yaml

from learning_wt.learning_dwt import learning_dwt_kwargs
from models.learning_dwt_repncb import LearningDWTRepNCB, refiner_kwargs
from utils.model_factory import build_denoiser_from_checkpoint


def test_noisy_s2d_projection_path_gradients_and_reload(tmp_path):
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    path = next((root/'configs').glob('*hf_cnn_depth5_dw1_residual_ll_no_norm_skip1x1.yaml'))
    config = yaml.safe_load(path.read_text())
    base = yaml.safe_load(path.with_name(path.name.replace('_skip1x1','')).read_text())
    assert config['output_dir'] != base['output_dir']
    assert {k:v for k,v in config.items() if k not in ('output_dir','refine_skip_projection')} == {k:v for k,v in base.items() if k != 'output_dir'}
    torch.manual_seed(2026)
    model = LearningDWTRepNCB(learning_dwt_kwargs(config), refine_config=refiner_kwargs(config))
    torch.manual_seed(2026)
    baseline = LearningDWTRepNCB(learning_dwt_kwargs(base), refine_config=refiner_kwargs(base))
    conv = model.refiner.skip_projection
    assert conv.in_channels == conv.out_channels == 16
    assert sum(p.numel() for p in conv.parameters()) == 272
    x = torch.randn(1,4,32,40)
    with torch.no_grad():
        torch.testing.assert_close(model(x),baseline(x),atol=0,rtol=0)
        conv.weight.mul_(0.7)
        conv.bias.fill_(0.02)
    captured = {}
    def capture(name):
        def hook(module, args):
            captured[name] = args[0].detach().clone()
        return hook
    handles = [conv.register_forward_pre_hook(capture('skip')),
               model.refiner.fusion.register_forward_pre_hook(capture('fusion'))]
    output = model(x)
    for h in handles:
        h.remove()
    torch.testing.assert_close(captured['skip'],torch.nn.functional.pixel_unshuffle(x,2))
    torch.testing.assert_close(captured['fusion'][:,16:],conv(captured['skip']))
    output.square().mean().backward()
    for p in conv.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    model.eval()
    deployed = model.deploy().eval()
    for shape in [(1,4,32,40),(1,4,31,37)]:
        x = torch.randn(shape)
        with torch.no_grad():
            torch.testing.assert_close(deployed(x),model(x),atol=5e-6,rtol=3e-5)
    for key,state in [('model',model.state_dict()),('model_deploy',deployed.state_dict())]:
        checkpoint=tmp_path/f'{key}.pth'
        torch.save({'args':config,key:state},checkpoint)
        restored,_=build_denoiser_from_checkpoint(str(checkpoint),'cpu')
        with torch.no_grad():
            torch.testing.assert_close(restored(x),deployed(x),atol=5e-6,rtol=3e-5)
