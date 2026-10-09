"""HF CNN training gradients, configuration isolation and checkpoint compatibility."""
from pathlib import Path

import pytest
import torch
import yaml

from learning_wt.learning_dwt import learning_dwt_kwargs
from models.learning_dwt_repncb import LearningDWTRepNCB, refiner_kwargs
from utils.model_factory import build_denoiser_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_'


@pytest.mark.parametrize('variant,hf_params', [('dw1',72), ('dw3',360), ('dw3_pw1',828), ('dw1_residual',72), ('dw3_residual',360), ('dw3_prelu_residual',396),
    ('dw3_skipdw1_residual',432), ('dw3_prelu_skipdw1_residual',468),
    ('repncb_prelu_residual',14328)])
def test_training_and_checkpoint_roundtrip(variant, hf_params, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    config = yaml.safe_load((ROOT/'configs'/f'{PREFIX}hf_cnn_depth5_{variant}_ll_no_norm.yaml').read_text())
    baseline = yaml.safe_load((ROOT/'configs'/f'{PREFIX}static_hf_depth5_soft_ll_no_norm.yaml').read_text())
    for key, value in baseline.items():
        # GPU assignment is an execution setting users may change independently.
        if key not in ('output_dir', 'dwt_threshold_mode', 'device'):
            assert config[key] == value
    assert config['resume'] is None and config['init_checkpoint'] is None
    assert config['output_dir'] != baseline['output_dir']
    model = LearningDWTRepNCB(learning_dwt_kwargs(config), refine_config=refiner_kwargs(config))
    wavelet = model.wavelet
    assert len(wavelet.processors) == 3
    for processor in wavelet.processors:
        assert processor.residual == variant.endswith("_residual")
        if variant.startswith('repncb'):
            assert isinstance(processor.block.activation, torch.nn.PReLU)
            assert sum(isinstance(m, torch.nn.PReLU) for m in processor.modules()) == 1
        else:
            assert processor.depthwise.kernel_size == ((3, 3) if variant.startswith("dw3") else (1, 1))
            assert isinstance(processor.relu, torch.nn.PReLU if 'prelu' in variant else torch.nn.ReLU)
        if 'skipdw1' in variant:
            assert processor.skip_projection.groups == 12
            sample = torch.randn(1,12,5,7)
            torch.testing.assert_close(processor.skip_projection(sample),sample,rtol=0,atol=0)
    assert sum(p.numel() for p in wavelet.processors.parameters()) == hf_params
    assert not hasattr(wavelet, 'hf_logits') and wavelet.predictor is None
    assert len({next(p.parameters()).data_ptr() for p in wavelet.processors}) == 3
    x = torch.randn(1,4,32,40)*0.2
    model(x).abs().mean().backward()
    for processor in wavelet.processors:
        grads = [p.grad for p in processor.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    for branch in (wavelet.ll_restorer, model.refiner):
        assert all(p.requires_grad and p.grad is not None for p in branch.parameters())
    model.eval()
    deployed = model.deploy().eval()
    for shape in ((1,4,32,40), (1,4,31,37)):
        image = torch.randn(shape)*0.2
        with torch.no_grad():
            torch.testing.assert_close(deployed(image),model(image),atol=3e-6,rtol=2e-5)
    for name, state in [('model',model.state_dict()), ('model_deploy',deployed.state_dict())]:
        path = tmp_path/f'{name}.pth'
        torch.save({'args':config,name:state},path)
        restored, _ = build_denoiser_from_checkpoint(str(path),device='cpu')
        with torch.no_grad():
            torch.testing.assert_close(restored(x),deployed(x),atol=3e-6,rtol=2e-5)
