from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
import pywt

from losses.wavelet_ll_loss import WaveletLLLoss


@pytest.mark.parametrize('basis', WaveletLLLoss.SUPPORTED)
@pytest.mark.parametrize('shape', [(32, 40), (33, 41)])
def test_multilevel_matches_pywavelets_and_has_gradients(basis, shape):
    torch.manual_seed(1)
    pred = torch.randn(2, 4, *shape, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(pred)
    criterion = WaveletLLLoss(basis, levels=2)
    p, t = pred.detach().numpy(), target.numpy()
    expected = []
    for _ in range(2):
        p = pywt.dwt2(p, basis, mode='symmetric', axes=(-2, -1))[0]
        t = pywt.dwt2(t, basis, mode='symmetric', axes=(-2, -1))[0]
        expected.append(np.abs(p - t).mean())
    loss = criterion(pred, target)
    assert loss.item() == pytest.approx(np.mean(expected), rel=1e-10)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0
    assert criterion(target, target).item() == 0


def test_haar_constant_error_standard_scale_and_gradcheck():
    criterion = WaveletLLLoss('haar', 3)
    # Each 2D LL decomposition doubles a constant: mean(2,4,8).
    assert criterion(torch.ones(1, 4, 16, 16), torch.zeros(1, 4, 16, 16)).item() == pytest.approx(14/3)
    pred = torch.randn(1, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(WaveletLLLoss('db2', 1), (pred, torch.zeros_like(pred)))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        WaveletLLLoss('unknown')
    with pytest.raises(ValueError):
        WaveletLLLoss(levels=0)
    with pytest.raises(ValueError):
        WaveletLLLoss('sym4', 3)(torch.ones(1, 4, 16, 16), torch.ones(1, 4, 16, 16))


def test_configs_only_change_auxiliary_and_output():
    base = yaml.safe_load(Path('configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml').read_text())
    configs = list(Path('configs').glob('train_sid_sony_mrlfn_paper_s2d_k4_n4_d32_wavelet_ll_l3_*_w010.yaml'))
    assert len(configs) == 5
    outputs = set()
    for path in configs:
        config = yaml.safe_load(path.read_text())
        outputs.add(config.pop('output_dir'))
        assert config.pop('wavelet_basis') in WaveletLLLoss.SUPPORTED
        assert config.pop('wavelet_levels') == 3
        assert config.pop('wavelet_loss_weight') == 0.1
        assert config == {k: v for k, v in base.items() if k != 'output_dir'}
    assert len(outputs) == 5
    assert base['output_dir'] not in outputs
