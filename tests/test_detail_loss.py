from pathlib import Path
import numpy as np
import pywt
import pytest
import torch
import yaml
from losses.detail_loss import GradientLoss, WaveletHFLoss


def test_sobel_ramp_and_signed_gradient():
    loss = GradientLoss()
    x = torch.arange(12, dtype=torch.float64).view(1, 1, 1, 12).expand(1, 4, 12, 12).clone().requires_grad_()
    g = loss.gradients(x).reshape(1, 4, 2, 12, 12)
    torch.testing.assert_close(g[:, :, 0, 1:-1, 1:-1], torch.ones(1, 4, 10, 10, dtype=torch.float64))
    assert g[:, :, 1].abs().max() == 0
    assert loss(x, x + 3).item() == 0
    value = loss(x, -x)
    assert value > 0
    value.backward()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize('basis', WaveletHFLoss.SUPPORTED)
@pytest.mark.parametrize('shape', [(32, 40), (33, 41)])
def test_hf_matches_pywt(basis, shape):
    torch.manual_seed(5)
    p = torch.randn(2, 4, *shape, dtype=torch.float64, requires_grad=True)
    t = torch.randn_like(p)
    criterion = WaveletHFLoss(basis, 2)
    a, b = p.detach().numpy(), t.numpy()
    expected = []
    for _ in range(2):
        a, ah = pywt.dwt2(a, basis, mode='symmetric')
        b, bh = pywt.dwt2(b, basis, mode='symmetric')
        expected.append(np.mean([np.abs(x-y).mean() for x, y in zip(ah, bh)]))
    value = criterion(p, t)
    assert value.item() == pytest.approx(np.mean(expected), rel=1e-10)
    value.backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert criterion(t, t) == 0


def test_hf_no_dc_and_gradcheck():
    loss = WaveletHFLoss('haar', 2)
    assert loss(torch.ones(1, 4, 16, 16), torch.zeros(1, 4, 16, 16)).abs() < 1e-6
    x = torch.randn(1, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    for criterion in (GradientLoss(), loss):
        assert torch.autograd.gradcheck(criterion, (x, torch.zeros_like(x)))
    with pytest.raises(ValueError):
        WaveletHFLoss('sym4', 3)(x, x)


def test_configs_preserve_baseline():
    root = Path('configs')
    name = 'train_sid_sony_mrlfn_paper_s2d_k4_n4_d32'
    base = yaml.safe_load((root / (name+'.yaml')).read_text())
    for suffix, additions in [('_g_sobel_w005', {'gradient_loss_weight': 0.05}),
                              ('_w_hf_haar_l2_w010', {'wavelet_hf_loss_weight': 0.1, 'wavelet_hf_basis': 'haar', 'wavelet_hf_levels': 2})]:
        config = yaml.safe_load((root / (name+suffix+'.yaml')).read_text())
        assert config.pop('output_dir') == base['output_dir']+suffix
        for key, value in additions.items():
            assert config.pop(key) == value
        assert config == {k:v for k,v in base.items() if k != 'output_dir'}
