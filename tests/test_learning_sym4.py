import numpy as np
import pywt
import pytest
import torch
from learning_wt.sym4_transform import PeriodizedWavelet
from learning_wt.learning_dwt import LearningDWT, band_layout, DWT_DEFAULTS, learning_dwt_kwargs
from utils.model_factory import build_denoiser_from_checkpoint


def test_sym4_coefficients_inverse_energy():
    torch.set_num_threads(2)
    transform = PeriodizedWavelet().double()
    for shape in ((16,24),(64,80)):
        x = torch.randn(1,4,*shape,dtype=torch.float64)
        atlas = transform.encode(x,3)
        coeffs = pywt.wavedec2(x.numpy(), 'sym4', mode='periodization', level=3, axes=(-2,-1))
        reference, _ = pywt.coeffs_to_array(coeffs, axes=(-2,-1))
        np.testing.assert_allclose(atlas.numpy(), reference, atol=1e-11)
        torch.testing.assert_close(transform.decode(atlas,3), x, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(atlas.square().sum(), x.square().sum(),atol=1e-8,rtol=1e-10)


def test_sym4_gradcheck():
    transform = PeriodizedWavelet().double()
    x = torch.randn(1,1,8,8,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: transform.decode(transform.encode(z,3),3),(x,))


@pytest.mark.parametrize('context', ['atlas','bandwise'])
def test_sym4_network_gradients_and_ll_bound(context):
    model = LearningDWT(width=8, context=context, magnitude_input=True,
                        condition_bands=context=='bandwise',ll_max_threshold=.01,pad_input=True)
    x = torch.randn(2,4,65,73)*.1
    y, aux = model(x,True)
    assert y.shape == x.shape
    assert aux['threshold'].shape == (2,4,72,80)
    ll = aux['bands'][0]
    assert aux['threshold'][...,ll.rows,ll.cols].max() <= .08
    assert torch.all(aux['threshold'] >= 0)
    y.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("shrink_mode", ["soft", "smooth"])
def test_checkpoint_factory_roundtrip(tmp_path, shrink_mode):
    args = {**DWT_DEFAULTS,'model':'learning_dwt','dwt_width':8, "dwt_shrink_mode":shrink_mode}
    model = LearningDWT(**learning_dwt_kwargs(args)).eval()
    path=tmp_path/'test.pth'
    torch.save({'model':model.state_dict(),'args':args},path)
    restored, meta=build_denoiser_from_checkpoint(str(path),'cpu')
    x=torch.randn(1,4,64,72)
    torch.testing.assert_close(model(x),restored(x))
    assert meta['learning_dwt']['wavelet']=='sym4'


def test_ll_off_and_identity():
    x=torch.randn(1,4,64,72)
    torch.testing.assert_close(LearningDWT(leak=1)(x),x,atol=1e-5,rtol=1e-5)
    _,aux=LearningDWT(shrink_ll=False)(x,True)
    ll=aux['bands'][0]
    assert aux['threshold'][...,ll.rows,ll.cols].count_nonzero()==0


def test_configs_preserve_reference_training_settings():
    from pathlib import Path
    import yaml
    base=yaml.safe_load(Path('configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml').read_text())
    keys=('batch_size','patch_size','epochs','learning_rate','warmup_epochs','seed',
          'ratios','raw_loss_weight','chromatic_loss_weight','synthesis','adam_epsilon')
    for variant in ('bandwise','atlas','no_ll'):
        config=yaml.safe_load(Path(f'configs/train_sid_sony_learning_dwt_sym4_l3_d64_{variant}.yaml').read_text())
        assert all(config[k]==base[k] for k in keys)
        assert config['dwt_wavelet']=='sym4' and config['dwt_levels']==3
        assert config['output_dir'] != base['output_dir']
        LearningDWT(**learning_dwt_kwargs(config))


def test_smooth_shrink_no_dead_threshold_gradient():
    from learning_wt.learning_dwt import smooth_shrink
    z=torch.tensor([-0.01,0.,0.02],dtype=torch.float64)
    threshold=torch.ones_like(z,requires_grad=True)
    restored=smooth_shrink(z,threshold)
    assert (restored*z>=0).all() and (restored.abs()<=z.abs()).all()
    restored.abs().sum().backward()
    assert (threshold.grad[[0,2]]<0).all()
    torch.testing.assert_close(smooth_shrink(z,torch.zeros_like(z)),z)
    torch.testing.assert_close(smooth_shrink(z,threshold,leak=1),z)
    assert torch.autograd.gradcheck(lambda t:smooth_shrink(z,t),(threshold,))
