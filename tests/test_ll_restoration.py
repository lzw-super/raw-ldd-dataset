from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from learning_wt.learning_dwt import LearningDWT, DWT_DEFAULTS, learning_dwt_kwargs
from utils.model_factory import build_denoiser_from_checkpoint


@pytest.mark.parametrize('mode,channels,fusion', [('ll_only',4,'residual'),('level_bands',16,'residual'),('ll_only',4,'concat_1x1'),('level_bands',16,'concat_1x1')])
def test_branch_inputs_hf_preservation_gradients_and_checkpoint(mode, channels, fusion, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(21)
    args={**DWT_DEFAULTS, 'model':'learning_dwt', 'dwt_width':8, 'dwt_context':'atlas',
          'dwt_condition_bands':False, 'dwt_ll_mode':mode, 'dwt_ll_width':8, 'dwt_ll_fusion':fusion}
    model=LearningDWT(**learning_dwt_kwargs(args))
    captured=[]
    hook=model.ll_restorer.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
    x=torch.randn(2,4,65,73)*.2
    y,aux=model(x,True)
    hook.remove()
    assert y.shape==x.shape
    assert captured[0].shape==(2,channels,9,10)
    expected=torch.cat([aux['atlas'][...,b.rows,b.cols] for b in aux['bands'][:channels//4]],dim=1)/8
    torch.testing.assert_close(captured[0],expected)
    ll=aux['bands'][0]
    assert aux['threshold'][...,ll.rows,ll.cols].count_nonzero()==0
    # The complete HF path is unchanged when only LL restoration is enabled.
    reference=LearningDWT(**learning_dwt_kwargs({**args,'dwt_ll_mode':'threshold','dwt_shrink_ll':False}))
    reference.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('ll_restorer.')})
    _,ref=reference(x,True)
    for band in aux['bands'][1:]:
        torch.testing.assert_close(aux['filtered_atlas'][...,band.rows,band.cols],ref['filtered_atlas'][...,band.rows,band.cols])
    (y-torch.rand_like(y)).abs().mean().backward()
    for module in (model.ll_restorer,model.predictor):
        for p in module.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert p.grad.abs().sum()>0
    cp=tmp_path/'model.pth';torch.save({'model':model.state_dict(),'args':args},cp)
    loaded,meta=build_denoiser_from_checkpoint(str(cp),'cpu')
    torch.testing.assert_close(loaded(x),model(x))
    assert meta['learning_dwt']['ll_mode']==mode


def test_ll_can_increase_coefficients_and_change_sign():
    model=LearningDWT(ll_mode='ll_only',ll_width=8)
    for p in model.ll_restorer.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.constant_(model.ll_restorer.layers[-1].bias,0.1)
    x=torch.full((1,4,64,64),-0.01)
    _,aux=model(x,True);ll=aux['bands'][0]
    original=aux['atlas'][...,ll.rows,ll.cols]
    restored=aux['filtered_atlas'][...,ll.rows,ll.cols]
    torch.testing.assert_close(restored,original+0.8)
    assert (restored>0).all()


def test_configs_change_only_experiment_fields():
    base=yaml.safe_load(Path('configs/train_sid_sony_learning_dwt_sym4_l3_d64_atlas.yaml').read_text())
    for suffix,mode in [('ll3_cnn','ll_only'),('ll3_fourbands_cnn','level_bands')]:
        config=yaml.safe_load(Path(f'configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_{suffix}.yaml').read_text())
        assert config.pop('dwt_ll_mode')==mode
        assert config.pop('dwt_ll_width')==32
        assert config.pop('dwt_ll_depth')==4
        assert config.pop('dwt_shrink_ll') is False
        assert config.pop('output_dir')!=base['output_dir']
        assert config=={k:v for k,v in base.items() if k not in ('output_dir','dwt_shrink_ll')}


def test_old_checkpoint_has_no_new_parameters(tmp_path):
    args={**DWT_DEFAULTS,'model':'learning_dwt','dwt_width':8}
    for k in ('dwt_ll_mode','dwt_ll_width','dwt_ll_depth'):args.pop(k)
    model=LearningDWT(**learning_dwt_kwargs(args))
    assert not any(k.startswith('ll_restorer.') for k in model.state_dict())
    path=tmp_path/'old.pth';torch.save({'model':model.state_dict(),'args':args},path)
    loaded,_=build_denoiser_from_checkpoint(str(path),'cpu')
    assert loaded.ll_mode=='threshold' and loaded.ll_restorer is None


def test_concat_fusion_initialization_and_learnable_channel_mixing():
    from learning_wt.learning_dwt import LLRestorationCNN
    branch=LLRestorationCNN(4,8,4,fusion="concat_1x1")
    x=torch.randn(2,4,8,8)
    cnn=branch.layers(x)
    torch.testing.assert_close(branch(x),x+cnn)
    # Verify concatenation order [CNN, LL], with no hidden addition after fusion.
    with torch.no_grad():
        branch.fusion.weight.zero_()
        branch.fusion.bias.fill_(0.25)
        for c in range(4):
            branch.fusion.weight[c,(c+1)%4+4,0,0]=2
    torch.testing.assert_close(branch(x),2*x[:,[1,2,3,0]]+0.25)


@pytest.mark.parametrize("branch", ["ll3", "ll3_fourbands"])
def test_concat_config_only_changes_fusion_and_output(branch):
    root=Path('configs')
    base=yaml.safe_load((root/f'train_sid_sony_learning_dwt_sym4_l3_d32_atlas_{branch}_cnn.yaml').read_text())
    variant=yaml.safe_load((root/f'train_sid_sony_learning_dwt_sym4_l3_d32_atlas_{branch}_cnn_concat1x1.yaml').read_text())
    assert variant.pop('dwt_ll_fusion')=='concat_1x1'
    assert variant.pop('output_dir')!=base.pop('output_dir')
    assert variant==base
