import copy
from pathlib import Path

import pytest
import torch
import yaml

from models.learning_dwt_repncb import RepNCB, PackedRAWRefiner, LearningDWTRepNCB
from learning_wt.learning_dwt import learning_dwt_kwargs
from utils.model_factory import build_denoiser_from_checkpoint


torch.set_num_threads(2)


@pytest.mark.parametrize('shape', [(1, 16, 1, 1), (2, 16, 7, 9)])
def test_fusion_including_bias_and_edges(shape):
    torch.manual_seed(7)
    block = RepNCB().double()
    with torch.no_grad():
        for p in block.parameters():
            p.uniform_(-.3, .3)
    x = torch.randn(shape, dtype=torch.float64) + 2
    fused = copy.deepcopy(block).switch_to_deploy()
    torch.testing.assert_close(block(x), fused(x), atol=1e-12, rtol=1e-12)
    assert hasattr(block, 'direct')
    torch.testing.assert_close(fused(x), fused.switch_to_deploy()(x))
    for name in ('gaussian', 'horizontal', 'vertical'):
        assert getattr(block, name).mask.sum() == 1


def test_original_input_skip_and_odd_crop():
    model = PackedRAWRefiner()
    with torch.no_grad():
        model.fusion.weight.zero_()
        model.fusion.bias.zero_()
        model.fusion.weight[:, 16:, 0, 0].copy_(torch.eye(16))
    noisy = torch.randn(2, 4, 17, 19)
    torch.testing.assert_close(model(torch.zeros_like(noisy), noisy), noisy, atol=0, rtol=0)


@pytest.mark.parametrize('pack_deploy', [False, True])
def test_full_model_gradient_and_checkpoint(tmp_path, pack_deploy):
    args = yaml.safe_load(Path('configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1_repncb.yaml').read_text())
    net = LearningDWTRepNCB(learning_dwt_kwargs(args))
    x = torch.rand(1, 4, 33, 41)
    y, aux = net(x, True)
    assert y.shape == aux['preliminary'].shape == x.shape
    (y - x).abs().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    fused = net.deploy()
    torch.testing.assert_close(fused(x), y, atol=2e-6, rtol=2e-5)
    assert hasattr(net.refiner.blocks[0], 'direct')
    checkpoint = {'args': args, 'model': net.state_dict()}
    if pack_deploy:
        checkpoint['model_deploy'] = fused.state_dict()
    path = tmp_path / 'checkpoint.pth'
    torch.save(checkpoint, path)
    restored, meta = build_denoiser_from_checkpoint(str(path), 'cpu')
    torch.testing.assert_close(restored(x), y, atol=2e-6, rtol=2e-5)
    assert meta['graph_state'] == 'deploy'


@pytest.mark.parametrize('k,width,blocks,source', [(1,16,4,'noisy'), (2,16,4,'preliminary'), (2,16,6,'noisy'), (3,8,2,'preliminary')])
def test_configurable_refiner(k, width, blocks, source, tmp_path):
    from models.learning_dwt_repncb import refiner_kwargs
    args = yaml.safe_load(Path('configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1_repncb.yaml').read_text())
    args.update(refine_s2d_factor=k, refine_width=width, refine_num_blocks=blocks, refine_skip_source=source)
    net = LearningDWTRepNCB(learning_dwt_kwargs(args), refine_config=refiner_kwargs(args))
    assert net.refiner.stem.in_channels == 4*k*k
    assert net.refiner.stem.out_channels == width
    assert net.refiner.head.in_channels == width
    assert net.refiner.head.out_channels == 4*k*k
    assert net.refiner.fusion.in_channels == 8*k*k
    assert net.refiner.fusion.out_channels == 4*k*k
    x = torch.rand(1,4,25,29)
    y = net(x)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    for deploy in (False, True):
        state = {'args':args, 'model':net.state_dict()}
        if deploy:
            state['model_deploy'] = net.deploy().state_dict()
        path = tmp_path/'model.pth'
        torch.save(state,path)
        loaded, meta = build_denoiser_from_checkpoint(str(path),'cpu')
        torch.testing.assert_close(loaded(x),y,atol=2e-6,rtol=2e-5)
        assert meta['refinement']['skip_source'] == source
        assert meta['feature_channels'] == width
        assert len(loaded.refiner.blocks) == blocks
    # Select only the bypass to distinguish I from I-, including odd-size padding.
    refiner = net.refiner
    with torch.no_grad():
        refiner.fusion.weight.zero_()
        refiner.fusion.bias.zero_()
        refiner.fusion.weight[:,4*k*k:,0,0].copy_(torch.eye(4*k*k))
    preliminary = torch.rand_like(x)
    torch.testing.assert_close(refiner(preliminary,x), x if source=='noisy' else preliminary, atol=0,rtol=0)


def test_ablation_configs():
    prefix = 'configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1_repncb'
    base = yaml.safe_load(Path(prefix+'.yaml').read_text())
    assert (base['dwt_width'],base['dwt_depth']) == (16,3)
    directories = {base['output_dir']}
    for suffix,key,value in [('k1','refine_s2d_factor',1), ('skip_preliminary','refine_skip_source','preliminary'), ('n6','refine_num_blocks',6)]:
        config = yaml.safe_load(Path(prefix+'_'+suffix+'.yaml').read_text())
        assert config[key] == value
        assert config['output_dir'] not in directories
        directories.add(config['output_dir'])
        assert {k for k in base if base[k]!=config[k]} == {key,'output_dir'}


@pytest.mark.parametrize('config', [{'s2d_factor':0},{'width':0},{'num_blocks':-1},{'skip_source':'invalid'}])
def test_invalid_refiner_settings(config):
    with pytest.raises(ValueError):
        PackedRAWRefiner(**config)
