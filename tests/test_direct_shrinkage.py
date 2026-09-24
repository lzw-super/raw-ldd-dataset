from pathlib import Path
import torch
import pytest
import yaml
from learning_wt.direct_shrinkage import DirectShrinkage
from learning_wt.learning_dwt import learning_dwt_kwargs
from models.learning_dwt_repncb import LearningDWTRepNCB,refiner_kwargs
from utils.model_factory import build_denoiser_from_checkpoint


@pytest.mark.parametrize('mode',['soft','firm','pwl'])
def test_modes(mode,tmp_path):
    torch.set_num_threads(2)
    path=Path('configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_'+mode+'.yaml')
    args=yaml.safe_load(path.read_text())
    model=LearningDWTRepNCB(learning_dwt_kwargs(args),refine_config=refiner_kwargs(args))
    x=torch.randn(1,4,33,41)*10
    y=model(x);y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    deployed=model.deploy()
    torch.testing.assert_close(deployed(x),y,atol=2e-5,rtol=2e-5)
    for dep in (False,True):
        cp={'args':args,'model':model.state_dict()}
        if dep:cp['model_deploy']=deployed.state_dict()
        p=tmp_path/'m.pth';torch.save(cp,p)
        loaded,_=build_denoiser_from_checkpoint(str(p),'cpu')
        torch.testing.assert_close(loaded(x),y,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('mode',['firm','pwl'])
def test_reference_formula(mode):
    m=DirectShrinkage(mode,normalize=False)
    z=torch.linspace(-6,6,401).reshape(1,1,1,-1).repeat(1,4,1,1)
    tau,w=m.coefficients();a=z.abs();expected=w[0,:,0].view(1,4,1,1)*a
    for j in range(tau.shape[-1]):expected=expected+w[0,:,j+1].view(1,4,1,1)*torch.relu(a-tau[0,:,j].view(1,4,1,1))
    torch.testing.assert_close(m(z,0),z.sign()*expected)
    m.freeze()
    import io,onnx
    class Band(torch.nn.Module):
        def forward(self,x):return m(x,0)
    f=io.BytesIO();torch.onnx.export(Band(),z,f,opset_version=17)
    ops={n.op_type for n in onnx.load_from_string(f.getvalue()).graph.node}
    assert not ops.intersection({'Div','Softplus','Sigmoid','Sign','Abs'})
