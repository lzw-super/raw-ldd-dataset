import pytest
import torch
from models.haar_hf_cnn import HighFrequencyCNN


@pytest.mark.parametrize('variant', ['dw1_dual_relu','dw3_dual_relu','dw1_dual_prelu'])
@pytest.mark.parametrize('bias', [0.0, -0.1, 0.1])
def test_signed_identity_initialization_and_soft_function(variant, bias):
    torch.set_num_threads(2)
    model = HighFrequencyCNN(variant, init_bias=bias)
    x = torch.linspace(-1,1,12*5*7).reshape(1,12,5,7)
    # Includes edges: 3x3 must be a signed delta kernel, not an all-ones box.
    torch.testing.assert_close(model.branches[0][0](x),x+bias)
    torch.testing.assert_close(model.branches[1][0](x),-x+bias)
    y = model(x)
    expected = torch.relu(x+bias)-torch.relu(-x+bias)
    torch.testing.assert_close(y,expected)
    if bias <= 0:
        torch.testing.assert_close(y,x.sign()*torch.relu(x.abs()+bias))
    if variant.endswith('prelu'):
        for branch in model.branches:
            assert branch[1].weight.shape == (12,)
            assert torch.equal(branch[1].weight,torch.zeros(12))
    assert model.branches[0][0].weight.data_ptr()!=model.branches[1][0].weight.data_ptr()
    y.square().mean().backward()
    for branch in model.branches:
        for parameter in branch.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum()>0
