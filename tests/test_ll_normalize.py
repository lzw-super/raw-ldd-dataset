import torch
import pytest
from models.learning_dwt_repncb import LearningDWTRepNCB


@pytest.mark.parametrize('normalize',[True,False])
def test_ll_input_output_scaling(normalize):
    torch.set_num_threads(2)
    model=LearningDWTRepNCB(dict(wavelet='haar',threshold_mode='band_channel',shrink_mode='soft',ll_normalize=normalize))
    x=torch.randn(1,4,32,40)
    captured=[]
    h=model.wavelet.ll_restorer.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
    y,aux=model.wavelet(x,True);h.remove()
    ll=aux['bands'][0];original=aux['atlas'][...,ll.rows,ll.cols]
    torch.testing.assert_close(captured[0],original*.125 if normalize else original)
    expected=model.wavelet.ll_restorer(captured[0])*(8 if normalize else 1)
    torch.testing.assert_close(aux['filtered_atlas'][...,ll.rows,ll.cols],expected)
    torch.testing.assert_close(model.deploy().wavelet(x),y,atol=2e-6,rtol=2e-5)
    if normalize:
        torch.testing.assert_close(model.wavelet.restore_ll(original),model.wavelet.ll_restorer(original/8)*8,rtol=0,atol=0)
