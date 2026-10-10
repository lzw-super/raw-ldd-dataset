"""The export median must remain exact as each input's zero mask changes."""
import numpy as np
import pytest
import torch

from learning_wt.bayes_shrink import estimate_noise_variance
from models.haar_bayes_export import nonzero_noise_variance


@pytest.mark.parametrize('size', [1, 2, 3, 4, 15, 16])
def test_nonzero_median_matches_quantile(size):
    for count in range(size + 1):
        x = torch.zeros(2, 4, 1, size)
        if count:
            x[..., :count] = torch.linspace(-1.3, 2.7, count)
        torch.testing.assert_close(nonzero_noise_variance(x), estimate_noise_variance(x),
                                   rtol=1e-12, atol=1e-14)


def test_onnx_zero_mask_is_input_dependent(tmp_path):
    ort = pytest.importorskip('onnxruntime')

    class Noise(torch.nn.Module):
        def forward(self, x):
            return nonzero_noise_variance(x)

    path = tmp_path / 'noise.onnx'
    torch.onnx.export(Noise(), torch.ones(2, 4, 4, 8), str(path),
                      input_names=['hh'], opset_version=17)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    for count in [0, 1, 2, 3, 15, 16, 31, 32]:
        x = torch.zeros(2, 4, 32)
        if count:
            x[..., :count] = torch.linspace(-1.3, 2.7, count)
        x = x.reshape(2, 4, 4, 8)
        actual = session.run(None, {'hh': x.numpy()})[0]
        np.testing.assert_allclose(actual, estimate_noise_variance(x).numpy(),
                                   rtol=1e-12, atol=1e-14)
