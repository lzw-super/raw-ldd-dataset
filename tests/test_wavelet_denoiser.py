import numpy as np
import pytest
from skimage.restoration import denoise_wavelet
from models.wavelet_denoiser import WaveletDenoiser


@pytest.mark.parametrize('wavelet', ['haar','db2','sym4','coif1'])
def test_matches_reference(wavelet):
    image = np.random.default_rng(4).normal(0.3, 0.05, (4, 65, 67)).astype(np.float32)
    actual = WaveletDenoiser(3, wavelet)(image)
    expected = np.stack([denoise_wavelet(c, wavelet=wavelet, wavelet_levels=3,
                                        method='BayesShrink', mode='soft', rescale_sigma=True) for c in image])
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert actual.shape == image.shape


def test_constant_and_gaussian_noise():
    clean = np.full((4,64,64), 0.3, np.float32)
    denoiser = WaveletDenoiser()
    np.testing.assert_allclose(denoiser(clean), clean, atol=1e-6)
    noisy = clean + np.random.default_rng(3).normal(0,0.05, clean.shape).astype(np.float32)
    assert np.mean((denoiser(noisy)-clean)**2) < np.mean((noisy-clean)**2)


def test_invalid():
    with pytest.raises(ValueError):
        WaveletDenoiser(0)
    with pytest.raises(ValueError):
        WaveletDenoiser(3, 'bior2.2')
    with pytest.raises(ValueError):
        WaveletDenoiser(3)(np.zeros((4,16,16)))
