"""Classical channel-wise DWT BayesShrink; no training or ground truth needed."""
import numpy as np
import pywt


class WaveletDenoiser:
    def __init__(self, levels=3, wavelet='sym4', boundary='symmetric'):
        if boundary not in ('symmetric', 'periodization'):
            raise ValueError('boundary must be symmetric or periodization')
        self.boundary = boundary
        self.wavelet = pywt.Wavelet(wavelet)
        if not self.wavelet.orthogonal:
            raise ValueError('BayesShrink requires an orthogonal wavelet (e.g. haar/db2/sym4/coif1)')
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise ValueError('levels must be a positive integer')
        self.levels = levels

    def __call__(self, image):
        """Denoise a floating CHW image, preserving shape and input range."""
        image = np.asarray(image)
        if image.ndim != 3 or min(image.shape) < 1 or not np.isfinite(image).all():
            raise ValueError('Expected finite, nonempty CHW image')
        if self.levels > pywt.dwtn_max_level(image.shape[-2:], self.wavelet):
            raise ValueError('Too many levels for this image size and wavelet')
        output = np.empty(image.shape, dtype=np.float32)
        for c, channel in enumerate(image):
            coeffs = pywt.wavedec2(channel.astype(np.float32), self.wavelet,
                                   mode=self.boundary, level=self.levels)
            # Robust Gaussian noise estimate from finest diagonal detail.
            hh = coeffs[-1][2]
            nonzero = hh[hh != 0]  # Ignore exact zeros introduced by symmetric padding.
            sigma = float(np.median(np.abs(nonzero))) / 0.6744897501960817 if nonzero.size else 0.0
            noise_var = sigma * sigma
            filtered = [coeffs[0]]  # Preserve coarsest approximation.
            for details in coeffs[1:]:
                subbands = []
                for detail in details:
                    variance = float(np.mean(detail.astype(np.float64) ** 2))
                    if noise_var == 0:
                        result = detail
                    elif variance <= noise_var:
                        result = np.zeros_like(detail)
                    else:
                        threshold = noise_var / np.sqrt(variance - noise_var)
                        result = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
                    subbands.append(result)
                filtered.append(tuple(subbands))
            restored = pywt.waverec2(filtered, self.wavelet, mode=self.boundary)
            output[c] = restored[:image.shape[1], :image.shape[2]]
        return output
