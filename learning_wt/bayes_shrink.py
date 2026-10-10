"""Parameter-free Torch counterpart of models.wavelet_denoiser.BayesShrink logic."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def estimate_noise_variance(hh):
    """One variance per image/RAW channel from nonzero finest HH coefficients."""
    rows = []
    for image in hh.detach().double():
        channels = []
        for channel in image:
            values = channel[channel != 0].abs()
            # quantile averages the two middle values, matching np.median.
            sigma = torch.quantile(values, 0.5) / 0.6744897501960817 if values.numel() else channel.new_zeros(())
            channels.append(sigma.square())
        rows.append(torch.stack(channels))
    return torch.stack(rows)[..., None, None]


def bayes_soft_shrink(detail, noise_variance):
    with torch.no_grad():
        observed_variance = detail.detach().double().square().mean((-2, -1), keepdim=True)
        signal_variance = observed_variance - noise_variance
        threshold = noise_variance / signal_variance.clamp_min(torch.finfo(torch.float64).tiny).sqrt()
        # For noise-only bands, a finite max threshold produces exactly zero.
        threshold = torch.where(signal_variance > 0, threshold,
                                detail.detach().double().abs().amax((-2, -1), keepdim=True))
        threshold = threshold.to(detail.dtype)
    filtered = F.relu(detail-threshold) - F.relu(-detail-threshold)
    return torch.where(noise_variance == 0, detail, filtered)
