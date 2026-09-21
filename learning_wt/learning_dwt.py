"""Packed-RAW wavelet threshold predictor with SID training integration.

Input/output: floating NCHW tensors [N, 4, H, W], NOT RGB or Bayer mosaic.
Fixed sym4 periodization analysis is channel-wise (Haar is also supported). Only LL is recursively decomposed. The ten
terminal bands are tiled without resizing into an H x W coefficient atlas.
A small sequential CNN predicts thresholds, not clean pixels. LL shrinkage
is enabled by default, with a smaller initial threshold. Optional LL restoration
branches predict recovered LL from LL alone or all four deepest-level bands. No calibrated sigma
is required. See README.md for assumptions, limits, and halo caveats.

Run: python learning_dwt.py --demo
Tests: python -m unittest -v learning_wt.test_learning_dwt
sym4 uses fixed PyWavelets filters in differentiable PyTorch convolutions.
Legacy standalone Haar helpers are retained for the original tests.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from .sym4_transform import PeriodizedWavelet
except ImportError:  # Direct demo invocation
    from sym4_transform import PeriodizedWavelet


@dataclass(frozen=True)
class Band:
    name: str
    level: int
    kind: str
    rows: slice
    cols: slice


def band_layout(height: int, width: int, levels: int = 3) -> list[Band]:
    """Terminal rectangles: LL_J then LH/HL/HH from coarse to fine.

    L/H refer to row filter first, column filter second; signs are defined
    by haar_dwt2, not inferred from another library's naming convention.
    """
    if levels < 1 or height < 1 or width < 1:
        raise ValueError("Positive dimensions and levels >= 1 required")
    factor = 2 ** levels
    if height % factor or width % factor:
        raise ValueError(f"H and W must be divisible by {factor}")
    bands = [Band(f"LL{levels}", levels, "LL",
                  slice(0, height // factor), slice(0, width // factor))]
    for level in range(levels, 0, -1):
        h, w = height // (2 ** level), width // (2 ** level)
        bands.extend([
            Band(f"LH{level}", level, "LH", slice(0, h), slice(w, 2*w)),
            Band(f"HL{level}", level, "HL", slice(h, 2*h), slice(0, w)),
            Band(f"HH{level}", level, "HH", slice(h, 2*h), slice(w, 2*w)),
        ])
    return bands


def haar_dwt2(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Orthonormal 2D Haar; scaling 1/2, not averaging with 1/4."""
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("DWT requires NCHW with even spatial dimensions")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return ((a+b+c+d)*0.5, (a-b+c-d)*0.5,
            (a+b-c-d)*0.5, (a-b-c+d)*0.5)


def haar_iwt2(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
    """Differentiable inverse, avoiding detached or in-place input edits."""
    if ll.ndim != 4 or not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError("Four identical NCHW band shapes required")
    a, b = (ll+lh+hl+hh)*0.5, (ll-lh+hl-hh)*0.5
    c, d = (ll+lh-hl-hh)*0.5, (ll-lh-hl+hh)*0.5
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    return torch.stack((top, bottom), dim=-2).flatten(-3, -2)


def dwt_atlas(x: Tensor, levels: int = 3) -> Tensor:
    """Recursive upper-left LL replacement, with no resampling."""
    band_layout(x.shape[-2], x.shape[-1], levels)

    def encode(current: Tensor, depth: int) -> Tensor:
        ll, lh, hl, hh = haar_dwt2(current)
        if depth > 1:
            ll = encode(ll, depth-1)
        return torch.cat((torch.cat((ll, lh), -1),
                          torch.cat((hl, hh), -1)), -2)
    return encode(x, levels)


def iwt_atlas(atlas: Tensor, levels: int = 3) -> Tensor:
    """Exact inverse of dwt_atlas up to floating point roundoff."""
    band_layout(atlas.shape[-2], atlas.shape[-1], levels)

    def decode(current: Tensor, depth: int) -> Tensor:
        h, w = current.shape[-2] // 2, current.shape[-1] // 2
        ll = current[..., :h, :w]
        if depth > 1:
            ll = decode(ll, depth-1)
        return haar_iwt2(ll, current[..., :h, w:],
                         current[..., h:, :w], current[..., h:, w:])
    return decode(atlas, levels)


def soft_shrink(z: Tensor, threshold: Tensor, leak: float = 0.0) -> Tensor:
    """0 <= leak <= 1; leak=0 is ordinary soft thresholding.

    threshold must be nonnegative. Output cannot amplify or reverse a
    coefficient. This restriction is intentional, but limits restoration.
    """
    shrunk = F.relu(z-threshold) - F.relu(-z-threshold)
    return leak*z + (1.0-leak)*shrunk


def smooth_shrink(z: Tensor, threshold: Tensor, leak: float = 0.0) -> Tensor:
    """Continuous half-gain threshold: no dead interval for nonzero coefficients."""
    magnitude = z.abs()
    gain = magnitude / (magnitude + threshold).clamp_min(torch.finfo(z.dtype).tiny)
    return leak*z + (1.0-leak)*z*gain


class ThresholdCNN(nn.Module):
    """Plain Conv/ReLU chain. depth counts ALL spatial convolutions.

    No BN or global pooling; optional signed/magnitude input and band conditioning.
    Output logits are converted into thresholds by the parent model.
    """
    def __init__(self, width: int = 16, depth: int = 4, magnitude_input: bool = False, bands: int = 0):
        super().__init__()
        if width < 1 or depth < 2:
            raise ValueError("width >= 1 and depth >= 2 required")
        self.magnitude_input = magnitude_input
        self.band_embedding = nn.Embedding(bands, width) if bands else None
        if self.band_embedding is not None:
            nn.init.zeros_(self.band_embedding.weight)
        layers: list[nn.Module] = [nn.Conv2d(8 if magnitude_input else 4, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth-2):
            layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.ReLU()])
        layers.append(nn.Conv2d(width, 4, 3, padding=1))
        self.layers = nn.Sequential(*layers)
        # Small, nonzero weights permit gradient flow into earlier layers
        # at the first update; thresholds initially remain near their bias.
        nn.init.normal_(self.layers[-1].weight, std=1e-3)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: Tensor, band_id: int | None = None) -> Tensor:
        if self.magnitude_input:
            x = torch.cat((x, x.abs()), dim=1)
        x = self.layers[0](x)
        if self.band_embedding is not None:
            if band_id is None:
                raise ValueError("Band-conditioned predictor requires bandwise context")
            x = x + self.band_embedding.weight[band_id].view(1, -1, 1, 1)
        for layer in self.layers[1:]:
            x = layer(x)
        return x


class LLRestorationCNN(nn.Module):
    """Restore LL with residual addition or learned concatenation/1x1 fusion.

    Inputs are normalized wavelet coefficients; the first four channels are LL.
    Both fusion modes start near the input without restricting LL to shrinkage.
    """
    def __init__(self, in_channels: int, width: int = 32, depth: int = 4, fusion: str = "residual", block_type: str = "conv3x3"):
        super().__init__()
        if width < 1 or depth < 2:
            raise ValueError("LL CNN requires width >= 1 and depth >= 2")
        if block_type not in ("conv3x3", "repmbconv", "repncb"):
            raise ValueError("LL block_type must be conv3x3, repmbconv or repncb")
        from models.rep_mbconv import RepMBConv
        layers = [nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            if block_type == "repncb":
                from models.learning_dwt_repncb import RepNCB
                # RepNCB includes its own PReLU; do not append a second activation.
                layers.append(RepNCB(width))
            else:
                block = RepMBConv(width) if block_type == "repmbconv" else nn.Conv2d(width, width, 3, padding=1)
                layers.extend([block, nn.ReLU()])
        layers.append(nn.Conv2d(width, 4, 3, padding=1))
        self.layers = nn.Sequential(*layers)
        nn.init.normal_(self.layers[-1].weight, std=1e-3)
        nn.init.zeros_(self.layers[-1].bias)
        if fusion not in ("residual", "concat_1x1"):
            raise ValueError("LL fusion must be residual or concat_1x1")
        self.fusion = nn.Conv2d(8, 4, 1) if fusion == "concat_1x1" else None
        if self.fusion is not None:
            # Start equivalent to residual addition, then learn all cross-channel weights.
            with torch.no_grad():
                self.fusion.weight.zero_()
                self.fusion.bias.zero_()
                for channel in range(4):
                    self.fusion.weight[channel, channel, 0, 0] = 1
                    self.fusion.weight[channel, channel + 4, 0, 0] = 1

    def forward(self, x: Tensor) -> Tensor:
        cnn_output = self.layers(x)
        if self.fusion is not None:
            return self.fusion(torch.cat((cnn_output, x[:, :4]), dim=1))
        return x[:, :4] + cnn_output


class LearningDWT(nn.Module):
    """User's atlas-threshold design plus explicit comparison switches.

    context='atlas': one CNN on the whole atlas, as requested. Artificial
      band seams induce artificial boundary interactions; a sufficient,
      aligned crop can exclude them for this local Haar/CNN configuration.
    context='bandwise': reuse the SAME CNN independently on the 10 bands.
      Prevents convolution across artificial seams. Can be scheduled serially.
    shrink_ll=True: LL receives its own learned, initially smaller threshold.
    normalize_bands=True: divide level-j coefficients by 2**j for CNN input
      and multiply predicted thresholds back by 2**j. This is static scale
      balancing, NOT noise standardization or a calibrated variance model.
    pad_input=False: require H,W divisible by 8. Optional right/bottom
      replication permits arbitrary sizes; diagnostics refer to PADDED atlas.
    """
    def __init__(self, width: int = 16, depth: int = 4,
                 context: str = "atlas", shrink_ll: bool = True,
                 leak: float = 0.0, normalize_bands: bool = True,
                 pad_input: bool = False, init_threshold: float = 0.01,
                 init_ll_threshold: float = 0.001, wavelet: str = "sym4", levels: int = 3,
                 magnitude_input: bool = False, condition_bands: bool = False,
                 ll_max_threshold: float | None = None, shrink_mode: str = "soft",
                 ll_mode: str = "threshold", ll_width: int = 32, ll_depth: int = 4,
                 ll_fusion: str = "residual", trainable_haar: bool = False,
                 haar_share_channels: bool = True, haar_share_levels: bool = True, ll_block_type: str = "conv3x3"):
        super().__init__()
        if context not in ("atlas", "bandwise"):
            raise ValueError("context must be atlas or bandwise")
        if not 0 <= leak <= 1:
            raise ValueError("leak must be in [0,1]")
        if not all(math.isfinite(t) and t > 0
                   for t in (init_threshold, init_ll_threshold)):
            raise ValueError("Initial thresholds must be finite and positive")
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise ValueError("levels must be a positive integer")
        self.levels = levels
        self.wavelet = wavelet
        if trainable_haar and wavelet != "haar":
            raise ValueError("trainable_haar requires wavelet=haar")
        self.transform = PeriodizedWavelet(wavelet) if wavelet != "haar" else None
        if trainable_haar:
            from .trainable_haar import TrainableHaar
            self.transform = TrainableHaar(haar_share_channels, haar_share_levels, levels)
        self.context, self.shrink_ll = context, shrink_ll
        self.leak, self.normalize_bands = leak, normalize_bands
        self.pad_input = pad_input
        if condition_bands and context != "bandwise":
            raise ValueError("condition_bands requires bandwise context")
        if ll_max_threshold is not None and (not math.isfinite(ll_max_threshold) or ll_max_threshold <= init_ll_threshold):
            raise ValueError("ll_max_threshold must be finite and exceed init_ll_threshold")
        if shrink_mode not in ("soft", "smooth"):
            raise ValueError("shrink_mode must be soft or smooth")
        self.shrink_mode = shrink_mode
        self.ll_max_threshold = ll_max_threshold
        self.predictor = ThresholdCNN(width, depth, magnitude_input, 1 + 3*levels if condition_bands else 0)
        self.depth = depth
        # 10 band identities x 4 CFA channels, in band_layout order.
        initial = torch.full((1 + 3*levels, 4), init_threshold)
        initial[0] = init_ll_threshold
        # Stable inverse softplus: t + log(1-exp(-t)).
        bias = initial + torch.log(-torch.expm1(-initial))
        if ll_max_threshold is not None:
            bias[0] = math.log(init_ll_threshold / (ll_max_threshold - init_ll_threshold))
        self.band_bias = nn.Parameter(bias)
        if ll_mode not in ("threshold", "ll_only", "level_bands"):
            raise ValueError("ll_mode must be threshold, ll_only or level_bands")
        self.ll_mode = ll_mode
        self.ll_restorer = (LLRestorationCNN(4 if ll_mode == "ll_only" else 16, ll_width, ll_depth, ll_fusion, ll_block_type)
                            if ll_mode != "threshold" else None)

    def _maps(self, atlas: Tensor, bands: list[Band]) -> tuple[Tensor, Tensor, Tensor]:
        h, w = atlas.shape[-2:]
        scale = atlas.new_ones((1, 1, h, w))
        bias = atlas.new_empty((1, 4, h, w))
        mask = atlas.new_ones((1, 1, h, w))
        for i, band in enumerate(bands):
            if self.normalize_bands:
                scale[..., band.rows, band.cols] = 2 ** band.level
            bias[..., band.rows, band.cols] = self.band_bias[i].view(1, 4, 1, 1)
            if band.kind == "LL" and (not self.shrink_ll or self.ll_mode != "threshold"):
                mask[..., band.rows, band.cols] = 0
        return scale, bias, mask

    def forward(self, noisy: Tensor, return_aux: bool = False):
        # Long-filter analysis/synthesis and tiny thresholds stay in FP32 under AMP.
        with torch.autocast(device_type=noisy.device.type, enabled=False):
            if noisy.dtype in (torch.float16, torch.bfloat16):
                noisy = noisy.float()
            return self._forward(noisy, return_aux)

    def _forward(self, noisy: Tensor, return_aux: bool = False):
        if noisy.ndim != 4 or noisy.shape[1] != 4 or not noisy.is_floating_point():
            raise ValueError("Input must be floating [N,4,H,W] packed RAW")
        if min(noisy.shape) < 1:
            raise ValueError("Empty dimensions are not supported")
        h, w = noisy.shape[-2:]
        factor = 2 ** self.levels
        pad_h, pad_w = (-h) % factor, (-w) % factor
        if (pad_h or pad_w) and not self.pad_input:
            raise ValueError(f"H,W must be multiples of {factor}; or enable pad_input")
        x = F.pad(noisy, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else noisy
        atlas = (self.transform.encode(x, self.levels) if self.transform is not None
                 else dwt_atlas(x, self.levels))
        bands = band_layout(*atlas.shape[-2:], self.levels)
        scale, bias, mask = self._maps(atlas, bands)
        normalized = atlas / scale
        if self.context == "atlas":
            logits = self.predictor(normalized)
        else:
            logits = torch.empty_like(atlas)
            for band_id, band in enumerate(bands):
                logits[..., band.rows, band.cols] = self.predictor(
                    normalized[..., band.rows, band.cols], band_id)
        positive = F.softplus(logits + bias)
        if self.ll_max_threshold is not None:
            ll = bands[0]
            # Clone before slice replacement to keep the autograd graph intact.
            positive = positive.clone()
            positive[..., ll.rows, ll.cols] = self.ll_max_threshold * torch.sigmoid(
                (logits + bias)[..., ll.rows, ll.cols])
        threshold = positive * scale * mask
        if self.shrink_mode == "smooth":
            # Half-gain at |Z|=T, unlike soft shrink no exactly dead interval.
            filtered = smooth_shrink(atlas, threshold, self.leak)
        else:
            filtered = soft_shrink(atlas, threshold, self.leak)
        if self.ll_restorer is not None:
            # Read ORIGINAL deepest-level bands; no thresholded inputs or GT.
            selected = bands[:1] if self.ll_mode == "ll_only" else bands[:4]
            ll_input = torch.cat([atlas[..., b.rows, b.cols] for b in selected], dim=1)
            ll_scale = float(2 ** self.levels)
            recovered_ll = self.ll_restorer(ll_input / ll_scale) * ll_scale
            ll = bands[0]
            filtered = filtered.clone()
            filtered[..., ll.rows, ll.cols] = recovered_ll
        restored = (self.transform.decode(filtered, self.levels) if self.transform is not None
                    else iwt_atlas(filtered, self.levels))[..., :h, :w]
        # No [0,1] clipping: signed read-noise residuals and gradient preserved.
        if return_aux:
            return restored, {"atlas": atlas, "threshold": threshold,
                              "filtered_atlas": filtered,
                              "padding": (pad_h, pad_w), "bands": bands}
        return restored


DWT_DEFAULTS = dict(dwt_width=64, dwt_depth=4, dwt_context="bandwise",
                    dwt_shrink_ll=True, dwt_leak=0.0, dwt_normalize_bands=True,
                    dwt_pad_input=True, dwt_init_threshold=0.01,
                    dwt_init_ll_threshold=0.001, dwt_wavelet="sym4", dwt_levels=3,
                    dwt_magnitude_input=True, dwt_condition_bands=True,
                    dwt_ll_max_threshold=0.01, dwt_shrink_mode="soft",
                    dwt_ll_mode="threshold", dwt_ll_width=32, dwt_ll_depth=4,
                    dwt_ll_fusion="residual", dwt_trainable_haar=False,
                    dwt_haar_share_channels=True, dwt_haar_share_levels=True, dwt_ll_block_type="conv3x3")


def learning_dwt_kwargs(options):
    values = vars(options) if not isinstance(options, dict) else options
    return {key[4:]: values.get(key, default) for key, default in DWT_DEFAULTS.items()}


def demo() -> None:
    """One smoke-training step on toy Gaussian pairs, NOT One Hour training."""
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = LearningDWT()
    clean = torch.rand(2, 4, 32, 48) * 0.2
    noisy = clean + torch.randn_like(clean) * 0.02
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    restored, aux = model(noisy, return_aux=True)
    loss = F.l1_loss(restored, clean)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(model)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Input/output/threshold:", tuple(noisy.shape), tuple(restored.shape),
          tuple(aux["threshold"].shape))
    print("Toy L1 before update:", float(loss.detach()))
    print("Use real dark-frame + Poisson pairs for research, not this toy noise.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run a toy optimizer step")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
