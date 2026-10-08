"""Two-stage packed-RAW denoiser with exactly reparameterizable smoothing blocks."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from learning_wt.learning_dwt import LearningDWT


REFINER_DEFAULTS = dict(refine_s2d_factor=2, refine_width=16,
                        refine_num_blocks=4, refine_skip_source="noisy",
                        refine_block_type="repncb", refine_input_fusion="none", refine_activation="prelu", refine_stem_activation="none", refine_skip_projection=False)


def refiner_kwargs(options):
    values = options if isinstance(options, dict) else vars(options)
    return {key[len("refine_"):]: values.get(key, default)
            for key, default in REFINER_DEFAULTS.items()}


class SpatialBranch(nn.Module):
    """Biased 1x1 followed by dense or fixed depthwise 3x3, with bias padding."""

    def __init__(self, channels, kernel=None):
        super().__init__()
        hidden = channels * 2 if kernel is None else channels
        self.project = nn.Conv2d(channels, hidden, 1)
        if kernel is None:
            self.spatial = nn.Conv2d(hidden, channels, 3)
        else:
            self.register_buffer('mask', torch.tensor(kernel, dtype=torch.float32).reshape(1, 1, 3, 3))
            self.scale = nn.Parameter(torch.randn(channels, 1, 1, 1) * 1e-3)
            self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # Project AFTER padding: outside-image values equal project.bias,
        # which is required for an exact spatially constant fused bias.
        y = self.project(F.pad(x, (1, 1, 1, 1)))
        if hasattr(self, 'spatial'):
            return self.spatial(y)
        return F.conv2d(y, self.scale * self.mask, self.bias, groups=x.shape[1])

    def equivalent(self):
        point = self.project.weight[:, :, 0, 0]
        if hasattr(self, 'spatial'):
            weight = torch.einsum('omhw,mi->oihw', self.spatial.weight, point)
            bias = self.spatial.bias + torch.einsum('omhw,m->o', self.spatial.weight, self.project.bias)
        else:
            spatial = self.scale * self.mask
            weight = spatial * point[:, :, None, None]
            bias = self.bias + spatial.sum((1, 2, 3)) * self.project.bias
        return weight, bias


class RepNCB(nn.Module):
    """Five linear branches -> sum -> channel-wise PReLU; deploy as 3x3 + PReLU."""

    def __init__(self, channels=16, deploy=False):
        super().__init__()
        self.activation = nn.PReLU(channels)
        if deploy:
            self.reparam_conv = nn.Conv2d(channels, channels, 3, padding=1)
        else:
            self.direct = nn.Conv2d(channels, channels, 3, padding=1)
            self.expand = SpatialBranch(channels)
            self.gaussian = SpatialBranch(channels, [[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])
            self.horizontal = SpatialBranch(channels, [[0, 0, 0], [1/4, 2/4, 1/4], [0, 0, 0]])
            self.vertical = SpatialBranch(channels, [[0, 1/4, 0], [0, 2/4, 0], [0, 1/4, 0]])

    def forward(self, x):
        if hasattr(self, 'reparam_conv'):
            y = self.reparam_conv(x)
        else:
            y = self.direct(x) + self.expand(x) + self.gaussian(x) + self.horizontal(x) + self.vertical(x)
        return self.activation(y)

    @torch.no_grad()
    def switch_to_deploy(self):
        if hasattr(self, 'reparam_conv'):
            return self
        weight, bias = self.direct.weight.clone(), self.direct.bias.clone()
        for name in ('expand', 'gaussian', 'horizontal', 'vertical'):
            w, b = getattr(self, name).equivalent()
            weight += w
            bias += b
        conv = nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1).to(device=weight.device, dtype=weight.dtype)
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
        self.reparam_conv = conv
        for name in ('direct', 'expand', 'gaussian', 'horizontal', 'vertical'):
            delattr(self, name)
        return self


class PackedRAWRefiner(nn.Module):
    """Configurable packed-RAW refinement with a concatenated input bypass."""

    def __init__(self, deploy=False, s2d_factor=2, width=16, num_blocks=4,
                 skip_source="noisy", block_type="repncb", input_fusion="none", stem_activation="none", skip_projection=False):
        super().__init__()
        for name, value in (("s2d_factor", s2d_factor), ("width", width), ("num_blocks", num_blocks)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if skip_source not in ("noisy", "preliminary", "fused"):
            raise ValueError("skip_source must be noisy, preliminary, or fused")
        if block_type not in ("repncb", "conv3x3"):
            raise ValueError("block_type must be repncb or conv3x3")
        if input_fusion not in ("none", "preliminary_difference"):
            raise ValueError("input_fusion must be none or preliminary_difference")
        if skip_source == "fused" and input_fusion == "none":
            raise ValueError("fused skip requires input fusion")
        self.input_fusion = input_fusion
        if input_fusion == "preliminary_difference":
            self.input_projection = nn.Conv2d(8, 4, 1)
        self.s2d_factor = s2d_factor
        self.skip_source = skip_source
        packed_channels = 4 * s2d_factor ** 2
        self.s2d = nn.PixelUnshuffle(s2d_factor) if s2d_factor > 1 else nn.Identity()
        self.stem = nn.Conv2d(packed_channels, width, 3, padding=1)
        if stem_activation not in ("none", "relu"):
            raise ValueError("stem_activation must be none or relu")
        self.stem_activation = nn.ReLU() if stem_activation == "relu" else nn.Identity()
        self.blocks = nn.Sequential(*(
            RepNCB(width, deploy) if block_type == "repncb" else
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.PReLU(width))
            for _ in range(num_blocks)))
        self.head = nn.Conv2d(width, packed_channels, 3, padding=1)
        self.fusion = nn.Conv2d(2 * packed_channels, packed_channels, 1)
        self.d2s = nn.PixelShuffle(s2d_factor) if s2d_factor > 1 else nn.Identity()
        # Optional channel mixing on the packed skip, before final concatenation.
        self.skip_projection = nn.Conv2d(packed_channels, packed_channels, 1) if skip_projection else nn.Identity()
        if skip_projection:
            # Preserve the original bypass initially; all weights remain trainable.
            nn.init.dirac_(self.skip_projection.weight)
            nn.init.zeros_(self.skip_projection.bias)

    def forward(self, preliminary, noisy):
        if preliminary.shape != noisy.shape or noisy.ndim != 4 or noisy.shape[1] != 4:
            raise ValueError('Expected two matching N x 4 x H x W packed RAW tensors')
        h, w = noisy.shape[-2:]
        padding = (0, (-w) % self.s2d_factor, 0, (-h) % self.s2d_factor)
        if padding[1] or padding[3]:
            preliminary = F.pad(preliminary, padding, mode='replicate')
            noisy = F.pad(noisy, padding, mode='replicate')
        refined_input = preliminary
        if self.input_fusion == "preliminary_difference":
            refined_input = self.input_projection(torch.cat((preliminary, noisy - preliminary), dim=1))
        packed_input = self.s2d(refined_input)
        features = self.head(self.blocks(self.stem_activation(self.stem(packed_input))))
        # The fused-input bypass branches AFTER S2D, sharing the stem input.
        if self.skip_source == "fused":
            packed_skip = packed_input
        else:
            skip = noisy if self.skip_source == "noisy" else preliminary
            packed_skip = self.s2d(skip)
        packed_skip = self.skip_projection(packed_skip)
        fused = self.fusion(torch.cat((features, packed_skip), dim=1))
        result = self.d2s(fused)
        return result if not (padding[1] or padding[3]) else result[..., :h, :w]


class LearningDWTRepNCB(nn.Module):
    def __init__(self, dwt_config=None, deploy=False, refine_config=None):
        super().__init__()
        config = dict(width=32, context='atlas', shrink_ll=False, pad_input=True,
                      wavelet='sym4', levels=3, magnitude_input=True, condition_bands=False,
                      shrink_mode='smooth', ll_mode='ll_only', ll_width=32,
                      ll_depth=4, ll_fusion='concat_1x1')
        config.update(dwt_config or {})
        self.wavelet = LearningDWT(**config)
        refine_options = dict(refine_config or {})
        activation = refine_options.pop("activation", "prelu")
        if activation not in ("prelu", "relu"):
            raise ValueError("refine_activation must be prelu or relu")
        self.refiner = PackedRAWRefiner(deploy=deploy, **refine_options)
        # This experiment switches ALL PReLUs, including the LL branch.
        if activation == "relu":
            for module in list(self.modules()):
                for name, child in list(module.named_children()):
                    if isinstance(child, nn.PReLU):
                        setattr(module, name, nn.ReLU())
        if deploy:
            self.switch_to_deploy()

    def forward(self, noisy, return_aux=False):
        preliminary = self.wavelet(noisy)
        restored = self.refiner(preliminary, noisy)
        return (restored, {'preliminary': preliminary}) if return_aux else restored

    @torch.no_grad()
    def switch_to_deploy(self):
        self.wavelet.freeze_hf_thresholds()
        if self.wavelet.threshold_mode in ("band_channel", "hf_cnn") and self.wavelet.wavelet == "haar":
            from models.fixed_raw_ops import FixedHaarConv, FixedSpaceDepth
            reference = next(self.parameters())
            if self.wavelet.transform is None:
                self.wavelet.transform = FixedHaarConv().to(reference)
            if self.refiner.s2d_factor == 2:
                self.refiner.s2d = FixedSpaceDepth().to(reference)
                self.refiner.d2s = FixedSpaceDepth(inverse=True).to(reference)
        from models.rep_mbconv import RepMBConv
        for block in list(self.modules()):
            if isinstance(block, (RepNCB, RepMBConv)):
                block.switch_to_deploy()
        return self

    def deploy(self):
        return copy.deepcopy(self).switch_to_deploy()
