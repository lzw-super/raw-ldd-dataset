"""Mobile Residual Local Feature Network for four-plane packed RAW.

The implementation mirrors ``LED-mine/led/archs/mrlfn_arch.py``.  During
training each :class:`ReparamConv3x3` uses three convolution branches; for
validation and inference they are fused exactly into one 3x3 convolution.

The optional paper-aligned path reconstructs the one-plane Bayer mosaic from
the repository's four row-major Bayer planes before applying the paper's
``k x k`` Space-to-Depth operation.  This distinction matters: applying
``PixelUnshuffle(k)`` directly to the four packed planes would create
``4*k*k`` channels, whereas Figure 6 specifies ``k*k`` channels.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn


class PackedBayerSpaceToDepth(nn.Module):
    """Apply mosaic-domain S2D to row-major four-plane packed Bayer RAW.

    Input planes must represent the 2x2 Bayer positions in raster order:
    ``[top-left, top-right, bottom-left, bottom-right]``.  SID Sony RGGB data
    in this repository therefore corresponds to ``[R, G1, G2, B]``.
    """

    def __init__(self, factor: int = 4):
        super().__init__()
        factor = int(factor)
        if factor < 2 or factor % 2:
            raise ValueError(f"S2D factor must be an even integer >= 2, got {factor}")
        self.factor = factor

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 4 or packed.shape[1] != 4:
            raise ValueError(f"Expected [N,4,H,W] packed Bayer RAW, got {tuple(packed.shape)}")
        mosaic_height = packed.shape[-2] * 2
        mosaic_width = packed.shape[-1] * 2
        if mosaic_height % self.factor or mosaic_width % self.factor:
            raise ValueError(
                f"Mosaic shape {(mosaic_height, mosaic_width)} must be divisible by "
                f"S2D factor {self.factor}"
            )
        mosaic = F.pixel_shuffle(packed, upscale_factor=2)
        return F.pixel_unshuffle(mosaic, downscale_factor=self.factor)


class PackedBayerDepthToSpace(nn.Module):
    """Inverse of :class:`PackedBayerSpaceToDepth`."""

    def __init__(self, factor: int = 4):
        super().__init__()
        factor = int(factor)
        if factor < 2 or factor % 2:
            raise ValueError(f"D2S factor must be an even integer >= 2, got {factor}")
        self.factor = factor

    def forward(self, reduced: torch.Tensor) -> torch.Tensor:
        expected_channels = self.factor**2
        if reduced.ndim != 4 or reduced.shape[1] != expected_channels:
            raise ValueError(
                f"Expected [N,{expected_channels},H,W] tensor before D2S, "
                f"got {tuple(reduced.shape)}"
            )
        mosaic = F.pixel_shuffle(reduced, upscale_factor=self.factor)
        return F.pixel_unshuffle(mosaic, downscale_factor=2)


class ReparamConv3x3(nn.Module):
    """Training-time three-convolution block that fuses to one 3x3 conv."""

    def __init__(self, channels: int, bias: bool = True, deploy: bool = False):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = int(channels)
        self.deploy_mode = bool(deploy)
        self.activation = nn.ReLU(inplace=True)

        if deploy:
            self.reparam_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        else:
            # A bias here could not be fused exactly at zero-padded borders.
            self.expand_conv = nn.Conv2d(channels, channels, 1, bias=False)
            self.spatial_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=bias)
            self.project_conv = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy_mode:
            return self.activation(self.reparam_conv(x))
        local_feature = x + self.expand_conv(x)
        refined_feature = self.project_conv(self.spatial_conv(local_feature))
        return self.activation(x + refined_feature)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the kernel and bias of the mathematically equivalent conv."""
        if self.deploy_mode:
            return self.reparam_conv.weight, self.reparam_conv.bias

        expand_weight = self.expand_conv.weight[:, :, 0, 0]
        spatial_weight = self.spatial_conv.weight
        project_weight = self.project_conv.weight[:, :, 0, 0]
        identity = torch.eye(self.channels, device=expand_weight.device, dtype=expand_weight.dtype)
        local_weight = expand_weight + identity

        equivalent_weight = torch.einsum("omhw,mi->oihw", spatial_weight, local_weight)
        equivalent_weight = torch.einsum("om,mihw->oihw", project_weight, equivalent_weight)
        equivalent_weight[:, :, 1, 1] += identity

        spatial_bias = (
            spatial_weight.new_zeros(self.channels)
            if self.spatial_conv.bias is None
            else self.spatial_conv.bias
        )
        project_bias = (
            spatial_weight.new_zeros(self.channels)
            if self.project_conv.bias is None
            else self.project_conv.bias
        )
        equivalent_bias = torch.matmul(project_weight, spatial_bias) + project_bias
        return equivalent_weight, equivalent_bias

    @torch.no_grad()
    def switch_to_deploy(self) -> "ReparamConv3x3":
        if self.deploy_mode:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam_conv = nn.Conv2d(self.channels, self.channels, 3, 1, 1, bias=True)
        reparam_conv = reparam_conv.to(device=kernel.device, dtype=kernel.dtype)
        reparam_conv.weight.copy_(kernel)
        reparam_conv.bias.copy_(bias)
        self.reparam_conv = reparam_conv
        del self.expand_conv
        del self.spatial_conv
        del self.project_conv
        self.deploy_mode = True
        return self


class MobileResidualLocalFeatureBlock(nn.Module):
    """The paper's mobile-optimized Residual Local Feature Block (mRLFB)."""

    def __init__(self, channels: int, bias: bool = True, deploy: bool = False):
        super().__init__()
        self.conv1 = ReparamConv3x3(channels, bias=bias, deploy=deploy)
        self.conv2 = ReparamConv3x3(channels, bias=bias, deploy=deploy)
        self.conv3 = ReparamConv3x3(channels, bias=bias, deploy=deploy)
        self.linear = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.conv3(self.conv2(self.conv1(x)))
        feature = self.linear(feature + x)
        return feature + x


class MRLFN(nn.Module):
    """Single-frame mRLFB network adapted to four-channel packed RAW.

    ``space_to_depth_factor=1`` preserves the original repository adaptation
    (no S2D/D2S) and its checkpoint shapes.  Setting it to the paper's
    ``k=4`` enables the Figure-6 path and also moves the shallow 1x1 fusion
    branch to the S2D output, as drawn in the paper.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        feature_channels: int = 16,
        num_blocks: int = 4,
        bias: bool = True,
        deploy: bool = False,
        space_to_depth_factor: int = 1,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or feature_channels <= 0:
            raise ValueError("Channel counts must all be positive")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.feature_channels = int(feature_channels)
        self.num_blocks = int(num_blocks)
        self.deploy_mode = bool(deploy)
        self.space_to_depth_factor = int(space_to_depth_factor)
        if self.space_to_depth_factor != 1 and (
            self.in_channels != 4 or self.out_channels != 4
        ):
            raise ValueError("Paper-aligned Bayer S2D/D2S requires four packed input/output planes")
        if self.space_to_depth_factor != 1 and (
            self.space_to_depth_factor < 2 or self.space_to_depth_factor % 2
        ):
            raise ValueError("space_to_depth_factor must be 1 or an even integer >= 2")

        self.paper_aligned_spatial = self.space_to_depth_factor != 1
        if self.paper_aligned_spatial:
            self.space_to_depth = PackedBayerSpaceToDepth(self.space_to_depth_factor)
            self.depth_to_space = PackedBayerDepthToSpace(self.space_to_depth_factor)
            network_in_channels = self.space_to_depth_factor**2
            network_out_channels = self.space_to_depth_factor**2
        else:
            self.space_to_depth = nn.Identity()
            self.depth_to_space = nn.Identity()
            network_in_channels = self.in_channels
            network_out_channels = self.out_channels

        self.shallow_conv = nn.Conv2d(network_in_channels, feature_channels, 3, 1, 1, bias=bias)
        self.blocks = nn.Sequential(
            *[
                MobileResidualLocalFeatureBlock(feature_channels, bias=bias, deploy=deploy)
                for _ in range(num_blocks)
            ]
        )
        self.deep_fusion = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1, bias=bias)
        shallow_fusion_channels = network_in_channels if self.paper_aligned_spatial else feature_channels
        self.shallow_fusion = nn.Conv2d(shallow_fusion_channels, feature_channels, 1, bias=bias)
        self.output_conv = nn.Conv2d(2 * feature_channels, network_out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected NCHW input with {self.in_channels} channels, got {tuple(x.shape)}"
            )
        reduced_raw = self.space_to_depth(x)
        shallow_feature = self.shallow_conv(reduced_raw)
        deep_feature = self.deep_fusion(self.blocks(shallow_feature))
        skip_source = reduced_raw if self.paper_aligned_spatial else shallow_feature
        shallow_feature = self.shallow_fusion(skip_source)
        reduced_output = self.output_conv(torch.cat((deep_feature, shallow_feature), dim=1))
        return self.depth_to_space(reduced_output)

    @torch.no_grad()
    def switch_to_deploy(self) -> "MRLFN":
        for module in self.modules():
            if isinstance(module, ReparamConv3x3):
                module.switch_to_deploy()
        self.deploy_mode = True
        return self

    @torch.no_grad()
    def deploy(self) -> "MRLFN":
        """Return a fused copy and leave the live training network untouched."""
        return deepcopy(self).switch_to_deploy()
