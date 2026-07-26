"""Standalone NAFNet architecture adapted from LED-mine.

Reference:
    /home/zhengwu/Desktop/graduate-workspace/LED-mine/led/archs/nafnet_arch.py

The original module depends on LED's ``ARCH_REGISTRY``.  This copy removes
only that framework dependency and keeps module/parameter names compatible
with the original NAFNet state dict.  ``NATNet`` is provided as a spelling
compatibility alias; the architecture's established name is ``NAFNet``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    """Channel-wise LayerNorm for an NCHW tensor."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _, channels, _, _ = x.size()
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / (variance + eps).sqrt()
        ctx.save_for_backward(normalized, variance, weight)
        return weight.view(1, channels, 1, 1) * normalized + bias.view(1, channels, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        _, channels, _, _ = grad_output.size()
        normalized, variance, weight = ctx.saved_tensors
        grad = grad_output * weight.view(1, channels, 1, 1)
        mean_grad = grad.mean(dim=1, keepdim=True)
        mean_grad_normalized = (grad * normalized).mean(dim=1, keepdim=True)
        grad_input = (
            grad - normalized * mean_grad_normalized - mean_grad
        ) / torch.sqrt(variance + eps)
        grad_weight = (grad_output * normalized).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Replace an activation with channel split and element-wise multiplication."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        DW_Expand: int = 2,
        FFN_Expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        depthwise_channels = channels * DW_Expand
        if depthwise_channels % 2:
            raise ValueError("channels * DW_Expand must be even for SimpleGate")
        ffn_channels = channels * FFN_Expand
        if ffn_channels % 2:
            raise ValueError("channels * FFN_Expand must be even for SimpleGate")

        self.conv1 = nn.Conv2d(channels, depthwise_channels, kernel_size=1, bias=True)
        self.dwconv2 = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            kernel_size=3,
            padding=1,
            groups=depthwise_channels,
            bias=True,
        )
        self.conv3 = nn.Conv2d(depthwise_channels // 2, channels, kernel_size=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(depthwise_channels // 2, depthwise_channels // 2, kernel_size=1, bias=True),
        )
        self.sg = SimpleGate()

        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        # Zero initialization makes every new NAFBlock an identity mapping at
        # initialization, which stabilizes optimization of the deep residual path.
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.dwconv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.downconv = nn.Conv2d(channels, 2 * channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downconv(x)


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.upconv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.upconv(x))


class NAFNet(nn.Module):
    """Activation-free encoder-decoder restoration network.

    For the LED SID packed-RAW configuration use:

    ``NAFNet(img_channel=4, width=32, middle_blk_num=2,
             enc_blk_nums=(2, 2, 2, 2), dec_blk_nums=(2, 2, 2, 2))``.
    """

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 16,
        middle_blk_num: int = 1,
        enc_blk_nums: Sequence[int] = (),
        dec_blk_nums: Sequence[int] = (),
    ):
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc_blk_nums and dec_blk_nums must have the same number of stages")
        if img_channel <= 0 or width <= 0 or middle_blk_num < 0:
            raise ValueError("img_channel and width must be positive; middle_blk_num must be non-negative")
        if any(number < 0 for number in (*enc_blk_nums, *dec_blk_nums)):
            raise ValueError("block counts must be non-negative")

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = width
        for block_count in enc_blk_nums:
            self.encoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(block_count))))
            self.downs.append(DownSample(channels))
            channels *= 2

        self.middle_blks = nn.Sequential(*(NAFBlock(channels) for _ in range(middle_blk_num)))

        for block_count in dec_blk_nums:
            self.ups.append(UpSample(channels))
            channels //= 2
            self.decoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(block_count))))

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.size()
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_width, 0, pad_height))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, height, width = inp.shape
        padded_input = self.check_image_size(inp)
        x = self.intro(padded_input)
        encoder_skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encoder_skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(encoder_skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + padded_input
        return x[:, :, :height, :width]


# Compatibility with the spelling used in the request.  Both names construct
# exactly the same module and keep the canonical NAFNet state-dict keys.
NATNet = NAFNet
NatNet = NAFNet


__all__ = [
    "LayerNorm2d",
    "SimpleGate",
    "NAFBlock",
    "DownSample",
    "UpSample",
    "NAFNet",
    "NATNet",
    "NatNet",
]
