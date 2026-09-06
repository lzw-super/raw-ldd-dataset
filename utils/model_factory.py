# -*- coding: utf-8 -*-
"""统一的去噪模型工厂。

从训练 checkpoint 还原 4 通道 packed-RAW 去噪器，自动推断模型类型与结构，
命令行参数可覆盖。兼容 ``train_sid_sony.py`` / ``train_phone.py`` 保存的可恢复
字典（state dict 在 ``"model"`` 键下）与官方发布的纯 state_dict。

设计目的：集中“如何从 checkpoint 还原模型”这一复杂决策，避免在
``qual_denoise_compare.py``、``test_denoise_sideld.py``、
``evaluate_synthetic_checkpoint.py`` 等多处重复实现（蓝本取自
``test_denoise_sideld.py`` 的 ``build_model``）。
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from models.ELD_models import UNetSeeInDark
from models.mrlfn_arch import MRLFN
from models.natnet_arch import NAFNet

# 各模型的默认结构：当 checkpoint 未记录、且 CLI 未显式覆盖时使用
_DEFAULT_WIDTH = 32
_DEFAULT_ENC_BLOCKS: tuple[int, ...] = (2, 2, 2, 2)
_DEFAULT_MIDDLE_BLOCKS = 2
_DEFAULT_DEC_BLOCKS: tuple[int, ...] = (2, 2, 2, 2)
_DEFAULT_MRLFN_FEATURES = 16
_DEFAULT_MRLFN_BLOCKS = 4


def _coerce_int_tuple(value: Any, default: Sequence[int]) -> tuple[int, ...]:
    """把 list / tuple / ndarray 统一成 int 元组；None 时回退到默认值。"""
    if value is None:
        return tuple(int(v) for v in default)
    return tuple(int(v) for v in value)


def build_denoiser_from_checkpoint(
    cp_dir: str,
    device: torch.device | str | None = None,
    model: str | None = None,
    model_width: int | None = None,
    encoder_blocks: Sequence[int] | None = None,
    middle_blocks: int | None = None,
    decoder_blocks: Sequence[int] | None = None,
    feature_channels: int | None = None,
    num_blocks: int | None = None,
    model_bias: bool | None = None,
    space_to_depth_factor: int | None = None,
) -> tuple[torch.nn.Module, dict]:
    """从 checkpoint 还原 4 通道 packed-RAW 去噪器。

    参数优先级：显式参数 > ``checkpoint["args"]`` > 默认值。

    - ``model``：``unet`` / ``nafnet`` / ``natnet`` / ``mrlfn``。
      为 None 时按 ``checkpoint["args"]["model"]`` 推断，再回退到 ``unet``。
    - 兼容两种 checkpoint：可恢复字典（权重在 ``"model"`` 键下）与官方纯 state_dict。

    返回 ``(net, meta)``：``meta`` 记录最终解析到的模型名与结构，供日志、
    结果 JSON 或图内标注使用。
    """
    checkpoint = torch.load(cp_dir, map_location="cpu")
    checkpoint_args: dict = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    # 1) 模型类型：显式 > checkpoint args > 默认 unet；natnet 归一化为 nafnet
    model_name = model or checkpoint_args.get("model", "unet")
    model_name = "nafnet" if model_name == "natnet" else model_name

    # 2) 结构超参：显式 > checkpoint args > 默认
    if model_width is not None:
        width = int(model_width)
    else:
        width = int(checkpoint_args.get("model_width", _DEFAULT_WIDTH))
    enc_blocks = _coerce_int_tuple(
        encoder_blocks if encoder_blocks is not None else checkpoint_args.get("encoder_blocks"),
        _DEFAULT_ENC_BLOCKS,
    )
    if middle_blocks is not None:
        mid_blocks = int(middle_blocks)
    else:
        mid_blocks = int(checkpoint_args.get("middle_blocks", _DEFAULT_MIDDLE_BLOCKS))
    dec_blocks = _coerce_int_tuple(
        decoder_blocks if decoder_blocks is not None else checkpoint_args.get("decoder_blocks"),
        _DEFAULT_DEC_BLOCKS,
    )
    mrlfn_features = int(
        feature_channels
        if feature_channels is not None
        else checkpoint_args.get("feature_channels", _DEFAULT_MRLFN_FEATURES)
    )
    mrlfn_blocks = int(
        num_blocks if num_blocks is not None else checkpoint_args.get("num_blocks", _DEFAULT_MRLFN_BLOCKS)
    )
    mrlfn_bias = bool(model_bias if model_bias is not None else checkpoint_args.get("model_bias", True))
    mrlfn_s2d_factor = int(
        space_to_depth_factor
        if space_to_depth_factor is not None
        else checkpoint_args.get("space_to_depth_factor", 1)
    )

    packaged_checkpoint = isinstance(checkpoint, dict) and (
        "model" in checkpoint or "model_deploy" in checkpoint
    )
    if packaged_checkpoint and model_name == "mrlfn" and "model_deploy" in checkpoint:
        state_dict = checkpoint["model_deploy"]
        weight_source = "model_deploy"
        build_deploy = True
    else:
        state_dict = checkpoint["model"] if packaged_checkpoint and "model" in checkpoint else checkpoint
        weight_source = "model" if packaged_checkpoint else "bare_state_dict"
        build_deploy = bool(
            model_name == "mrlfn"
            and isinstance(state_dict, dict)
            and any(".reparam_conv." in key for key in state_dict)
        )

    # Parameterless S2D/D2S modules leave no state-dict keys.  For a bare
    # paper-aligned checkpoint, infer k from the Figure-6 stem/head widths.
    if (
        model_name == "mrlfn"
        and space_to_depth_factor is None
        and "space_to_depth_factor" not in checkpoint_args
        and isinstance(state_dict, dict)
    ):
        stem_weight = state_dict.get("shallow_conv.weight")
        output_weight = state_dict.get("output_conv.weight")
        if stem_weight is not None and output_weight is not None and stem_weight.shape[1] > 4:
            inferred_factor = math.isqrt(int(stem_weight.shape[1]))
            if inferred_factor**2 == stem_weight.shape[1] == output_weight.shape[0]:
                mrlfn_s2d_factor = inferred_factor

    # 3) 按模型名构造网络
    if model_name == "unet":
        net = UNetSeeInDark(in_nc=4, out_nc=4, nf=width)
    elif model_name == "nafnet":
        net = NAFNet(
            img_channel=4,
            width=width,
            enc_blk_nums=enc_blocks,
            middle_blk_num=mid_blocks,
            dec_blk_nums=dec_blocks,
        )
    elif model_name == "mrlfn":
        net = MRLFN(
            in_channels=4,
            out_channels=4,
            feature_channels=mrlfn_features,
            num_blocks=mrlfn_blocks,
            bias=mrlfn_bias,
            deploy=build_deploy,
            space_to_depth_factor=mrlfn_s2d_factor,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name!r} (expected unet / nafnet / natnet / mrlfn)")

    # 4) 载入权重：可恢复字典取 "model" 键，否则视为纯 state_dict
    net.load_state_dict(state_dict, strict=True)
    if model_name == "mrlfn" and not build_deploy:
        net = net.deploy()
        weight_source += "->fused"

    if device is not None:
        net = net.to(device)
    net.eval()

    meta = {
        "model": model_name,
        "model_width": width,
        "encoder_blocks": list(enc_blocks),
        "middle_blocks": mid_blocks,
        "decoder_blocks": list(dec_blocks),
        "feature_channels": mrlfn_features,
        "num_blocks": mrlfn_blocks,
        "model_bias": mrlfn_bias,
        "space_to_depth_factor": mrlfn_s2d_factor,
        "graph_state": "deploy" if model_name == "mrlfn" else "native",
        "weight_source": weight_source,
    }
    return net, meta
