#!/usr/bin/env python3
"""Calculate parameter counts and convolutional MACs/FLOPs for SID U-Net.

The FLOP convention used here is explicit: one multiply-accumulate is one MAC
and approximately two FLOPs.  Counts cover Conv2d, ConvTranspose2d and Linear
layers.  Activation, pooling, concatenation, crop and padding operations are
not included, which makes the result deterministic and easy to compare.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Sequence

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.ELD_models import UNetSeeInDark
from models.mrlfn_arch import MRLFN
from models.natnet_arch import NAFNet
from utils.argparse_compat import add_boolean_optional_argument


def _shape(value: torch.Tensor) -> list[int]:
    return [int(dimension) for dimension in value.shape]


def calculate_model_info(
    model: nn.Module,
    input_shape: Sequence[int],
    device: torch.device | str | None = None,
) -> Dict[str, Any]:
    """Profile a model with one real forward pass.

    Args:
        model: Model to profile.
        input_shape: NCHW input shape, including batch size.
        device: Profiling device. Defaults to the model parameter device.
    """
    input_shape = tuple(int(value) for value in input_shape)
    if len(input_shape) != 4 or any(value <= 0 for value in input_shape):
        raise ValueError(f"input_shape must contain four positive NCHW values, got {input_shape}")

    parameters = list(model.parameters())
    if device is None:
        device = parameters[0].device if parameters else torch.device("cpu")
    device = torch.device(device)
    model_device = parameters[0].device if parameters else device
    if model_device != device:
        raise ValueError(f"model is on {model_device}, but profiling device is {device}")

    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    buffer_elements = sum(buffer.numel() for buffer in model.buffers())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    module_counts = Counter(type(module).__name__ for module in model.modules() if module is not model)

    macs = 0
    profiled_layers = 0
    first_compute_shape: list[int] | None = None
    handles: list[Any] = []

    def count_operations(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs, profiled_layers, first_compute_shape
        input_tensor = inputs[0]
        if first_compute_shape is None:
            first_compute_shape = _shape(input_tensor)

        if isinstance(module, nn.Conv2d):
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            macs += int(output.numel() * kernel_ops)
        elif isinstance(module, nn.ConvTranspose2d):
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
            macs += int(input_tensor.numel() * kernel_ops)
        elif isinstance(module, nn.Linear):
            macs += int(output.numel() * module.in_features)
        profiled_layers += 1

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(module.register_forward_hook(count_operations))

    was_training = model.training
    model.eval()
    dtype = parameters[0].dtype if parameters else torch.float32
    dummy_input = torch.zeros(input_shape, device=device, dtype=dtype)
    try:
        with torch.inference_mode():
            output = model(dummy_input)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    if not isinstance(output, torch.Tensor):
        raise TypeError("The profiler currently expects a single Tensor model output")

    flops = 2 * macs
    return {
        "model_name": type(model).__name__,
        "input_shape": list(input_shape),
        "effective_first_layer_input_shape": first_compute_shape,
        "output_shape": _shape(output),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "non_trainable_parameters": int(total_parameters - trainable_parameters),
        "buffer_elements": int(buffer_elements),
        "parameter_and_buffer_bytes": int(parameter_bytes + buffer_bytes),
        "model_size_fp32_mib": (parameter_bytes + buffer_bytes) / (1024**2),
        "profiled_layers": int(profiled_layers),
        "module_counts": dict(sorted(module_counts.items())),
        "macs": int(macs),
        "gmacs": macs / 1e9,
        "flops": int(flops),
        "gflops": flops / 1e9,
        "flops_convention": (
            "1 MAC = 1 multiply-accumulate ~= 2 FLOPs; counts Conv2d, "
            "ConvTranspose2d and Linear only"
        ),
    }


def format_model_info(info: Dict[str, Any]) -> str:
    """Format the profiler result for terminal output."""
    lines = [
        "=" * 72,
        f"Model:                         {info['model_name']}",
        f"Requested input (NCHW):        {info['input_shape']}",
        f"Effective first-layer input:   {info['effective_first_layer_input_shape']}",
        f"Output (NCHW):                 {info['output_shape']}",
        f"Parameters:                    {info['total_parameters']:,}",
        f"Trainable parameters:          {info['trainable_parameters']:,}",
        f"Non-trainable parameters:      {info['non_trainable_parameters']:,}",
        f"FP32 parameter/buffer size:    {info['model_size_fp32_mib']:.2f} MiB",
        f"Profiled compute layers:       {info['profiled_layers']}",
        f"MACs / forward:                {info['macs']:,} ({info['gmacs']:.3f} GMACs)",
        f"FLOPs / forward:               {info['flops']:,} ({info['gflops']:.3f} GFLOPs)",
        f"Convention:                    {info['flops_convention']}",
        "=" * 72,
    ]
    return "\n".join(lines)


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=["unet", "nafnet", "natnet", "mrlfn"], default="unet")
    parser.add_argument("--input-shape", type=int, nargs=4, default=[1, 4, 512, 512], metavar=("N", "C", "H", "W"))
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=4)
    parser.add_argument(
        "--features",
        type=int,
        default=None,
        help="Feature width; inferred from checkpoint, else 16 for MRLFN and 32 for U-Net/NAFNet",
    )
    parser.add_argument("--encoder-blocks", type=int, nargs="+", default=[2, 2, 2, 2])
    parser.add_argument("--middle-blocks", type=int, default=2)
    parser.add_argument("--decoder-blocks", type=int, nargs="+", default=[2, 2, 2, 2])
    parser.add_argument("--num-blocks", type=int, default=None, help="MRLFN block count N; inferred, else 4")
    add_boolean_optional_argument(parser, "--model-bias", default=None)
    parser.add_argument(
        "--space-to-depth-factor",
        type=int,
        default=None,
        help="MRLFN mosaic-domain S2D/D2S factor; inferred from checkpoint, else disabled",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=None, help="Optional bare or training checkpoint")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--metrics-jsonl",
        default=None,
        help="Optionally append one model_info metadata record to an existing training metrics JSONL",
    )
    args = parser.parse_args()

    checkpoint_args: dict[str, Any] = {}
    if args.checkpoint:
        checkpoint_metadata = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(checkpoint_metadata, dict):
            checkpoint_args = checkpoint_metadata.get("args", {})
    if args.features is None:
        key, fallback = ("feature_channels", 16) if args.model == "mrlfn" else ("model_width", 32)
        args.features = int(checkpoint_args.get(key, fallback))
    if args.num_blocks is None:
        args.num_blocks = int(checkpoint_args.get("num_blocks", 4))
    if args.model_bias is None:
        args.model_bias = bool(checkpoint_args.get("model_bias", True))
    if args.space_to_depth_factor is None:
        args.space_to_depth_factor = int(checkpoint_args.get("space_to_depth_factor", 1))

    if args.input_shape[1] != args.in_channels:
        raise ValueError("--input-shape channel count must equal --in-channels")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if args.model == "unet":
        model = UNetSeeInDark(in_nc=args.in_channels, out_nc=args.out_channels, nf=args.features)
        architecture_config = {
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "features": args.features,
        }
    elif args.model in {"nafnet", "natnet"}:
        if args.in_channels != args.out_channels:
            raise ValueError("NAFNet's global image residual requires equal input and output channels")
        model = NAFNet(
            img_channel=args.in_channels,
            width=args.features,
            enc_blk_nums=tuple(args.encoder_blocks),
            middle_blk_num=args.middle_blocks,
            dec_blk_nums=tuple(args.decoder_blocks),
        )
        architecture_config = {
            "img_channel": args.in_channels,
            "width": args.features,
            "enc_blk_nums": args.encoder_blocks,
            "middle_blk_num": args.middle_blocks,
            "dec_blk_nums": args.decoder_blocks,
        }
    else:
        model = MRLFN(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_channels=args.features,
            num_blocks=args.num_blocks,
            bias=args.model_bias,
            deploy=False,
            space_to_depth_factor=args.space_to_depth_factor,
        )
        architecture_config = {
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "feature_channels": args.features,
            "num_blocks": args.num_blocks,
            "bias": args.model_bias,
            "space_to_depth_factor": args.space_to_depth_factor,
            "deploy": True,
        }
    if args.checkpoint and args.model != "mrlfn":
        _load_checkpoint(model, args.checkpoint)
    elif args.model == "mrlfn":
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_deploy" in checkpoint:
                model = MRLFN(
                    in_channels=args.in_channels,
                    out_channels=args.out_channels,
                    feature_channels=args.features,
                    num_blocks=args.num_blocks,
                    bias=args.model_bias,
                    deploy=True,
                    space_to_depth_factor=args.space_to_depth_factor,
                )
                model.load_state_dict(checkpoint["model_deploy"], strict=True)
            else:
                state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
                bare_is_deploy = isinstance(state_dict, dict) and any(
                    ".reparam_conv." in key for key in state_dict
                )
                if bare_is_deploy:
                    model = MRLFN(
                        in_channels=args.in_channels,
                        out_channels=args.out_channels,
                        feature_channels=args.features,
                        num_blocks=args.num_blocks,
                        bias=args.model_bias,
                        deploy=True,
                        space_to_depth_factor=args.space_to_depth_factor,
                    )
                    model.load_state_dict(state_dict, strict=True)
                else:
                    model.load_state_dict(state_dict, strict=True)
                    model = model.deploy()
        else:
            model = model.deploy()
    model.to(device)
    info = calculate_model_info(model, args.input_shape, device)
    info["architecture_config"] = architecture_config
    info["graph_state"] = "deploy" if args.model == "mrlfn" else "native"
    if args.checkpoint:
        info["checkpoint"] = str(args.checkpoint)
    print(format_model_info(info))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON: {output_path}")
    if args.metrics_jsonl:
        metrics_path = Path(args.metrics_jsonl)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"record_type": "model_info", "model_info": info}
        with metrics_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Appended model_info record: {metrics_path}")


if __name__ == "__main__":
    main()
