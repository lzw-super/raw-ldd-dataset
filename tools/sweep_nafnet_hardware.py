#!/usr/bin/env python3
"""NAFNet 硬件相关指标扫描工具。

本脚本针对 SID 索尼 packed-RAW 去噪用的 NAFNet (见 models/natnet_arch.py)，
在「初始通道数 width」与「深度」两个维度上做网格扫描，输出对 FPGA / ASIC
实现最关心的几类指标：

  1. 参数量 (parameters) —— 决定权重存储 (BRAM / 权重 ROM / DRAM)；
  2. 卷积+线性 MACs / FLOPs —— 决定算力需求 (DSP / MAC 单元数量)；
  3. 最大中间特征图 (max single feature map) —— 决定单层激活缓冲大小；
  4. U-Net 跳连总足迹 (sum of encoder skips) —— 决定需要同时驻留的激活总量；
  5. 逐分辨率特征图表 —— 直观看每一层激活分布；
  6. FPGA / ASIC 资源估算 —— 在给定并行度与时钟下的 DSP / BRAM / 面积 / 功耗。

关于「深度」的两种解读，本脚本同时支持：
  - depth = 每个 encoder/decoder 阶段的 NAFBlock 数量 (NAFNet 论文惯例,
    即 enc_blk_nums / dec_blk_nums 的取值)。当前 tiny 模型为 1。
  - depth = encoder/decoder 的下采样阶段数 (即 enc_blk_nums 列表长度)。
    当前 tiny 模型为 4 阶段。

FLOP 口径与 tools/calculate_model_info.py 保持一致：
  1 MAC = 1 次乘累加 ≈ 2 FLOPs, 仅统计 Conv2d / ConvTranspose2d / Linear。
激活、池化、拼接、SimpleGate、LayerNorm 的算术不计入 MACs (确定且可比)。

用法示例:
  python tools/sweep_nafnet_hardware.py \
      --input-shape 1 4 512 512 \
      --output-json experiments/sid_sony_nafnet_tiny/hw_sweep.json \
      --output-report experiments/sid_sony_nafnet_tiny/hw_sweep_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.natnet_arch import NAFNet  # noqa: E402

# 不同数据类型下单元素占用的字节数 (用于把「元素数」换算成「存储字节」)。
DTYPE_BYTES: dict[str, int] = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}


# --------------------------------------------------------------------------- #
# 1. MACs 统计 (口径与 calculate_model_info.py 完全一致)
# --------------------------------------------------------------------------- #
def count_conv_linear_macs(model: nn.Module, input_shape: Sequence[int],
                           device: torch.device) -> tuple[int, int]:
    """用 forward hook 统计 Conv2d / ConvTranspose2d / Linear 的 MACs。

    返回 (macs, profiled_layers)。
    """
    macs = 0
    profiled_layers = 0
    handles: list[Any] = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...],
             output: torch.Tensor) -> None:
        nonlocal macs, profiled_layers
        inp = inputs[0]
        if isinstance(module, nn.Conv2d):
            kernel_ops = (module.kernel_size[0] * module.kernel_size[1]
                          * module.in_channels // module.groups)
            macs += int(output.numel() * kernel_ops)
        elif isinstance(module, nn.ConvTranspose2d):
            kernel_ops = (module.kernel_size[0] * module.kernel_size[1]
                          * module.out_channels // module.groups)
            macs += int(inp.numel() * kernel_ops)
        elif isinstance(module, nn.Linear):
            macs += int(output.numel() * module.in_features)
        profiled_layers += 1

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    dummy = torch.zeros(input_shape, device=device, dtype=torch.float32)
    try:
        with torch.inference_mode():
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return int(macs), int(profiled_layers)


# --------------------------------------------------------------------------- #
# 2. 中间特征图统计: 单层最大特征图 + 逐分辨率分布
# --------------------------------------------------------------------------- #
def profile_feature_maps(model: nn.Module, input_shape: Sequence[int],
                         device: torch.device) -> dict[str, Any]:
    """对所有子模块挂 hook, 记录每个 input / output 张量的元素数。

    返回:
      max_single_elements        : 单个中间张量的最大元素数 (单层激活缓冲上界)
      max_single_tensor          : 该张量的 (模块名, 形状)
      all_tensors                : 全部捕获到的 (name, shape, numel) 列表
    """
    max_elements = 0
    max_tensor: tuple[str, list[int]] | None = None
    captured: list[tuple[str, list[int], int]] = []
    handles: list[Any] = []

    def record(name: str, tensor: torch.Tensor) -> None:
        nonlocal max_elements, max_tensor
        if not torch.is_tensor(tensor):
            return
        numel = int(tensor.numel())
        shape = [int(d) for d in tensor.shape]
        captured.append((name, shape, numel))
        if numel > max_elements:
            max_elements = numel
            max_tensor = (name, shape)

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...],
             output: Any) -> None:
        name = type(module).__name__
        if inputs and torch.is_tensor(inputs[0]):
            record(f"{name}.input", inputs[0])
        if torch.is_tensor(output):
            record(f"{name}.output", output)
        elif isinstance(output, (list, tuple)):
            for idx, item in enumerate(output):
                if torch.is_tensor(item):
                    record(f"{name}.output[{idx}]", item)

    # 对所有子模块 (含叶子层) 挂钩, 捕获每个算子的输入/输出。
    for module in model.modules():
        handles.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    dummy = torch.zeros(input_shape, device=device, dtype=torch.float32)
    try:
        with torch.inference_mode():
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return {
        "max_single_elements": max_elements,
        "max_single_tensor": max_tensor,
        "all_tensors": captured,
    }


def per_resolution_table(width: int, height: int, width_px: int,
                         num_stages: int, dtype_bytes: int) -> list[dict[str, Any]]:
    """解析地给出每个分辨率层的「基础特征图」大小。

    NAFNet 是 U-Net 结构, 第 r 个 encoder 阶段 (r=0..num_stages-1) 的通道数
    为 width * 2^r, 空间分辨率为 (H/2^r, W/2^r)。最深层 (middle) 通道数为
    width * 2^num_stages, 空间 (H/2^num_stages, W/2^num_stages)。

    NAFBlock 内部会有 DW_Expand=2 与 FFN_Expand=2 的扩张, 因此「单层最大」
    约为基础特征图的 2 倍; 这里同时给出 base 与 expanded(2x) 两列。
    """
    rows: list[dict[str, Any]] = []
    for stage in range(num_stages + 1):
        channels = width * (2 ** stage)
        h = height // (2 ** stage)
        w = width_px // (2 ** stage)
        base = channels * h * w
        rows.append({
            "stage": stage,
            "role": "middle" if stage == num_stages else f"enc/dec {stage}",
            "channels": channels,
            "spatial": [h, w],
            "base_elements": base,
            "expanded2x_elements": base * 2,
            "base_bytes_fp16": base * dtype_bytes,
        })
    return rows


def unet_skip_footprint(width: int, height: int, width_px: int,
                        num_stages: int, dtype_bytes: int) -> dict[str, Any]:
    """U-Net 跳连需要同时驻留的激活总量。

    encoder 会依次产生 num_stages 个 skip 特征 (stage 0..num_stages-1),
    它们在 decoder 阶段被逐个消费, 因此在解码过程中这些 skip 必须同时保留
    在片上 (或外存)。这里给出它们的元素数之和与字节大小。
    """
    skip_elements = 0
    per_stage: list[int] = []
    for stage in range(num_stages):
        channels = width * (2 ** stage)
        h = height // (2 ** stage)
        w = width_px // (2 ** stage)
        elem = channels * h * w
        per_stage.append(elem)
        skip_elements += elem
    return {
        "skip_elements": skip_elements,
        "skip_bytes_fp16": skip_elements * dtype_bytes,
        "per_stage_elements": per_stage,
    }


# --------------------------------------------------------------------------- #
# 3. 参数量按子模块分组
# --------------------------------------------------------------------------- #
def param_breakdown(model: NAFNet) -> dict[str, int]:
    """按 intro / encoders / downs / middle / decoders / ups / ending 分组统计参数。"""
    groups: dict[str, int] = OrderedDict()
    groups["intro"] = 0
    groups["encoders"] = 0
    groups["downs"] = 0
    groups["middle_blks"] = 0
    groups["decoders"] = 0
    groups["ups"] = 0
    groups["ending"] = 0

    for name, param in model.named_parameters():
        if name.startswith("intro"):
            groups["intro"] += param.numel()
        elif name.startswith("encoders"):
            groups["encoders"] += param.numel()
        elif name.startswith("downs"):
            groups["downs"] += param.numel()
        elif name.startswith("middle_blks"):
            groups["middle_blks"] += param.numel()
        elif name.startswith("decoders"):
            groups["decoders"] += param.numel()
        elif name.startswith("ups"):
            groups["ups"] += param.numel()
        elif name.startswith("ending"):
            groups["ending"] += param.numel()
    return groups


# --------------------------------------------------------------------------- #
# 4. 单个配置的完整剖析
# --------------------------------------------------------------------------- #
def build_config(width: int, blocks_per_stage: int, num_stages: int,
                 middle_blocks: int, img_channel: int) -> dict[str, Any]:
    """构造一组扫描配置的描述 (不实例化模型)。"""
    enc = tuple([blocks_per_stage] * num_stages)
    dec = tuple([blocks_per_stage] * num_stages)
    return {
        "width": width,
        "blocks_per_stage": blocks_per_stage,
        "num_stages": num_stages,
        "middle_blocks": middle_blocks,
        "img_channel": img_channel,
        "enc_blk_nums": list(enc),
        "dec_blk_nums": list(dec),
        "label": f"w{width}_d{blocks_per_stage}_s{num_stages}",
    }


def profile_one(config: dict[str, Any], input_shape: Sequence[int],
                device: torch.device) -> dict[str, Any]:
    """实例化模型并跑一次完整剖析。"""
    model = NAFNet(
        img_channel=config["img_channel"],
        width=config["width"],
        enc_blk_nums=tuple(config["enc_blk_nums"]),
        middle_blk_num=config["middle_blocks"],
        dec_blk_nums=tuple(config["dec_blk_nums"]),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, profiled_layers = count_conv_linear_macs(model, input_shape, device)
    fm = profile_feature_maps(model, input_shape, device)

    _, _, in_h, in_w = (int(v) for v in input_shape)
    dtype_bytes = DTYPE_BYTES["fp16"]
    resolution_table = per_resolution_table(config["width"], in_h, in_w,
                                            config["num_stages"], dtype_bytes)
    skip = unet_skip_footprint(config["width"], in_h, in_w,
                               config["num_stages"], dtype_bytes)
    breakdown = param_breakdown(model)

    max_single = fm["max_single_elements"]
    result = {
        "config": config,
        "input_shape": list(input_shape),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable),
        "weight_bytes_fp16": int(total_params * DTYPE_BYTES["fp16"]),
        "weight_bytes_int8": int(total_params * DTYPE_BYTES["int8"]),
        "macs": int(macs),
        "gmacs": macs / 1e9,
        "gflops": (2 * macs) / 1e9,
        "profiled_layers": int(profiled_layers),
        "max_single_feature_map_elements": int(max_single),
        "max_single_feature_map_bytes_fp16": int(max_single * DTYPE_BYTES["fp16"]),
        "max_single_feature_map_tensor": fm["max_single_tensor"],
        "unet_skip_footprint": skip,
        "per_resolution": resolution_table,
        "param_breakdown": breakdown,
    }
    # 释放模型显存/内存
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# 5. FPGA / ASIC 资源估算 (规划级, 非综合后精确值)
# --------------------------------------------------------------------------- #
# FPGA (Xilinx UltraScale+ 级别) 单资源假设
FPGA_BRAM_KB = 36 / 8 * 1024          # 36Kb BRAM ≈ 4 KiB
FPGA_BRAM_BITS = 36 * 1024
FPGA_INT16_MAC_PER_DSP = 1            # FP16/INT16: 每 DSP 约 1 个 MAC
FPGA_INT8_MAC_PER_DSP = 2             # INT8: 每 DSP 2 个 MAC
FPGA_LUT_PER_MAC = 12                 # 数据通路 + 互连, 粗略
FPGA_FF_PER_MAC = 18

# ASIC (28nm 规划级) 单资源假设
ASIC_MAC_KGE_INT8 = 2.0               # INT8 MAC 单元 ≈ 2.0 KGE (含寄存器)
ASIC_MAC_KGE_INT16 = 3.0              # INT16/FP16 MAC 单元 ≈ 3.0 KGE
ASIC_SRAM_UM2_PER_BIT_28NM = 0.6      # 28nm 单端口 SRAM ≈ 0.6 µm²/bit (粗略)
ASIC_CONTROL_KGE_FIXED = 50           # 控制/调度/LayerNorm/池化 等固定开销
ASIC_DYNAMIC_UW_PER_MHZ_PER_MAC = 12  # 每 MAC 每 MHz 动态功耗 (粗略)
ASIC_LEAKAGE_MW = 30                  # 28nm 漏电粗略


def estimate_fpga(result: dict[str, Any], clock_mhz: float,
                  parallel_macs: int, int8: bool = False) -> dict[str, Any]:
    """估算 FPGA 资源: 给定并行 MAC 数与时钟, 推算吞吐与资源占用。

    思路: 流水化加速器每周期可完成 parallel_macs 次 MAC。单张图像的 MAC 总数
    除以「每秒 MAC 数 = parallel_macs * clock_mhz * 1e6」得到单帧推理时延。
    """
    mac_per_image = result["macs"]
    mac_per_sec = parallel_macs * clock_mhz * 1e6
    latency_ms = mac_per_image / mac_per_sec * 1e3
    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")

    weight_bytes = result["weight_bytes_int8"] if int8 else result["weight_bytes_fp16"]
    act_bytes = result["max_single_feature_map_bytes_fp16"]  # 单层激活缓冲 (双缓冲约 2x)
    # 权重 + 激活双缓冲所需 BRAM 块数
    weight_bram = (weight_bytes * 8 + FPGA_BRAM_BITS - 1) // FPGA_BRAM_BITS
    act_bram = (act_bytes * 2 * 8 + FPGA_BRAM_BITS - 1) // FPGA_BRAM_BITS  # 输入+输出双缓冲

    mac_per_dsp = FPGA_INT8_MAC_PER_DSP if int8 else FPGA_INT16_MAC_PER_DSP
    dsp = -(-parallel_macs // mac_per_dsp)  # 向上取整
    lut = parallel_macs * FPGA_LUT_PER_MAC
    ff = parallel_macs * FPGA_FF_PER_MAC

    return {
        "platform": "FPGA",
        "clock_mhz": clock_mhz,
        "parallel_macs": parallel_macs,
        "quant": "int8" if int8 else "fp16/int16",
        "latency_ms_per_frame": latency_ms,
        "fps": fps,
        "dsp48": int(dsp),
        "weight_bram_36k": int(weight_bram),
        "act_bram_36k": int(act_bram),
        "total_bram_36k": int(weight_bram + act_bram),
        "lut_est": int(lut),
        "ff_est": int(ff),
        "weight_kib": weight_bytes / 1024,
        "act_buffer_kib": act_bytes * 2 / 1024,
    }


def estimate_asic(result: dict[str, Any], clock_mhz: float,
                  parallel_macs: int, int8: bool = True) -> dict[str, Any]:
    """估算 ASIC (28nm 规划级) 资源: 面积、SRAM、功耗。"""
    mac_per_image = result["macs"]
    mac_per_sec = parallel_macs * clock_mhz * 1e6
    latency_ms = mac_per_image / mac_per_sec * 1e3
    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")

    mac_kge = ASIC_MAC_KGE_INT8 if int8 else ASIC_MAC_KGE_INT16
    mac_area_kge = parallel_macs * mac_kge
    logic_kge = mac_area_kge + ASIC_CONTROL_KGE_FIXED

    # SRAM: 权重 (若放片上) + 激活缓冲 (双缓冲)
    weight_bits = result["total_parameters"] * (8 if int8 else 16)
    act_bits = result["max_single_feature_map_elements"] * 16 * 2  # 双缓冲, fp16
    weight_um2 = weight_bits * ASIC_SRAM_UM2_PER_BIT_28NM
    act_um2 = act_bits * ASIC_SRAM_UM2_PER_BIT_28NM
    sram_mm2 = (weight_um2 + act_um2) * 1e-6  # 含逻辑外围会更大, 这里是裸存储
    logic_mm2 = logic_kge * 1e-3 * 0.8        # 28nm 约 0.8 mm²/KGE (粗略)
    total_mm2 = logic_mm2 + sram_mm2 * 1.25   # SRAM 加 25% 外围/译码开销

    # 功耗: 动态 + 漏电
    dynamic_uw = parallel_macs * clock_mhz * ASIC_DYNAMIC_UW_PER_MHZ_PER_MAC
    dynamic_mw = dynamic_uw * 1e-3
    power_mw = dynamic_mw + ASIC_LEAKAGE_MW

    return {
        "platform": "ASIC",
        "process": "28nm (规划级估算)",
        "clock_mhz": clock_mhz,
        "parallel_macs": parallel_macs,
        "quant": "int8" if int8 else "int16/fp16",
        "latency_ms_per_frame": latency_ms,
        "fps": fps,
        "logic_kge": int(logic_kge),
        "logic_mm2": logic_mm2,
        "sram_weight_bits": int(weight_bits),
        "sram_act_bits": int(act_bits),
        "sram_mm2": sram_mm2,
        "total_mm2": total_mm2,
        "power_mw": power_mw,
    }


def required_parallelism_for_fps(result: dict[str, Any], target_fps: float,
                                 clock_mhz: float) -> int:
    """反推: 为达到 target_fps, 在给定时钟下所需的并行 MAC 单元数。"""
    budget_s = 1.0 / target_fps
    mac_per_sec_needed = result["macs"] / budget_s
    parallel = int(-(-(mac_per_sec_needed / (clock_mhz * 1e6)) // 1))  # 向上取整
    return max(parallel, 1)


# --------------------------------------------------------------------------- #
# 6. 主入口: 扫描 + 汇总 + 输出
# --------------------------------------------------------------------------- #
def build_sweep_configs(widths: Iterable[int], blocks: Iterable[int],
                        stages_list: Iterable[int], middle_blocks: int,
                        img_channel: int) -> list[dict[str, Any]]:
    """构建扫描配置集合 (去重)。"""
    seen: set[tuple[int, int, int]] = set()
    configs: list[dict[str, Any]] = []
    for w in widths:
        for bps in blocks:
            for st in stages_list:
                key = (w, bps, st)
                if key in seen:
                    continue
                seen.add(key)
                configs.append(build_config(w, bps, st, middle_blocks, img_channel))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NAFNet 深度/通道扫描 + FPGA/ASIC 资源估算",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-shape", type=int, nargs=4, default=[1, 4, 512, 512],
                        metavar=("N", "C", "H", "W"))
    parser.add_argument("--img-channel", type=int, default=4)
    parser.add_argument("--middle-blocks", type=int, default=2)
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 12, 16],
                        help="初始通道数候选")
    parser.add_argument("--blocks-per-stage", type=int, nargs="+",
                        default=[1, 2, 3, 4],
                        help="每个 enc/dec 阶段的 NAFBlock 数 (NAFNet 惯例下的深度)")
    parser.add_argument("--stages", type=int, nargs="+", default=[4],
                        help="encoder/decoder 下采样阶段数候选 (另一种深度解读)")
    parser.add_argument("--clock-mhz", type=float, default=200.0,
                        help="FPGA/ASIC 估算用时钟")
    parser.add_argument("--target-fps", type=float, default=30.0,
                        help="反推并行度所用目标帧率")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-report", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但不可用")

    configs = build_sweep_configs(
        args.widths, args.blocks_per_stage, args.stages,
        args.middle_blocks, args.img_channel,
    )
    print(f"[sweep] 共 {len(configs)} 个配置, 输入 {args.input_shape}")

    results: list[dict[str, Any]] = []
    for idx, config in enumerate(configs, 1):
        result = profile_one(config, args.input_shape, device)
        results.append(result)
        print(f"  [{idx:2d}/{len(configs)}] {config['label']:>16s}  "
              f"params={result['total_parameters']:>10,}  "
              f"GMACs={result['gmacs']:>6.3f}  "
              f"maxFM={result['max_single_feature_map_elements']:>12,}")

    # ---- FPGA/ASIC 估算: 对每个配置反推 30fps 所需并行度, 再估资源 ----
    hw_estimates: list[dict[str, Any]] = []
    for result in results:
        p_int8 = required_parallelism_for_fps(result, args.target_fps, args.clock_mhz)
        # 把并行度凑成常用 2 的幂 (向上), 便于对照 PE 阵列规模
        p_int8_pow2 = 1 << (p_int8 - 1).bit_length() if p_int8 > 0 else 1
        fpga_int8 = estimate_fpga(result, args.clock_mhz, p_int8_pow2, int8=True)
        fpga_fp16 = estimate_fpga(result, args.clock_mhz, p_int8_pow2, int8=False)
        asic_int8 = estimate_asic(result, args.clock_mhz, p_int8_pow2, int8=True)
        hw_estimates.append({
            "label": result["config"]["label"],
            "config": result["config"],
            "required_parallel_int8_ceil": p_int8,
            "parallel_pow2": p_int8_pow2,
            "fpga_int8": fpga_int8,
            "fpga_fp16": fpga_fp16,
            "asic_int8": asic_int8,
        })

    payload = {
        "input_shape": list(args.input_shape),
        "clock_mhz": args.clock_mhz,
        "target_fps": args.target_fps,
        "dtype_bytes": DTYPE_BYTES,
        "results": results,
        "hardware_estimates": hw_estimates,
    }

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"[sweep] JSON 已保存: {out_json}")

    if args.output_report:
        report = render_report(payload)
        out_md = Path(args.output_report)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")
        print(f"[sweep] 报告已保存: {out_md}")


def render_report(payload: dict[str, Any]) -> str:
    """把扫描结果渲染成 Markdown 报告 (中文)。"""
    lines: list[str] = []
    input_shape = payload["input_shape"]
    clock = payload["clock_mhz"]
    fps = payload["target_fps"]
    lines.append("# NAFNet 深度/通道扫描 与 FPGA/ASIC 资源分析\n")
    lines.append(f"- 输入 (NCHW): `{input_shape}`")
    lines.append(f"- 估算时钟: {clock} MHz; 目标帧率: {fps} fps")
    lines.append("- FLOP 口径: 1 MAC ≈ 2 FLOPs, 仅统计 Conv/Linear\n")

    # ---- 主结果表 ----
    lines.append("## 1. 参数量 / 计算量 / 最大特征图 总览\n")
    header = ("| 配置 | width | 深度(block/stage) | 阶段数 | 参数量 | "
              "权重(fp16) | GMACs | GFLOPs | 单层最大特征图(元素) | 单层最大(fp16) | "
              "跳连总足迹(元素) |")
    sep = "|" + "---|" * 11
    lines.append(header)
    lines.append(sep)
    for r in payload["results"]:
        c = r["config"]
        lines.append(
            f"| {c['label']} | {c['width']} | {c['blocks_per_stage']} | "
            f"{c['num_stages']} | {r['total_parameters']:,} | "
            f"{r['weight_bytes_fp16']/1024:.1f} KiB | {r['gmacs']:.3f} | "
            f"{r['gflops']:.3f} | {r['max_single_feature_map_elements']:,} | "
            f"{r['max_single_feature_map_bytes_fp16']/1024:.1f} KiB | "
            f"{r['unet_skip_footprint']['skip_elements']:,} |"
        )
    lines.append("")

    # ---- 逐分辨率特征图 (取一个代表配置, 例如当前 baseline) ----
    # 找 baseline: w16 d1 s4; 找不到就用第一个
    base = next((r for r in payload["results"]
                 if r["config"]["width"] == 16
                 and r["config"]["blocks_per_stage"] == 1
                 and r["config"]["num_stages"] == 4), payload["results"][0])
    lines.append(f"## 2. 逐分辨率特征图分布 (代表配置: {base['config']['label']})\n")
    lines.append("stage = 下采样次数; base = 该层基础特征图; expanded2x = NAFBlock 内部扩张后峰值。\n")
    lines.append("| stage | 角色 | 通道 | 空间 | base(元素) | base(fp16) | expanded2x(元素) |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in base["per_resolution"]:
        lines.append(
            f"| {row['stage']} | {row['role']} | {row['channels']} | "
            f"{row['spatial'][0]}x{row['spatial'][1]} | {row['base_elements']:,} | "
            f"{row['base_bytes_fp16']/1024:.1f} KiB | {row['expanded2x_elements']:,} |"
        )
    lines.append("")
    lines.append(f"U-Net 跳连需同时驻留的激活: "
                 f"{base['unet_skip_footprint']['skip_elements']:,} 元素 "
                 f"({base['unet_skip_footprint']['skip_bytes_fp16']/1024:.1f} KiB @fp16)\n")

    # ---- 硬件估算表 ----
    lines.append("## 3. FPGA 资源估算 (目标 %g fps @ %g MHz, INT8 量化)\n" % (fps, clock))
    lines.append("并行度 = 为达到目标帧率在给定时钟下所需的 MAC 单元数 (向上取整到 2 的幂)。\n")
    lines.append("| 配置 | 所需并行(ceil) | 并行(2^) | DSP48 | 权重BRAM(36K) | "
                 "激活BRAM(36K) | 总BRAM | LUT估 | FF估 | 单帧时延(ms) | 实际FPS |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for hw in payload["hardware_estimates"]:
        f = hw["fpga_int8"]
        lines.append(
            f"| {hw['label']} | {hw['required_parallel_int8_ceil']} | "
            f"{hw['parallel_pow2']} | {f['dsp48']} | {f['weight_bram_36k']} | "
            f"{f['act_bram_36k']} | {f['total_bram_36k']} | {f['lut_est']:,} | "
            f"{f['ff_est']:,} | {f['latency_ms_per_frame']:.2f} | {f['fps']:.1f} |"
        )
    lines.append("")

    lines.append("## 4. ASIC 资源估算 (28nm 规划级, INT8, 同上并行度)\n")
    lines.append("| 配置 | 逻辑(KGE) | 逻辑(mm²) | 权重SRAM(bit) | 激活SRAM(bit) | "
                 "SRAM(mm²) | 总面积(mm²) | 功耗(mW) | 单帧时延(ms) | 实际FPS |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for hw in payload["hardware_estimates"]:
        a = hw["asic_int8"]
        lines.append(
            f"| {hw['label']} | {a['logic_kge']:,} | {a['logic_mm2']:.2f} | "
            f"{a['sram_weight_bits']:,} | {a['sram_act_bits']:,} | "
            f"{a['sram_mm2']:.2f} | {a['total_mm2']:.2f} | {a['power_mw']:.0f} | "
            f"{a['latency_ms_per_frame']:.2f} | {a['fps']:.1f} |"
        )
    lines.append("")

    lines.append("## 5. 说明与假设\n")
    lines.append("- 上述 FPGA/ASIC 数字为 **规划级粗估**, 用于横向对比不同深度/通道配置, "
                 "非综合/流片后精确值。实际资源取决于量化方案、PE 阵列拓扑、流水深度、"
                 "LayerNorm/SimpleGate 的实现方式等。")
    lines.append("- MAC 统计不含 LayerNorm / SimpleGate / 池化 / 元素乘 的算术, "
                 "硬件实现时这些会额外消耗 LUT/逻辑, 但相对卷积量级很小。")
    lines.append("- 「单层最大特征图」指任一中间张量的最大元素数; 折叠式加速器据此设计激活缓冲。")
    lines.append("- 「跳连总足迹」指 U-Net encoder 各 skip 在解码期需同时保留的激活量。")
    lines.append("- 权重 BRAM 假设全片上; 若放外存 (DRAM) 则 BRAM 仅需激活缓冲, 但带宽成为瓶颈。\n")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
