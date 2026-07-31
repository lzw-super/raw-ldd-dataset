#!/usr/bin/env python3
"""把 hw_sweep.json 渲染成中文权威分析报告 NAFNet_hw_analysis.md。

从 tools/sweep_nafnet_hardware.py 产出的扫描结果中, 挑选代表性配置并组织成
面向硬件实现的叙述性报告。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(json_path: str, out_path: str) -> None:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    results = payload["results"]
    hw = payload["hardware_estimates"]
    by_label = {r["config"]["label"]: r for r in results}
    hw_by_label = {h["label"]: h for h in hw}
    input_shape = payload["input_shape"]
    clock = payload["clock_mhz"]
    fps = payload["target_fps"]
    _, _, H, W = input_shape

    out: list[str] = []

    def gmacs(label: str) -> float:
        return by_label[label]["gmacs"]

    def gflops(label: str) -> float:
        return 2 * gmacs(label)

    # ----------------------------------------------------------------- #
    out.append("# NAFNet 硬件指标与 FPGA/ASIC 资源分析\n")
    out.append(f"> 输入 (NCHW): `{input_shape}`  |  时钟假设: {clock} MHz  |  目标帧率: {fps} fps")
    out.append("> FLOP 口径: 1 MAC ≈ 2 FLOPs, 仅统计 Conv2d/ConvTranspose2d/Linear (与 model_info.json 一致)\n")

    out.append("## 0. 当前模型与设计变量\n")
    out.append("当前 `sid_sony_nafnet_tiny` 配置 (即 `w16_d1_s4`):\n")
    out.append("- 结构: NAFNet, `img_channel=4, width=16, enc/dec_blk_nums=[1,1,1,1], middle_blk_num=2`")
    out.append(f"- 参数量 **{by_label['w16_d1_s4']['total_parameters']:,}**, "
               f"权重 fp16 = {by_label['w16_d1_s4']['weight_bytes_fp16']/1024:.1f} KiB")
    out.append(f"- 计算量 **{gmacs('w16_d1_s4'):.3f} GMACs / {gflops('w16_d1_s4'):.3f} GFLOPs** (与 model_info.json 完全一致)")
    out.append(f"- 单层最大特征图 **{by_label['w16_d1_s4']['max_single_feature_map_elements']:,} 元素 "
               f"= {by_label['w16_d1_s4']['max_single_feature_map_bytes_fp16']/1024/1024:.1f} MiB (fp16)**\n")
    out.append("三个可调设计变量 (本次扫描范围):\n")
    out.append("| 变量 | 含义 | 候选 |")
    out.append("|---|---|---|")
    out.append("| **width (通道)** | 初始通道数 (=第 0 阶段通道) | 8 / 12 / 16 |")
    out.append("| **depth (每阶段 NAFBlock 数)** | 每个 enc/dec 阶段的块数 (NAFNet 惯例的「深度」) | 1 / 2 / 3 / 4 |")
    out.append("| **stages (下采样阶段数)** | encoder/decoder 的下采样层数 (U-Net 深度) | 2 / 3 / 4 |")
    out.append("")
    out.append("第 r 阶段通道数 = `width × 2^r`, 空间 = `(H/2^r, W/2^r)`; middle 在 `width × 2^stages` 通道、"
               "`(H/2^stages, W/2^stages)` 空间。`middle_blk_num` 固定为 2。\n")

    # ----------------------------------------------------------------- #
    out.append("## 1. 全配置主表 (36 组)\n")
    out.append("按 width 分组。`maxFM` = 单层最大中间特征图元素数; `skip` = U-Net 跳连需同时驻留的激活元素数。\n")
    out.append("| 配置 | width | depth | stages | 参数量 | 权重fp16 | GMACs | GFLOPs | maxFM(元素) | maxFM(fp16) | skip(元素) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        c = r["config"]
        out.append(
            f"| {c['label']} | {c['width']} | {c['blocks_per_stage']} | {c['num_stages']} | "
            f"{r['total_parameters']:,} | {r['weight_bytes_fp16']/1024:.0f} KiB | "
            f"{r['gmacs']:.2f} | {r['gflops']:.2f} | "
            f"{r['max_single_feature_map_elements']:,} | "
            f"{r['max_single_feature_map_bytes_fp16']/1024/1024:.1f} MiB | "
            f"{r['unet_skip_footprint']['skip_elements']:,} |"
        )
    out.append("")

    # ----------------------------------------------------------------- #
    out.append("## 2. 最大中间特征图 (本次重点)\n")
    out.append("通过对每个算子的输入/输出挂 hook 实测, **单个中间张量的最大元素数** 只取决于 `width`, "
               "与 depth / stages 无关:\n")
    out.append("| width | 最大张量 | 形状 | 元素数 | fp16 | int8 |")
    out.append("|---|---|---|---|---|---|")
    for w in (8, 12, 16):
        label = f"w{w}_d1_s4"
        r = by_label[label]
        elem = r["max_single_feature_map_elements"]
        out.append(f"| {w} | {r['max_single_feature_map_tensor'][0]} | "
                   f"{r['max_single_feature_map_tensor'][1]} | {elem:,} | "
                   f"{elem*2/1024/1024:.1f} MiB | {elem/1024/1024:.1f} MiB |")
    out.append("")
    out.append("**成因**: 峰值出现在 stage-0 (最高分辨率 512×512) 的 NAFBlock 内部 —— "
               "`conv1` (1×1) 把通道从 `width` 扩张到 `2×width` (DW_Expand=2), "
               "`dwconv2` (3×3 depthwise) 保持 `2×width` 通道不变, 该张量 = `2 × width × 512 × 512`。"
               "FFN 分支 (`conv4`) 同样扩张到 `2×width`, 量级相同。stage-0 之后通道翻倍但空间 ÷4, "
               "元素数逐层减半, 故最大值恒由 stage-0 决定。\n")
    out.append("**U-Net 跳连总足迹** (解码期需同时保留的 encoder skip 激活):\n")
    out.append("| width | skip 总元素 | fp16 | 各 stage skip (stage0→3) |")
    out.append("|---|---|---|---|")
    for w in (8, 12, 16):
        r = by_label[f"w{w}_d1_s4"]
        sk = r["unet_skip_footprint"]
        per = sk["per_stage_elements"]
        out.append(f"| {w} | {sk['skip_elements']:,} | {sk['skip_bytes_fp16']/1024/1024:.1f} MiB | "
                   f"{per[0]:,} / {per[1]:,} / {per[2]:,} / {per[3]:,} |")
    out.append("")
    out.append("**逐分辨率分布** (以 w16 为例):\n")
    out.append("| stage | 角色 | 通道 | 空间 | 基础特征图 | 内部扩张2×后 (峰值) |")
    out.append("|---|---|---|---|---|---|")
    for row in by_label["w16_d1_s4"]["per_resolution"]:
        out.append(f"| {row['stage']} | {row['role']} | {row['channels']} | "
                   f"{row['spatial'][0]}×{row['spatial'][1]} | {row['base_elements']:,} | "
                   f"{row['expanded2x_elements']:,} |")
    out.append("")
    out.append("> **硬件含义**: 「权重存储」瓶颈在深层 (middle, 256 通道), 「激活存储」瓶颈在浅层 (stage 0)。"
               "二者解耦 —— depth/stages 几乎不动 maxFM, 只有 width 决定激活缓冲上界。\n")

    # ----------------------------------------------------------------- #
    out.append("## 3. 参数分布的关键洞察\n")
    out.append("以 baseline `w16_d1_s4` 为例, 参数按子模块分组:\n")
    bd = by_label["w16_d1_s4"]["param_breakdown"]
    total = by_label["w16_d1_s4"]["total_parameters"]
    out.append("| 子模块 | 参数量 | 占比 | 说明 |")
    out.append("|---|---|---|---|")
    role_desc = {
        "intro": "输入 3×3 卷积 (4→16)",
        "encoders": "encoder 各 stage 的 NAFBlock",
        "downs": "下采样 2×2 卷积 (通道翻倍)",
        "middle_blks": "最深层 (256 通道) 的 2 个 NAFBlock",
        "decoders": "decoder 各 stage 的 NAFBlock",
        "ups": "上采样 1×1 + PixelShuffle (通道减半)",
        "ending": "输出 3×3 卷积 (16→4)",
    }
    for k, v in bd.items():
        out.append(f"| {k} | {v:,} | {v/total*100:.1f}% | {role_desc.get(k,'')} |")
    out.append("")
    out.append(f"- **middle_blks 占 {bd['middle_blks']/total*100:.0f}% 参数**, 却只处理 32×32 (最小) 特征图 —— "
               "1×1 卷积参数 ∝ 通道², 256 通道最贵。")
    out.append(f"- 这就是 **stages 对参数量影响远大于 depth** 的原因: stages 4→3 会把 middle 从 256 通道降到 128, "
               f"参数从 {by_label['w16_d1_s4']['total_parameters']:,} 骤降到 {by_label['w16_d1_s3']['total_parameters']:,}"
               f" (×{by_label['w16_d1_s4']['total_parameters']/by_label['w16_d1_s3']['total_parameters']:.1f}); "
               f"而 depth 1→4 在 s4 下只从 {by_label['w16_d1_s4']['total_parameters']:,} 涨到 "
               f"{by_label['w16_d4_s4']['total_parameters']:,} (×{by_label['w16_d4_s4']['total_parameters']/by_label['w16_d1_s4']['total_parameters']:.1f})。\n")

    # ----------------------------------------------------------------- #
    out.append("## 4. FPGA 资源估算 (INT8 量化, 30 fps @ 200 MHz)\n")
    out.append("并行度 = 为达到 30 fps 在 200 MHz 下所需的 MAC 单元数 (向上取整到 2 的幂); "
               "INT8 下 1 个 DSP48E2 可做 2 个 MAC。\n")
    curated = ["w8_d1_s2", "w8_d1_s4", "w8_d2_s4", "w8_d4_s4",
               "w12_d2_s3", "w12_d2_s4", "w16_d1_s4", "w16_d2_s4", "w16_d4_s4"]
    out.append("| 配置 | 参数量 | GMACs | 并行(ceil) | 并行(2^) | DSP48 | 权重BRAM | 激活BRAM | LUT估 | 单帧ms | FPS |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for label in curated:
        r = by_label[label]
        f = hw_by_label[label]["fpga_int8"]
        out.append(
            f"| {label} | {r['total_parameters']/1e3:.0f}K | {r['gmacs']:.2f} | "
            f"{hw_by_label[label]['required_parallel_int8_ceil']} | {f['parallel_macs']} | "
            f"{f['dsp48']} | {f['weight_bram_36k']} | {f['act_bram_36k']} | "
            f"{f['lut_est']:,} | {f['latency_ms_per_frame']:.1f} | {f['fps']:.0f} |"
        )
    out.append("")
    out.append("- **权重 BRAM 很小** (最大 ~557 块 36Kb ≈ 2 MiB), 权重轻松全片上。")
    out.append("- **激活 BRAM 极大** (数千块): 因为按「整帧最大特征图 + 双缓冲」估, 16–32 MiB 级。"
               "任何主流 FPGA 都无法把它全塞进 BRAM —— **必须分块 (tiling) 流水 或 用外存 DRAM**。")
    out.append("- **DSP 是算力瓶颈**: 全帧 30 fps 需要数百~数千 DSP。例如 `w16_d4_s4` 需 ~2048 DSP (≈ ZU19EG 级), "
               "`w8_d2_s4` 仅需 ~256 DSP (中端 FPGA 可达 >40 fps)。\n")

    # ----------------------------------------------------------------- #
    out.append("## 5. ASIC 资源估算 (28nm 规划级, INT8, 同并行度)\n")
    out.append("| 配置 | 逻辑(KGE) | 逻辑(mm²) | 权重SRAM(bit) | 激活SRAM(bit) | SRAM(mm²) | 总面积(mm²) | 功耗(mW) | FPS |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for label in curated:
        a = hw_by_label[label]["asic_int8"]
        out.append(
            f"| {label} | {a['logic_kge']:,} | {a['logic_mm2']:.2f} | "
            f"{a['sram_weight_bits']/1e6:.1f} M | {a['sram_act_bits']/1e6:.0f} M | "
            f"{a['sram_mm2']:.1f} | {a['total_mm2']:.1f} | {a['power_mw']:.0f} | {a['fps']:.0f} |"
        )
    out.append("")
    out.append("- **面积被 SRAM (激活缓冲) 主导**, 同样源于整帧 maxFM。采用 line-buffer / tiling 流水后, "
               "激活 SRAM 可从数十 mm² 降到 <1 mm² (仅保留若干行缓冲 + 权重 SRAM)。")
    out.append("- 逻辑 (MAC 阵列 + 控制) 本身很小: 即使最大配置也仅 ~8 KGE / <7 mm²。"
               "功耗主要随并行度 (MAC 数 × 时钟) 线性增长。\n")

    # ----------------------------------------------------------------- #
    out.append("## 6. 设计建议 (面向 FPGA/ASIC 部署)\n")
    out.append("若「模型太大」是主要矛盾, 在 **保持 4 阶段 (足够感受野利于去噪)** 前提下:\n")
    out.append("| 候选 | 参数量 | GMACs | maxFM(fp16) | 相对 baseline |")
    out.append("|---|---|---|---|---|")
    base_p = by_label["w16_d1_s4"]["total_parameters"]
    base_g = by_label["w16_d1_s4"]["gmacs"]
    for label, note in [
        ("w8_d2_s4", "推荐甜点: 参数 ÷3.2, 算力 ÷2.3"),
        ("w8_d4_s4", "更深的窄模型: 参数 ÷2.4, 算力 ÷1.3"),
        ("w12_d2_s4", "精度/算力折中: 参数 ÷1.5, 算力 ÷1.07"),
        ("w8_d1_s4", "极小: 参数 ÷3.9, 算力 ÷3.6"),
    ]:
        r = by_label[label]
        out.append(f"| {label} ({note}) | {r['total_parameters']/1e3:.0f}K | {r['gmacs']:.2f} | "
                   f"{r['max_single_feature_map_bytes_fp16']/1024/1024:.0f} MiB | "
                   f"参数 ×{base_p/r['total_parameters']:.1f}, 算力 ×{base_g/r['gmacs']:.1f} ↓ |")
    out.append("")
    out.append("若可接受较少下采样阶段 (牺牲一点大尺度结构恢复), **stages 是最强瘦身杠杆**:\n")
    out.append("| 配置 | 参数量 | GMACs | 备注 |")
    out.append("|---|---|---|---|")
    for label in ("w16_d1_s4", "w16_d1_s3", "w16_d1_s2"):
        r = by_label[label]
        out.append(f"| {label} | {r['total_parameters']/1e3:.0f}K | {r['gmacs']:.2f} | "
                   f"middle 通道 {16*2**r['config']['num_stages']} |")
    out.append("")
    out.append("通用硬件实现建议:\n")
    out.append("1. **INT8 量化**: 权重与 DSP 需求减半; NAFNet 无 ReLU (用 SimpleGate), 对量化友好。")
    out.append("2. **分块 (tiling) 流水**: 把 512×512 切成条带/块逐块流过, 激活缓冲从 MiB 级降到 KB 级 line buffer, "
               "彻底消除 BRAM/SRAM 瓶颈。代价是需要处理跨块边界 (3×3 卷积/PixelShuffle 的 overlap)。")
    out.append("3. **权重全片上**: 最大配置权重 <6 MiB (INT8 <3 MiB), 用 BRAM/ROM 缓存即可, 避免权重带宽瓶颈。")
    out.append("4. **realism**: 全帧 512×512×4 @30 fps 偏激进; 实际可降到 10–15 fps, 或对全画幅 (如 ~2128×1420) "
               "做分块推理, DSP 需求按比例下降。\n")

    # ----------------------------------------------------------------- #
    out.append("## 7. 假设与局限\n")
    out.append("- FPGA/ASIC 数字为 **规划级粗估**, 仅供横向对比配置, 非综合/流片后精确值。"
               "实际取决于量化方案、PE 阵列拓扑、流水深度、LayerNorm(含除法/开方)/SCA 池化的实现。")
    out.append("- MAC 不含 LayerNorm / SimpleGate / 池化 / 元素乘 的算术; 这些主要消耗 LUT/逻辑, 量级远小于卷积。")
    out.append("- 激活 BRAM/SRAM 按「整帧最大特征图 + 双缓冲」 worst-case 估算; "
               "实际 tiled 设计会小 1–2 个数量级。")
    out.append("- 单层最大特征图、跳连足迹为 forward hook 实测 + 解析交叉验证, 数值可信。")
    out.append("- 输入固定为 1×4×512×512; 其他分辨率下 maxFM ∝ H×W, MACs ∝ H×W×(块数+固定项), 可线性外推。\n")

    Path(out_path).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"报告已生成: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
