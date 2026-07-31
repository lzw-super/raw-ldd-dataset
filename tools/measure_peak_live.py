#!/usr/bin/env python3
"""实测 NAFNet 前向过程中「需要同时保存的最大特征图」。

对每个叶子算子 (Conv2d / AdaptiveAvgPool2d / Linear 等) 挂 forward hook, 记录其
输入/输出元素数; 再按控制流判定该算子执行瞬间「还有哪些 skip 没被消费」,
求出每一时刻的存活激活 = 当前算子(in+out) + NAFBlock 局部残差
+ 不与算子输入重叠的存活 skip + 输入长残差, 取全局最大。

口径是整帧、逐算子、卷积 input/output ping-pong；允许元素级算子覆盖已无后续
用途的输入。它是硬件缓冲 liveness 模型，不是 PyTorch allocator 的内存统计。

skip 的存活区间 (按分辨率层 r, r=0 最浅):
  - encoders[r] 内部算子: skip[0..r-1] 已产生 (skip[r] 还没产生)
  - downs[r]:               skip[0..r] 存活
  - middle_blks:            skip[0..S-1] 全存活
  - ups[j]  (产生 res r=S-1-j): skip[0..r] 存活 (skip[r] 在 up 之后才被 + 消费)
  - decoders[j] (res r=S-1-j):   skip[0..r-1] 存活 (skip[r] 已消费)
  - intro / ending: 无 skip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.natnet_arch import NAFNet  # noqa: E402
from tools.sweep_nafnet_hardware import full_frame_live_activation_peak  # noqa: E402


def measure(width: int, blocks: int, stages: int, middle: int,
            img_channel: int, input_shape, device) -> dict:
    model = NAFNet(img_channel=img_channel, width=width,
                   enc_blk_nums=tuple([blocks] * stages),
                   middle_blk_num=middle,
                   dec_blk_nums=tuple([blocks] * stages)).to(device)
    names = {id(m): n for n, m in model.named_modules()}

    S = stages
    # 1) 实测各 skip 大小 (encoder 阶段输出) + 输入残差
    skip_numel = [0] * S
    input_residual_numel = [0]

    def enc_hook(r):
        def _h(_m, _inp, out):
            skip_numel[r] = int(out.numel())
        return _h

    for r in range(S):
        getattr(model, "encoders")[r].register_forward_hook(enc_hook(r))
    model.intro.register_forward_hook(
        lambda m, inp, _out: input_residual_numel.__setitem__(0, int(inp[0].numel())))

    # 2) 实测每个叶子算子的 (in_numel, out_numel) 与所在 section
    ops: list[dict] = []

    def leaf_hook(m, inp, out):
        name = names.get(id(m), "")
        first = name.split(".")[0]
        in_n = int(inp[0].numel()) if inp and torch.is_tensor(inp[0]) else 0
        # 取输出元素数 (PixelShuffle 等也算, 但量级小)
        if torch.is_tensor(out):
            out_n = int(out.numel())
        elif isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]):
            out_n = int(out[0].numel())
        else:
            out_n = 0
        ops.append({"name": name, "section": first, "in": in_n, "out": out_n})

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                          nn.AdaptiveAvgPool2d, nn.PixelShuffle)):
            m.register_forward_hook(leaf_hook)

    # 跑一次前向
    was = model.training
    model.eval()
    dummy = torch.zeros(input_shape, device=device)
    with torch.inference_mode():
        model(dummy)
    model.train(was)

    # 3) 按 section 名解析分辨率 r 与存活 skip 集合
    def live_skip_set(section: str, rest: str) -> set[int]:
        # rest = name 去掉首段后的剩余, 取第一个数字作为 index
        idx = -1
        for tok in rest.split("."):
            if tok.isdigit():
                idx = int(tok); break
        if section == "intro" or section == "ending":
            return set()
        if section == "encoders":
            r = idx
            return set(range(r))                 # skip[0..r-1]
        if section == "downs":
            r = idx
            return set(range(r + 1))             # skip[0..r]
        if section == "middle_blks":
            return set(range(S))                 # 全部
        if section in ("ups", "decoders"):
            j = idx
            r = (S - 1) - j
            if section == "ups":
                return set(range(r + 1))         # skip[0..r]
            return set(range(r))                 # decoders: skip[0..r-1]
        return set()

    resid = input_residual_numel[0]
    rows = []
    for op in ops:
        rest = op["name"][len(op["section"]) + 1:] if op["name"] else ""
        live = live_skip_set(op["section"], rest)
        live_skip_sum = sum(skip_numel[i] for i in live)
        # downs[r] 的输入本身就是 skip[r]；同一物理 buffer 不应作为算子输入
        # 和 skip 重复计数。
        aliased_skip = 0
        if op["section"] == "downs":
            idx = next((int(tok) for tok in rest.split(".") if tok.isdigit()), -1)
            if idx in live:
                aliased_skip = skip_numel[idx]

        # NAFBlock 的输入必须跨越主分支保存到残差加法。进入 FFN 分支后，
        # 同尺寸的 y 承担同一角色，因此每个块内叶子算子都额外保留一个 B。
        block_residual = 0
        if op["section"] in ("encoders", "middle_blks", "decoders"):
            tokens = op["name"].split(".")
            is_block_leaf = any(part in tokens for part in
                                ("conv1", "dwconv2", "conv3", "conv4", "conv5", "sca"))
            if is_block_leaf:
                if op["section"] == "encoders":
                    stage = int(tokens[1])
                elif op["section"] == "decoders":
                    stage = S - 1 - int(tokens[1])
                else:
                    stage = S
                block_residual = (
                    width * (2 ** stage)
                    * (input_shape[2] // (2 ** stage))
                    * (input_shape[3] // (2 ** stage))
                )
        transient = op["in"] + op["out"]          # 当前算子双缓冲 (读入+写出)
        extra_skip_sum = live_skip_sum - aliased_skip
        total = transient + block_residual + extra_skip_sum + resid
        rows.append({**op, "live_skip_idx": sorted(live),
                     "live_skip_sum": live_skip_sum,
                     "aliased_skip": aliased_skip,
                     "extra_skip_sum": extra_skip_sum,
                     "block_residual": block_residual,
                     "transient": transient,
                     "residual": resid, "total": total})

    # keep-all 模式 (所有 skip 全程存活, naive)
    all_skip = sum(skip_numel)
    for row in rows:
        row["total_keepall"] = (row["transient"] + row["block_residual"]
                                + all_skip - row["aliased_skip"] + resid)

    peak = max(rows, key=lambda r: r["total"])
    peak_ka = max(rows, key=lambda r: r["total_keepall"])
    return {
        "config": {"width": width, "blocks": blocks, "stages": stages, "middle": middle},
        "skip_numel": skip_numel, "all_skip": all_skip, "input_residual": resid,
        "rows": rows, "peak": peak, "peak_keepall": peak_ka,
    }


def fmt(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--middle", type=int, default=2)
    ap.add_argument("--img-channel", type=int, default=4)
    ap.add_argument("--input-shape", type=int, nargs=4, default=[1, 4, 512, 512])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    r = measure(args.width, args.blocks, args.stages, args.middle,
                args.img_channel, args.input_shape, torch.device(args.device))
    S = args.stages
    A = args.width * args.input_shape[2] * args.input_shape[3]
    print(f"\n===== 配置 w{args.width}_d{args.blocks}_s{S}  (A=width×H×W={fmt(A)}) =====")
    print("\n[1] 实测 skip 特征图 (encoder 阶段输出, 下采样前):")
    print(f"    {'r':>2} {'通道':>6} {'空间':>10} {'元素数':>12} {'fp16':>9}")
    for i, n in enumerate(r["skip_numel"]):
        ch = args.width * (2 ** i)
        sp = f"{args.input_shape[2]//(2**i)}×{args.input_shape[3]//(2**i)}"
        print(f"    {i:>2} {ch:>6} {sp:>10} {fmt(n):>12} {n*2/1024/1024:>7.1f}MiB")
    print(f"    全部 skip 之和 = {fmt(r['all_skip'])} ({r['all_skip']*2/1024/1024:.1f} MiB fp16)")
    print(f"    输入残差 padded_input = {fmt(r['input_residual'])} (全程驻留)")

    print("\n[2] 存活激活最大的前 12 个算子 (物理 buffer 去重):")
    print(f"    {'算子':<34}{'in+out':>12}{'块残差':>12}{'额外skip':>12}{'长残差':>10}{'瞬时总计':>14}")
    top = sorted(r["rows"], key=lambda x: -x["total"])[:12]
    for row in top:
        sk = ",".join(map(str, row["live_skip_idx"])) if row["live_skip_idx"] else "—"
        print(f"    {row['name'][:34]:<34}{fmt(row['transient']):>12}"
              f"{fmt(row['block_residual']):>12}{fmt(row['extra_skip_sum']):>12}"
              f"{fmt(row['residual']):>10}{fmt(row['total']):>14}  skip={sk}")

    p, pk = r["peak"], r["peak_keepall"]
    print("\n[3] 峰值汇总:")
    print(f"    立即释放调度峰值 = {fmt(p['total'])} 元素 = {p['total']*2/1024/1024:.1f} MiB(fp16)")
    print(f"      └ 出现在算子 {p['name']}  | 双缓冲={fmt(p['transient'])} "
          f"块残差={fmt(p['block_residual'])} 额外skip={fmt(p['extra_skip_sum'])} "
          f"长残差={fmt(p['residual'])}")
    print(f"      └ 当时存活 skip 层 = {p['live_skip_idx'] or '无'}")
    print(f"    keep-all(naive) 峰值 = {fmt(pk['total_keepall'])} 元素 = {pk['total_keepall']*2/1024/1024:.1f} MiB(fp16)")
    print(f"    旧报告里的 4A (仅扩张张量 in+out) = {fmt(4*A)} 元素 = {4*A*2/1024/1024:.1f} MiB(fp16)")
    print(f"    stage-0 正确峰值 5A+长残差       = {fmt(5*A + r['input_residual'])} 元素")

    analytic = full_frame_live_activation_peak(
        args.width, args.img_channel, args.input_shape[2], args.input_shape[3],
        args.stages, args.blocks, args.middle,
    )
    print(f"    解析 liveness 交叉验证           = {fmt(analytic['peak_elements'])} 元素 "
          f"({analytic['peak_location']})")
    if p["total"] != analytic["peak_elements"]:
        raise RuntimeError("hook 与解析 liveness 峰值不一致")


if __name__ == "__main__":
    main()
