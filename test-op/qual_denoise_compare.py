# -*- coding: utf-8 -*-
"""
SID Sony 去噪定性可视化脚本
============================================================
用途：
    加载训练好的 UNetSeeInDark 模型（experiments/sid_sony_paper_fair/checkpoints/latest.pth），
    在 SID Sony 测试集上挑若干张图做去噪，生成 **单张** 对比图：
        左列 = 含噪输入（Noisy input，即短曝光×放大倍率后的 RAW）
        右列 = 去噪结果（Denoised，模型输出并经照度校正）
    每行一张测试图，多行纵向堆叠为一张 PNG，便于直观看去噪性能。

说明：
    - 复用仓库的 SIDEvalDataset 数据加载、UNetSeeInDark 模型、ELDIlluminanceCorrect
      照度校正与 rggb_to_srgb 显示转换，与官方 test_denoise_sideld.py 的评估流程一致。
    - 默认对中心区域做固定大小裁剪（--crop-size 512）展示，因为整图缩放会平均掉噪声、
      看不出去噪差异；裁剪是 SID/ELD 论文里定性对比的标准做法。
      如需查看整图，加 --full。
    - 使用 torch.inference_mode() 推理：SID packed RAW 为 1424x2128，保留 autograd 图
      会撑爆显存，inference_mode 数值等价且显著降低峰值显存。
    - checkpoint 由 train_sid_sony.py 保存为可恢复格式（字典里 "model" 键存放权重）；
      同时兼容官方发布的纯 state_dict 权重。

产物：
    experiments/sid_sony_paper_fair/qualitative/denoise_comparison_ratio{R}.png
"""
import os
# 限制 OpenMP 线程数，避免在 CPU 预处理时与 GPU 推理争抢资源
os.environ.setdefault("OPENMP_NUM_THREADS", "16")

import sys
import random
import argparse

import numpy as np
import torch
import matplotlib
# 无界面环境强制使用 Agg 后端，避免 tk/Qt 依赖报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tqdm import tqdm


def _setup_cjk_font() -> None:
    """注册一个支持中文的字体，避免标题/标签里的中文渲染成方框（缺字形）。

    优先使用系统已安装的中文字体；找不到则保持默认字体（此时中文会缺字形，
    但英文正常）。同时关闭 unicode 减号的字形替换。
    """
    # 候选字体文件（按优先级），均为常见中文/CJK 字体
    candidates = [
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Droid 回退字体（纯 ttf，覆盖 CJK）
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",     # Noto Sans CJK
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                # 让 matplotlib 优先用该中文字体，其次回退到 DejaVu Sans（拉丁字符）
                plt.rcParams["font.family"] = [name, "DejaVu Sans"]
                break
            except Exception as e:  # 注册失败则尝试下一个候选
                print(f"[warn] 注册字体 {path} 失败: {e}")
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 把仓库根目录加入搜索路径，使 utils/ models/ datasets/ 可被导入
# （本脚本位于 raw_image_denoising/test-op/ 下）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import rggb_to_srgb, ELDIlluminanceCorrect, PMN_metric  # noqa: E402
from utils.model_factory import build_denoiser_from_checkpoint  # noqa: E402
from datasets.real_dataset import SIDEvalDataset  # noqa: E402


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------
def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """从 checkpoint 还原去噪模型并切换到实际部署使用的 eval 图。

    模型类型与结构默认从 checkpoint 的 ``args`` 自动推断；命令行
    ``--model / --model-width / --encoder-blocks / --middle-blocks /
    --decoder-blocks`` 可强制覆盖。解析结果写入 ``args.resolved_meta``，
    供日志与图内标注使用。实际构建逻辑统一收敛到 utils.model_factory。
    """
    model, meta = build_denoiser_from_checkpoint(
        args.cp_dir,
        device=args.device,
        model=args.model,
        model_width=args.model_width,
        encoder_blocks=args.encoder_blocks,
        middle_blocks=args.middle_blocks,
        decoder_blocks=args.decoder_blocks,
        feature_channels=args.feature_channels,
        num_blocks=args.num_blocks,
    )
    args.resolved_meta = meta
    return model


# ---------------------------------------------------------------------------
# RAW -> sRGB 显示转换
# ---------------------------------------------------------------------------
def raw_to_srgb_uint8(img_tensor: torch.Tensor, wb: np.ndarray, ccm: np.ndarray,
                      gamma: float = 3.0) -> np.ndarray:
    """把单张 packed RAW [1,4,H,W]（RGGB）转成 sRGB uint8 [H,W,3] 用于可视化。

    与 test_denoise_sideld.py 的可视化保持一致：gamma=3，format='rggb'。
    """
    x = img_tensor.detach().cpu()
    if x.dim() == 4:
        x = x[0]  # [4,H,W]
    x = x.clamp(0.0, 1.0).permute(1, 2, 0).numpy()  # [H,W,4]
    return rggb_to_srgb(x, wb=wb, ccm=ccm, gamma=gamma, format="rggb", uint8=True)


# ---------------------------------------------------------------------------
# 单张图去噪并取出（含噪 / 去噪 / GT）显示画面
# ---------------------------------------------------------------------------
@torch.inference_mode()
def denoise_one(model, dataset, idx, device, crop_size, full, crop_dy, crop_dx, gamma):
    """对数据集中第 idx 张图做去噪，返回单行对比所需的三个 sRGB 画面。

    返回 dict: {noisy, dn, gt, psnr, name, iso}
    三个画面均为 sRGB uint8；若 full=False 则为同一坐标的中心裁剪区域（严格对齐）。
    """
    data = dataset[idx]
    imgs_lr = data["lr"].to(device)   # [1,4,H,W] 含噪输入
    imgs_hr = data["hr"].to(device)   # [1,4,H,W] 干净 GT（用于照度校正、PSNR 与参考显示）

    # 模型前向 + ELD 照度校正（补偿黑电平放大误差），再 clamp 到合法范围
    imgs_dn = model(imgs_lr)
    imgs_dn = ELDIlluminanceCorrect()(imgs_dn, imgs_hr)
    imgs_dn = torch.clamp(imgs_dn, 0.0, 1.0)

    # 度量去噪结果与 GT 的 PSNR/SSIM（仅作图注展示）
    metric = PMN_metric(imgs_dn.cpu(), imgs_hr.cpu())
    psnr = float(metric["psnr"])

    # 白平衡 / 色彩矩阵，用于 RAW->sRGB
    wb = data["wb"].numpy()
    ccm = data["ccm"].numpy()

    # 三幅全图 sRGB：含噪输入 / 去噪结果 / 干净 GT，均使用同一组 wb、ccm、gamma
    noisy_full = raw_to_srgb_uint8(imgs_lr, wb, ccm, gamma=gamma)
    dn_full = raw_to_srgb_uint8(imgs_dn, wb, ccm, gamma=gamma)
    gt_full = raw_to_srgb_uint8(imgs_hr, wb, ccm, gamma=gamma)

    # 三者尺寸一致，用同一裁剪参数保证区域严格对齐
    crop_args = (crop_size, full, crop_dy, crop_dx)
    return {
        "noisy": _crop_or_full(noisy_full, *crop_args),
        "dn": _crop_or_full(dn_full, *crop_args),
        "gt": _crop_or_full(gt_full, *crop_args),
        "psnr": psnr,
        "name": str(data["name"]),
        "iso": int(data["iso"]),
    }


def _crop_or_full(rgb: np.ndarray, crop_size: int, full: bool, dy: int, dx: int):
    """对 sRGB 画面做中心裁剪（或整图返回）。含噪/去噪/GT 使用同一区域以严格对齐。"""
    if full:
        return rgb
    h, w = rgb.shape[:2]
    cs = min(crop_size, h, w)
    y0 = int((h - cs) / 2) + dy
    x0 = int((w - cs) / 2) + dx
    y0 = max(0, min(y0, h - cs))
    x0 = max(0, min(x0, w - cs))
    return rgb[y0:y0 + cs, x0:x0 + cs]


# ---------------------------------------------------------------------------
# 拼图
# ---------------------------------------------------------------------------
def make_comparison_figure(rows, ratio, out_path, dpi, full, model_label=""):
    """把多行 (noisy, dn, gt) 拼成单张三列对比 PNG 并保存。

    每行：含噪输入 | 去噪结果 | 干净参考 GT；多行纵向堆叠。
    rows: list of dict {noisy, dn, gt, psnr, name, iso}
    """
    n = len(rows)
    panel_h, panel_w = rows[0]["noisy"].shape[:2]

    # 让每个面板在最终图里尽量接近 1:1 像素显示，保证裁剪细节清晰
    panel_in_w = panel_w / dpi
    panel_in_h = panel_h / dpi
    fig_w = 3 * panel_in_w + 0.7   # 三列 + 左侧行标签留白
    fig_h = n * panel_in_h + 0.5   # n 行 + 顶部列标题留白
    fig, axes = plt.subplots(n, 3, figsize=(fig_w, fig_h), dpi=dpi)
    if n == 1:
        axes = axes[np.newaxis, :]

    for r, row in enumerate(rows):
        ax_l, ax_m, ax_r = axes[r, 0], axes[r, 1], axes[r, 2]
        ax_l.imshow(row["noisy"], interpolation="nearest")
        ax_m.imshow(row["dn"], interpolation="nearest")
        ax_r.imshow(row["gt"], interpolation="nearest")
        for ax in (ax_l, ax_m, ax_r):
            ax.set_axis_off()
        # 去噪面板左下角标注 PSNR（去噪 vs GT），便于参考
        ax_m.text(
            0.015, 0.02, f"PSNR {row['psnr']:.2f} dB",
            transform=ax_m.transAxes, color="yellow", fontsize=11,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55),
        )
        # 左侧标注图片名 / ISO / 放大倍率 / 模型名（便于横向对比不同模型产物）
        row_label = f"{row['name']}\nISO {row['iso']} | x{ratio}"
        if model_label:
            row_label += f" | {model_label}"
        ax_l.text(
            -0.02, 0.5, row_label,
            transform=ax_l.transAxes, color="black", fontsize=10,
            ha="right", va="center", rotation=90,
        )

    # 顶部列标题
    axes[0, 0].set_title("含噪输入  Noisy input", fontsize=13, pad=8)
    axes[0, 1].set_title("去噪结果  Denoised", fontsize=13, pad=8)
    axes[0, 2].set_title("干净参考  GT", fontsize=13, pad=8)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.01,
                        wspace=0.03, hspace=0.05)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SID Sony 去噪定性可视化（单张左右对比图）")
    parser.add_argument("--cp-dir", default="experiments/sid_sony_paper_fair/checkpoints/latest.pth",
                        help="checkpoint 路径")
    parser.add_argument("--model", choices=["unet", "nafnet", "natnet", "mrlfn"], default=None,
                        help="强制模型类型；留空则按 checkpoint 的 args 自动推断（推荐）")
    parser.add_argument("--model-width", type=int, default=None,
                        help="特征通道数；留空则用 checkpoint 记录值，再回退到 32")
    parser.add_argument("--encoder-blocks", type=int, nargs="+", default=None,
                        help="NAFNet 各编码阶段 block 数；留空则用 checkpoint 记录值")
    parser.add_argument("--middle-blocks", type=int, default=None,
                        help="NAFNet 中间 block 数；留空则用 checkpoint 记录值")
    parser.add_argument("--decoder-blocks", type=int, nargs="+", default=None,
                        help="NAFNet 各解码阶段 block 数；留空则用 checkpoint 记录值")
    parser.add_argument("--feature-channels", type=int, default=None,
                        help="MRLFN 特征深度 d；留空则用 checkpoint 记录值")
    parser.add_argument("--num-blocks", type=int, default=None,
                        help="MRLFN mRLFB 数量 N；留空则用 checkpoint 记录值")
    parser.add_argument("--device", default="cuda:0", help="推理设备，如 cuda:0 / cpu")
    parser.add_argument("--ratio", type=int, default=100, choices=[100, 250, 300],
                        help="SID 评估放大倍率（越大噪声越强、去噪难度越高）")
    parser.add_argument("--num", type=int, default=4, help="抽样展示的图片数量")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="直接指定测试集下标（优先于 --num 的均匀抽样）")
    parser.add_argument("--crop-size", type=int, default=512, help="中心裁剪边长（像素）")
    parser.add_argument("--full", action="store_true", help="展示整图而非中心裁剪")
    parser.add_argument("--crop-dy", type=int, default=0, help="裁剪中心纵向偏移（便于对准感兴趣区域）")
    parser.add_argument("--crop-dx", type=int, default=0, help="裁剪中心横向偏移")
    parser.add_argument("--gamma", type=float, default=3.0, help="RAW->sRGB 显示 gamma")
    parser.add_argument("--dpi", type=int, default=120, help="输出 PNG 的 DPI")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--out-dir", default="experiments/sid_sony_paper_fair/qualitative",
                        help="输出目录（当 --out-path 为空时，在其下按 ratio 自动命名）")
    parser.add_argument("--out-prefix", default="",
                        help="产物文件名前缀，用于区分不同 checkpoint（如 official_）；留空则无前缀")
    parser.add_argument("--out-path", default="",
                        help="对比图完整保存路径（含文件名）。留空则用 --out-dir/<prefix>denoise_comparison_ratio{R}.png")
    args = parser.parse_args()

    # 注册中文字体，确保图中的中文标题/标签正常显示
    _setup_cjk_font()

    # 固定随机种子，保证可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    print(f"[info] device = {device}, ratio = {args.ratio}, crop = "
          f"{'full' if args.full else args.crop_size}")

    # 1) 加载模型；MRLFN 会优先读取 model_deploy，旧训练权重则先融合
    model = build_model(args)
    meta = args.resolved_meta
    if meta["model"] == "mrlfn":
        print(
            f"[info] resolved model = mrlfn N={meta['num_blocks']} d={meta['feature_channels']} "
            f"graph={meta['graph_state']} weights={meta['weight_source']}"
        )
    else:
        print(
            f"[info] resolved model = {meta['model']} width={meta['model_width']} "
            f"enc={meta['encoder_blocks']} mid={meta['middle_blocks']} "
            f"dec={meta['decoder_blocks']} graph={meta['graph_state']}"
        )

    # 2) 构建评估集（构造时会缓存该 ratio 下所有 RAW，与官方评估一致）
    #    注意：SIDEvalDataset 内部使用相对路径 ./resources、./infos，必须在仓库根目录运行
    valid_set = SIDEvalDataset(clip_low=False, clip_high=True, eval_ratio=args.ratio)

    # 3) 选图：优先用 --indices，否则在该 ratio 评估集上均匀抽样 num 张
    total = len(valid_set)
    if args.indices:
        idxs = [i for i in args.indices if 0 <= i < total]
    else:
        k = min(args.num, total)
        idxs = np.linspace(0, total - 1, k).round().astype(int).tolist()
    print(f"[info] 共 {total} 张候选，选用下标: {idxs}")

    # 4) 逐张去噪 + 收集显示画面（含噪 / 去噪 / GT）
    rows = []
    for i, idx in enumerate(idxs):
        row = denoise_one(
            model, valid_set, idx, device,
            crop_size=args.crop_size, full=args.full,
            crop_dy=args.crop_dy, crop_dx=args.crop_dx, gamma=args.gamma,
        )
        rows.append(row)
        print(f"[{i + 1}/{len(idxs)}] idx={idx} {row['name']} ISO{row['iso']}  PSNR={row['psnr']:.2f} dB")

    # 5) 拼成单张左右对比图并保存
    #    优先使用 --out-path 指定的完整路径；否则按 --out-dir + ratio 自动命名
    if args.out_path:
        out_path = args.out_path
    else:
        out_path = os.path.join(
            args.out_dir, f"{args.out_prefix}denoise_comparison_ratio{args.ratio}.png"
        )
    model_label = f"{meta['model']} ({meta['graph_state']})"
    make_comparison_figure(rows, args.ratio, out_path, args.dpi, args.full, model_label=model_label)
    print(f"\n[done] 对比图已保存: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
