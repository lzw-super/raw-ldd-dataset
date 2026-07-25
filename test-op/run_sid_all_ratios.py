# -*- coding: utf-8 -*-
"""
SID (Sony A7S2) 全量评估脚本 —— 复现 CVPR'25《Noise Modeling in One Hour》

说明：
- 在 SID Sony 数据集上分别评估 ratio = 100 / 250 / 300 三档放大倍率。
- 复用官方 test_denoise_sideld.py 的模型构建、数据加载与度量逻辑，
  仅在单进程内循环三档 ratio，并汇总打印 PSNR / SSIM。
- 对每个 ratio 额外保存少量去噪结果 PNG（plot_res），便于目视检查。

运行前置条件：
  1) checkpoints/sonya7s2.pth          （已从 Google Drive 下载）
  2) resources/SonyA7S2/*.npy|pkl       （dark shadings，已下载）
  3) infos/SID_evaltest.info            （由 get_dataset_infos.py 生成）
  4) 数据集位于 /home/shared_files/dataset/SID/Sony （long/ short/）
"""
import os
os.environ["OPENMP_NUM_THREADS"] = "16"

# 将仓库根目录加入搜索路径，使 utils/ models/ datasets/ 可被导入
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import random
import numpy as np
import torch

from utils.utils import *
from utils.imgproc import *
from datasets.real_dataset import SIDEvalDataset
from models.ELD_models import UNetSeeInDark
from test_denoise_sideld import valid_one_ep  # 复用官方评估主循环


def build_model(device, cp_dir):
    """加载预训练 U-Net 并切换到 eval 模式"""
    model = UNetSeeInDark().to(device)
    model.load_state_dict(torch.load(cp_dir, map_location="cpu"), strict=True)
    model.eval()
    return model


def build_loader(eval_ratio):
    """按指定 ratio 构建 SID 评估 DataLoader"""
    valid_set = SIDEvalDataset(clip_low=False, clip_high=True, eval_ratio=eval_ratio)
    return torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cp_dir", type=str, default="./checkpoints/sonya7s2.pth")
    parser.add_argument("--ratios", type=int, nargs="+", default=[100, 250, 300])
    parser.add_argument("--plot_res", action="store_true", help="是否保存去噪可视化 PNG")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # 固定随机种子，保证可复现
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # 模型只加载一次，三档 ratio 共用
    model = build_model(args.device, args.cp_dir)

    print(f"\n{'='*60}\nSID Sony 评估（checkpoint={args.cp_dir}, device={args.device}）\n{'='*60}")
    results = {}
    for ratio in args.ratios:
        args.task = "sonya7s2"
        args.testset_type = "sid"
        args.eval_ratio = ratio
        valid_loader = build_loader(ratio)

        # 每个 ratio 仅保存第一档(ratio==100)的前几张可视化，避免占用过多磁盘
        do_plot = args.plot_res and (ratio == args.ratios[0])
        psnr, ssim = valid_one_ep(model, valid_loader, args, plot_res=do_plot)
        results[ratio] = (psnr, ssim)
        print(f"[ratio={ratio:>3d}]  PSNR={psnr:.4f} dB   SSIM={ssim:.4f}")

    print(f"\n{'='*60}\n汇总结果\n{'='*60}")
    print(f"{'ratio':>8} | {'PSNR(dB)':>10} | {'SSIM':>8}")
    print("-" * 34)
    for ratio, (psnr, ssim) in results.items():
        print(f"{ratio:>8} | {psnr:>10.4f} | {ssim:>8.4f}")


if __name__ == "__main__":
    main()
