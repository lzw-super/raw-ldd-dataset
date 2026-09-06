#!/bin/bash
# =====================================================================
# SID (Sony A7S2) 复现评估一键脚本
# 论文：Noise Modeling in One Hour (CVPR'25, Sony Research)
# 仓库：https://github.com/SonyResearch/raw_image_denoising
# ---------------------------------------------------------------------
# 用法：
#   bash test-op/run_sid_eval.sh            # 默认用 GPU0，评估 ratio 100/250/300
#   bash test-op/run_sid_eval.sh 1          # 指定 GPU 编号，例如 1
#   bash test-op/run_sid_eval.sh 0 250      # 指定 GPU 与单个 ratio
#
# 产物：
#   - 终端打印各 ratio 的 PSNR / SSIM 汇总
#   - test_res/sonya7s2/sid/ 下保存前若干张去噪可视化 PNG（仅 ratio100）
# =====================================================================

set -e  # 任一步失败即终止

# ---- 进入仓库根目录（脚本位于 raw_image_denoising/test-op/ 下）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ---- 运行环境：复用已装齐依赖的 LED-ICCV23 conda 环境 ----
PY=/home/zhengwu/anaconda3/envs/LED-ICCV23/bin/python
GPU=${1:-0}                 # 第 1 个参数：GPU 编号，默认 0
RATIOS=${2:-100 250 300}    # 第 2 个参数：ratio 列表，默认三档全跑

echo "==== 运行目录: $(pwd) ===="
echo "==== Python:   $PY ===="
echo "==== GPU:      $GPU ===="
echo "==== RATIOS:   $RATIOS ===="

export CUDA_VISIBLE_DEVICES=$GPU
$PY test-op/run_sid_all_ratios.py --ratios $RATIOS --plot_res
