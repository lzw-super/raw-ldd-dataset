#!/bin/bash
# =====================================================================
# SID Sony 去噪定性可视化一键脚本
# ---------------------------------------------------------------------
# 作用：加载指定 checkpoint，在 SID Sony 测试集上挑几张图做去噪，
#       生成 **单张** 左右对比图（左：含噪输入，右：去噪结果）。
#
# 用法：
#   bash test-op/run_qual_compare.sh              # 默认 GPU0、ratio=100、4 张图、中心裁剪 512
#   bash test-op/run_qual_compare.sh 1            # 指定 GPU 编号，例如 1
#   bash test-op/run_qual_compare.sh 0 300        # 指定 GPU 与放大倍率（100/250/300，越大越难）
#   bash test-op/run_qual_compare.sh 0 100 6      # 指定 GPU、ratio、抽样图片数
#
# 切换不同模型（unet / nafnet / mrlfn 等）：
#   ★ 推荐：靠 checkpoint 自动推断模型类型，无需指定 MODEL，只换 CP_DIR：
#     CP_DIR=experiments/sid_sony_nafnet_tiny/checkpoints/latest.pth bash test-op/run_qual_compare.sh 0 100
#   ★ 或用 MODEL 显式强制覆盖模型类型（unet / nafnet / natnet / mrlfn）：
#     MODEL=nafnet CP_DIR=experiments/sid_sony_nafnet_tiny/checkpoints/latest.pth bash test-op/run_qual_compare.sh 0 300
#   ★ NAFNet 结构高级覆盖（经 EXTRA 透传给 argparse）：
#     EXTRA="--model-width 16 --encoder-blocks 1 1 1 1 --middle-blocks 2 --decoder-blocks 1 1 1 1" \
#       bash test-op/run_qual_compare.sh 0 100
#   ★ MRLFN N=4,d=16 的 checkpoint 通常无需覆盖；裸权重可显式指定：
#     MODEL=mrlfn EXTRA="--feature-channels 16 --num-blocks 4" \
#       bash test-op/run_qual_compare.sh 0 100
#
# 修改下方“可调参数”区块里的 CP_DIR / OUT_PATH 即可切换模型权重与输出图像地址。
# =====================================================================

set -e  # 任一步失败即终止

# =====================================================================
# 可调参数（按需修改）
# =====================================================================
# 模型权重路径（支持 train_sid_sony.py 保存的可恢复字典，或官方纯 state_dict）
# 可通过环境变量覆盖，便于切换不同模型权重而无需改脚本：
#   CP_DIR=experiments/sid_sony_nafnet_tiny/checkpoints/latest.pth bash test-op/run_qual_compare.sh 0 100
CP_DIR="${CP_DIR:-experiments/sid_sony_paper_fair/checkpoints/latest.pth}"

# 模型类型（unet / nafnet / natnet / mrlfn）。
#   ★ 推荐留空：脚本会从 checkpoint 的 args 自动推断模型类型与结构，无需手动指定；
#     仅在需要强制覆盖（例如官方纯 state_dict 未记录 args 时）才填值。
MODEL="${MODEL:-}"

# 对比图输出完整路径（含文件名）。
#   ★ 建议留空（OUT_PATH=""）：此时文件名会自动带上 ratio，
#     例如 ratio=300 → OUT_DIR/denoise_comparison_ratio300.png，不会与 ratio 不符。
#   若想自定义文件名，填完整路径，如
#     OUT_PATH="experiments/sid_sony_paper_fair/qualitative/my_compare.png"
#     （注意：填了之后文件名就固定，不再随 ratio 变化）
OUT_PATH=""
# 当 OUT_PATH 留空时使用的输出目录（文件名按 ratio 自动生成）
# 可通过环境变量显式指定，便于把不同模型（unet / nafnet-tiny）的产物分目录存放，避免同名覆盖：
#   OUT_DIR=experiments/sid_sony_nafnet_tiny/qualitative bash test-op/run_qual_compare.sh 0 100
OUT_DIR="${OUT_DIR:-experiments/sid_sony_paper_fair/qualitative}"

# 产物文件名前缀（如 official_），用于在同一 OUT_DIR 下区分不同 checkpoint 的产物，避免同名覆盖：
#   OUT_PREFIX=official_ bash test-op/run_qual_compare.sh 0 100
OUT_PREFIX="${OUT_PREFIX:-}"

# 推理设备（在脚本内受 $1 GPU 参数控制，一般无需在此改）
DEVICE="cuda:0"

# 附加可视化参数（裁剪大小/偏移、指定下标、--full 整图等）
#   例：EXTRA="--crop-size 384 --crop-dy -100 --crop-dx 120"
#       EXTRA="--indices 0 10 20 30"
#       EXTRA="--full"
EXTRA=${EXTRA:-}
# =====================================================================

# ---- 命令行位置参数（仍保留，便于快速覆盖）----
GPU=${1:-0}                 # 第 1 个参数：GPU 编号，默认 0
RATIO=${2:-100}             # 第 2 个参数：放大倍率 100/250/300，默认 100
NUM=${3:-6}                 # 第 3 个参数：抽样图片数，默认 4

# ---- 进入仓库根目录（脚本位于 raw_image_denoising/test-op/ 下）----
# 重要：SIDEvalDataset 使用相对路径 ./resources、./infos，必须在仓库根目录运行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ---- 运行环境：复用已装齐依赖的 LED-ICCV23 conda 环境 ----
PY=/home/zhengwu/anaconda3/envs/LED-ICCV23/bin/python

# 计算实际生效的输出路径：OUT_PATH 非空就用它，否则按 OUT_DIR + 前缀 + ratio 自动命名
if [ -n "$OUT_PATH" ]; then
    EFFECTIVE_OUT="$OUT_PATH"
else
    EFFECTIVE_OUT="${OUT_DIR}/${OUT_PREFIX}denoise_comparison_ratio${RATIO}.png"
fi

echo "==== 运行目录:      $(pwd) ===="
echo "==== Python:        $PY ===="
echo "==== GPU:           $GPU ===="
echo "==== CP_DIR:        $CP_DIR ===="
echo "==== MODEL:         ${MODEL:-（自动推断）} ===="
echo "==== RATIO:         $RATIO ===="
echo "==== NUM:           $NUM ===="
echo "==== EXTRA:         $EXTRA ===="
echo "==== OUT_DIR:       $OUT_DIR ===="
echo "==== OUT_PREFIX:    ${OUT_PREFIX:-（无）} ===="
echo "==== 实际输出路径:   $EFFECTIVE_OUT ===="

export CUDA_VISIBLE_DEVICES=$GPU

# 仅当 MODEL 非空时透传 --model；留空则交给 checkpoint 自动推断（推荐）。
# NAFNet 结构等高级覆盖仍可经 EXTRA 环境变量传入（如 --model-width）。
MODEL_ARGS=()
[ -n "$MODEL" ] && MODEL_ARGS+=(--model "$MODEL")

$PY test-op/qual_denoise_compare.py \
    --cp-dir "$CP_DIR" \
    --out-path "$OUT_PATH" \
    --out-dir "$OUT_DIR" \
    --out-prefix "$OUT_PREFIX" \
    --device "$DEVICE" \
    --ratio "$RATIO" \
    --num "$NUM" \
    "${MODEL_ARGS[@]}" \
    $EXTRA
