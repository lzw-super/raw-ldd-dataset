# 小波初步去噪 + Rep-NCB 精修实验

配置：`configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1_repncb.yaml`。
模型：`models/learning_dwt_repncb.py`，注册名 `learning_dwt_repncb`。

第一阶段完整沿用第三方案：sym4，三层递归分解 LL；LL3 输入 CNN，CNN 输出与原 LL3 拼接后经 1×1 卷积恢复 LL；高频继续学习阈值。IWT 得到初步去噪图 I-。第二阶段如下（实际张量使用 NCHW）：

```text
I- [N,4,H,W]
 -> PixelUnshuffle(2) [N,16,H/2,W/2]
 -> Conv3×3(16,16)
 -> Rep-NCB(16) × 4
 -> Conv3×3(16,16)
 -> concat with PixelUnshuffle(2)(original I) [N,32,H/2,W/2]
 -> Conv1×1(32,16)
 -> PixelShuffle(2) [N,4,H,W]
```

S2D 直接作用于 packed RAW，不先还原 Bayer mosaic。奇数尺寸在右/下边缘 replicate padding 后处理，输出裁回原尺寸。stem/head/fusion 不额外添加激活；各 Rep-NCB 使用逐通道 PReLU。末端融合是图中指定的通道拼接，不额外添加全局残差或输出裁剪。

## Rep-NCB

参考本地 `ref-pdf` 中 ECB 论文与其[官方代码](https://github.com/xindongzhang/ECBSR/blob/main/models/ecb.py)。五个线性分支求和，之后接 PReLU：

1. 普通 3×3 卷积，16→16。
2. 1×1 扩张 16→32，再 3×3 压缩 32→16。
3. 1×1 卷积 16→16，再逐通道高斯平滑。
4. 1×1 卷积 16→16，再逐通道水平平滑。
5. 1×1 卷积 16→16，再逐通道垂直平滑。

固定核严格采用用户给出的数值：

```text
G  = [[1,2,1], [2,4,2], [1,2,1]] / 16
Gh = [[0,0,0], [1,2,1], [0,0,0]] / 4
Gv = [[0,1,0], [0,2,0], [0,1,0]] / 4
```

核注册为 buffer，不学习其矩阵元素；每个固定核分支具有可学习的前置 1×1 权重/偏置、逐输出通道缩放和末端偏置。没有额外 identity 分支。固定平滑核提供平滑方向先验；其余分支仍可学习细节恢复，因此不强制最终卷积为低通。

对固定核分支，融合权重为 `W[o,i,h,w] = scale[o] * G[h,w] * W1x1[o,i]`，融合偏置为 `b[o] = scale[o] * sum(G) * b1x1[o] + b_spatial[o]`。这三个核的和都是 1，不能忽略前置偏置项。实现先对输入补零再做 1×1 投影，使投影后的边界为前置偏置，确保边缘也可以精确融合。

`model.deploy()` 复制模型并融合四个 Rep-NCB，每个变为单个 3×3 卷积 + PReLU，保留训练模型。DWT、IWT、阈值网络及其他非线性运算仍然保留，不能把整个模型融合为一个卷积。训练脚本自动保存训练态和部署态参数；验证及评测自动使用部署图。

## 手动训练与评测

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1_repncb.yaml
```

该配置从头联合训练两阶段，没有冻结或 detach 小波网络；优化最终输出的原有 RAW/chromatic 损失，不额外监督 I-。独立输出目录防止覆盖第三方案。原第三方案 checkpoint 不能直接作为整个两阶段模型的 `--resume` 或 `--init-checkpoint`；本配置未自动加载已有实验权重。

```bash
for ratio in 100 250 300; do
  conda run --no-capture-output -n LED-ICCV23 python test_denoise_sideld.py \
    --cp-dir experiments/sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1_repncb/checkpoints/best.pth \
    --testset-type sid --eval-ratio "$ratio" --device cuda:0 --num-workers 2 \
    --result-json "experiments/sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1_repncb/sid_x${ratio}.json"
done
```

定性脚本 `test-op/qual_denoise_compare.py` 也支持从 checkpoint 自动识别新模型。比较时使用相同数据对、训练预算和 best-PSNR 选择规则；训练 loss 更低并不保证真实 pair 的 PSNR 更高。目前只验证实现和数值一致性，尚未完整训练本实验。

## 精修网络消融配置

这四组均保留当前阈值网络 `dwt_width: 16`、`dwt_depth: 3`；LL 恢复 CNN 仍为宽度 32、深度 4。文件名中的 `d32` 是沿用的实验命名，实际结构以 YAML 为准。训练由用户离线执行。

以下后缀均相对于 `configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1_repncb`：

| 配置后缀 | refine_s2d_factor | refine_width | refine_num_blocks | refine_skip_source |
|---|---:|---:|---:|---|
| `.yaml`（基准） | 2 | 16 | 4 | noisy |
| `_k1.yaml` | 1 | 16 | 4 | noisy |
| `_skip_preliminary.yaml` | 2 | 16 | 4 | preliminary |
| `_n6.yaml` | 2 | 16 | 6 | noisy |

`refine_skip_source: noisy` 使用原图 I，`preliminary` 使用小波输出 I-，且梯度仍通过此旁路回传小波网络。这里的“残差”沿用现有通道拼接 + 1×1 融合，不是直接逐元素相加。三组仅改变对应一个参数和输出目录，可以用上面的训练命令替换配置路径分别启动。

通用通道规则：令 K=`refine_s2d_factor`、D=`refine_width`、N=`refine_num_blocks`，则 S2D 输出 4K² 通道，stem 为 4K²→D，N 个 Rep-NCB 均为 D→D，head 为 D→4K²，末端拼接 8K² 通道并由 1×1 输出 4K²，D2S 返回 4 通道。K=1 时 S2D/D2S 使用 Identity，无空间降采样，主干保持 H×W，因此相同 D、N 下显存和计算量通常更大。K、D、N 必须为正整数。

参数也可通过 CLI 覆盖：`--refine-s2d-factor`、`--refine-width`、`--refine-num-blocks`、`--refine-skip-source`。checkpoint 中的参数用于自动恢复结构；早期没有这些字段的 checkpoint 按 K=2、D=16、N=4、noisy 还原，保留旧权重兼容性。

K=1、D=16 时：stem 4→16，主干 16→16，head 16→4，与 4 通道旁路拼接为 8 通道，fusion 8→4。K=2、D=16 的原有结构不变。此前 head 输出 D 的 K=1 版本权重与修正后的结构不兼容，应使用修正后的配置从头训练。

## 标准 3×3 卷积对照

`configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_conv3x3.yaml` 基于 `_repncb_w32.yaml`，新增 `refine_block_type: conv3x3` 并设置独立输出目录。默认 `refine_block_type: repncb` 保持原结构和已有 checkpoint 兼容；也支持 CLI `--refine-block-type`。

本组四个主干块均为普通 `Conv3×3(32→32, bias=True, padding=1) + PReLU(32)`，从头训练单个卷积，没有多分支或固定平滑核。其余小波网络、K=2、输入旁路 I、stem 16→32、head 32→16、fusion 32→16、训练超参数均与 w32 基准一致。保留激活以单独比较多分支重参数化训练与标准卷积训练；两者部署时的主干算子结构相同。标准卷积组沿用 checkpoint 保存/加载流程，部署复制不会改变其卷积权重。
