# LL3直接恢复：两个对比实验

基线为 `configs/train_sid_sony_learning_dwt_sym4_l3_d64_atlas.yaml`，实际阈值CNN宽度32。
新增两组均保留sym4、三级仅递归LL、periodization边界和高频smooth阈值处理。
只替换LL3的处理方式：用独立CNN输出恢复后的LL3，不再对LL3进行阈值收缩。

| 实验 | 分支输入（通道拼接） | 分支输出 | 完整模型参数量 |
|---|---|---|---:|
| 基线 | 系数atlas预测阈值 | LL3收缩结果 | 22,028 |
| LL3-only | 原始LL3，N×4×H/8×W/8 | 恢复LL3，N×4×H/8×W/8 | 42,864 |
| LL3-fourbands | 原始LL3/LH3/HL3/HH3，N×16×H/8×W/8 | 恢复LL3，N×4×H/8×W/8 | 46,320 |

第二组使用阈值处理前、同一级同分辨率的四个子带，沿通道维拼接，不在空间维拼图。
LL分支不直接读取第1/2级细节子带；整个模型的高频路径仍使用原atlas阈值CNN。

## CNN与输出定义

两个分支均为4个3×3卷积，隐藏宽度32，前三层ReLU，末层线性：

```
组1：4  → 32 → 32 → 32 → 4
组2：16 → 32 → 32 → 32 → 4
```

LL分支内部对输入子带除以 `2**levels`（三级为8），输出恢复尺度：

```
LL_restored = 8 * (LL_noisy / 8 + CNN(branch_input / 8))
```

采用有符号残差参数化，分支整体输出的是恢复后的LL；不是阈值或乘法gate。
它可增加/减小系数、改变符号，没有LL阈值上限或输出clamp。
末层小随机权重、零bias使分支从接近LL恒等映射开始，并让首步梯度进入前层。
将LL_restored放回系数atlas，仅替换LL3；其他9个终端子带保持原阈值网络结果。
最后IWT还原整张packed RAW。非8倍数输入延续原补齐/裁回策略。

## 训练目标与对比口径

LL CNN和原高频阈值CNN**联合端到端训练**，不冻结原网络。
监督仍作用于IWT后的最终RAW，保持原损失：

```
L_total = 0.6 * L_raw + 0.4 * L_chromatic
```

没有添加额外LL监督项或改变损失权重，以便与基线比较。
两份配置从当前atlas基线复制；batch32、epoch1000、patch256、LR0.0002、warmup10、
seed2026、真实验证设置都不变。独立实验目录，建议从头训练各组，不混用初始化条件。
模型增加了参数，因此效果差异包含额外CNN容量的影响，不是严格等参数量消融。

比较每epoch日志中的 **train_l1**（RAW L1），不是 **train_loss**（加权总损失）。
0.01高于0.006；PSNR则越高越好。训练L1下降是否转化为真实去噪提升，仍需观察real_psnr。

## 运行命令

组1，只有LL3输入：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn.yaml
```

组2，第三层四子带输入：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_fourbands_cnn.yaml
```

对应输出目录：

```
experiments/sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn
experiments/sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_fourbands_cnn
```

新增参数：

```yaml
dwt_ll_mode: ll_only  # threshold（旧模型）/ ll_only / level_bands
dwt_ll_width: 32
dwt_ll_depth: 4
dwt_shrink_ll: false
```

`dwt_ll_mode`非threshold时LL始终走恢复CNN，无论旧`dwt_shrink_ll`开关如何设置。
此时`dwt_ll_max_threshold`仅为兼容保留，不限制恢复LL；`dwt_leak`也仅作用于高频收缩路径。
旧checkpoint无新增字段时默认threshold，不新增LL模块参数，保持严格加载兼容。
新checkpoint自动保存/恢复LL分支结构，可直接用于原`test_denoise_sideld.py`：

```bash
conda run --no-capture-output -n LED-ICCV23 python test_denoise_sideld.py \
  --cp-dir experiments/sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn/checkpoints/best.pth \
  --eval-ratio 100 --device cuda:0
```

只从各自的checkpoint进行resume。旧threshold-only权重不能严格加载为新分支模型；
本次没有提供默认的部分权重迁移，避免影响从头训练对比。

## 验证范围

已检查分支输入确实来自原始对应子带、LL阈值旁路、高频系数保持不变、
两个网络的梯度、任意尺寸补齐、允许LL系数增加/翻转符号、新旧checkpoint加载与配置一致性。
仅执行功能测试及CUDA前反向检查，未启动正式训练；是否降低训练L1等待手动实验结果。

## 第三组：CNN输出与LL拼接，再通过1×1卷积融合

以第一组LL3-only为基准，新增 `dwt_ll_fusion: concat_1x1`；只改变融合方式与输出目录。
输入CNN的是LL3，CNN仍输出4通道。将它与原始LL3沿通道拼成8通道后，
由带bias的可训练1×1卷积输出4通道恢复LL3：

```
x = LL3 / 8
f = CNN(x)                          # N×4×H/8×W/8
LL_restored = 8 * Conv1x1(cat(f, x)) # 8通道→4通道
```

拼接顺序为 `[CNN输出, 原始LL]`，1×1卷积后不再额外加LL，也不加激活或clamp。
卷积可以学习跨CFA通道融合及偏置；初始化权重为 `[I,I]`、bias为0，
使初始融合等价于第一组残差相加，避免随机融合导致初始亮度偏移，同时两路都有梯度。
训练中这些权重均可更新，不固定为相加。

其余高频处理、最终RAW监督、优化器和训练参数与第一组一致；
新增36个可训练参数，完整模型共42,900个参数。默认`dwt_ll_fusion: residual`，
已有两组及其checkpoint的结构不变。该融合选项也支持`dwt_ll_mode: level_bands`，
但本次提供的第三组配置明确使用`ll_only`，便于比较融合方式。

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1.yaml
```

输出目录：`experiments/sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_cnn_concat1x1`。
仅进行梯度、checkpoint及CUDA功能检查，正式训练由用户手动启动。

## 第四组：四子带输入CNN + LL拼接 + 1×1融合

在第三组基础上，将CNN输入换为原始LL3/LH3/HL3/HH3沿通道拼接的16通道。
CNN先输出4通道，再与原始LL3拼为8通道，通过可训练1×1卷积输出恢复LL3：

```
x = cat(LL3, LH3, HL3, HH3) / 8    # 16通道
f = CNN(x)                        # 16→32→32→32→4
LL_restored = 8 * Conv1x1(cat(f, LL3/8))  # 8→4
```

配置：`dwt_ll_mode: level_bands`、`dwt_ll_fusion: concat_1x1`。
这里保留CNN提取局部特征，再做1×1融合，不是直接将原始16通道送入单层1×1卷积。
高频处理、训练损失、初始化方式及其他超参数与前三组一致；总参数46,356。
与第二组比较融合方式，与第三组比较CNN输入信息。

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_fourbands_cnn_concat1x1.yaml
```

独立输出目录：`experiments/sid_sony_learning_dwt_sym4_l3_d32_atlas_ll3_fourbands_cnn_concat1x1`。
未启动正式训练，供手动执行。
