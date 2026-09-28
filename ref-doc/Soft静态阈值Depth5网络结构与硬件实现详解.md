# Haar + 静态 Soft 阈值 + LL-RepNCB depth5：结构与硬件实现详解

## 1. 分析范围与版本

分析日期：2026-09-28。依据当前 YAML、Python 实现及实际 clean ONNX，不以配置名或旧注释推断结构。

- 配置：`configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft.yaml`
- 模型：`models/learning_dwt_repncb.py`
- 小波、阈值与 LL 恢复：`learning_wt/learning_dwt.py`
- 固定卷积变换：`models/fixed_raw_ops.py`
- 导出：`tools/export_sid_onnx.py`
- ONNX：`experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft/onnx/learning_dwt_repncb_best_deploy_1x4x360x640_conv2x2_clean.onnx`

硬件复现建议以本文件分析的 clean ONNX 及其权重为基准。下文的“一致”指实数公式等价、浮点实现允许舍入差异，不承诺不同累加顺序逐位一致。

## 2. 配置中真正生效的结构

| 参数 | 当前值 | 实际作用 |
|---|---|---|
| dwt_wavelet / dwt_levels | haar / 3 | 固定 Haar，三级仅递归分解 LL |
| dwt_threshold_mode | band_channel | 9 个高频子带 × 4 RAW 通道独立阈值 |
| dwt_shrink_mode / dwt_leak | soft / 0 | 对系数直接 soft threshold，没有 smooth gain |
| dwt_normalize_bands | true | 高频阈值离线乘对应层的 2^level |
| dwt_pad_input | true | 右/下补边到 8 的倍数，最后裁回 |
| dwt_ll_mode | ll_only | LL CNN 只接收原始 LL3，不接收高频 |
| dwt_ll_width / dwt_ll_depth | 32 / 5 | 输入卷积 + 3 个 Rep-NCB + 输出卷积 |
| dwt_ll_block_type | repncb | 中间块内置逐通道 PReLU |
| dwt_ll_fusion | concat_1x1 | CNN 输出与归一化原 LL3 拼接，1×1 学习融合 |
| refine_s2d_factor / refine_width | 2 / 32 | packed RAW 直接 S2D，主干宽 32 |
| refine_num_blocks | 5 | **精修主干也是 5 个 Rep-NCB** |
| refine_skip_source | noisy | 末端旁路来自原输入 I |
| refine_input_fusion | 未写，默认 none | 不存在前置 `[I−,I−I−]` 融合 |
| refine_activation | 未写，默认 prelu | 8 个 Rep-NCB 的激活为 PReLU |

`dwt_width:16`、`dwt_depth:3`、`dwt_magnitude_input:true` 是动态阈值 CNN 的历史配置字段，本组没有实例化该 CNN，不产生相关卷积或幅度输入拼接。`dwt_context:atlas` 不意味着部署必须拼 atlas。LL 初始阈值/阈值上限在本组不构成 LL 收缩操作，LL 直接由网络恢复。Haar 核本组不学习。

配置里的“four Rep-NCB”“two internal LL”以及“CNN predicts thresholds”等注释已经滞后。实际 LL 中 3 个、精修中 5 个，共 **8 个 Rep-NCB**。

## 3. 输入输出约定

输入 I 为浮点 NCHW `[N,4,H,W]`，示例为 `[1,4,360,640]`，这是 packed RAW 尺寸，不是 Bayer mosaic 尺寸。硬件输入不应再做一次 Bayer→packed，除非你的传感器接口尚未完成该步骤。

卷积与固定变换均按通道索引 c=0,1,2,3 工作。该训练配置的损失注释采用 `[R,G1,G2,B]` 顺序；仓库中也有其他 RAW 打包函数使用不同约定，因此接入时应对照实际数据管线和校验样本，不能仅根据 RGGB 名称重排。`chromatic_channel_order: [0,1,3,2]` 属于损失，不是 ONNX 中的通道置换。

黑电平扣除、白电平归一化、曝光比例处理、噪声合成、白平衡/ISP 均不在这个模型 ONNX 内。部署输入必须与已有评测管线数值尺度一致。本模型中间系数允许负数，最终输出也不做 `[0,1]` clamp。GT 亮度校正和 PSNR 的裁剪不属于推理网络。

NCHW 逻辑线性地址可写为 `(((n*C+c)*H+y)*W+x)`；硬件可内部改成 NHWC/分块布局，但所有 Split/Concat、PReLU 和阈值通道对应必须一起映射。

## 4. 完整数据流与张量尺寸

记 LL0=I，Hj 代表该层三个高频子带，带撇号表示已处理。

```text
I → DWT1 → LL1 → DWT2 → LL2 → DWT3 → LL3
      └ HF1→soft       └ HF2→soft       └ HF3→soft
LL3 → /8 → LL CNN → 与 LL3/8 拼接 → 1×1 → ×8 → LL3'
(LL3',HF3') → IWT3 → LL2'
(LL2',HF2') → IWT2 → LL1'
(LL1',HF1') → IWT1 → I−
I− → S2D → Conv16→32 → 5×(Conv32→32+PReLU) → Conv32→16 → F
I  → S2D ────────────────────────────────────────────────┐
                               concat(F,S2D(I)) → Conv1×1 32→16 → D2S → Y
```

| 阶段 | 通用尺寸 | 360×640 示例 |
|---|---|---|
| I | N×4×H×W | 1×4×360×640 |
| LL1/LH1/HL1/HH1 每个 | N×4×H/2×W/2 | 1×4×180×320 |
| LL2/LH2/HL2/HH2 每个 | N×4×H/4×W/4 | 1×4×90×160 |
| LL3/LH3/HL3/HH3 每个 | N×4×H/8×W/8 | 1×4×45×80 |
| LL CNN 隐层 | N×32×H/8×W/8 | 1×32×45×80 |
| LL 融合拼接 | N×8×H/8×W/8 | 1×8×45×80 |
| I− | N×4×H×W | 1×4×360×640 |
| S2D(I−)、S2D(I) 每个 | N×16×H/2×W/2 | 1×16×180×320 |
| 精修隐层 | N×32×H/2×W/2 | 1×32×180×320 |
| 精修末端拼接 | N×32×H/2×W/2 | 1×32×180×320 |
| 输出 Y | N×4×H×W | 1×4×360×640 |

这两个所谓“残差连接”都是**Concat + 学习卷积**，不是直接逐元素相加。没有额外 `Y += I` 或 `Y += I−`。

## 5. Haar DWT/IWT：系数、相位和路由

### 5.1 单个 2×2 块

对每个 c 单独取 `a=I[c,2y,2x]`、`b=I[c,2y,2x+1]`、`c0=I[c,2y+1,2x]`、`d=I[c,2y+1,2x+1]`：

```
LL = (a+b+c0+d)/2
LH = (a-b+c0-d)/2
HL = (a+b-c0-d)/2
HH = (a-b-c0+d)/2
```

采用正交 Haar 的 1/2 缩放，不是均值池化的 1/4。LH/HL 命名以本代码为准，不要直接套用其他库的方向命名。各层都以当前 LL 的左上角为偶数坐标起点，只递归 LL，高频不继续分解。

固定 Conv 权重为 `[16,4,2,2]`，stride=2、padding=0、dilation=1、groups=1、无 bias。只有 `weight[4*b+c,c,:,:]` 非零，b=0/1/2/3 分别为 LL/LH/HL/HH，权重为上述符号矩阵乘 1/2。

输出为 **band-major**：`[LL_c0..c3, LH_c0..c3, HL_c0..c3, HH_c0..c3]`。Split 沿 C 分成 `[4,4,4,4]`。Split 在硬件上可以是地址视图/分流，无须算术，但是否需要物理搬运取决于存储布局。

### 5.2 逆变换

输入必须先按 `[恢复LL, 已收缩LH, 已收缩HL, 已收缩HH]` 沿通道拼接。恢复像素：

```
a  = (LL+LH+HL+HH)/2
b  = (LL-LH+HL-HH)/2
c0 = (LL+LH-HL-HH)/2
d  = (LL-LH-HL+HH)/2
```

部署为权重 `[16,4,2,2]` 的 ConvTranspose，stride=2、padding=0、output_padding=0、无 bias。每个 2×2 输出块互不重叠，不需要块间叠加，但每个像素内部要累加四个子带贡献。该权重按 PyTorch/ONNX ConvTranspose 的 `[Cin,Cout/groups,kH,kW]` 解释，不能按普通 Conv 权重布局读取。

固定零权重可在定制硬件上跳过：Haar 本质是加减树与乘 1/2，S2D/D2S 则是路由。这是对当前通用 Conv 表达的特化，不要求硬件真的执行大量乘零。

## 6. 静态高频 Soft Threshold：重点非卷积路径

每个高频子带、每个 RAW 通道有一个标量阈值，空间和 batch 共享。共 36 个训练 logit：`hf_logits[9,4]`。

训练参数换算为 `T[j,c]=softplus(p[j,c])*2^level`。部署表 `fixed_hf_thresholds` 已包含该 scale；硬件直接加载 T，不再执行 softplus、exp、log 或额外层级缩放。初始归一化阈值为 0.01，训练后必须使用 checkpoint 的实际值，不能用初始值代替。

表行顺序由深到浅：

```
0 LH3, 1 HL3, 2 HH3,
3 LH2, 4 HL2, 5 HH2,
6 LH1, 7 HL1, 8 HH1
```

分解过程由浅到深，层 j 取偏移 `3*(3-j)`。广播形式 `[1,4,1,1]`，每通道重复使用同一个 T，无须生成全尺寸阈值图。

每个系数 z 的处理：

```
p = max(z-T,0)
n = max(-z-T,0)
z' = p-n
```

等价分段公式：z>T 时 z'=z-T；z<−T 时 z'=z+T；其余 z'=0。在 z=±T 时输出 0。T 非负，leak=0，没有旁路残留。

当前 clean ONNX 每个子带导出 3 Sub、2 Relu、1 Neg；九个子带共 27 Sub、18 Relu、9 Neg。不含 gain、Div、Abs、Sign、Clip 或系数逐元素 Mul。定制硬件可用比较器、加减和 mux 实现这条分段公式，不必机械复刻每个 ONNX 节点；需要规定最小负数取反的溢出行为。

每级处理的系数数目：第一级 691,200；第二级 172,800；第三级 43,200；合计 907,200（N=1）。相同数目的输出高频系数需保存或流式送至重建端。

## 7. LLRestorationCNN 的准确结构和尺度

LL3 先除以 8，记 `q=LL3/8`。这是模型尺度的一部分，即使高频 scale 已离线折叠，LL 的这个操作仍然存在。

| 顺序 | 运算 | 输出通道 | 激活 |
|---|---|---:|---|
| 1 | 标准 3×3，4→32，bias | 32 | ReLU |
| 2 | 部署 Rep-NCB 3×3，32→32，bias | 32 | PReLU，32 个斜率 |
| 3 | 同上 | 32 | PReLU |
| 4 | 同上 | 32 | PReLU |
| 5 | 标准 3×3，32→4，bias | 4 | 无 |
| 6 | concat(CNN输出,q)，C 轴 | 8 | 无 |
| 7 | 标准 1×1，8→4，bias | 4 | 无 |
| 8 | 乘 8 | 4 | 无 |

全部 3×3 stride=1、zero padding=1；1×1 stride=1、padding=0。深度 5 不包含末端 1×1 融合。计算 `LL3'=8*Conv1x1([CNN(q),q])`，不是简单 `LL3+CNN(LL3)`。初始化时融合权重近似加法，并不代表训练后的融合仍是加法。

PReLU：`f(x)=x`（x≥0），`f(x)=alpha[c]*x`（x<0）。alpha 来自训练权重，逐通道、跨空间共享，不能用常数 0.25 替代；也不能未经评估改成 ReLU。

如需离线去掉 /8 和 ×8：将第一层 W 改为 W/8、bias 不变；融合权重分为 `[A,B]`，改成 `[8A,B]`，融合 bias 改成 8b；旁路直接传 LL3。这是后续优化公式，当前 clean ONNX 尚未这样折叠。

## 8. Rep-NCB 训练结构与部署结构

训练包含五个线性分支求和再 PReLU：普通 3×3；1×1 扩张32→64后稠密3×3压缩64→32；以及三个1×1后接固定高斯/水平/垂直平滑核的分支。固定核分支还有可学习逐通道 scale 和 bias。

高斯核为 `[[1,2,1],[2,4,2],[1,2,1]]/16`，方向核分别为 `[[0,0,0],[1,2,1],[0,0,0]]/4` 及其转置。前置 1×1 偏置填充的边界贡献在融合时计入 bias。

部署时这五个分支精确合成一个普通 3×3+bias，PReLU 保留。因此硬件不需要执行这些平滑分支、分支加法或通道扩张；直接读 clean ONNX 的 reparam_conv 权重即可。八个块都已融合。

## 9. 精修网络与 S2D/D2S 的准确地址映射

`I−` 是完整小波恢复图，输入主路；旁路是原始 I，没有曝光校正或原图差值操作。

```
Z = S2D(I−)                         # 16ch
F = Conv3×3_32→16(5个RepNCB(Conv3×3_16→32(Z)))
B = S2D(I)                          # 16ch
P = Conv1×1_32→16(concat(F,B))
Y = D2S(P)                          # 4ch
```

stem/head 不带额外激活，只有五个块后的 PReLU。所有精修 3×3 stride=1、padding=1、有 bias，末端 1×1 有 bias。Concat 的前 16 通道是 F，后 16 通道是 B，颠倒会改变输出。

S2D(K=2) 定义：

`Z[n,4*c+2*dy+dx,y,x] = I[n,c,2*y+dy,2*x+dx]`，dy/dx∈{0,1}。

输出是 **channel-major**：先通道0的00/01/10/11四个位置，再通道1，以此类推。这与 Haar 输出的 band-major 不同。

D2S 使用严格逆映射：`Y[n,c,2*y+dy,2*x+dx]=P[n,4*c+2*dy+dx,y,x]`。固定卷积权重只有上述位置为1，其余为0，无 bias，stride=2、无 padding；S2D 使用 Conv，D2S 使用 ConvTranspose。

D2S 不是插值：没有双线性权重、没有相邻像素混合。由于 S2D 直接作用于 packed RAW，各通道各自收集邻域，而不是先还原 Bayer mosaic。

## 9.1 卷积后的激活位置汇总（硬件逐层对应）

以下是本配置的**部署态**顺序，Rep-NCB 的训练分支已经融合成单个 3×3 卷积。PReLU 属于 Rep-NCB 内部，块外不再追加 ReLU；表中“无”表示卷积输出直接交给下一操作，不做截断。

### LLRestorationCNN

所有层空间尺寸均为 45×80（输入 packed RAW 为360×640时）。

| 次序 | 代码模块 | 卷积 | 紧随激活 | 激活后去向 |
|---|---|---|---|---|
| 1 | `wavelet.ll_restorer.layers.0` | 标准3×3，4→32 | **ReLU**，`layers.1` | 第1个LL Rep-NCB |
| 2 | `wavelet.ll_restorer.layers.2.reparam_conv` | 融合3×3，32→32 | **PReLU**，`layers.2.activation` | 第2个LL Rep-NCB |
| 3 | `wavelet.ll_restorer.layers.3.reparam_conv` | 融合3×3，32→32 | **PReLU**，`layers.3.activation` | 第3个LL Rep-NCB |
| 4 | `wavelet.ll_restorer.layers.4.reparam_conv` | 融合3×3，32→32 | **PReLU**，`layers.4.activation` | 标准输出卷积 |
| 5 | `wavelet.ll_restorer.layers.5` | 标准3×3，32→4 | **无** | 与原始归一化LL3拼接 |
| 6 | `wavelet.ll_restorer.fusion` | 标准1×1，8→4 | **无** | 乘8，然后参与IWT3 |

```text
LL3/8
 → Conv3×3(4→32) → ReLU
 → Conv3×3(32→32) → PReLU     # LL Rep-NCB 1
 → Conv3×3(32→32) → PReLU     # LL Rep-NCB 2
 → Conv3×3(32→32) → PReLU     # LL Rep-NCB 3
 → Conv3×3(32→4)              # 无激活
 → concat(上述输出, LL3/8)
 → Conv1×1(8→4)               # 无激活
 → ×8
```

### 精修网络

学习卷积的空间尺寸均为180×320。主路S2D和旁路S2D也没有激活。

| 次序 | 代码模块 | 卷积 | 紧随激活 | 激活后去向 |
|---|---|---|---|---|
| 1 | `refiner.stem` | 标准3×3，16→32 | **无** | 第1个精修Rep-NCB |
| 2 | `refiner.blocks.0.reparam_conv` | 融合3×3，32→32 | **PReLU**，`blocks.0.activation` | 第2个精修Rep-NCB |
| 3 | `refiner.blocks.1.reparam_conv` | 融合3×3，32→32 | **PReLU**，`blocks.1.activation` | 第3个精修Rep-NCB |
| 4 | `refiner.blocks.2.reparam_conv` | 融合3×3，32→32 | **PReLU**，`blocks.2.activation` | 第4个精修Rep-NCB |
| 5 | `refiner.blocks.3.reparam_conv` | 融合3×3，32→32 | **PReLU**，`blocks.3.activation` | 第5个精修Rep-NCB |
| 6 | `refiner.blocks.4.reparam_conv` | 融合3×3，32→32 | **PReLU**，`blocks.4.activation` | head输出卷积 |
| 7 | `refiner.head` | 标准3×3，32→16 | **无** | 与S2D(I)拼接 |
| 8 | `refiner.fusion` | 标准1×1，32→16 | **无** | D2S，然后直接输出 |

```text
S2D(I−)
 → Conv3×3(16→32)             # stem，无激活
 → [Conv3×3(32→32) → PReLU] × 5
 → Conv3×3(32→16)             # head，无激活
 → concat(上述输出, S2D(I))
 → Conv1×1(32→16)             # fusion，无激活
 → D2S                       # 无激活、无输出clamp
```

两套学习网络合计 **1个ReLU、8个PReLU**。每个PReLU具有独立的32个通道斜率，总计256个斜率值，需从对应权重读取。高频soft threshold中的18个ReLU属于收缩公式，不属于上述CNN激活。Haar DWT/IWT、固定S2D/D2S卷积后不额外添加激活，否则会破坏有符号系数或像素重排结果。

## 10. Padding、裁剪与固定尺寸导出

- 小波入口将 H/W 右侧和下侧补到 8 的倍数，使用 replicate（复制边缘），不是 zero padding；重建后裁回原尺寸。
- 精修入口独立对 I− 和 I 右/下补到 2 的倍数，也是 replicate；D2S 后裁回。
- 网络所有标准 3×3 卷积的边缘则是 zero padding=1，和上述 replicate 有本质区别。
- 当前 H=360、W=640 均可被8整除，两个空间补边及最终裁剪在 clean 图中均不需要。
- 不能将360×640静态 ONNX 当作任意尺寸模型。分块硬件需要在内部 tile 边界保留真实邻域，而不是把每块边缘当成整图边界；分解相位也必须按全图坐标对齐。多级 CNN 的 halo 应按完整依赖关系计算并做整图/分块对照，本文不武断指定固定 halo。

## 11. 内存、调度与非卷积操作实现

分解顺序是 DWT1→DWT2→DWT3，重建顺序相反。HF1'、HF2' 在等待深层 LL 恢复期间仍有后续消费者；不可覆盖。原输入 I 也必须保留或重读，供精修旁路使用。

| 张量/集合，N=1 | 元素数 | FP32 字节 | MiB |
|---|---:|---:|---:|
| 输入 I 或 I− | 921,600 | 3,686,400 | 3.516 |
| 第一级三个 HF | 691,200 | 2,764,800 | 2.637 |
| 第二级三个 HF | 172,800 | 691,200 | 0.659 |
| 第三级三个 HF | 43,200 | 172,800 | 0.165 |
| 全部九个 HF | 907,200 | 3,628,800 | 3.461 |
| LL3 | 14,400 | 57,600 | 0.055 |
| LL 32ch 隐层 | 115,200 | 460,800 | 0.439 |
| 精修32ch隐层/末端Concat | 1,843,200 | 7,372,800 | 7.031 |
| 36项阈值表 | 36 | 144 | — |

这些是单张量容量，不是峰值显存，不能把表中数值简单相加当作实际运行峰值。可采用 ping-pong、行缓存或分块调度；是否原地覆盖需检查消费者生命周期。阈值表在 clean ONNX 可能被常量折叠分散为九个广播常量，不必强制硬件重建一个9×4张量。

Split/Concat 本身不做数值运算。若生产者/消费者支持分区地址，可以通过地址映射消除拷贝；若只能接受连续输入，就需要 DMA 或拼接缓冲。五个 Concat 分别为：LL融合一次、三级IWT各一次、精修融合一次。它们不是 atlas 的空间拼接。

训练/return_aux 调试仍可能构建 atlas 和阈值图；默认部署逐层处理，不需要这些缓冲。不要把训练图的 Scatter、阈值 map 生成搬到硬件。

## 12. 当前 clean 图的算子解释

实际图含19 Conv：3个固定DWT + 2个固定S2D + 6个LL学习卷积（含融合）+ 8个精修学习卷积（含融合）。4个ConvTranspose是3个IWT+1个D2S。

19 Relu 中18个来自九组 soft threshold，1个来自LL首层。8 PRelu对应3个LL块+5个精修块。其余27 Sub、9 Neg来自soft；1 Div和1 Mul分别是LL /8和×8。3 Split和5 Concat的用途如上。

此模型没有 BN、池化、动态注意力、Softmax、Sigmoid、动态阈值CNN或网络内部量化模块。部署的 softplus 已离线完成。完整图仍有一个常量 Div，不能误认为还有高频动态除法。

## 13. 硬件数值与验收建议

1. 保留有符号中间值。Haar HF、卷积输出和最终输出都可能为负；soft阈值之前不能饱和到0。
2. Haar的1/2是精确二进制比例，但整数右移的舍入/截断与浮点不同。理论上 |I|≤M 时单层系数可达2M，三级LL可达8M；学习网络激活范围需用真实校准数据测量，不能据此界定整个网络范围。
3. 若量化 z，阈值 T 必须换算到同一层/同一通道的尺度。Concat 两路的量化尺度可能不同，需要重标定或后端支持，不能仅拼接不同标度的整数。
4. 固定Haar/重排权重可以专用加减/布线实现，但训练卷积权重、bias、PReLU斜率不可当成固定先验。
5. 验收顺序：单块Haar正反变换→S2D/D2S相位→阈值边界±T→LL归一化和拼接→各层中间值→整网输出。
6. 测试零输入、带符号小数、常量图、单通道脉冲、棋盘格、接近阈值的系数、奇数尺寸PyTorch边界和真实SID三种ratio。零输入整网未必输出零，因为学习卷积有bias。
7. 保留 FP32 ONNX 作为参考；FP16/定点误差应独立记录最大/均值误差与真实PSNR。不同舍入顺序或NPU融合不可仅以少量随机图判断无损。

本文不修改网络、参数或导出文件，也不假定特定NPU的布局或算子融合策略。以下附录由实际 ONNX 自动提取，便于实现逐节点校核。

## 14. Rep-NCB 训练结构、部署结构与融合数学证明

本章逐项对应 `models/learning_dwt_repncb.py` 中的 `SpatialBranch.forward()`、`SpatialBranch.equivalent()`、`RepNCB.forward()` 和 `RepNCB.switch_to_deploy()`。推导针对本项目实现，不假设存在额外 identity 分支、BN 或分支内激活。

### 14.1 训练态：五个仿射分支加和，然后激活

令输入输出通道数均为 C，当前网络 C=32，扩张通道数 M=2C=64。

```text
                          ┌ direct:   Conv3×3 C→C ───────────────────────────┐
                          ├ expand:   Pad→Conv1×1 C→2C→Conv3×3 2C→C ─────────┤
X ────────────────────────┼ gaussian: Pad→Conv1×1 C→C→DWConv3×3(scale·G) ────┼→ Sum → PReLU → Y
                          ├ horizontal: Pad→Conv1×1 C→C→DWConv3×3(scale·Gh) ┤
                          └ vertical: Pad→Conv1×1 C→C→DWConv3×3(scale·Gv) ───┘
```

- direct 的 3×3 为普通稠密卷积，stride=1、padding=1，有可学习 bias。
- 其他四个分支先对 **X 补一圈零**，再执行带 bias 的 1×1，最后执行 padding=0 的 3×3。这样所有分支输出均为 N×C×H×W。
- expand 的空间卷积是稠密卷积，不是 depthwise；中间没有激活。
- 三个固定核分支先通过稠密 1×1 混合通道，再逐输出通道做 depthwise 3×3；核的数值固定，但每个输出通道的 scale、前置1×1权重/偏置、末端bias均学习。三个分支各有独立参数。
- Sum 是逐元素求和，只有求和后接一次逐通道 PReLU。
- 本模块没有额外 `+X`；不要与 LL 外部融合旁路、精修旁路或 RepMBConv 的 identity 分支混淆。

固定核为：

\[
G=\frac1{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix},\quad
G_h=\frac14\begin{bmatrix}0&0&0\\1&2&1\\0&0&0\end{bmatrix},\quad
G_v=\frac14\begin{bmatrix}0&1&0\\0&2&0\\0&1&0\end{bmatrix}.
\]

三个核的元素和均为1。它们只提供训练分支的结构先验，融合后总卷积核不要求对称、非负或低通。

### 14.2 统一符号：使用互相关约定

PyTorch Conv2d 使用互相关，不对存储权重翻转。令空间偏移 \(r=(r_y,r_x)\in\{-1,0,1\}^2\)，\(\bar X\) 表示图像外部补零后的输入，则标准3×3定义为：

\[
\mathcal C(W,b;X)_o(p)=\sum_i\sum_r W_{o,i,r}\bar X_i(p+r)+b_o.
\]

batch 维独立，以下省略。1×1 权重去掉最后两个大小为1的空间维。若硬件实现真正的翻核卷积，需要按硬件约定转换权重，不能直接套用相反索引。

### 14.3 direct 分支

记其权重/偏置为 \(W^d,b^d\)，则：

\[
Y^d=\mathcal C(W^d,b^d;X).
\]

该分支已经是目标形式，不需要转换。

### 14.4 expand 分支：1×1 与3×3融合

记前置1×1为 \(A\in\mathbb R^{M\times C}\)、偏置 \(a\in\mathbb R^M\)，后置3×3为 \(B\in\mathbb R^{C\times M\times3\times3}\)、偏置 \(b\in\mathbb R^C\)。

因为先对 X 补零再投影，包含边界扩展位置 q 在内，都有：

\[
U_m(q)=\sum_i A_{m,i}\bar X_i(q)+a_m.
\]

代入第二层：

\[
\begin{aligned}
Y^e_o(p)
&=\sum_m\sum_r B_{o,m,r}U_m(p+r)+b_o\\
&=\sum_i\sum_r\left(\sum_m B_{o,m,r}A_{m,i}\right)\bar X_i(p+r)
 +\sum_m\sum_r B_{o,m,r}a_m+b_o.
\end{aligned}
\]

故该分支等价参数为：

\[
\boxed{W^e_{o,i,r}=\sum_m B_{o,m,r}A_{m,i}},\qquad
\boxed{b^e_o=b_o+\sum_m\sum_r B_{o,m,r}a_m}.
\]

偏置项与空间位置 p 无关，因此可以用普通卷积的一维 bias 向量表达。对应代码：

```python
W_e = torch.einsum('omhw,mi->oihw', B, A)
b_e = b + torch.einsum('omhw,m->o', B, a)
```

这是在通道维消去中间变量，而非把两个空间3×3核串联合成5×5；前一层是1×1，因此有效感受野仍为3×3。

### 14.5 固定平滑核分支融合

对任意一个固定核 \(K\in\{G,G_h,G_v\}\)，记前置1×1权重为 \(P_{o,i}\)、偏置为 \(q_o\)，可学习逐通道scale为 \(s_o\)，末端偏置为 \(t_o\)。depthwise空间卷积不跨输出通道：

\[
\begin{aligned}
Y^K_o(p)
&=s_o\sum_r K_r\left(\sum_i P_{o,i}\bar X_i(p+r)+q_o\right)+t_o\\
&=\sum_i\sum_r (s_oK_rP_{o,i})\bar X_i(p+r)
 +s_oq_o\sum_rK_r+t_o.
\end{aligned}
\]

因此：

\[
\boxed{W^K_{o,i,r}=s_oK_rP_{o,i}},\qquad
\boxed{b^K_o=t_o+s_oq_o\sum_rK_r}.
\]

本项目三个核均满足 \(\sum_r K_r=1\)，可以进一步写成 \(b^K_o=t_o+s_oq_o\)。代码保留通用的 sum(K) 形式，而不是硬编码1：

```python
spatial = scale * mask                 # [C,1,3,3]
W_K = spatial * P[:, :, None, None]    # [C,C,3,3]
b_K = t + spatial.sum((1,2,3)) * q
```

单个此类分支的空间核受到可分离结构约束，但多分支加上自由学习的 direct 稠密核后，总核没有上述限制。尤其不能漏掉 \(s_oq_o\)：平滑核不是和为零的边缘核。

### 14.6 为什么边界也成立：bias padding 是必要条件

本实现做的是：

```python
U = Conv1x1(F.pad(X, (1,1,1,1)))
Y = Conv3x3(U, padding=0)
```

在扩展区域，X=0，所以 U 等于前置1×1的偏置。每个输出位置，包括角点，都看到9个位置上的偏置，因此前面偏置求和公式对所有 p 成立。

若错误地改成“先投影，再对 U 补零”，扩展区域 U 就是0，而不是偏置。此时正确的偏置贡献变成：

\[
\sum_m\sum_{r:\ p+r\ \text{在原图内}} B_{o,m,r}a_m+b_o,
\]

它随边缘/内部位置变化，通常不能压成一个空间恒定 bias。举例：前置偏置为1、后置3×3九个权重全为1，X=0；正确bias-padding在每个位置输出9，错误zero-padding在一般角点只输出4。融合后加常量9无法同时复现错误实现的内部9和角点4。

因此训练期间这个边界处理必须保持，不能只在部署时补救。若前置bias恰好为0，问题可能被测试掩盖；验证应使用非零bias。

### 14.7 五个分支相加的融合证明

各分支已写成同一 padding/stride/output shape 的 \(\mathcal C(W^j,b^j;X)\)。利用有限求和的分配律：

\[
\begin{aligned}
Z_o(p)
&=\sum_{j\in\{d,e,G,G_h,G_v\}}\mathcal C(W^j,b^j;X)_o(p)\\
&=\sum_i\sum_r\left(\sum_jW^j_{o,i,r}\right)\bar X_i(p+r)+\sum_jb^j_o.
\end{aligned}
\]

定义：

\[
\boxed{W^{deploy}=W^d+W^e+W^G+W^{G_h}+W^{G_v}},
\quad
\boxed{b^{deploy}=b^d+b^e+b^G+b^{G_h}+b^{G_v}}.
\]

则对于任意输入、任意有效输出位置（包括边界），在实数运算下：

\[
Z^{train}=\mathcal C(W^{deploy},b^{deploy};X)=Z^{deploy}.
\]

两侧施加相同PReLU，得到最终模块输出一致。硬件只需存储 \([C,C,3,3]\) 权重、C个bias和C个PReLU斜率。

### 14.8 PReLU 为什么不融合进线性卷积

\[
Y_o(p)=\begin{cases}Z_o(p),&Z_o(p)\ge0,\\\alpha_oZ_o(p),&Z_o(p)<0.\end{cases}
\]

当 \(\alpha_o\ne1\) 时，分段选择取决于输入，不能用一个固定卷积权重表达对所有输入的两种斜率。因此部署结构为：

```text
X → Conv3×3(C→C, stride=1, padding=1, bias=True) → PReLU(C) → Y
```

硬件可以把Conv与PReLU安排成一个融合执行单元以减少读写，但数学上仍须在累加和bias之后做符号判断和负半轴缩放。这与“把PReLU吸收到卷积权重”不同。

分支内部没有激活，是上述推导成立的重要条件；若在1×1与3×3之间加入ReLU/PReLU，一般不能再按此公式精确融合。多个Rep-NCB之间的PReLU同样阻止把整串块融合为单个线性卷积。

### 14.9 参数数量与部署成本

不计固定核buffer，训练可学习参数为：

| 部分 | 参数数 |
|---|---:|
| direct | 9C²+C |
| expand 的1×1与3×3 | 20C²+3C |
| 每个固定核分支 | C²+3C |
| 三个固定核分支合计 | 3C²+9C |
| PReLU | C |
| 训练合计 | **32C²+14C** |
| 部署3×3+bias+PReLU | **9C²+2C** |

C=32时，每块训练有 **33,216** 个参数，部署为 **9,280** 个参数。八个块合计训练265,728、部署74,240；这里仅统计Rep-NCB，不包含网络首尾卷积、阈值等。

部署卷积每个输出像素位置约需9C²次MAC（不含bias/激活）。融合消除了分支、中间扩张特征和五路相加的运行时开销；参数减少来自代数合并，不是剪枝或近似。训练执行还存在显式补边及中间张量，其实际MAC/访存不能仅以参数量代替。

### 14.10 代码转换、权重存储与硬件验收

`switch_to_deploy()` 先克隆direct权重，再累加四个分支的等价权重/偏置，建立 `reparam_conv`，删除训练分支，保留activation。重复调用不再融合。外层模型 `deploy()` 先深拷贝再转换，保留原训练模型；训练checkpoint与融合checkpoint有不同key，不应互换严格加载。

权重精度保持原设备/dtype。浮点加法非结合，训练分支计算与融合卷积的累加顺序不同，通常不逐位相同；测试用误差容限。定点部署应先在高精度下完成融合，再量化融合后的核与bias，并重新校准PReLU输入范围；分别量化分支后再融合一般不等于先融合后量化。

建议验证：

1. 在float64下给所有分支设置非零随机bias、scale，测试1×1、奇数H/W及常规尺寸，确认边界融合误差接近舍入量级。
2. 保留负输入和负scale，不要只测正数或初始化状态。
3. 检查输出前的Z，以及PReLU后的Y，分开定位融合与激活问题。
4. 使用实际训练权重验证整个LL与精修网络，而非仅单块随机权重。
5. 核对硬件的卷积权重布局、互相关方向、padding=1、bias精度和每通道PReLU斜率。

本项目已有 `tests/test_learning_dwt_repncb.py` 中的 `test_fusion_including_bias_and_edges` 覆盖非零随机偏置、1×1边界、float64融合一致性、原训练图保留及重复转换。数学证明不替代目标硬件的量化/溢出测试。

## 附录A：实际模型指纹与统计

- SHA256：`9c48930039e5abaf6c5b287541fed8567850abde32ddcf69cbebfd5225acfefe`
- 节点总数：103（含Constant）。
- 算子计数：`{'Constant': 7, 'Conv': 19, 'Split': 3, 'Sub': 27, 'Relu': 19, 'Neg': 9, 'Div': 1, 'PRelu': 8, 'Concat': 5, 'Mul': 1, 'ConvTranspose': 4}`

- 原导出校验最大绝对误差：7.748603820800781e-07；来自已有导出报告，本轮未重新导出。

## 附录B：实际学习后的阈值（已含层级 scale）

列按RAW通道索引排列；不可用其他checkpoint的阈值替换。

| 子带 | c0 | c1 | c2 | c3 |
|---|---:|---:|---:|---:|
| LH1 | 0.003313774243 | 0.002401772887 | 0.002903770655 | 0.004766329657 |
| HL1 | 0.003546473105 | 0.003931569401 | 0.003077999689 | 0.006203135941 |
| HH1 | 0.01267493796 | 0.01128497906 | 0.01049843803 | 0.01215688977 |
| LH2 | 0.003280629637 | 0.002004210372 | 0.001968861325 | 0.004461020697 |
| HL2 | 0.004142025486 | 0.002556048799 | 0.002513123676 | 0.006711874157 |
| HH2 | 0.00219805981 | 0.002425812418 | 0.002266067546 | 0.003801558632 |
| LH3 | 0.005742522888 | 0.001772818388 | 0.001688282005 | 0.01137555204 |
| HL3 | 0.008774359711 | 0.002622586442 | 0.002559163142 | 0.01412712224 |
| HH3 | 0.003692030441 | 0.002087003551 | 0.002097887453 | 0.005855304189 |

## 附录C：非Constant节点逐项清单

输出尺寸按当前固定输入推导。Conv/ConvTranspose权重和bias位于节点输入指定的initializer/Constant；完整名字以ONNX为准。

| 节点 | 类型 | 输入节点张量 | 输出尺寸 |
|---|---|---|---|
| `/wavelet/Conv` | Conv | `noisy_raw` | [1, 16, 180, 320] |
| `/wavelet/Split` | Split | `/wavelet/Conv_output_0` | [1, 4, 180, 320]; [1, 4, 180, 320]; [1, 4, 180, 320]; [1, 4, 180, 320] |
| `/wavelet/Sub` | Sub | `/wavelet/Split_output_1` | [1, 4, 180, 320] |
| `/wavelet/Relu` | Relu | `/wavelet/Sub_output_0` | [1, 4, 180, 320] |
| `/wavelet/Neg` | Neg | `/wavelet/Split_output_1` | [1, 4, 180, 320] |
| `/wavelet/Sub_1` | Sub | `/wavelet/Neg_output_0` | [1, 4, 180, 320] |
| `/wavelet/Relu_1` | Relu | `/wavelet/Sub_1_output_0` | [1, 4, 180, 320] |
| `/wavelet/Sub_2` | Sub | `/wavelet/Relu_output_0`<br>`/wavelet/Relu_1_output_0` | [1, 4, 180, 320] |
| `/wavelet/Sub_3` | Sub | `/wavelet/Split_output_2` | [1, 4, 180, 320] |
| `/wavelet/Relu_2` | Relu | `/wavelet/Sub_3_output_0` | [1, 4, 180, 320] |
| `/wavelet/Neg_1` | Neg | `/wavelet/Split_output_2` | [1, 4, 180, 320] |
| `/wavelet/Sub_4` | Sub | `/wavelet/Neg_1_output_0` | [1, 4, 180, 320] |
| `/wavelet/Relu_3` | Relu | `/wavelet/Sub_4_output_0` | [1, 4, 180, 320] |
| `/wavelet/Sub_5` | Sub | `/wavelet/Relu_2_output_0`<br>`/wavelet/Relu_3_output_0` | [1, 4, 180, 320] |
| `/wavelet/Sub_6` | Sub | `/wavelet/Split_output_3` | [1, 4, 180, 320] |
| `/wavelet/Relu_4` | Relu | `/wavelet/Sub_6_output_0` | [1, 4, 180, 320] |
| `/wavelet/Neg_2` | Neg | `/wavelet/Split_output_3` | [1, 4, 180, 320] |
| `/wavelet/Sub_7` | Sub | `/wavelet/Neg_2_output_0` | [1, 4, 180, 320] |
| `/wavelet/Relu_5` | Relu | `/wavelet/Sub_7_output_0` | [1, 4, 180, 320] |
| `/wavelet/Sub_8` | Sub | `/wavelet/Relu_4_output_0`<br>`/wavelet/Relu_5_output_0` | [1, 4, 180, 320] |
| `/wavelet/Conv_1` | Conv | `/wavelet/Split_output_0` | [1, 16, 90, 160] |
| `/wavelet/Split_1` | Split | `/wavelet/Conv_1_output_0` | [1, 4, 90, 160]; [1, 4, 90, 160]; [1, 4, 90, 160]; [1, 4, 90, 160] |
| `/wavelet/Sub_9` | Sub | `/wavelet/Split_1_output_1` | [1, 4, 90, 160] |
| `/wavelet/Relu_6` | Relu | `/wavelet/Sub_9_output_0` | [1, 4, 90, 160] |
| `/wavelet/Neg_3` | Neg | `/wavelet/Split_1_output_1` | [1, 4, 90, 160] |
| `/wavelet/Sub_10` | Sub | `/wavelet/Neg_3_output_0` | [1, 4, 90, 160] |
| `/wavelet/Relu_7` | Relu | `/wavelet/Sub_10_output_0` | [1, 4, 90, 160] |
| `/wavelet/Sub_11` | Sub | `/wavelet/Relu_6_output_0`<br>`/wavelet/Relu_7_output_0` | [1, 4, 90, 160] |
| `/wavelet/Sub_12` | Sub | `/wavelet/Split_1_output_2` | [1, 4, 90, 160] |
| `/wavelet/Relu_8` | Relu | `/wavelet/Sub_12_output_0` | [1, 4, 90, 160] |
| `/wavelet/Neg_4` | Neg | `/wavelet/Split_1_output_2` | [1, 4, 90, 160] |
| `/wavelet/Sub_13` | Sub | `/wavelet/Neg_4_output_0` | [1, 4, 90, 160] |
| `/wavelet/Relu_9` | Relu | `/wavelet/Sub_13_output_0` | [1, 4, 90, 160] |
| `/wavelet/Sub_14` | Sub | `/wavelet/Relu_8_output_0`<br>`/wavelet/Relu_9_output_0` | [1, 4, 90, 160] |
| `/wavelet/Sub_15` | Sub | `/wavelet/Split_1_output_3` | [1, 4, 90, 160] |
| `/wavelet/Relu_10` | Relu | `/wavelet/Sub_15_output_0` | [1, 4, 90, 160] |
| `/wavelet/Neg_5` | Neg | `/wavelet/Split_1_output_3` | [1, 4, 90, 160] |
| `/wavelet/Sub_16` | Sub | `/wavelet/Neg_5_output_0` | [1, 4, 90, 160] |
| `/wavelet/Relu_11` | Relu | `/wavelet/Sub_16_output_0` | [1, 4, 90, 160] |
| `/wavelet/Sub_17` | Sub | `/wavelet/Relu_10_output_0`<br>`/wavelet/Relu_11_output_0` | [1, 4, 90, 160] |
| `/wavelet/Conv_2` | Conv | `/wavelet/Split_1_output_0` | [1, 16, 45, 80] |
| `/wavelet/Split_2` | Split | `/wavelet/Conv_2_output_0` | [1, 4, 45, 80]; [1, 4, 45, 80]; [1, 4, 45, 80]; [1, 4, 45, 80] |
| `/wavelet/Sub_18` | Sub | `/wavelet/Split_2_output_1` | [1, 4, 45, 80] |
| `/wavelet/Relu_12` | Relu | `/wavelet/Sub_18_output_0` | [1, 4, 45, 80] |
| `/wavelet/Neg_6` | Neg | `/wavelet/Split_2_output_1` | [1, 4, 45, 80] |
| `/wavelet/Sub_19` | Sub | `/wavelet/Neg_6_output_0` | [1, 4, 45, 80] |
| `/wavelet/Relu_13` | Relu | `/wavelet/Sub_19_output_0` | [1, 4, 45, 80] |
| `/wavelet/Sub_20` | Sub | `/wavelet/Relu_12_output_0`<br>`/wavelet/Relu_13_output_0` | [1, 4, 45, 80] |
| `/wavelet/Sub_21` | Sub | `/wavelet/Split_2_output_2` | [1, 4, 45, 80] |
| `/wavelet/Relu_14` | Relu | `/wavelet/Sub_21_output_0` | [1, 4, 45, 80] |
| `/wavelet/Neg_7` | Neg | `/wavelet/Split_2_output_2` | [1, 4, 45, 80] |
| `/wavelet/Sub_22` | Sub | `/wavelet/Neg_7_output_0` | [1, 4, 45, 80] |
| `/wavelet/Relu_15` | Relu | `/wavelet/Sub_22_output_0` | [1, 4, 45, 80] |
| `/wavelet/Sub_23` | Sub | `/wavelet/Relu_14_output_0`<br>`/wavelet/Relu_15_output_0` | [1, 4, 45, 80] |
| `/wavelet/Sub_24` | Sub | `/wavelet/Split_2_output_3` | [1, 4, 45, 80] |
| `/wavelet/Relu_16` | Relu | `/wavelet/Sub_24_output_0` | [1, 4, 45, 80] |
| `/wavelet/Neg_8` | Neg | `/wavelet/Split_2_output_3` | [1, 4, 45, 80] |
| `/wavelet/Sub_25` | Sub | `/wavelet/Neg_8_output_0` | [1, 4, 45, 80] |
| `/wavelet/Relu_17` | Relu | `/wavelet/Sub_25_output_0` | [1, 4, 45, 80] |
| `/wavelet/Sub_26` | Sub | `/wavelet/Relu_16_output_0`<br>`/wavelet/Relu_17_output_0` | [1, 4, 45, 80] |
| `/wavelet/Div` | Div | `/wavelet/Split_2_output_0` | [1, 4, 45, 80] |
| `/wavelet/ll_restorer/layers/layers.0/Conv` | Conv | `/wavelet/Div_output_0` | [1, 32, 45, 80] |
| `/wavelet/ll_restorer/layers/layers.1/Relu` | Relu | `/wavelet/ll_restorer/layers/layers.0/Conv_output_0` | [1, 32, 45, 80] |
| `/wavelet/ll_restorer/layers/layers.2/reparam_conv/Conv` | Conv | `/wavelet/ll_restorer/layers/layers.1/Relu_output_0` | [1, 32, 45, 80] |
| `/wavelet/ll_restorer/layers/layers.2/activation/PRelu` | PRelu | `/wavelet/ll_restorer/layers/layers.2/reparam_conv/Conv_output_0` | [1, 32, 45, 80] |
| `/wavelet/ll_restorer/layers/layers.3/reparam_conv/Conv` | Conv | `/wavelet/ll_restorer/layers/layers.2/activation/PRelu_output_0` | [1, 32, 45, 80] |
| `/wavelet/ll_restorer/layers/layers.3/activation/PRelu` | PRelu | `/wavelet/ll_restorer/layers/layers.3/reparam_conv/Conv_output_0` | [1, 32, 45, 80] |
| `/wavelet/ll_restorer/layers/layers.4/reparam_conv/Conv` | Conv | `/wavelet/ll_restorer/layers/layers.3/activation/PRelu_output_0` | [1, 32, 45, 80] |
| `/wavelet/ll_restorer/layers/layers.4/activation/PRelu` | PRelu | `/wavelet/ll_restorer/layers/layers.4/reparam_conv/Conv_output_0` | [1, 32, 45, 80] |
| `/wavelet/ll_restorer/layers/layers.5/Conv` | Conv | `/wavelet/ll_restorer/layers/layers.4/activation/PRelu_output_0` | [1, 4, 45, 80] |
| `/wavelet/ll_restorer/Concat` | Concat | `/wavelet/ll_restorer/layers/layers.5/Conv_output_0`<br>`/wavelet/Div_output_0` | [1, 8, 45, 80] |
| `/wavelet/ll_restorer/fusion/Conv` | Conv | `/wavelet/ll_restorer/Concat_output_0` | [1, 4, 45, 80] |
| `/wavelet/Mul` | Mul | `/wavelet/ll_restorer/fusion/Conv_output_0` | [1, 4, 45, 80] |
| `/wavelet/Concat` | Concat | `/wavelet/Mul_output_0`<br>`/wavelet/Sub_20_output_0`<br>`/wavelet/Sub_23_output_0`<br>`/wavelet/Sub_26_output_0` | [1, 16, 45, 80] |
| `/wavelet/ConvTranspose` | ConvTranspose | `/wavelet/Concat_output_0` | [1, 4, 90, 160] |
| `/wavelet/Concat_1` | Concat | `/wavelet/ConvTranspose_output_0`<br>`/wavelet/Sub_11_output_0`<br>`/wavelet/Sub_14_output_0`<br>`/wavelet/Sub_17_output_0` | [1, 16, 90, 160] |
| `/wavelet/ConvTranspose_1` | ConvTranspose | `/wavelet/Concat_1_output_0` | [1, 4, 180, 320] |
| `/wavelet/Concat_2` | Concat | `/wavelet/ConvTranspose_1_output_0`<br>`/wavelet/Sub_2_output_0`<br>`/wavelet/Sub_5_output_0`<br>`/wavelet/Sub_8_output_0` | [1, 16, 180, 320] |
| `/wavelet/ConvTranspose_2` | ConvTranspose | `/wavelet/Concat_2_output_0` | [1, 4, 360, 640] |
| `/refiner/s2d/Conv` | Conv | `/wavelet/ConvTranspose_2_output_0` | [1, 16, 180, 320] |
| `/refiner/stem/Conv` | Conv | `/refiner/s2d/Conv_output_0` | [1, 32, 180, 320] |
| `/refiner/blocks/blocks.0/reparam_conv/Conv` | Conv | `/refiner/stem/Conv_output_0` | [1, 32, 180, 320] |
| `/refiner/blocks/blocks.0/activation/PRelu` | PRelu | `/refiner/blocks/blocks.0/reparam_conv/Conv_output_0` | [1, 32, 180, 320] |
| `/refiner/blocks/blocks.1/reparam_conv/Conv` | Conv | `/refiner/blocks/blocks.0/activation/PRelu_output_0` | [1, 32, 180, 320] |
| `/refiner/blocks/blocks.1/activation/PRelu` | PRelu | `/refiner/blocks/blocks.1/reparam_conv/Conv_output_0` | [1, 32, 180, 320] |
| `/refiner/blocks/blocks.2/reparam_conv/Conv` | Conv | `/refiner/blocks/blocks.1/activation/PRelu_output_0` | [1, 32, 180, 320] |
| `/refiner/blocks/blocks.2/activation/PRelu` | PRelu | `/refiner/blocks/blocks.2/reparam_conv/Conv_output_0` | [1, 32, 180, 320] |
| `/refiner/blocks/blocks.3/reparam_conv/Conv` | Conv | `/refiner/blocks/blocks.2/activation/PRelu_output_0` | [1, 32, 180, 320] |
| `/refiner/blocks/blocks.3/activation/PRelu` | PRelu | `/refiner/blocks/blocks.3/reparam_conv/Conv_output_0` | [1, 32, 180, 320] |
| `/refiner/blocks/blocks.4/reparam_conv/Conv` | Conv | `/refiner/blocks/blocks.3/activation/PRelu_output_0` | [1, 32, 180, 320] |
| `/refiner/blocks/blocks.4/activation/PRelu` | PRelu | `/refiner/blocks/blocks.4/reparam_conv/Conv_output_0` | [1, 32, 180, 320] |
| `/refiner/head/Conv` | Conv | `/refiner/blocks/blocks.4/activation/PRelu_output_0` | [1, 16, 180, 320] |
| `/refiner/s2d_1/Conv` | Conv | `noisy_raw` | [1, 16, 180, 320] |
| `/refiner/Concat` | Concat | `/refiner/head/Conv_output_0`<br>`/refiner/s2d_1/Conv_output_0` | [1, 32, 180, 320] |
| `/refiner/fusion/Conv` | Conv | `/refiner/Concat_output_0` | [1, 16, 180, 320] |
| `/refiner/d2s/ConvTranspose` | ConvTranspose | `/refiner/fusion/Conv_output_0` | [1, 4, 360, 640] |
