# 计算量与 NPU 延迟差异：Haar / MRLFN / SplitterNet 热点分析

日期：2026-10-08。第1–6节分析已有测速；第7节测试等价部署优化，第8节测试新增高频 CNN。所有实验均未重新训练。

## 1. 结论与口径

**“计算量更低，所以必然更快”需要算子类型、访存方式、融合程度及硬件利用率相近才成立。这里三个模型不满足这些条件。** 计算量统计没有错；它统计的是算法运算次数，不能直接预测 NPU 时间。

计算量来自[各模型推理计算量与峰值特征内存](各模型推理计算量与峰值特征内存.md)及其 JSON；延迟来自[W8A8 测速报告](Qualcomm_AI_Hub_W8A8同NPU测速对比.md)。五个配置当前的 SHA256 均与测速时保存值一致。统一外部输入 `[1,4,360,640]`，Samsung Galaxy S24 / SM8650 / Hexagon V75，QAIRT 2.50。

| 模型 | 卷积/矩阵 GMAC | 总 GOp | 整网实测均值 / ms | profiler 条目数 |
|---|---:|---:|---:|---:|
| MRLFN | 7.4908 | 15.0543 | 0.7723 | 49 |
| Haar 无 LL 归一化 | 3.4054 | 6.8489 | 1.1976 | 111 |
| SplitterNet | 3.8666 | 7.8266 | 2.6029 | 492 |

Haar 的 GOp 约为 MRLFN 的 45.5%，延迟却为其 1.55 倍；SplitterNet 的 GOp 约为 52.0%，延迟为其 3.37 倍。以下用实际热点解释这一差异。

### 逐层结果不能当成整网延迟分解

原始 `execution_detail` 的 `execution_time` **全部为 0**，有效字段是 `execution_cycles`。本报告取相同节点三轮周期数的算术平均，以全部节点平均周期之和为分母计算百分比；完整精度见 JSON/CSV。

**本文所有算子/模块百分比均为逐层 profiling 周期占比，不是整网 wall-clock 时间占比。** Haar 第一轮运行日志中，正常 100 次推理的报告值是 1168 μs；另行执行的逐层 profiling 为 2 次，报告值是 8552 μs。这说明插桩和逐层测量显著改变了开销，不能用“占比 × 1.198 ms”冒充单算子的毫秒耗时，也不能由此直接预言删除一个模块能节省多少整网时间。

profile 中不少 Conv 的周期为 0，而紧随其后的 PReLU/ReLU 或 Add 有周期，符合融合/归属合并的现象。**0 不代表该卷积免费，PReLU 的条目也不一定只测了激活。** 因此 CNN 统计按整个模块归组，避免把被融合卷积错误归结成“激活很慢”。111/49/492 是报告条目数，含零值、边界和融合节点，不是独立 kernel 启动次数。

## 2. 为什么 Ops 少仍然慢

### 2.1 运算次数没有表达实际数据读写

可以用下面的近似关系理解单个阶段，但不能把它视为精确预测器：

```text
阶段时间 ≈ max(计算量 / 有效计算吞吐, 实际搬运字节数 / 有效带宽)
          + 调度、同步、布局转换等未被重叠隐藏的开销
```

你的统计明确把 `cat/split/slice/pad/reshape` 等计为 0 算术 Ops；它们仍可能真实读写、重排张量。ReLU、Sub、Neg 算术很少，却可能反复扫描完整特征图。一次具有权重复用的稠密 3×3 卷积，运算多，但硬件可以并行完成大量乘加；多个分开的点运算不一定享有同样的吞吐。

这属于硬件执行层面的合理解释。当前日志没有实际 DRAM 字节数或硬件利用率计数器，**尚不能定量断言这些节点已经饱和了外部内存带宽**；可以确定的是，不能按 GOp 等比例缩放延迟。

### 2.2 Haar：算术集中在 CNN，逐层热点却在高频处理

Haar 普通网络卷积约占总 GOp 的 97.2%；Soft Threshold 只有 5.4432 MOp，约占总 GOp 的 **0.0795%**。但是三层 Soft 路径在逐层 profile 中占约 **53.03%** 的周期，LL CNN 加精修 CNN 归属条目仅约 **4.91%**。

Soft 目前逐子带执行：

```text
z: [1,4,h,w]
y = relu(z - t) - relu(-z - t)
```

一级分解的三个高频子带各为 `[1,4,180,320]`；每个子带都有两个带阈值的 ReLU 分支、Neg 和最终 Sub。当前实际记录中，减阈值的 Sub 部分为 0 周期、归属到后续 ReLU，但 Neg 和两个分支相减仍有独立非零条目。三子带重复以上图结构，伴随 Split/Slice 与后续 Concat，形成大量低算术量的张量遍历。

三层高频元素数之比是 16:4:1，第一级就占总高频元素的 76.19%，因此一级成为主要热点符合张量规模。4 通道窄子带是否导致特定 HTP 对齐/向量化效率下降，是值得测试的假设；本次数据没有直接给出硬件通道利用率，不能作为已证实原因。

### 2.3 MRLFN：更规则的 CNN 路径，更少的图碎片

MRLFN 的部署主干包含规则的 32 通道卷积、ReLU 与残差，相比 Haar 的三层小波分解和逐子带处理，编译后的条目更少（49 对 111）。这与更好的融合和更少的中间处理相符，但仅凭条目数仍不能证明精确的启动开销。

还需纠正一个可能的误解：**MRLFN 的 k=4 在 mosaic 域操作，相对四通道 packed RAW 实际只降低空间分辨率 2 倍**。因此 MRLFN 主干与 Haar 精修主干都在约 `180×320` 上处理，不能把此次速度差解释成“MRLFN 比 Haar 多下采样了一倍”。代码见 [mrlfn_arch.py](../models/mrlfn_arch.py)。

MRLFN 本身也有明显重排热点，其 S2D/D2S 并不免费；区别在于计算量增加主要来自规则 CNN，而没有 Haar 当前这一长串逐子带阈值路径。

### 2.4 SplitterNet：Pad、分支与高分辨率张量处理

SplitterNet 的 profiler 条目为 **492**。所有名字以 `/Pad` 结尾的条目合计 **45.08%** 的平均周期。其中根 `/Pad` 占 **21.43%**，`/down.0.1/Pad` 占 **6.26%**。这是非常直接的证据：它的大量时间热点属于算法 Ops 表中计为 0 的处理路径。

代码 [paper_denoisers.py](../models/paper_denoisers.py) 显示：

- 输入因四级下采样把高度从 360 补到 368（replicate / ONNX edge）。面积仅增加约 2.22%，不足以单独解释相对 MRLFN 的 3.37 倍延迟。
- `ReflectConv` 在每次卷积前显式执行 reflect padding，不能默认当成零填充卷积的廉价内置 padding。
- 下采样分支数增长到 16 个，随后有多个注意力、拼接、上采样、裁剪与 skip 融合路径。
- 高分辨率部分仍保留 32 通道特征，LeakyReLU 与 Add 命名条目也有较大周期。

上述记录支持优先检查 padding 与碎片化分支执行；由于节点可能融合，不能断言 `/Pad` 条目的全部周期都仅用于复制边缘像素。

### 2.5 结构性峰值内存也不等于访存总量

你的文档中的 3T、7T、17T 是结构性特征保留预算，不是 NPU 的实际流量或峰值 workspace。Haar 可以保持较低峰值容量，却多次读写同一大小的特征图；MRLFN 可以保留更多连接，同时在计算密集卷积中高效复用特征。**容量小与累计数据搬运少，是两个不同指标。** MRLFN 的 7T 还按指定规则分别预留了两个引用同一张量的连接位置，不能用于推导它实际搬运了两份数据。

## 3. Haar 最慢的模块

以下分类互斥，周期平均值四舍五入为整数。CNN 包含其激活和融合卷积，避免漏计融合后的归属。

| 模块/路径 | 三轮平均周期 | 周期占比 |
|---|---:|---:|
| 一级高频 Soft Threshold | 3,405,473 | 39.47% |
| 二级高频 Soft Threshold | 920,789 | 10.67% |
| 输入/输出布局条目 | 806,380 | 9.35% |
| 三层 IDWT 固定反卷积 | 707,578 | 8.20% |
| 小波子带 Split/Slice | 607,710 | 7.04% |
| 精修末端 D2S | 395,031 | 4.58% |
| 三层 DWT 固定卷积 | 364,991 | 4.23% |
| 精修两次 S2D | 364,563 | 4.23% |
| 精修 CNN（含激活、融合卷积） | 353,582 | 4.10% |
| 三级高频 Soft Threshold | 249,137 | 2.89% |
| IDWT 子带拼接 | 228,975 | 2.65% |
| 精修末端拼接 | 88,598 | 1.03% |
| LL CNN（含激活、融合） | 70,049 | 0.81% |
| Input/Output 条目 | 64,696 | 0.75% |
| 常量条目 | 0 | 0.00% |

最优先检查的是**一级高频 Soft Threshold**，而不是继续减少 LL CNN 通道数。LL 路径仅占上述逐层周期约 0.81%，精修 CNN 约 4.10%；削弱它们未必能明显降低实际整网延迟，却会改变恢复能力。

当前这份无 LL 归一化的 clean 源 ONNX **没有 Div，也没有在线 softplus、gain 或阈值 CNN**。热点已经从旧模型中的除法转移到 Soft 路径、子带拆拼和变换/重排，不能沿用旧版本“优化 Div”的定位结论。

## 4. Haar 最慢的单条目

表内形状按对应源 ONNX 的逻辑 NCHW 解释；后端内部可采用不同布局。边界的 `_0231` 是编译后条目命名，源 ONNX 中没有同名节点，布局归因结合其命名及编译器的布局规则判断。

| 排名 | 条目 | 三轮平均周期 | 周期占比 | 对应处理 |
|---|---|---:|---:|---|
| 1 | `noisy_raw_0231` | 487,017 | 5.64% | 输入边界布局处理 |
| 2 | `/wavelet/Sub_2` | 419,630 | 4.86% | L1 第1高频子带双分支相减，4×180×320 |
| 3 | `/wavelet/Sub_5` | 414,150 | 4.80% | L1 第2高频子带双分支相减，4×180×320 |
| 4 | `/wavelet/Sub_8` | 412,052 | 4.78% | L1 第3高频子带双分支相减，4×180×320 |
| 5 | `/refiner/d2s/ConvTranspose` | 395,031 | 4.58% | 末端 D2S，16×180×320 → 4×360×640 |
| 6 | `/wavelet/ConvTranspose_2` | 393,327 | 4.56% | 最后一级 IDWT，恢复到 4×360×640 |
| 7 | `output_0_0231` | 319,363 | 3.70% | 输出边界布局处理 |
| 8 | `/wavelet/Relu` | 299,890 | 3.48% | L1 带阈值 ReLU 分支，4×180×320 |
| 9 | `/wavelet/Conv` | 298,432 | 3.46% | L1 DWT，4×360×640 → 16×180×320 |
| 10 | `/wavelet/Relu_4` | 297,513 | 3.45% | L1 带阈值 ReLU 分支，4×180×320 |
| 11 | `/wavelet/Relu_2` | 296,967 | 3.44% | L1 带阈值 ReLU 分支，4×180×320 |
| 12 | `/wavelet/Relu_3` | 295,770 | 3.43% | L1 带阈值 ReLU 分支，4×180×320 |
| 13 | `/wavelet/Relu_5` | 295,389 | 3.42% | L1 带阈值 ReLU 分支，4×180×320 |
| 14 | `/wavelet/Relu_1` | 295,263 | 3.42% | L1 带阈值 ReLU 分支，4×180×320 |
| 15 | `/wavelet/ConvTranspose_1` | 234,250 | 2.72% | L2 IDWT，恢复到 4×180×320 |
| 16 | `/wavelet/Concat_2` | 228,975 | 2.65% | L1 重建前拼成 16×180×320 |
| 17 | `/refiner/s2d/Conv` | 197,063 | 2.28% | 初步去噪图 I− 的 S2D |
| 18 | `/refiner/s2d_1/Conv` | 167,499 | 1.94% | 原始 RAW I 的 S2D |

六个 L1 ReLU 命名条目分别对应三个子带的正/负阈值分支，不是 CNN 末端的普通单个激活。在源图中它们前面都有减阈值 Sub；前置 Sub 在 profile 中为 0，解释条目时应保留融合归属这一可能。

完整 111 条记录及三个任务分别的周期见 [haar_hotspots.csv](qaihub_w8a8_20261007/haar_hotspots.csv)；其他模型的节点与聚合结果见 [hotspots.json](qaihub_w8a8_20261007/hotspots.json)。

## 5. 建议的优化顺序

以下为初始热点分析提出的优化方案。优先级1的部署对照与实测结果补充在第7节；其他优先级仍是待验证建议，不能承诺加速比例。

### 优先级 1：减少高频处理的分支与完整特征遍历

首先可对照以下实数等价写法，保持每子带、每 RAW 通道独立阈值：

```text
原式：soft(z,t) = ReLU(z-t) - ReLU(-z-t)
候选：soft(z,t) = z - min(max(z,-t),t)       # t >= 0
```

证明分三段：`z>t` 得到 `z-t`；`-t<=z<=t` 得到 0；`z<-t` 得到 `z+t`。`-t` 可离线计算，候选数据通路只有 Max、Min、Sub，避免输入 Neg 和双 ReLU 分支。阈值是广播张量，可用 ONNX Min/Max 表达；不能想当然把 12 通道不同阈值塞入要求标量边界的 [ONNX Clip 接口](https://onnx.ai/onnx/operators/onnx__Clip.html)。

两种表达式在实数域等价，W8A8 的中间量化尺度与舍入可能不同，需要重新校准及输出误差检查。Min/Max 是否被 QAIRT 高效实现、是否真正更快，必须在同一 NPU 上实测；只减少源码算子数量不构成速度证明。

第二个对照是每层把三个 HF 子带作为**连续的 12 通道张量**统一处理，阈值为 `[1,12,1,1]`，仍然使用原来的 36 个学习参数。当前 DWT 已一次输出 16 通道（4 LL + 12 HF），应优先直接切出 HF 范围，避免先拆三个4通道子带再重新 Concat。这样减少图分支和多次调度的机会，同时不需要恢复十子带 atlas。

也可评估对完整16通道处理、给LL设置零阈值，但这会增加处理元素数，应作为独立 A/B，而不是默认比12通道更快。

### 优先级 2：把最后一级 IDWT 与紧接的 S2D 合成

当前主路径存在：

```text
[LL1恢复结果, HF1] (16×180×320)
  → IDWT 2×2 / stride 2 → I− (4×360×640)
  → S2D 2×2 / stride 2 → 16×180×320
  → stem 3×3 (16→32)
```

最后的 IDWT 与 S2D 都在同一个不重叠2×2块上做固定线性变换。令 `c` 是16维子带向量，`S` 是块内Haar合成矩阵，`P` 是S2D排列矩阵，则：

```text
S2D(IDWT(c)) = P S c = M c
```

因此这两个固定操作的组合可改成低分辨率的 **16→16 固定1×1矩阵 M**，直接避免中间完整 I− 图像的物化。若两个操作之间没有激活/裁剪，进一步可把 `M` 折入后续 stem：

```text
W_new[o,j,dy,dx] = Σ_i W_stem[o,i,dy,dx] × M[i,j]
b_new = b_stem
```

本模型该段满足无激活、固定Haar和S2D、非重叠2×2这些结构条件；具体实现仍需核对 LH/HL 顺序、通道排列及边界。因为 M 无偏置，stem 的零 padding 也可保持一致。原 RAW 旁路的 S2D 仍然需要保留。

这是代数层面的等价候选，并非已经完成的优化。FP32 应做严格误差检查；INT8 因删去中间量化边界而可能产生差异，需要重新量化。该候选针对表中的 `/wavelet/ConvTranspose_2` 和 `/refiner/s2d/Conv`，其周期合计约 6.84%，不能直接换算为同等整网加速。

### 优先级 3：尝试 NHWC 的模型 I/O

输入/输出布局命名条目合计约 9.35% 的逐层周期。Qualcomm 官方说明默认保持源模型边界布局可能增加转置，并提供以下编译选项：[官方 Compile Options](https://workbench.aihub.qualcomm.com/docs/hub/api.html#compile-options)。

```text
--force_channel_last_input noisy_raw --force_channel_last_output output_0
```

这是边界契约的改变，外部输入需要相应使用 `[1,360,640,4]`，并确认实际产物接口。若相机应用原本就产生/消费 NHWC packed RAW，有机会避免往返转换；如果只是把转换移到 CPU，就必须把 CPU 转换计入端到端测试，不能算白得的加速。对三个模型比较时也应统一边界契约。

### 优先级 4：再评估固定重排实现和 SplitterNet padding

精修的两次 S2D 与末端 D2S 合计约 8.80% 周期。可以对照原生重排和当前2×2固定卷积，但本项目此前使用卷积形式是为了实际 NPU 兼容性和速度，因此不能凭“重排0 Ops”就认为切回 PixelShuffle 必然更快。

SplitterNet 应首先对照高分辨率 edge/reflect Pad 的部署方式，而不是只裁剪卷积通道。把 reflect 改成 zero padding 会改变网络函数，不是等价导出优化；移除输入 padding 也须处理16倍数尺寸、上采样对齐及裁剪。这类实验应明确与原模型区分。

## 6. 复现这次统计

```bash
python tools/analyze_sid_qaihub_hotspots.py
```

脚本只读本地三轮 profile，校验节点集合一致、全部节点在NPU、分组周期合计等于总量，输出 JSON 和 CSV；不请求 API token，不提交新云端任务。

原始数据：[Haar 第一轮](qaihub_w8a8_20261007/haar_ll_no_norm/profile_1.json)、[第二轮](qaihub_w8a8_20261007/haar_ll_no_norm/profile_2.json)、[第三轮](qaihub_w8a8_20261007/haar_ll_no_norm/profile_3.json)。正常推理与逐层测量差异的证据位于 `qaihub_w8a8_20261007/haar_ll_no_norm/profile_logs_1/j5q43m6ng_runtime.log`。

## 7. 优先级1实测：四组 Haar 对照与 SplitterNet clean（2026-10-08）

本节实际实现并导出了优先级1的四组部署图，使用同一个 Haar `best.pth` 和同一套36个高频阈值。四组采用 **2×2交叉对照**，其中 A 是控制组；不新增训练、不改变 LL CNN 或精修网络。其他优先级（例如 IDWT+S2D 折叠）没有混入本次实验。

### 7.1 四组配置与导出实现

| 组别 | 高频组织 | 收缩公式 | 高频处理节点数 | 整图节点数 |
|---|---|---|---:|---:|
| A | 每层3个4通道子带分别处理 | 双 ReLU | 54 | 94 |
| B | 每层3个4通道子带分别处理 | Min/Max | 27 | 67 |
| C | 每层12通道HF一次处理 | 双 ReLU | 18 | 58 |
| D | 每层12通道HF一次处理 | Min/Max | 9 | 49 |

公式分别是：

```text
双 ReLU: y = relu(z-t) - relu(-z-t)
Min/Max: y = z - min(max(z,-t),t)
```

按原计算量文档的标量计数规则推导，A/C总量均约6.848928 GOp，B/D约6.8462064 GOp：Min/Max仅把907,200个高频系数的每元素6 Ops变为3 Ops，总算术量减少约0.0397%。12通道合并本身不减少算术量。因此即使速度明显变化，也主要体现执行组织差异，而不是GOp显著下降。

负阈值 `-t` 在导出时固化为常量。C/D 从 DWT 的16通道输出直接分成 `[4 LL, 12 HF]`，没有先拆三个子带再重新拼起来；阈值从 `[3,4]` 按 band-major 顺序展平为 `[1,12,1,1]`。层次仍按原 LL 递归，阈值存储仍使用原模型的 deepest-first 顺序。

实现见 [haar_soft_export.py](../models/haar_soft_export.py)；导出入口见 [export_sid_npu_ablation.py](../tools/export_sid_npu_ablation.py)。四份 YAML 是**部署导出配置**，交给这个新导出脚本使用，不是 `train_sid_sony.py` 的训练配置。

- Haar A：4通道逐子带 + 双 ReLU：[haar_soft_band_relu.yaml](../configs/deploy/haar_soft_band_relu.yaml)
- Haar B：4通道逐子带 + Min/Max：[haar_soft_band_minmax.yaml](../configs/deploy/haar_soft_band_minmax.yaml)
- Haar C：12通道按层合并 + 双 ReLU：[haar_soft_level_relu.yaml](../configs/deploy/haar_soft_level_relu.yaml)
- Haar D：12通道按层合并 + Min/Max：[haar_soft_level_minmax.yaml](../configs/deploy/haar_soft_level_minmax.yaml)

所有图均使用相同 onnxsim 0.4.36 清理常量；图节点减少不直接等于 NPU kernel 数减少。输出为标准 ONNX 算子，没有自定义算子。

### 7.2 SplitterNet：固定 shape 能清理什么，不能删除什么

配置：[splitternet_static_clean.yaml](../configs/deploy/splitternet_static_clean.yaml)。沿用原 `experiments/sid_sony_splitternet/checkpoints/best.pth`，输入仍为 `[1,4,360,640]`，不通过改变输入尺寸来获取不公平的延迟优势。

原导出 **2117节点 → clean 488节点**。固定 shape 使常量形状计算可折叠，但不会让边界数据依赖消失。

| 操作 | 清理前 | 清理后 | 含义 |
|---|---:|---:|---|
| Constant | 880 | 0 | 常量转入 initializer 等形式；并非常量数据消失 |
| ConstantOfShape | 79 | 0 | 已知形状的常量构造离线计算 |
| Shape / Gather | 45 / 45 | 0 / 0 | 固定尺寸，无需在线查询 |
| Reshape / Transpose / Cast | 158 / 79 / 79 | 0 / 0 / 0 | 此图中用于常量/形状构造的部分被折叠 |
| Div | 15 | 0 | 原分支尺寸计算被折叠 |
| Slice | 141 | 46 | 保留有效拆分和裁剪 |
| Pad | 79 | 79 | 仍然影响图像边界的数值，保留 |
| Conv / ConvTranspose | 96 / 15 | 96 / 15 | 原CNN结构不变 |

剩余79个Pad中：**1个edge**把输入高度360补到368，**78个reflect**来自30个下采样卷积和16个瓶颈分支各3个ReflectConv。固定尺寸能预计算 padding 的大小，不能把边界像素的复制/反射值预计算成与输入无关的常量。

剩余46个Slice中，30个用于15次通道二分，15个用于转置卷积后的有效裁剪，最后1个恢复360×640输出尺寸。转置卷积 k=3、stride=2、padding=0 的尺寸是 `2n+1`，所以这些裁剪仍有作用。将reflect改成zero、删去输入pad或删去有效crop，都会改变当前模型的函数。

因此本轮 clean 是**固定形状常量折叠与冗余操作消除**，没有篡改边界行为。使用 [ONNX Simplifier](https://github.com/onnxsim/onnxsim) 完成图清理，再独立进行数值验证。QAIRT 原编译流程也会优化源图，因此不能从2117→488直接推断设备速度按同比例提升。

### 7.3 导出正确性

五份新图分别与原检查点的 PyTorch 输出对比，ORT 禁用图优化，测试 uniform、signed_raw、zeros、tiny、border_ramp、corner_impulses 六类输入。所有检查通过：Haar 四组最大全图绝对误差均不超过 **1.789×10⁻⁶**，SplitterNet clean 不超过 **8.345×10⁻⁷**，并通过 ONNX checker。

这验证的是FP32等价导出，不是INT8去噪质量。Min/Max重写会改变中间量化边界，本轮仍按之前约定只比较W8A8速度，不报告PSNR。

完整验证、算子清单、配置及权重/ONNX哈希见 [export_reports.json](qaihub_priority1_20261008/export_reports.json)。五份新ONNX位于 [experiments/npu_priority1_ablation/onnx](../experiments/npu_priority1_ablation/onnx)。

### 7.4 本轮 NPU 实测

与上一轮统一：Samsung Galaxy S24（SM8650 / Hexagon V75），QNN DLC，QAIRT 2.50，W8A8及量化I/O，NCHW `[1,4,360,640]`，随机校准仅用于速度测试。四组Haar、clean SplitterNet为新编译；MRLFN和原SplitterNet复用上轮编译产物，**重新提交三轮测速**，共7个条目、21个任务。此处比较同一型号NPU，不保证同一台物理手机。

| 模型/组别 | 全部样本均值 / ms | P50 / ms | P95 / ms | 三轮各自均值 / ms |
|---|---:|---:|---:|---|
| Haar A：逐子带双ReLU | 1.2001 | 1.1810 | 1.2639 | 1.1991 / 1.2170 / 1.1842 |
| Haar B：逐子带Min/Max | 1.4959 | 1.4760 | 1.5605 | 1.4808 / 1.5172 / 1.4897 |
| Haar C：12通道双ReLU | 0.8123 | 0.7880 | 1.0040 | 0.8327 / 0.7948 / 0.8095 |
| Haar D：12通道Min/Max | 0.7979 | 0.7800 | 0.8838 | 0.7858 / 0.8173 / 0.7906 |
| MRLFN（本轮重测） | 0.7682 | 0.7460 | 0.9153 | 0.7700 / 0.7660 / 0.7687 |
| SplitterNet 原导出（本轮重测） | 2.5911 | 2.5670 | 2.7273 | 2.5749 / 2.6025 / 2.5958 |
| SplitterNet clean | 2.5952 | 2.5710 | 2.7505 | 2.5983 / 2.5656 / 2.6217 |

四组因子对照（延迟下降为正值，负值表示变慢）：

| 对照 | 隔离的改动 | 延迟下降 |
|---|---|---:|
| Haar A：逐子带双ReLU → Haar B：逐子带Min/Max | 只换Min/Max | -24.65% |
| Haar A：逐子带双ReLU → Haar C：12通道双ReLU | 只合并12通道 | 32.31% |
| Haar C：12通道双ReLU → Haar D：12通道Min/Max | 合并后再换Min/Max | 1.78% |
| Haar B：逐子带Min/Max → Haar D：12通道Min/Max | Min/Max基础上合并12通道 | 46.66% |
| Haar A：逐子带双ReLU → Haar D：12通道Min/Max | 两个改动组合 | 33.51% |

本次四组中最快的是 **Haar D：12通道Min/Max：0.7979 ms**，相对A的延迟下降 **33.51%**。其耗时是本轮MRLFN的 **1.039 倍**，是SplitterNet clean的 **0.307 倍**。

SplitterNet clean相对原导出的延迟变化：下降 **-0.16%**（负数表示变慢）。
尽管源ONNX节点大幅减少，实际端侧延迟差异较小，本次短时三轮不足以证明稳定加速；这与QAIRT已对原图执行常量折叠/形状优化的解释相符。clean的确定收益是更简洁的源图与更明确的固定尺寸部署接口，而不是已证实的大幅NPU加速。
原版和clean首轮的端侧profiler均为492个条目，其中446个同名；主要名称差异来自拆分/裁剪转换后的节点命名。这进一步说明源ONNX清理并没有同等减少最终NPU执行图，但不据此声称两个量化产物逐位相同。

逐子带Min/Max（B）没有兑现“算子更少就更快”的预期：实测比A更慢。其首轮逐层记录中，一级Max/Min条目各约0.58–0.61百万周期，而A的对应ReLU命名条目约0.30百万周期；这些条目的融合边界不同，只能辅助定位，不能用算子数直接推算延迟。应按整网实测选择表达方式。

最快两组的均值差不足5%；本次可报告均值排名，但少量短时轮次不足以证明在不同手机/温度下仍保持同样优劣。

均值/P50/P95取三轮合计300个原始推理样本，保留首个较慢样本；平台 `estimated_inference_time` 另存于 JSON。本轮所有横向速度比都使用本轮重测的基线，不能把不同轮次的微小差别解读成确定收益。

**7个条目的W8A8编译与21个测速任务均成功。** 编译日志实际含 `--weights_bitwidth 8 --act_bitwidth 8`；所有任务日志确认SM-S921U1 / SM8650及同一QAIRT版本，profiler条目均为NPU。实际量化接口见 [compiled_specs.json](qaihub_priority1_20261008/compiled_specs.json)，证据见 [execution_audit.json](qaihub_priority1_20261008/execution_audit.json)。

逐层记录的交叉检查（Haar选择HF收缩命名条目，SplitterNet选择Pad命名条目）：

| 条目 | 各轮profiler条目数 | 所选路径平均周期 | 逐层周期占比 |
|---|---|---:|---:|
| Haar A：逐子带双ReLU | 111 / 111 / 111 | 4,577,192 | 52.87% |
| Haar B：逐子带Min/Max | 84 / 84 / 84 | 6,563,235 | 61.69% |
| Haar C：12通道双ReLU | 69 / 69 / 69 | 1,575,848 | 32.97% |
| Haar D：12通道Min/Max | 60 / 60 / 60 | 2,184,918 | 40.42% |
| SplitterNet 原导出（本轮重测） | 492 / 492 / 492 | 7,250,584 | 45.07% |
| SplitterNet clean | 492 / 492 / 492 | 7,330,203 | 45.35% |

这些周期占比仍具有第1节所述插桩限制，不能替代上表整网毫秒结果。节点融合也可能改变周期归属。 本轮D的高频逐层周期高于C，但整网均值略低，也再次说明不能把逐层周期总和直接换算成正常推理时间；C/D的细小差距不作为稳定优劣结论。

| 条目 | 编译任务 | 三轮测速任务 |
|---|---|---|
| Haar A：逐子带双ReLU | [jp3oeyqlp](https://workbench.aihub.qualcomm.com/jobs/jp3oeyqlp) | [j57oxq6vg](https://workbench.aihub.qualcomm.com/jobs/j57oxq6vg) / [jp4evz88g](https://workbench.aihub.qualcomm.com/jobs/jp4evz88g) / [jpx0ywm3p](https://workbench.aihub.qualcomm.com/jobs/jpx0ywm3p) |
| Haar B：逐子带Min/Max | [jgod3jex5](https://workbench.aihub.qualcomm.com/jobs/jgod3jex5) | [jprxez90p](https://workbench.aihub.qualcomm.com/jobs/jprxez90p) / [jp2ol2jrg](https://workbench.aihub.qualcomm.com/jobs/jp2ol2jrg) / [jpy869n8g](https://workbench.aihub.qualcomm.com/jobs/jpy869n8g) |
| Haar C：12通道双ReLU | [jpv2vjzjg](https://workbench.aihub.qualcomm.com/jobs/jpv2vjzjg) | [jp0olnk9p](https://workbench.aihub.qualcomm.com/jobs/jp0olnk9p) / [jp8jzl8k5](https://workbench.aihub.qualcomm.com/jobs/jp8jzl8k5) / [jgk63jdw5](https://workbench.aihub.qualcomm.com/jobs/jgk63jdw5) |
| Haar D：12通道Min/Max | [jgj3ejkxp](https://workbench.aihub.qualcomm.com/jobs/jgj3ejkxp) | [j5q43jwng](https://workbench.aihub.qualcomm.com/jobs/j5q43jwng) / [jglw3j7jp](https://workbench.aihub.qualcomm.com/jobs/jglw3j7jp) / [jp3oey83p](https://workbench.aihub.qualcomm.com/jobs/jp3oey83p) |
| MRLFN（本轮重测） | [jp0olv09p](https://workbench.aihub.qualcomm.com/jobs/jp0olv09p) | [j5wyqjx6g](https://workbench.aihub.qualcomm.com/jobs/j5wyqjx6g) / [jg9ow68lg](https://workbench.aihub.qualcomm.com/jobs/jg9ow68lg) / [jp1oer325](https://workbench.aihub.qualcomm.com/jobs/jp1oer325) |
| SplitterNet 原导出（本轮重测） | [jgk639ew5](https://workbench.aihub.qualcomm.com/jobs/jgk639ew5) | [jgd6oj0ep](https://workbench.aihub.qualcomm.com/jobs/jgd6oj0ep) / [j5wyqjx3g](https://workbench.aihub.qualcomm.com/jobs/j5wyqjx3g) / [jg9ow68wg](https://workbench.aihub.qualcomm.com/jobs/jg9ow68wg) |
| SplitterNet clean | [jgzzr1vkg](https://workbench.aihub.qualcomm.com/jobs/jgzzr1vkg) | [jpv2vjekg](https://workbench.aihub.qualcomm.com/jobs/jpv2vjekg) / [jgj3ejovp](https://workbench.aihub.qualcomm.com/jobs/jgj3ejovp) / [jpe6kj8og](https://workbench.aihub.qualcomm.com/jobs/jpe6kj8og) |

### 7.5 复现

项目导出环境需有 `onnxsim==0.4.36`；云端SDK继续使用单独环境的 `qai-hub==0.56.0`，避免影响训练环境依赖。

```bash
python tools/export_sid_npu_ablation.py \
  configs/deploy/haar_soft_band_relu.yaml \
  configs/deploy/haar_soft_band_minmax.yaml \
  configs/deploy/haar_soft_level_relu.yaml \
  configs/deploy/haar_soft_level_minmax.yaml \
  configs/deploy/splitternet_static_clean.yaml

# 在已安装 qai-hub 的环境中执行，token不写入配置
read -r -s -p 'AI Hub API token: ' QAI_HUB_API_TOKEN
export QAI_HUB_API_TOKEN
python tools/benchmark_sid_qaihub.py --watch \
  --sources-json ref-doc/qaihub_priority1_20261008/sources.json \
  --output-dir ref-doc/qaihub_priority1_20261008
unset QAI_HUB_API_TOKEN

# 下载完成后，本地汇总，无需认证
python tools/summarize_sid_qaihub.py \
  --output-dir ref-doc/qaihub_priority1_20261008
```

已有任务目录会恢复状态和下载，不重复提交。如果要独立重测，换一个新的 `--output-dir`；如果重新导出导致ONNX哈希改变，需要同步更新sources清单中的哈希后再运行。脚本会检查该哈希，防止把旧产物当成新图测速。

每个条目目录中保存 `model.dlc`、三份`profile_N.json`及compile/profile日志；任务和源文件映射见 [jobs.json](qaihub_priority1_20261008/jobs.json)、[sources.json](qaihub_priority1_20261008/sources.json)。


## 8. 用逐级小 CNN 替换高频收缩：四组未训练速度实验

### 8.1 实验范围与精确结构

基于 `train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm.yaml` 对应的 `best.pth`，保留已训练的 LLRestorationCNN 和精修网络。Haar 为三级递归分解，每次继续分解原始 LL；只将同级 LH/HL/HH 的 12 个通道交给一个小 CNN，随后与恢复后的 LL 拼接并逆变换。三级各有一个独立 CNN，不共享参数，不进行十张子图的 atlas 拼接。

实现见 [haar_hf_cnn.py](../models/haar_hf_cnn.py)。令该级高频输入为 `z`，四组结构严格如下：

| 组别 | 公式 | 卷积细节 | 三级 HF 参数总数 |
|---|---|---|---:|
| 1：DW1 | `ReLU(DW1×1(z))` | 12→12，groups=12 | 72 |
| 2：DW3 | `ReLU(DW3×3(z))` | 12→12，groups=12，零填充1 | 360 |
| 3：DW3+PW1 | `PW1×1(ReLU(DW3×3(z)))` | DW 同组2；PW 为12→12、groups=1，末尾无激活 | 828 |
| 4：DW1残差 | `z + ReLU(DW1×1(z))` | DW 同组1，相加后无激活 | 72 |

所有卷积均有 bias、stride=1。三级输入分别为 `[1,12,180,320]`、`[1,12,90,160]`、`[1,12,45,80]`；输出形状不变。组1/2没有子带间或 RAW 通道间混合，组3的 PW 可以混合全部12个通道。第一级对应最高分辨率。

新 CNN 使用 PyTorch Conv2d 默认随机初始化，seed=2026；每级参数独立。组1和组4的 DW 初始权重完全相同，只区别残差相加。已导出初始 HF 参数文件 `*_hf_untrained.pth` 便于复现，未更新原始 checkpoint，也没有训练或 PSNR 测试。

**这四组是新的非等价网络，不是第7节的等价改写。** 组1/2的 HF 输出非负，无法直接表达原 Soft Threshold 的负系数；组4满足输出≥输入，对正系数不能直接执行向零收缩。组3的线性 PW 输出允许正负值，但当前也未学到收缩功能。速度优劣不能说明去噪效果；后续训练需要单独验证这些结构限制。这里按要求保留四种结构。

### 8.2 参数与计算量

统计对象为实际部署 ONNX，Rep-NCB 已融合。卷积按稠密运算计数，包括固定 Haar/S2D/D2S 核中的零权重；1 MAC=2 Op，另计 bias、激活和逐元素算术。Split/Concat/形状操作记0算术量，但它们可能产生实际访存和调度开销。固定变换核不计可学习参数。

“部署学习系数”包括已冻结的36个学习阈值，因此原始 Soft 为86,408个 CNN 参数 + 36个阈值 = **86,444**。新 CNN 替换这36个阈值，不应将原阈值重复计入。该口径不同于保留全部重参数化分支的训练态参数量。

| 组别 | 部署学习系数总数 | 其中HF参数 | 卷积 GMAC | 总 GOp |
|---|---:|---:|---:|---:|
| 原始逐子带 Soft | 86,444 | 36 | 3.4054272 | 6.8489280 |
| 合并12通道 Min/Max（D） | 86,444 | 36 | 3.4054272 | 6.8462064 |
| 1：DW1 + ReLU | 86,480 | 72 | 3.4063344 | 6.8471136 |
| 2：DW3 + ReLU | 86,768 | 360 | 3.4135920 | 6.8616288 |
| 3：DW3 + ReLU + PW1 | 87,236 | 828 | 3.4244784 | 6.8843088 |
| 4：DW1 + ReLU + 残差 | 86,480 | 72 | 3.4063344 | 6.8480208 |

四组 HF 的 MAC 分别为907,200、8,164,800、19,051,200、907,200。DW1每级参数为12个权重+12个偏置；DW3每级为108+12；PW额外144+12。组4相比组1多907,200次逐元素加法，没有增加参数。整网大部分卷积工作在 LL 和精修网络，故四组整网计算量差异很小。

逐节点统计和计数口径见 [complexity.json](qaihub_hf_cnn_20261008/complexity.json)，统计脚本见 [profile_hf_cnn_onnx.py](../tools/profile_hf_cnn_onnx.py)。这些是算法操作数，不代表 NPU 编译后指令数。

### 8.3 导出验证与同 NPU 测速

输入固定 `[1,4,360,640]`，模型中的 Haar/逆 Haar 和 S2D/D2S 继续使用2×2卷积/转置卷积。四份 clean 图都没有 Div/Abs/Sign；组1/2有46个节点，组3/4有49个节点。FP32 ONNX 分别与**自身未训练 CNN 的 PyTorch 输出**核对，覆盖均匀随机、有符号输入、全零、极小值、边界斜坡和角点脉冲六类输入，全部通过；最大绝对误差不超过1.79e-6。该检查不表示与原 Soft 模型输出一致，也不是 INT8 精度验证。

测速设备仍为 Samsung Galaxy S24 / SM-S921U1 / SM8650 / Hexagon V75，QAIRT 2.50.0.260828221209。W8A8，平台随机校准，只比较速度；实际量化 IO 为 uint8 并带 scale/zero_point。四组分别编译，另外复用第7节原始逐子带双ReLU和合并12通道Min/Max的编译产物，但本节两条基线的测速均重新执行。每项3轮、每轮100个样本，均值/P50/P95统计全部300个样本，不剔除首个较慢样本。同设备型号不保证同一台物理手机。

| 组别 | 均值 / ms | P50 / ms | P95 / ms | 三轮各自均值 / ms | 相对原始延迟下降 | 相对D延迟下降 |
|---|---:|---:|---:|---|---:|---:|
| 原始逐子带 Soft | 1.2057 | 1.1860 | 1.3401 | 1.2047 / 1.2067 / 1.2056 | 0.00% | -50.99% |
| 合并12通道 Min/Max（D） | 0.7985 | 0.7850 | 0.8340 | 0.8055 / 0.7883 / 0.8018 | 33.77% | 0.00% |
| 1：DW1 + ReLU | 0.5945 | 0.5800 | 0.6894 | 0.5929 / 0.5959 / 0.5947 | 50.69% | 25.55% |
| 2：DW3 + ReLU | 0.6190 | 0.5910 | 0.8023 | 0.6090 / 0.6293 / 0.6189 | 48.66% | 22.48% |
| 3：DW3 + ReLU + PW1 | 0.6024 | 0.5800 | 0.6975 | 0.5891 / 0.6181 / 0.5999 | 50.04% | 24.56% |
| 4：DW1 + ReLU + 残差 | 0.5982 | 0.5690 | 0.7580 | 0.6020 / 0.5971 / 0.5954 | 50.39% | 25.09% |

四个 CNN 组中本次均值最低的是 **1：DW1 + ReLU：0.5945 ms**。相对本轮原始 Soft 延迟下降 **50.69%**；相对合并高频的D组延迟下降 **25.55%**（负数表示变慢）。

四组算术量变化不足1%，速度差异应结合NPU执行方式判断，不能按参数量或MAC直接排序。这里测到的是这些初始权重、随机校准和当前编译器生成产物的速度；训练并重新量化后仍需复测。均值差距很小的组不能凭三轮短时测试断言稳定领先。

**6项编译产物、18个测速任务均成功，全部逐层执行条目为NPU。** 实际编译日志确认权重/激活8bit，设备型号和QAIRT版本一致。证据见 [execution_audit.json](qaihub_hf_cnn_20261008/execution_audit.json)、[compiled_specs.json](qaihub_hf_cnn_20261008/compiled_specs.json)、[summary.json](qaihub_hf_cnn_20261008/summary.json)。

| 组别 | 编译任务 | 三轮测速任务 |
|---|---|---|
| 原始逐子带 Soft | [jp3oeyqlp](https://workbench.aihub.qualcomm.com/jobs/jp3oeyqlp) | [j56onkmy5](https://workbench.aihub.qualcomm.com/jobs/j56onkmy5) / [jp3oey7np](https://workbench.aihub.qualcomm.com/jobs/jp3oey7np) / [jgod3jwk5](https://workbench.aihub.qualcomm.com/jobs/jgod3jwk5) |
| 合并12通道 Min/Max（D） | [jgj3ejkxp](https://workbench.aihub.qualcomm.com/jobs/jgj3ejkxp) | [jpv2vjmrg](https://workbench.aihub.qualcomm.com/jobs/jpv2vjmrg) / [jgj3ejyep](https://workbench.aihub.qualcomm.com/jobs/jgj3ejyep) / [jpe6kjxvg](https://workbench.aihub.qualcomm.com/jobs/jpe6kjxvg) |
| 1：DW1 + ReLU | [jp0olnonp](https://workbench.aihub.qualcomm.com/jobs/jp0olnonp) | [jg9ow628g](https://workbench.aihub.qualcomm.com/jobs/jg9ow628g) / [jp1oer175](https://workbench.aihub.qualcomm.com/jobs/jp1oer175) / [jgd6oj4zp](https://workbench.aihub.qualcomm.com/jobs/jgd6oj4zp) |
| 2：DW3 + ReLU | [jp8jzljo5](https://workbench.aihub.qualcomm.com/jobs/jp8jzljo5) | [j57oxqn9g](https://workbench.aihub.qualcomm.com/jobs/j57oxqn9g) / [jp4evz41g](https://workbench.aihub.qualcomm.com/jobs/jp4evz41g) / [jpx0ywrlp](https://workbench.aihub.qualcomm.com/jobs/jpx0ywrlp) |
| 3：DW3 + ReLU + PW1 | [j5q43j4og](https://workbench.aihub.qualcomm.com/jobs/j5q43j4og) | [j5m93jk9g](https://workbench.aihub.qualcomm.com/jobs/j5m93jk9g) / [jgn13jqqp](https://workbench.aihub.qualcomm.com/jobs/jgn13jqqp) / [jprxezd7p](https://workbench.aihub.qualcomm.com/jobs/jprxezd7p) |
| 4：DW1 + ReLU + 残差 | [jglw3j8mp](https://workbench.aihub.qualcomm.com/jobs/jglw3j8mp) | [jp2ol2dqg](https://workbench.aihub.qualcomm.com/jobs/jp2ol2dqg) / [jpy8692lg](https://workbench.aihub.qualcomm.com/jobs/jpy8692lg) / [jp0oln9np](https://workbench.aihub.qualcomm.com/jobs/jp0oln9np) |

高频命名路径的插桩周期参考：

| 组别 | profiler条目数（各轮） | HF命名路径平均周期 | HF周期占比 |
|---|---|---:|---:|
| 原始逐子带 Soft | 111 / 111 / 111 | 4,582,723 | 53.01% |
| 合并12通道 Min/Max（D） | 60 / 60 / 60 | 2,188,020 | 40.28% |
| 1：DW1 + ReLU | 57 / 57 / 57 | 97,514 | 2.98% |
| 2：DW3 + ReLU | 57 / 57 / 57 | 112,494 | 3.41% |
| 3：DW3 + ReLU + PW1 | 60 / 60 / 60 | 52,186 | 1.56% |
| 4：DW1 + ReLU + 残差 | 60 / 60 / 60 | 181,360 | 5.33% |

CNN组选择 `/processors.` 路径，Soft组选择高频 Min/Max/ReLU/Sub/Neg 路径。卷积与激活融合可能使Conv记录0周期而时间计入激活条目；这些不是独立算子的实际毫秒耗时。插桩周期仅用于定位，不得乘整网正常延迟推算该部分时间。


### 8.4 配置和复现

四个配置为**部署测速配置**，由导出脚本读取，不是 `train_sid_sony.py` 的训练配置：

- [haar_hf_cnn_dw1.yaml](../configs/deploy/haar_hf_cnn_dw1.yaml)
- [haar_hf_cnn_dw3.yaml](../configs/deploy/haar_hf_cnn_dw3.yaml)
- [haar_hf_cnn_dw3_pw1.yaml](../configs/deploy/haar_hf_cnn_dw3_pw1.yaml)
- [haar_hf_cnn_dw1_residual.yaml](../configs/deploy/haar_hf_cnn_dw1_residual.yaml)

```bash
# 导出环境：torch / onnx / onnxruntime / onnxsim==0.4.36
python tools/export_sid_npu_ablation.py \
  configs/deploy/haar_hf_cnn_dw1.yaml \
  configs/deploy/haar_hf_cnn_dw3.yaml \
  configs/deploy/haar_hf_cnn_dw3_pw1.yaml \
  configs/deploy/haar_hf_cnn_dw1_residual.yaml \
  --output-dir experiments/npu_hf_cnn_ablation/onnx

python tools/profile_hf_cnn_onnx.py

# 单独的 qai-hub==0.56.0 环境；API token 不写入配置
read -r -s -p 'AI Hub API token: ' QAI_HUB_API_TOKEN
export QAI_HUB_API_TOKEN
python tools/benchmark_sid_qaihub.py --watch \
  --sources-json ref-doc/qaihub_hf_cnn_20261008/sources.json \
  --output-dir ref-doc/qaihub_hf_cnn_20261008
unset QAI_HUB_API_TOKEN

python tools/summarize_sid_qaihub.py \
  --output-dir ref-doc/qaihub_hf_cnn_20261008
```

已有任务目录可恢复下载；独立重测需更换输出目录。重新导出后需同步 `sources.json` 的源文件SHA256。导出报告见 [export_reports.json](qaihub_hf_cnn_20261008/export_reports.json)，全部映射、任务和原始日志见 [jobs.json](qaihub_hf_cnn_20261008/jobs.json)、[sources.json](qaihub_hf_cnn_20261008/sources.json)。ONNX及初始参数位于 `experiments/npu_hf_cnn_ablation/onnx/`；各组 W8A8 DLC 位于本节结果目录的对应子目录。


### 8.5 四组从头训练配置（depth7 基线）

新增训练入口 `dwt_threshold_mode: hf_cnn`，用 `dwt_hf_cnn_variant` 选择四种结构，与测速实验共用 `HighFrequencyCNN`。三级独立网络直接处理有符号、未归一化的 LH/HL/HH 拼接输入，不使用阈值 logits、幅值输入、band scale 或 atlas 预测。每次只对原始 LL 继续分解；LL CNN 和精修网络与 HF CNN 一起从头训练，固定 Haar 核不学习。

本次训练配置参考用户指定的 `static_hf_depth7_soft_ll_no_norm.yaml`：LL 宽度32、深度7（五个内部Rep-NCB），精修宽度32、五个Rep-NCB、S2D=2，LL不归一化。数据、损失（0.6 raw + 0.4 chromatic）、1000 epochs、batch size32、学习率0.0002、seed2026及验证/保存策略均沿用该参考配置。此前8.3节测速使用depth5 checkpoint，**不能将其参数量和时间直接视为这些depth7训练配置的实测结果**。

四份配置的 `resume`、`init_checkpoint` 均为 `null`，输出目录各自独立：

| 高频结构 | 训练配置 |
|---|---|
| DW1×1 + ReLU | [dw1](../configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth7_dw1_ll_no_norm.yaml) |
| DW3×3 + ReLU | [dw3](../configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth7_dw3_ll_no_norm.yaml) |
| DW3×3 + ReLU + PW1×1 | [dw3_pw1](../configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth7_dw3_pw1_ll_no_norm.yaml) |
| DW1×1 + ReLU + 输入残差 | [dw1_residual](../configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth7_dw1_residual_ll_no_norm.yaml) |

逐个选择配置手动启动，也可以在项目根目录依次运行四组：

```bash
for variant in dw1 dw3 dw3_pw1 dw1_residual; do
  conda run --no-capture-output -n LED-ICCV23 python train_sid_sony.py \
    --config "configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth7_${variant}_ll_no_norm.yaml" || break
done
```

这是完整随机初始化训练，不加载此前测速用的随机HF参数或旧checkpoint。结构仍保持8.1节的输出符号限制。新模型配置会随训练checkpoint保存，现有模型工厂可据此恢复训练权重或融合后的部署权重。配置中保留的 `dwt_width/dwt_depth/dwt_context` 及阈值设置不控制该模式下的小CNN，其结构由 `dwt_hf_cnn_variant` 决定。

已通过四组反向传播（HF/LL/精修梯度）、独立层参数、非整除尺寸补边、训练态/部署态输出一致性和两种checkpoint格式重载检查；相关回归测试共23项通过。没有启动数据集训练。
