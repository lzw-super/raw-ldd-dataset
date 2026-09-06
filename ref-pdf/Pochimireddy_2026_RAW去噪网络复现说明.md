# Pochimireddy 等（CVPR 2026）RAW 去噪网络复现说明

> 2026-09-02 补充：当前 `raw_image_denoising` 仓库已新增论文图 6 对齐的单帧版本，包含
> Bayer mosaic 域 `k=4` S2D/D2S，并将浅层 1×1 旁路改为直接连接 S2D 输出。独立训练配置为
> `configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml`；旧配置默认
> `space_to_depth_factor=1`，继续作为无 S2D/D2S 基线并兼容已有 checkpoint。本文下方原有
> LED 路径和算力统计主要描述历史无 S2D 版本，不能套用于新增版本。

新增版本的数据流为：

```text
[B,4,H,W] [R,G1,G2,B] packed RAW
  → 恢复 [B,1,2H,2W] Bayer mosaic
  → S2D(k=4): [B,16,H/2,W/2]
  → 3×3 Conv + 4×mRLFB + fusion
  → D2S(k=4)
  → [B,4,H,W] packed RAW
```

这与论文图 6 的 `h×w×1 → h/k×w/k×k²` 一致；不能直接在现有四通道 packed
张量上做 `PixelUnshuffle(4)`，否则会错误地产生 64 通道。论文对齐配置采用
`N=4,d=32,k=4`、batch 16、packed patch 256×256、Adam `1e-4`、无 warmup cosine
annealing 和 `0.6 Lraw + 0.4 Lchromatic`。论文未公开训练总步数和 cosine 终止学习率，
因此配置显式沿用 257600 step 的既有 LED 预算，并取 TensorFlow cosine 默认终值 0；
Adam epsilon 取 TensorFlow/Keras 默认的 `1e-7`。训练数据仍是本仓库 SID clean RAW 与
dark-frame 噪声合成，因为论文自建的 Set1/Set2 和手机实拍训练集未公开。

启动命令：

```bash
python train_sid_sony.py \
  --config configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml
```

## 1. 历史无 S2D/D2S 版本的实现范围

本节及后续原有统计描述此前的无 S2D/D2S 实现；新增论文对齐版本见文首补充。本历史实现复现论文 *Efficient Real-Time Raw-to-Raw Denoising for Extreme Low-Light Ultra HD Video on Mobile Devices* 的单帧基础网络、训练态结构重参数化模块，以及 RAW 重建/色差联合损失，并接入本仓库现有的 SID Sony A7S2 数据管线。

按照当前任务的训练和数据特点，以下部分不实现：

- 网络内部的 Space-to-Depth（S2D）与 Depth-to-Space（D2S）；
- 双帧输入和多帧卷积；
- 知识蒸馏、INT16 量化、分辨率重构及移动端打包部署；
- 论文自建的合成视频/实拍视频数据生成流程。

SID 数据加载器已经将 Bayer 图像打包成四通道 patch。这个既有的 `[R, G1, B, G2]` 数据表示仍然保留，但网络内部不再额外执行论文的 S2D/D2S。

## 2. 结构对应关系

论文图 6 的单帧结构在本实现中的映射如下：

| 论文结构 | 当前实现 | 说明 |
| --- | --- | --- |
| S2D | 省略 | 直接输入四通道 SID packed RAW |
| 浅层 3×3 Conv | `MRLFN.shallow_conv` | 4 通道映射到特征深度 `d` |
| `N` 个 mRLFB | `MRLFN.blocks` | 默认采用论文 Model A：`N=4, d=32` |
| 深层 3×3 Conv | `MRLFN.deep_fusion` | 聚合级联块输出 |
| 浅层 1×1 Conv | `MRLFN.shallow_fusion` | 保留浅层纹理信息 |
| Concat + 1×1 Conv | `MRLFN.output_conv` | 输出四通道 clean RAW |
| D2S | 省略 | 输出与输入 patch 同尺寸 |

单个 mRLFB 严格保留图 6 的两个块内跳连。令块输入为 `x`，三个重参数卷积的连续输出为 `f3`，则：

```text
f3  = RepConv3(RepConv2(RepConv1(x)))
mid = f3 + x
out = Conv1x1(mid) + x
```

这里没有额外添加输入 RAW 到最终 clean RAW 的全局残差；论文图中可见的残差连接位于 mRLFB 和特征融合路径内。

## 3. 结构重参数化

训练时，每个 mRLFB 中原本的 `3×3 Conv + ReLU` 使用论文图 7 的多分支模块替代：

```text
y = ReLU(x + Conv1x1_b(Conv3x3(x + Conv1x1_a(x))))
```

推理前将其解析融合为：

```text
y = ReLU(Conv3x3_fused(x))
```

设两个 1×1 权重分别为 `Wa`、`Wb`，中间 3×3 权重为 `W3`，则先把局部跳连写成 `A = I + Wa`，再进行卷积核复合，最后在中心位置加入全局恒等核：

```text
Kmid   = W3 ○ A
Kfused = Wb ○ Kmid + I_center
bfused = Wb · b3 + bb
```

其中 `○` 表示连续卷积的核合成。为了使零填充边界也能严格等价，`Conv1x1_a` 不使用 bias；位于 3×3 之后的 bias 可以精确融合。测试会同时检查边界像素，当前 float32 前向的最大误差处于 `1e-6` 量级。

网络提供两种转换接口：

- `network.deploy()`：返回已融合的副本，不破坏训练中的网络及优化器状态；
- `network.switch_to_deploy()`：原地删除训练分支并替换为单个 3×3 Conv。

`RAWImageDenoisingModel.save()` 已扩展为识别所有提供 `deploy()` 的网络。训练 checkpoint 会同时保存：

- `params`：训练态多分支权重，可继续训练；
- `params_deploy`：推理态单 3×3 权重，不能直接恢复训练分支。

加载 `params_deploy` 时必须使用 `network_g.deploy: true`，以保证模型结构和 state dict 的键一致。

## 4. Loss

论文的最终目标为：

```text
L = 0.6 * Lraw + 0.4 * Lchromatic
```

`Lraw` 是预测 packed RAW 与 GT 的 L1。色差项先根据 SID 的 `[R, G1, B, G2]` 顺序计算：

```text
Gavg = (G1 + G2) / 2
DBG  = B - Gavg
DRG  = R - Gavg
Lchromatic = L1(DBG_pred, DBG_gt) + L1(DRG_pred, DRG_gt)
```

实现类为 `RawReconstructionChromaticLoss`。为与仓库现有 loss 的数值尺度和 patch 尺寸解耦，`reduction: mean` 会先对每个样本的空间/通道元素取均值，再对 batch 取均值；两个色差分量分别取空间均值后相加。这样 256 与 1024 patch 不会仅因像素数变化而改变梯度尺度。

## 5. LED 多虚拟相机 + SID clean RAW 训练配置

训练入口：

```bash
python led/train.py -opt options/LED/pretrain/MRLFN_CVPR26_Paper_Setting.yaml
```

该配置参考 `options/LED/pretrain/CVPR20_ELD_Setting.yaml`，使用的不是 SID short/long 成对训练方式，而是 LED 的虚拟相机预训练方式：

```text
SID long-exposure clean RAW patch
        ↓ RAWGTDataset
随机曝光 ratio（100–300）
        ↓ VirtualNoisyPairGenerator
从 IC0–IC4 随机选择一台虚拟相机，在线生成 ptrqc noisy RAW
        ↓ MRLFN
预测 clean RAW，并计算重建/色差联合 loss
```

相关数据与噪声设置如下：

- clean 数据：`datasets/ICCV23-LED/Sony_train_long_patches`；
- 数据类型：`RAWGTDataset`，读取 SID 长曝光 clean RAW；
- patch：数据准备阶段已有 `4×512×512` packed-RAW patch，训练时由 `RAWGTDataset` 随机裁成论文使用的 `4×256×256`；
- ratio：每个样本在 `[100,300]` 内随机采样；
- 虚拟相机数量：5，对应 `IC0`–`IC4`；
- 噪声类型：`ptrqc`，包含 photon shot、Tukey-Lambda read、row、quantization 和 color-bias noise；
- 虚拟相机参数：从 `VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth` 加载；
- 每个 iteration 随机选择一个 `camera_id`，同一 batch 使用该虚拟相机生成 noisy/clean pair；
- MRLFN 本身没有 RepNR 的相机专属 CSA 分支，因此不同虚拟相机共同更新同一套 MRLFN 权重。

网络、训练和 loss 配置如下：

- 输入/输出：4 通道 packed RAW；
- 特征深度：`d=32`；
- mRLFB 数量：`N=4`；
- patch：随机裁剪的 `256×256` packed RAW；
- batch size：16（单卡配置，因此 global batch size 也是 16）；
- optimizer：Adam，初始学习率 `1e-4`；
- scheduler：单周期 `CosineAnnealingRestartLR`，初始学习率 `1e-4`、`eta_min=1e-7`；
- loss 权重：`wr=0.6, wc=0.4`；
- 总迭代数：257600。

训练启动时会通过 THOP 统计 `1×4×256×256` 输入：默认 `N=4,d=32` 的训练态为 151,908 参数、9.890 GMACs；融合部署态为 126,948 参数、8.280 GMACs（16.559 GFLOPs）。

当前配置已经对齐论文明确报告的 256×256 packed-RAW patch、batch size 16、Adam `1e-4`、cosine schedule 和联合 loss，同时保留 LED 的 SID clean RAW + 5 台虚拟相机在线噪声流程。论文没有报告总 epoch/iteration 和 cosine 最低学习率，因此 257600 iteration 沿用 LED 训练预算，`eta_min=1e-7` 是本实现显式补充的工程值。

如果实际 GPU 无法容纳 batch 16，可以临时减小单卡 batch；但这会偏离论文设置，必要时可使用梯度累积保持 effective batch size 16（当前训练器尚未实现梯度累积）：

```bash
python led/train.py \
  -opt options/LED/pretrain/MRLFN_CVPR26_Paper_Setting.yaml \
  --force_yml datasets:train:batch_size_per_gpu=8
```

论文 Model B 可通过把 `network_g:feature_channels` 改为 16 获得；当前默认不做教师蒸馏，只训练对应宽度的单模型。

## 6. 不同 N/d 配置的模型规模与计算量

### 6.1 论文配置与本节扩展配置

论文明确给出了以下两种主干宽度配置：

| 论文名称 | N | d | 论文中的其他处理 | 本复现中的对应关系 |
| --- | ---: | ---: | --- | --- |
| Model A / 单帧 Model A* | 4 | 32 | `k=4`，并作为蒸馏 teacher | 默认训练配置；本复现省略网络内部 S2D/D2S |
| Model B / 单帧 Model B* | 4 | 16 | `k=4`，由 Model A 蒸馏得到 | 可直接把 `feature_channels` 改为 16，但当前不执行蒸馏 |
| GT 生成大模型 | 16 | 未说明 | 用于消除 burst average 后的残余噪声 | 论文没有报告 `d`，不能唯一确定其参数量和计算量 |

Model AQ、Model BRQ 不会改变基础 `N/d` 的含义，但涉及量化或分辨率重构，不属于本次统计。为了展示深度与宽度的缩放规律，下面额外扫描 `N∈{2,4,8,16}`、`d∈{16,32}`。除 `N=4,d=16/32` 外，其余组合均是当前实现上的工程扩展，不应写成论文报告过的模型。

### 6.2 统计口径与解析公式

本节参数扫描与当前训练配置一致，统一采用 `1×4×256×256` packed RAW；该 patch 对应 `512×512` 个传感器 Bayer 样本。统计约定如下：

- 参数量包含卷积 weight 和 bias；
- MACs 只统计 `Conv2d` 乘加，与仓库 THOP 日志口径一致，不计 ReLU、逐元素加法和 Concat；
- `1 MAC = 2 FLOPs`，因此表中的 GFLOPs 是 GMACs 的两倍；
- 训练态包含图 7 的三个卷积分支，部署态已将每个重参数模块融合为单个 3×3 Conv；
- FP32 权重体积按 `部署参数量 × 4 Byte / 2^20` 计算，不包含框架、激活、优化器和临时 buffer；
- `MAC/RAW pixel` 的分母是 `512×512` 个传感器样本，而非 `256×256` 个 packed 像素；
- 理论最长感受野只考虑最长的 3×3 卷积路径。当前没有 S2D，故为 `(6N+5)×(6N+5)` packed 像素。

令 packed patch 高宽为 `H,W`，在当前四通道输入/输出、所有卷积 stride=1 的实现下，可以得到：

```text
P_train  = (34N + 10)d² + (7N + 47)d + 4
P_deploy = (28N + 10)d² + (4N + 47)d + 4

MAC_train(H,W)  = HW[(34N + 10)d² + 44d]
MAC_deploy(H,W) = HW[(28N + 10)d² + 44d]
```

这些公式已使用逐层 `Conv2d` forward hook 对下列全部配置进行核对，解析结果和实际执行图的卷积 MAC 计数完全一致。

### 6.3 参数量比较

| 配置来源 | N | d | 重参数模块数 `3N` | 训练态参数 | 部署态参数 | 参数下降 | 部署 FP32 权重 | 最长感受野（packed） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 扩展配置 | 2 | 16 | 6 | 20,948 | 17,780 | 15.1% | 0.068 MiB | 17×17 |
| 扩展配置 | 2 | 32 | 6 | 81,828 | 69,348 | 15.3% | 0.265 MiB | 17×17 |
| 论文 Model B* 同结构 | 4 | 16 | 12 | 38,580 | 32,244 | 16.4% | 0.123 MiB | 29×29 |
| 论文 Model A* 同结构（默认） | 4 | 32 | 12 | 151,908 | 126,948 | 16.4% | 0.484 MiB | 29×29 |
| 扩展配置 | 8 | 16 | 24 | 73,844 | 61,172 | 17.2% | 0.233 MiB | 53×53 |
| 扩展配置 | 8 | 32 | 24 | 292,068 | 242,148 | 17.1% | 0.924 MiB | 53×53 |
| 扩展配置（非论文 GT 模型定值） | 16 | 16 | 48 | 144,372 | 119,028 | 17.6% | 0.454 MiB | 101×101 |
| 扩展配置（非论文 GT 模型定值） | 16 | 32 | 48 | 572,388 | 472,548 | 17.4% | 1.803 MiB | 101×101 |

### 6.4 256×256 packed patch 计算量比较

| 配置来源 | N | d | 训练态 GMACs | 部署态 GMACs | 部署态 GFLOPs | MACs 下降 | 部署态 MAC/RAW pixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 扩展配置 | 2 | 16 | 1.355 | 1.153 | 2.307 | 14.9% | 4,400 |
| 扩展配置 | 2 | 32 | 5.327 | 4.521 | 9.043 | 15.1% | 17,248 |
| 论文 Model B* 同结构 | 4 | 16 | 2.496 | 2.093 | 4.186 | 16.1% | 7,984 |
| 论文 Model A* 同结构（默认） | 4 | 32 | 9.890 | 8.280 | 16.559 | 16.3% | 31,584 |
| 扩展配置 | 8 | 16 | 4.777 | 3.972 | 7.944 | 16.9% | 15,152 |
| 扩展配置 | 8 | 32 | 19.017 | 15.796 | 31.591 | 16.9% | 60,256 |
| 扩展配置（非论文 GT 模型定值） | 16 | 16 | 9.341 | 7.730 | 15.460 | 17.2% | 29,488 |
| 扩展配置（非论文 GT 模型定值） | 16 | 32 | 37.271 | 30.828 | 61.656 | 17.3% | 117,600 |

### 6.5 SID 整帧计算量外推

SID Sony A7S2 的完整网络输入为 `1×4×1424×2128` packed RAW，对应 `2848×4256` 传感器 RAW。因为当前网络全程保持空间分辨率，卷积 MACs 与 `H×W` 成正比。对论文两个主干配置的部署态外推如下：

| 对应结构 | 部署态参数 | 256² patch | SID 整帧 GMACs | SID 整帧 GFLOPs | 理想 30 fps 算术吞吐需求 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model B*：`N=4,d=16` | 32,244 | 2.093 GMACs | 96.775 | 193.550 | 5.806 TFLOP/s |
| Model A*：`N=4,d=32` | 126,948 | 8.280 GMACs | 382.832 | 765.665 | 22.970 TFLOP/s |

这里的整帧结果是当前“省略 S2D/D2S”版本的理论卷积量，不能与论文采用 `k=4` S2D、分辨率重构、FP16/INT16 和特定 NPU 后报告的手机延迟直接比较。它也不包含 tiled inference 的 overlap 重复计算；若使用 crop/merge，实际 MACs 还要乘以脚本报告的 `processed_pixel_ratio`。

### 6.6 配置选择结论

- `d` 是主要计算量杠杆。固定 `N=4` 时，从 `d=16` 增至 `d=32`，部署参数由 32,244 增至 126,948（约 3.94 倍），patch MACs 由 2.093 增至 8.280 GMACs（约 3.96 倍），符合卷积复杂度近似随 `d²` 增长的规律。
- `N` 基本呈线性增长。固定 `d=32` 时，从 `N=4` 增至 `N=8`，部署参数和 MACs 分别约增至 1.91 倍。
- 重参数化在这些配置上减少约 15%–17% 的参数和 MACs，更重要的收益是把每个训练模块的 `1×1→3×3→1×1` 与两次 skip 变成单个 3×3 Conv，从而减少中间特征读写和算子调度；真实延迟收益不能只由 MACs 下降比例推断。
- `N=4,d=16` 可用于快速验证训练流程，但不再作为质量导向的 FPGA/ASIC 首选；面向硬件的配置建议见 6.7。若优先贴近论文 Model A 的容量，则使用默认 `N=4,d=32`。`N=16` 更适合研究大感受野或教师/GT 生成模型，不适合作为本次无 S2D 的整帧实时默认值。
- 参数表不等于训练显存表。训练态还会保存多分支激活和反向梯度，实际峰值显存需在目标 batch、混合精度设置和 GPU 上实测。

### 6.7 面向 FPGA / ASIC 的 N 与 d 取舍

#### 6.7.1 先给结论

如果目标是**把部署态参数量控制在约 100–200K**，当前论文配置 `N=4,d=32` 已经满足要求：部署态为
126,948 参数，FP32/INT16/INT8 权重分别只占约 0.484/0.242/0.121 MiB。因此，仅从权重容量看，
不需要牺牲 `N` 或 `d`，应先保留该配置作为质量基线。

如果还需要进一步降低资源，则应按真正的瓶颈选择：

- 延迟、计算量、特征搬运或流水线级数受限：优先小幅减 `N`，保留 `d=32`；
- 片上 SRAM、单层特征图或并行乘加阵列受限：优先把 `d` 从 32 小幅降至 28 或 24；
- 不建议直接使用“很窄、很深”的 `d=16,N≥14` 来凑回相同参数量。它虽然减少单层特征宽度，却显著增加串行层数、累计特征读写和部署复杂度，并且 `d=16` 存在更明确的质量风险；
- 综合 RAW 去噪质量、特征 SRAM 和实时性，本实现建议的尝试顺序是 `N=4,d=32` → `N=5,d=28` → `N=6,d=24`。若首要目标是最低延迟，则改试 `N=3,d=32`。

#### 6.7.2 减 N 和减 d 分别牺牲什么

部署态参数的主项近似为 `28Nd²`，因此 `N` 是线性杠杆，`d` 是二次杠杆。但参数量相同并不表示
延迟、SRAM 和去噪能力相同：

| 影响项 | 减少 N，保留 d | 减少 d，保留 N |
| --- | --- | --- |
| 参数/MACs | 近似随 `N` 线性下降 | 近似随 `d²` 下降，压缩更快 |
| 理论感受野 | 每减一个 mRLFB，边长减少 6 个 packed 像素 | 不变 |
| 单层特征图 SRAM | 基本不变，仍与原 `d` 相同 | 随 `d` 线性下降 |
| 串行 3×3 层数 | 每减一个块减少 3 层 | 不变 |
| 通道表征/中间秩 | 保留 | 每一层都形成更窄的瓶颈 |
| 累计激活读写 | 随层数下降 | 每层下降，但若增加 N 补容量，可能反而上升 |
| 硬件并行度 | 通道阵列保持不变，较易复用现有设计 | DSP/PE 数可下降，但需注意 8/16 通道对齐 |

以默认 `N=4,d=32` 为基线，分别“去掉一个块”和“保留块数但减小宽度”，可以得到两组几乎
相同规模的受控比较：

| 配置 | 部署参数 | 256² GMACs | 感受野 | INT16 单特征图 | 3×3 串行层 | `N×d` 搬运代理 | FP32/FP16 实测中位延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 减 N：`N=3,d=32` | 98,148 | 6.401 | 23×23 | 4.0 MiB | 11 | 96 | 2.718/2.060 ms |
| 减 d：`N=4,d=28` | 97,416 | 6.349 | 29×29 | 3.5 MiB | 14 | 112 | 2.839/2.351 ms |

两者都比默认模型少约 23% 参数和 MACs。`N=3,d=32` 保留通道容量且串行层更少，适合延迟优先；
`N=4,d=28` 保留 29×29 感受野并把特征 SRAM 降低 12.5%，适合内存优先。上述 GPU 上 `d=28`
还会受到非 8/16 倍数通道对齐的影响；定制 FPGA 可以按 28 路设计 PE，但通用 NPU/DSP 阵列通常更偏好
`d=16/24/32`。

#### 6.7.3 对 RAW 去噪质量的影响

`d` 对 RAW 去噪不是普通的“隐藏层大小”。四通道 packed Bayer 输入同时包含不同色彩位置、亮度结构、
shot/read noise、行噪声和色偏。中间通道需要并行保存边缘方向、纹理频率、跨 Bayer 通道相关性及不同
噪声强度下的恢复线索。降低 `d` 会在所有层持续限制这些特征的维度；极暗区域中信号与噪声高度重叠，
过窄的通道瓶颈更容易产生过平滑、伪色或细节恢复不足。

`N` 主要增加非线性变换次数和空间上下文。当前最长理论感受野为 `(6N+5)²` packed 像素；由于一个
packed 像素对应 2×2 Bayer 样本，默认 `N=4` 的 29×29 packed 感受野大致覆盖 58×58 个传感器采样跨度。
更大的上下文有利于区分弱纹理与随机噪声，也可能改善 LED `ptrqc` 中的 row/color-bias 等结构噪声；但在
局部 shot/read noise 已被覆盖后，继续增加 `N` 通常出现边际收益，而串行延迟和数据搬运仍持续增加。

论文结果也提示不能过度压缩宽度：固定 `N=4` 时，Model A* (`d=32`) 在论文 Synthetic→Real
测试上的 PSNR/SSIM 为 60.54/0.9886；经蒸馏的 Model B* (`d=16`) 为 58.63/0.9880，即 PSNR
低 1.91 dB。该比较包含论文自己的数据、S2D 和蒸馏设置，不能直接当作当前 SID 实现的精确降幅；论文在
CRVD 上也出现 Model B* 高于 Model A* 的结果，说明训练和域匹配会影响排序。但它至少说明：即便加入蒸馏，
把 `d` 从 32 直接减半也不是无损操作。同一张论文表中，单帧运行时间只从 22.95 ms 降到 19.30 ms
（约 15.9%），也说明实际硬件收益不一定等于 `d²` 的理论计算缩放。因此本实现把 `d=24/28` 作为优先
候选，而不是先跳到 `d=16`。

#### 6.7.4 等参数预算配置实测

新增的 `scripts/profile_mrlfn_scaling.py` 会构造融合部署态网络，用 THOP 统计 `1×4×256×256` 输入的
MACs，并测量推理延迟和 CUDA 峰值临时分配：

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/zhengwu/anaconda3/envs/LED-ICCV23/bin/python \
  scripts/profile_mrlfn_scaling.py --device cuda:0 --warmup 20 --repeats 100
```

受控配置和推荐候选可以显式指定：

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/zhengwu/anaconda3/envs/LED-ICCV23/bin/python \
  scripts/profile_mrlfn_scaling.py \
  --configs 3x32 4x28 4x32 5x28 6x24 --device cuda:0
```

本次实测环境为 RTX 2080 Ti、PyTorch 1.13.1+cu117、batch 1。每个延迟值经过 20 次预热并取 100 次
CUDA Event 测量的中位数。网络使用随机权重，但卷积执行图、参数量、MACs、激活形状和运行时间不依赖
训练质量。`峰值` 是相对已常驻模型权重与输入的额外 CUDA allocated memory，包含 PyTorch/cuDNN 临时
张量，不能直接等同于 FPGA/ASIC SRAM；它只用于观察配置间趋势。

| 预算组 | N | d | 部署参数 | GMACs | 感受野 | INT16 权重 | INT16 单特征图 | `N×d` | FP32 中位延迟 | FP16 中位延迟 | FP16 峰值增量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ≈100K | 3 | 32 | 98,148 | 6.401 | 23×23 | 191.7 KiB | 4.0 MiB | 96 | 2.714 ms | 2.080 ms | 20.0 MiB |
| ≈100K | 6 | 24 | 104,236 | 6.788 | 41×41 | 203.6 KiB | 3.0 MiB | 144 | 3.824 ms | 3.814 ms | 18.0 MiB |
| ≈100K | 14 | 16 | 104,564 | 6.791 | 89×89 | 204.2 KiB | 2.0 MiB | 224 | 6.126 ms | 7.365 ms | 10.0 MiB |
| ≈125K | 4 | 32 | 126,948 | 8.280 | 29×29 | 247.9 KiB | 4.0 MiB | 128 | 3.103 ms | 2.914 ms | 20.0 MiB |
| ≈125K | 8 | 24 | 136,684 | 8.902 | 53×53 | 267.0 KiB | 3.0 MiB | 192 | 5.261 ms | 5.560 ms | 18.0 MiB |
| ≈125K | 16 | 16 | 119,028 | 7.730 | 101×101 | 232.5 KiB | 2.0 MiB | 256 | 8.986 ms | 10.633 ms | 10.0 MiB |
| ≈175K | 6 | 32 | 184,548 | 12.038 | 41×41 | 360.4 KiB | 4.0 MiB | 192 | 4.422 ms | 4.077 ms | 20.0 MiB |
| ≈175K | 10 | 24 | 169,132 | 11.016 | 65×65 | 330.3 KiB | 3.0 MiB | 240 | 6.595 ms | 5.918 ms | 18.0 MiB |
| ≈175K | 24 | 16 | 176,884 | 11.488 | 149×149 | 345.5 KiB | 2.0 MiB | 384 | 11.145 ms | 14.151 ms | 10.0 MiB |

同一预算组内 MACs 和权重接近，但“窄而深”明显更慢。例如约 100K 组中，`N=14,d=16` 相比
`N=3,d=32` 的 FP32/FP16 延迟分别约为 2.26/3.54 倍，`N×d` 搬运代理也从 96 增至 224。
其原因是深层依赖不能跨层并行、算子启动更多、每层都要重新读写特征，而且窄卷积不一定能充分利用计算阵列。
FPGA/ASIC 的绝对延迟不会等于 GPU 结果，但复用同一卷积阵列时仍会面对相同的串行层数和数据流问题；若把
所有层完全展开，则更多的 `N` 会改为消耗更多 PE、流水级和 line buffer，而不是免费获得大感受野。

#### 6.7.5 权重很小，真正的内存瓶颈是激活

100–200K 参数对应的权重体积约为：FP32 0.381–0.763 MiB、INT16 0.191–0.381 MiB、INT8
0.095–0.191 MiB，通常可以全部放进 FPGA BRAM/URAM 或 ASIC SRAM。相比之下，一个 INT16 中间特征图为：

| d | 256×256 tile 单特征图 | SID 1424×2128 packed 整帧单特征图 |
| ---: | ---: | ---: |
| 32 | 4.0 MiB | 184.953 MiB |
| 28 | 3.5 MiB | 161.834 MiB |
| 24 | 3.0 MiB | 138.715 MiB |
| 16 | 2.0 MiB | 92.477 MiB |

实际执行还需要输入、输出、残差保留、双缓冲或卷积 workspace。因此，即使网络只有 100K 参数，不做切块也
不代表能把 SID 整帧特征放进片上 SRAM。当前实现为训练方便省略了论文的 S2D/D2S；若目标真正转向 FPGA/
ASIC 整帧部署，至少需要以下两条路线之一：

1. 使用带 halo 的 tile/streaming 推理，把中间特征限制在小 tile 或若干行，并避免物化最终 Concat；
2. 恢复论文的 S2D/D2S 或等效的分辨率重构，在低空间分辨率上运行主干。

对完全展开的流式 3×3 流水线，每层仅考虑两行 INT16 line buffer 的粗略下界为
`2×W×d×2 Byte`；部署态共有 `3N+2` 个 3×3 层，所以总 line-buffer 项近似与
`(3N+2)d` 成正比。在 256 宽 tile 上，`N=3,d=32`、`N=6,d=24`、`N=14,d=16` 的该项约为
352、480、704 KiB，尚未计入块残差延迟和边界缓存。这再次说明：增加大量 `N` 来补偿较小的 `d`，未必会
降低完全展开硬件的总 SRAM；若复用单个卷积引擎，则 line buffer 可复用，但中间特征的外存流量和总周期数会上升。

#### 6.7.6 推荐配置与验证顺序

| 使用目标 | 推荐 N/d | 部署参数 | 选择理由 | 主要风险 |
| --- | --- | ---: | --- | --- |
| 论文质量基线/100–200K 首选 | `N=4,d=32` | 126,948 | 已训练、宽度和感受野均保留，权重已足够小 | 单特征图 SRAM仍为 d=32 |
| 最低延迟、接近 100K | `N=3,d=32` | 98,148 | 保留通道容量，串行层最少；虽略低于 100K，但更利于 16/32 通道阵列 | 感受野降至 23×23 |
| 轻度降低 SRAM 的平衡点 | `N=5,d=28` | 119,480 | 特征图降低 12.5%，感受野升至 35×35；实测 FP32/FP16 为 3.450/2.770 ms | 28 通道对部分硬件不对齐 |
| SRAM 更紧的平衡点 | `N=6,d=24` | 104,236 | 特征图降低 25%，仍保留 d=16 以上容量，感受野 41×41 | 串行延迟和搬运高于默认模型 |
| 质量优先且允许接近 200K | `N=6,d=32` | 184,548 | 保留 d=32 并扩大上下文，仍在部署参数预算内 | MACs 和延迟明显增加 |
| 不建议作为第一版 | `N≥14,d=16` | 约 100–200K | 单层特征图最小、感受野很大 | 通道瓶颈、串行延迟、累计搬运和质量风险均较高 |

以上实验是结构/资源实验，不能替代 SID 上重新训练后的 PSNR/SSIM。不同 `N/d` 的 state dict 尺寸不同，
不能直接加载当前 `N=4,d=32` 权重作公平质量比较。建议优先完整训练 `N=3,d=32`、`N=4,d=28`、
`N=5,d=28` 和 `N=6,d=24`，并保持相同数据、随机种子、iteration、cosine schedule 和验证集。可以通过
现有配置直接覆盖网络尺寸，例如：

```bash
python led/train.py \
  -opt options/LED/pretrain/MRLFN_CVPR26_Paper_Setting.yaml \
  --force_yml name=LED_Pretrain_MRLFN_N5_D28 \
              network_g:num_blocks=5 \
              network_g:feature_channels=28
```

最终选择应使用目标 FPGA/ASIC 的 INT16/INT8 综合结果、tile 大小、片上 SRAM、外存带宽和时钟频率重新测量；
GPU 排名只用于识别串行深度与通道宽度的趋势，不应换算成硬件帧率。

## 7. 部署态验证

训练保存的 `params_deploy` 可用部署态配置验证：

```bash
python led/test.py \
  -opt options/SID/MRLFN_SonyA7S2_deploy.yaml \
  --force_yml path:pretrain_network_g=experiments/LED_Pretrain_MRLFN_CVPR26_Paper_Setting/models/net_g_latest.pth
```

部署配置只包含 SID 验证集，不会把训练集误当成测试集遍历；其 `param_key_g` 已设为 `params_deploy`。

### 7.1 自采 RAW / BIN 可视化

`quick_op/08_visualize_mrlfn_rgb.sh` 是面向 MRLFN 的自采数据测试入口。它默认读取
`experiments/LED_Pretrain_MRLFN_CVPR26_Paper_Setting/models/net_g_latest.pth`，先通过
`scripts/prepare_led_deploy_checkpoint.py` 融合所有训练态重参数分支，再调用既有
`quick_op/05_visualize_rgb.sh` 处理 `test_data_mine` 下的相机 RAW 容器、无头 `.raw` 和 BIN。

```bash
bash quick_op/08_visualize_mrlfn_rgb.sh
```

默认相机 RAW / 无头 `.raw` 数字增益为 `RATIO=100`，对应训练 exposure ratio 的下界；BIN
仍使用 `BIN_RATIO=2`，可根据实际采集曝光覆盖。例如：

```bash
CUDA_ID=0 RATIO=200 BIN_RATIO=4 bash quick_op/08_visualize_mrlfn_rgb.sh
```

默认生成的融合 checkpoint 位于 `tmp/led_deploy_checkpoints/`，结果位于
`inference/qualitative/net_g_latest_mrlfn_test_data_mine/`。每个 `.comparison.png` 均左侧为
输入 RAW 的可视化，右侧为融合部署态 MRLFN 的输出。

### 7.2 原始全分辨率 SID crop–merge 实测

#### 7.2.1 测试设置

本节复用提交 `ae54f1531099fa01e64530a16da6431a29a7310a` 中的 overlap-crop（halo exchange）
实现。它不是直接把无重叠 patch 拼起来，而是让相邻输入 tile 共享 `overlap` 个 packed 像素；每个内部边缘
获得 `overlap/2` 的上下文，网络输出后丢弃这部分 halo，仅把可靠中心区写入最终输出。

评测直接从 SID Sony 的原始 short/long ARW 解码，不使用训练用 256 patch，也不先把验证图裁小：

- 验证图：100× 29 张、250× 29 张、300× 35 张，共 93 对；
- 网络输入：`1×4×1424×2128` packed RAW；
- 传感器原始分辨率：`2848×4256`；
- checkpoint：`experiments/LED_Pretrain_MRLFN_CVPR26_Paper_Setting/models/net_g_latest.pth`；
- checkpoint 中的训练态 `params` 在加载后自动融合成 126,948 参数的部署图；
- 模式：`crop`，扫描 `overlap={80,100,128}`；
- tile：512 为主要部署候选，同时补测训练 patch 尺寸 256；
- MRLFN 没有下采样或 pooling phase 约束，因此 `alignment=1`；
- FP32、batch 1、RTX 2080 Ti；全图 merge 后统一进行 illumination correction，PSNR crop border 为 2。

复现实验命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/zhengwu/anaconda3/envs/LED-ICCV23/bin/python \
  scripts/evaluate_crop_merge_sid.py \
  -p experiments/LED_Pretrain_MRLFN_CVPR26_Paper_Setting/models/net_g_latest.pth \
  -opt options/base/network_g/mrlfn.yaml \
  --dataset-options options/base/dataset/test/SID_SonyA7S2_val_split.yaml \
  --dataset-root /home/shared_files/dataset/SID/Sony \
  --input-format raw \
  --ratios 100 250 300 \
  --tile-sizes 256 512 \
  --overlaps 80 100 128 \
  --methods crop \
  --alignment 1 \
  --device cuda:0 \
  --param-key params \
  --output-dir results/mrlfn_crop_merge_sid_raw_fullres_overlap_80_100_128
```

逐图结果保存在 `details.csv`，各倍率和总体汇总在 `summary.csv`，完整配置与元数据在 `report.json`。

#### 7.2.2 实际速度、显存和精度

整帧部署态基线平均为 382.832 GMACs、0.197 s/图、1914.73 MiB CUDA 峰值显存。六组 crop 配置的
93 张总体均值如下：

| Tile/overlap | Halo | Tile 数 | 实际计算倍率 | GMACs/图 | 时间/图 | 相对整帧时间 | 峰值显存 | 显存下降 | PSNR 损失 | 输出相对整帧 PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256/80 | 40 | 117 | 2.063× | 789.671 | 0.482 s | 2.447× | 41.87 MiB | 97.81% | 1.14e-9 dB | 177.63 dB |
| 256/100 | 50 | 140 | 2.595× | 993.485 | 0.592 s | 3.005× | 41.87 MiB | 97.81% | 1.08e-9 dB | 176.06 dB |
| 256/128 | 64 | 204 | 3.837× | 1468.780 | 0.864 s | 4.385× | 41.87 MiB | 97.81% | -1.87e-9 dB | 169.16 dB |
| **512/80** | **40** | **20** | **1.344×** | **514.626** | **0.229 s** | **1.163×** | **165.99 MiB** | **91.33%** | **3.80e-9 dB** | **168.67 dB** |
| 512/100 | 50 | 24 | 1.495× | 572.387 | 0.257 s | 1.306× | 165.99 MiB | 91.33% | 3.04e-11 dB | 166.95 dB |
| 512/128 | 64 | 24 | 1.652× | 632.254 | 0.273 s | 1.383× | 165.99 MiB | 91.33% | 5.98e-10 dB | 171.58 dB |

这里的 `时间/图` 是当前评测脚本的稳态网络路径均值：tile 在 GPU 上顺序执行，每个 tile 输出回传 CPU，
再写入 CPU 端的整图 buffer；不含 RAW 解码和数据加载，但包含 tile 的 GPU→CPU 回传与 merge。它适合比较
同一软件实现中的配置，不等同于 RTL 的纯卷积时延；硬件可将可靠中心区直接写回片上 SRAM/DDR，并用双缓冲
隐藏部分搬运时间。折算后整帧约 5.08 fps，512/80 约 4.36 fps，256/80 约 2.07 fps。

`PSNR 损失` 是 tiled PSNR 相对整帧 PSNR 的差；微小负值表示浮点计算次序使 tiled 结果高出约
`10^-9 dB`，不代表真实质量提升。六组输出相对整帧的平均 MAE 位于 `1.08e-10`–`1.23e-9`，所有
93×6 次推理的最大绝对误差不超过 `1.0133e-6`。因此三种 overlap 均没有可测的 RAW 去噪质量损失。

最佳速度/资源平衡是 **512/80**：相比整帧只慢 16.3%，显存降低约 11.53 倍；若必须把软件工作集继续
压小，则 256/80 将显存降低约 45.73 倍，但 117 次小 tile 调度和 2.063× 重复计算使其慢至整帧的
2.45 倍。增大 overlap 没有继续改善精度，只增加重复计算，所以在本次扫描范围内应选择 80。

512/80 在不同曝光倍率上的精度如下，说明总体结论不是由某一个 ratio 主导：

| SID ratio | 图像数 | Noisy PSNR | 整帧 MRLFN PSNR | 512/80 PSNR | PSNR 损失 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100× | 29 | 30.2019 dB | 40.6921 dB | 40.6921 dB | 1.06e-9 dB |
| 250× | 29 | 26.6139 dB | 37.9968 dB | 37.9968 dB | 3.26e-9 dB |
| 300× | 35 | 25.3580 dB | 35.5894 dB | 35.5894 dB | 6.52e-9 dB |
| 总体 | 93 | 27.2601 dB | 37.9312 dB | 37.9312 dB | 3.80e-9 dB |

#### 7.2.3 为什么 overlap 80 已经足够

当前 `N=4` 部署图的最长感受野为 29×29 packed 像素，半径为 14。只要内部拼接边界两侧各有至少
14 像素上下文，crop 输出在数学上就与整帧相同，因此理论最小 overlap 是 28。扫描中的 overlap 80/100/128
分别提供 40/50/64 像素 halo，均远大于 14；实测的 `10^-6` 最大绝对误差来自 cuDNN 对不同张量形状选择的
浮点卷积路径，而非 tile 接缝。

一般地，当前 MRLFN 的感受野半径为 `3N+2`，保证完整上下文的理论 overlap 下界为 `6N+4`。因此：

| N | 理论最小 overlap | overlap 80 是否覆盖 |
| ---: | ---: | --- |
| 4 | 28 | 是 |
| 5 | 34 | 是 |
| 6 | 40 | 是 |
| 12 | 76 | 是 |
| 14 | 88 | 否，应使用至少 88/96/100 |
| 16 | 100 | 恰好需要 100 |

这也说明 6.7 中推荐的 `N=5,d=28` 和 `N=6,d=24` 可以继续使用 overlap 80；若改成很深的
`N≥14`，必须重新扫描 overlap。对当前 `N=4`，7.3 已进一步在完整数据集上验证硬件友好的 overlap 32
（halo 16）可以保持等价，而 overlap 24 会产生接缝误差；因此工程下限更新为 32。

#### 7.2.4 FPGA / ASIC 实现可行性

crop 模式对硬件很友好：它没有 Hann 权重、除法或重叠累加，merge 只需要根据 tile 坐标丢弃 halo，然后把
中心矩形直接写入输出地址。MRLFN 部署图也只含普通 3×3/1×1 Conv、ReLU、局部加法和 Concat，没有全局
attention、normalization 或 pooling；因此 tile 之间没有隐藏的全局依赖，功能等价性已经由 93 张整图验证。

内存方面具有较高可行性：

- 126,948 个 INT16 权重约 248 KiB，可完整驻留片上；
- 256/512 tile 的 PyTorch 峰值显存不是 RTL SRAM 需求，硬件可以通过逐行卷积和 buffer 复用大幅降低；
- 若把 14 个部署态 3×3 层完全流水展开，仅按每层两行 INT16、`d=32` 估算，256/512 宽的 line-buffer
  下界约为 448/896 KiB，另需残差延迟、输入输出 FIFO 和边界缓存；
- 若复用一个卷积引擎并在层间保存完整特征，一个 `d=32` INT16 feature map 在 256/512 tile 下为
  4/16 MiB，ping-pong 至少约 8/32 MiB。256 tile 更可能完全放入高端 FPGA BRAM/URAM；512 tile 通常需要
  更积极的 strip streaming、层融合或外部 DDR。ASIC 可以按目标面积配置更大的片上 SRAM。

但 crop 只解决工作集，不会降低总算术量，overlap 还会增加计算。512/80 为 514.626 GMACs/帧，若要求
30 fps，需要约 15.44 TMAC/s，即按 `1 MAC=2 ops` 计约 30.88 TOPS 的持续有效吞吐；256/80 则需要
约 23.69 TMAC/s（47.38 TOPS）。这还没有计入 DMA、边界调度和利用率损失。因此：

- **FPGA 功能实现可行，低帧率/离线实现可行；当前 N=4,d=32、无 S2D 的 UHD 30 fps 对单片 FPGA 很激进。**
  需要大量 DSP、较高频率和高利用率，普通中端 FPGA 很难仅靠 crop 达到实时；
- **ASIC/NPU 可行性高于 FPGA，但仍需约 31 TOPS 的有效吞吐。** 需要留出利用率和带宽余量，标称 TOPS
  不能直接当成持续性能；
- 若目标是实时，应把 crop 与 INT16/INT8、蒸馏/缩宽、恢复 S2D 或等效低分辨率主干结合。以当前公式估算，
  `N=4,d=16` 的算术量约为 `d=32` 的四分之一，但需重新训练并验证前述质量风险；
- 可以用双缓冲让下一 tile 的 DMA 与当前 tile 计算重叠，并缓存相邻 tile 的输入 halo，减少外存重复读取；
  overlap 区域的网络计算仍会重复，除非进一步改为全宽 strip/line streaming；
- 对 16 通道/像素并行的数据通路，overlap 80 和 128 均为 16 的整数倍；`tile=512, overlap=80` 的有效中心
  宽度 432 也是 16 的整数倍。overlap 100 的中心宽度 412 不满足这一对齐，通常需要 padding 或尾部 mask；
- 边缘 tile 在当前软件中可以是较小形状。固定尺寸硬件建议把边缘补齐到 256/512，再用坐标 mask 丢弃无效输出；
- illumination correction 是 merge 后的全图标量归约。若产品链路需要它，应增加一次 reduction/scale pass；
  它不能在每个 tile 内独立计算，否则会产生不同亮度系数和接缝。

综合判断：**crop-merge 已经证明可以把高分辨率 MRLFN 的内存问题转化为可控 tile 工作集，而且在 overlap 80
下没有精度代价；但要实现 FPGA/ASIC 实时，下一优先级应是降低每像素 MACs 和设计流式数据通路，而不是继续
增大 overlap。** GPU 软件仍推荐 `tile=512, overlap=80`；片上 SRAM 更紧的硬件原型则根据 7.3 的结果优先
验证 `tile=128, overlap=32`，极限 SRAM 场景再使用 `tile=80, overlap=32`。

### 7.3 80/100/128 小 tile 与低 overlap 扫描

#### 7.3.1 扫描设计

为了进一步压缩工作集，本轮保持上一轮三档 overlap 比例约 31.25%、39.1%、50%，将它们缩放到
`tile={80,100,128}`，并取 crop 模式需要的相邻偶数：

| Tile | 第一档 | 第二档 | 第三档 |
| ---: | ---: | ---: | ---: |
| 80 | 24（30.0%） | 32（40.0%） | 40（50.0%） |
| 100 | 32（32.0%） | 40（40.0%） | 50（50.0%） |
| 128 | 40（31.25%） | 50（39.06%） | 64（50.0%） |

注意，比例只适合生成扫描候选，不能代替感受野约束。当前 `N=4` 的理论最小 overlap 是 28；因此 80/24
的 halo 只有 12，低于感受野半径 14，预期会发生接缝误差。其余组合至少为 overlap 32。主扫描仍使用
全部 93 张 `2848×4256` 传感器 RAW（packed 输入 `1424×2128`），命令如下：

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/zhengwu/anaconda3/envs/LED-ICCV23/bin/python \
  scripts/evaluate_crop_merge_sid.py \
  -p experiments/LED_Pretrain_MRLFN_CVPR26_Paper_Setting/models/net_g_latest.pth \
  -opt options/base/network_g/mrlfn.yaml \
  --dataset-options options/base/dataset/test/SID_SonyA7S2_val_split.yaml \
  --dataset-root /home/shared_files/dataset/SID/Sony \
  --input-format raw \
  --ratios 100 250 300 \
  --tile-overlap-pairs \
    80:24 80:32 80:40 \
    100:32 100:40 100:50 \
    128:40 128:50 128:64 \
  --methods crop \
  --alignment 1 \
  --device cuda:0 \
  --param-key params \
  --output-dir results/mrlfn_crop_merge_sid_raw_fullres_small_tiles
```

评测脚本新增了 `--tile-overlap-pairs TILE:OVERLAP [...]`，用于表达这种非笛卡尔积扫描，避免重复执行整帧
基线。由于所有安全配置都表明 overlap 32 已经足够，另对未包含在比例扫描中的 128/32 完成了同样的 93 张
补充实验，结果位于 `results/mrlfn_crop_merge_sid_raw_fullres_small_tiles_128_overlap32/`。

#### 7.3.2 全量实测结果

下面均为 93 张平均；显存倍率以 1914.73 MiB 整帧峰值为基准。粗体的 128/32 是补充的低 overlap 点，
80/24 因违反感受野下界标为不安全。

| Tile/overlap | Halo / core | Tile 数 | 计算倍率 | GMACs/图 | 时间/图 | 峰值显存 | 显存缩小 | PSNR 损失 | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80/24（不安全） | 12 / 56 | 988 | 2.014× | 771.203 | 3.730 s | 5.35 MiB | 357.89× | **2.521e-3 dB** | **2.609e-2** |
| 80/32 | 16 / 48 | 1350 | 2.745× | 1050.695 | 5.054 s | 5.35 MiB | 357.89× | 6.50e-8 dB | 9.537e-7 |
| 80/40 | 20 / 40 | 1944 | 3.948× | 1511.290 | 7.297 s | 5.35 MiB | 357.89× | 4.56e-8 dB | 9.537e-7 |
| 100/32 | 16 / 68 | 672 | 2.125× | 813.563 | 2.550 s | 6.75 MiB | 283.75× | 2.86e-9 dB | 4.768e-7 |
| 100/40 | 20 / 60 | 864 | 2.729× | 1044.752 | 3.237 s | 6.75 MiB | 283.75× | -4.54e-8 dB | 7.153e-7 |
| 100/50 | 25 / 50 | 1247 | 3.939× | 1507.902 | 4.630 s | 6.75 MiB | 283.75× | -2.02e-8 dB | 8.345e-7 |
| **128/32（补测、推荐）** | **16 / 96** | **345** | **1.750×** | **669.771** | **1.464 s** | **10.84 MiB** | **176.69×** | **-8.35e-8 dB** | **4.768e-7** |
| 128/40 | 20 / 88 | 425 | 2.097× | 802.618 | 1.626 s | 10.84 MiB | 176.69× | -8.35e-8 dB | 4.768e-7 |
| 128/50 | 25 / 78 | 532 | 2.659× | 1018.082 | 2.032 s | 10.84 MiB | 176.69× | 1.94e-8 dB | 8.345e-7 |
| 128/64 | 32 / 64 | 782 | 3.925× | 1502.739 | 2.954 s | 10.84 MiB | 176.69× | -8.35e-8 dB | 4.768e-7 |

负的 `PSNR 损失` 仍只是浮点计算次序造成的微小正向变化。除 80/24 外，八个比例扫描配置和补测的
128/32 的最大绝对误差均不超过 `9.537e-7`，相对整帧 PSNR 均高于 155 dB，可以视为完全等价。

80/24 则不能用平均值掩盖问题：总体平均损失为 0.002521 dB，但单图 PSNR 变化位于约
`-0.0254` 至 `+0.0422` dB，最大像素误差达到 0.0261，相对整帧 PSNR 只有 72.96 dB。按倍率看，
100× 平均变化为 +0.001526 dB，250×/300× 平均损失分别为 0.004570/0.004176 dB；正负抵消恰好说明
这是随图像内容变化的接缝误差，而不是真实质量增益。因此 **当前 N=4 部署图不得使用 overlap 24，工程下限
应取对齐友好的 overlap 32。**

#### 7.3.3 性能与硬件结论

极小 tile 显著降低显存，但顺序 PyTorch 推理被大量小 kernel、GPU→CPU 回传和 merge 调度主导：

- 128/32 相对同次评测的整帧慢 7.18 倍；100/32 慢 12.61 倍；80/32 慢 25.00 倍；
- 128/32 比按比例得到的 128/40 少 80 个 tile，GMACs 下降 16.6%，实测时间下降约 10.0%；
- 与上一轮 256/80 相比，128/40 的 GMACs 很接近（802.618 对 789.671），但 425 次而非 117 次网络调用
  使软件时延达到 1.626 s 而不是 0.482 s；
- 因此 GPU/桌面软件仍推荐 512/80；若必须把峰值工作集压到约 11 MiB，本轮推荐 128/32；只有在 SRAM
  极端紧张时才考虑 80/32，其 5.35 MiB PyTorch 峰值伴随 2.745× 重复计算和极差的软件吞吐。

以 INT16、14 个 3×3 层完全流水展开的两行 line-buffer 下界估算，80/100/128 tile 宽度分别约需
140/175/224 KiB；再加约 248 KiB 权重、残差延迟和 FIFO。若复用卷积引擎并保存完整 `d=32` feature map，
单张 80/100/128 tile 约为 400/625/1024 KiB，ping-pong 约为 0.78/1.22/2.00 MiB。这说明小 tile 在
FPGA/ASIC 的片上存储方面确实可行，但算术量仍是瓶颈：

- 128/32 在 30 fps 下约需 40.19 TOPS；
- 100/32 约需 48.81 TOPS；
- 80/32 约需 63.04 TOPS；
- 对比 512/80 的约 30.88 TOPS，小 tile 实际让 UHD 30 fps 的 FPGA 吞吐目标更困难。

所以本轮进一步确认：**tile 大小由片上存储上限决定，overlap 则应由感受野下限决定，两者不应保持固定比例。**
面向 FPGA，可用 80/32 验证极小 SRAM 数据通路，但当前已验证的平衡点是 128/32，并可把 256/32 作为
下一步待验证点；面向 ASIC，还应通过更大 tile、strip/line streaming 和 halo
缓存减少重复计算，再结合缩小 N/d、量化和 S2D 降低 TOPS。

## 8. 文件清单

当前 `raw_image_denoising` 仓库新增：

- `models/mrlfn_arch.py`：包含 packed Bayer ↔ mosaic、`k=4` S2D/D2S 和论文图 6 浅层旁路；
- `configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml`：论文对齐版独立训练配置；
- `train_sid_sony.py`：支持 `space_to_depth_factor`、精确 `max_steps`、cosine 终值和 Adam epsilon；
- `tests/test_mrlfn.py`：覆盖 S2D/D2S 可逆性、主干形状、融合等价性、checkpoint 恢复和 scheduler。

以下是历史 LED 仓库文件：

- `led/archs/mrlfn_arch.py`：MRLFN、mRLFB、可重参数 3×3 模块；
- `led/losses/mrlfn_loss.py`：RAW 重建与色差联合 loss；
- `options/base/network_g/mrlfn.yaml`：网络基础配置；
- `options/LED/pretrain/MRLFN_CVPR26_Paper_Setting.yaml`：SID clean RAW + 5 台虚拟相机的论文规格预训练配置；
- `options/SID/SID_SonyA7S2_CVPR26_MRLFN_Setting.yaml`：可选的真实 SID short/long 成对训练配置；
- `options/SID/MRLFN_SonyA7S2_deploy.yaml`：融合权重验证配置；
- `quick_op/08_visualize_mrlfn_rgb.sh`：自采 RAW / `.raw` / BIN 的 MRLFN 融合部署态可视化脚本；
- `scripts/profile_mrlfn_scaling.py`：面向 FPGA/ASIC 的等参数量 N/d、MACs、激活内存与 CUDA 延迟扫描；
- `scripts/evaluate_crop_merge_sid.py`：原始全分辨率 SID 的整帧/crop 精度、MACs、速度与显存评测，支持显式
  `TILE:OVERLAP` 配对扫描；
- `tests/test_evaluate_crop_merge_sid.py`：MRLFN 训练权重自动部署融合及 crop 等价性测试；
- `tests/test_mrlfn.py`：结构、loss、严格加载与融合等价性测试。

## 9. 已知差异与后续工作

新增配置已经补齐单帧基础网络的 S2D/D2S，但当前结果仍应理解为论文基础网络在 SID 单图
patch 训练上的结构复现，而非论文完整 UHD 视频移动端指标复现。论文自建 Set1/Set2、手机实拍
微调数据和完整训练预算未公开；双帧输入、空间分辨率 restructuring、Model A→B 蒸馏、INT16
量化校准及对应 NPU 算子评测也仍未实现。这些步骤会改变质量、延迟和功耗，不能由当前 SID
单帧对照实验直接外推。
