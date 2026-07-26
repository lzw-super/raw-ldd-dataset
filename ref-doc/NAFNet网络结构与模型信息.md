# NAFNet 网络结构、设计理念与模型信息

## 1. 名称与参考实现

本文描述用户所称的 “NATNet”。指定参考文件中的正式类名和论文通用名称均为
**NAFNet**，因此代码主类保持为 `NAFNet`；本仓库同时提供 `NATNet` 和
`NatNet` 别名，便于兼容不同拼写。

参考源文件：

```text
/home/zhengwu/Desktop/graduate-workspace/LED-mine/led/archs/nafnet_arch.py
```

本仓库独立实现：

```text
raw_image_denoising/models/natnet_arch.py
```

迁移时移除了 `led.utils.registry.ARCH_REGISTRY` 框架依赖，保留了原模型的
模块名称、参数名称和前向计算。使用相同权重对随机的
`[1, 4, 65, 79]` 输入进行验证，新旧实现输出最大绝对误差为 `0.0`，
340 个 state-dict key 全部兼容。

## 2. 本文统计的模型配置

源文件构造函数的字面默认值为 `img_channel=3`、`width=16`、一个 middle
block，且 encoder/decoder block 列表为空；这只是可实例化的接口默认值，
不是 LED 在 SID 上实际使用的完整网络。

本文所说的原始 **NAFNet32** 使用 LED 配置文件
`options/base/network_g/nafnet.yaml` 中的参数：

```python
NAFNet(
    img_channel=4,
    width=32,
    enc_blk_nums=(2, 2, 2, 2),
    middle_blk_num=2,
    dec_blk_nums=(2, 2, 2, 2),
)
```

- 输入和输出都是 4 通道 2×2 Bayer packed RAW；
- 基础通道数为 32；
- 4 个编码层级，每级 2 个 NAFBlock；
- bottleneck 包含 2 个 NAFBlock；
- 4 个解码层级，每级 2 个 NAFBlock；
- 总计 18 个 NAFBlock。

## 3. 整体模型结构

```text
4-channel packed RAW
  │
  ├─ Intro: 3×3 Conv, 4 → 32
  │
  ├─ Encoder 1: 2×NAFBlock(C=32)  ───────────────┐
  │    └─ 2×2 stride-2 Conv, 32 → 64             │ additive skip
  ├─ Encoder 2: 2×NAFBlock(C=64)  ────────────┐  │
  │    └─ 2×2 stride-2 Conv, 64 → 128         │  │
  ├─ Encoder 3: 2×NAFBlock(C=128) ─────────┐  │  │
  │    └─ 2×2 stride-2 Conv, 128 → 256     │  │  │
  ├─ Encoder 4: 2×NAFBlock(C=256) ──────┐  │  │  │
  │    └─ 2×2 stride-2 Conv, 256 → 512  │  │  │  │
  │                                     │  │  │  │
  ├─ Middle: 2×NAFBlock(C=512)          │  │  │  │
  │                                     │  │  │  │
  ├─ Up 1: 1×1 Conv + PixelShuffle → C=256 + skip
  │    └─ Decoder 1: 2×NAFBlock(C=256)
  ├─ Up 2: 1×1 Conv + PixelShuffle → C=128 + skip
  │    └─ Decoder 2: 2×NAFBlock(C=128)
  ├─ Up 3: 1×1 Conv + PixelShuffle → C=64 + skip
  │    └─ Decoder 3: 2×NAFBlock(C=64)
  ├─ Up 4: 1×1 Conv + PixelShuffle → C=32 + skip
  │    └─ Decoder 4: 2×NAFBlock(C=32)
  │
  ├─ Ending: 3×3 Conv, 32 → 4
  └─ Global residual: output = network_output + input
```

对于 `[1, 4, 512, 512]` 输入，各层特征尺寸如下：

|位置|输出尺寸 NCHW|
|---|---|
|Intro / Encoder 1|`1×32×512×512`|
|Down 1 / Encoder 2|`1×64×256×256`|
|Down 2 / Encoder 3|`1×128×128×128`|
|Down 3 / Encoder 4|`1×256×64×64`|
|Down 4 / Middle|`1×512×32×32`|
|Up 1 / Decoder 1|`1×256×64×64`|
|Up 2 / Decoder 2|`1×128×128×128`|
|Up 3 / Decoder 3|`1×64×256×256`|
|Up 4 / Decoder 4|`1×32×512×512`|
|Ending / 最终输出|`1×4×512×512`|

输入高宽不是 16 的整数倍时，模型只在右侧和底部补到最近的 16 倍数，
最后再裁回原尺寸。512 可以被 16 整除，因此 NAFNet 不会产生额外补边。

## 4. NAFBlock 内部结构

设输入为 `X ∈ R^(C×H×W)`，默认 `DW_Expand=2`、`FFN_Expand=2`。

### 4.1 空间与通道混合分支

```text
X
 └─ LayerNorm2d
    └─ 1×1 Conv: C → 2C
       └─ 3×3 Depthwise Conv: 2C → 2C
          └─ SimpleGate: split(2C) → C ⊙ C
             └─ Simplified Channel Attention
                ├─ AdaptiveAvgPool2d(1)
                └─ 1×1 Conv: C → C
             └─ feature × attention
                └─ 1×1 Conv: C → C
                   └─ Dropout/Identity
```

该分支输出通过可学习缩放参数 `beta` 加回输入：

```text
Y = X + beta × Branch1(X)
```

### 4.2 门控前馈分支

```text
Y
 └─ LayerNorm2d
    └─ 1×1 Conv: C → 2C
       └─ SimpleGate: split(2C) → C ⊙ C
          └─ 1×1 Conv: C → C
             └─ Dropout/Identity
```

最终输出为：

```text
Output = Y + gamma × Branch2(Y)
```

`beta` 和 `gamma` 都初始化为 0。因此训练开始时每个 NAFBlock 近似恒等映射，
可以在较深网络中稳定传递信号与梯度。

## 5. 核心设计理念

### 5.1 用 SimpleGate 替代传统激活函数

NAF 是 “Nonlinear Activation Free” 的缩写。网络不使用 ReLU、GELU 等
传统逐点激活，而是将通道一分为二并逐元素相乘：

```text
SimpleGate(X) = X1 ⊙ X2
```

乘法门控本身仍提供非线性表达能力，同时避免额外激活层。

### 5.2 深度卷积降低空间计算

NAFBlock 先通过 1×1 卷积扩展通道，再使用 3×3 depthwise convolution
提取空间信息。相比同通道数的普通 3×3 卷积，深度卷积显著减少参数与 MAC。

### 5.3 简化通道注意力

SCA 使用全局平均池化和单个 1×1 卷积生成通道权重，没有复杂的多层注意力
子网络。它增强全局通道建模，但也引入 `AdaptiveAvgPool2d` 和逐元素乘法。

### 5.4 加法 skip 而不是通道拼接

encoder 和 decoder 同尺度特征直接相加，不像当前 U-Net 那样 `concat`
后再用卷积压缩。这避免 decoder 输入通道翻倍，降低卷积与特征读写开销。

### 5.5 局部与全局双重残差

- 每个 NAFBlock 内有两条带 `beta/gamma` 的局部残差；
- 整个网络最终执行 `output + input` 的图像级全局残差。

这使网络主要学习噪声修正量，有利于保持 RAW 图像的低频亮度和结构。

## 6. 参数量与计算量

### 6.1 统计口径

本仓库 `tools/calculate_model_info.py` 的主要口径为：

- 一次乘加记为 1 MAC；
- 一次乘加近似记为 2 FLOPs；
- MAC/FLOPs 统计 Conv2d、ConvTranspose2d 和 Linear；
- 不统计 LayerNorm、池化、门控、逐元素加乘、PixelShuffle、padding/crop；
- 参数量通过 `sum(parameter.numel())` 计算，包含 LayerNorm、`beta/gamma`
  等全部可训练参数。

另外使用 LED 环境中的 THOP 做交叉验证。THOP 在本模型上会额外统计
AdaptiveAvgPool，因此其 MAC 略高；THOP 返回的参数量会遗漏部分自定义参数，
参数量应以 PyTorch 直接求和结果为准。

### 6.2 NAFNet32，512×512 packed RAW

|指标|结果|
|---|---:|
|总参数量|7,600,228（7.600 M）|
|可训练参数量|7,600,228|
|不可训练参数量|0|
|FP32 参数/Buffer 大小|28.99 MiB|
|NAFBlock 数量|18|
|Conv2d 数量|118|
|LayerNorm2d 数量|36|
|AdaptiveAvgPool2d 数量|18|
|PixelShuffle 数量|4|
|卷积 MACs|35.042 GMAC|
|卷积 FLOPs，2 FLOPs/MAC|70.084 GFLOPs|
|THOP MACs，含其支持的池化操作|35.106 GMAC|

卷积 MAC 和参数的结构分布：

|部分|参数量|卷积 GMAC|
|---|---:|---:|
|Intro + Ending|2,340|0.604|
|4 级 Encoder NAFBlock|1,250,240|13.451|
|4 个 DownSample|697,280|2.147|
|Middle NAFBlock|3,703,808|3.241|
|4 个 UpSample|696,320|2.147|
|4 级 Decoder NAFBlock|1,250,240|13.451|
|合计|7,600,228|35.042|

### 6.3 与当前 U-Net32 的同口径对比

输入均为 `[1, 4, 512, 512]`，使用本仓库卷积 hook 统计：

|模型|参数量|FP32 大小|卷积 GMAC|卷积 GFLOPs|
|---|---:|---:|---:|---:|
|UNetSeeInDark32|7,760,484|29.60 MiB|51.457|102.914|
|NAFNet32|7,600,228|28.99 MiB|35.042|70.084|
|NAFNet 相对变化|-2.1%|-2.1%|-31.9%|-31.9%|

因此 NAFNet32 主要降低的是计算量，参数和 checkpoint 大小只小约 2%。
若目标是数量级上的轻量化，还需要进一步降低 `width` 或 block 数量。

使用 THOP 的另一组同口径结果为：

```text
UNetSeeInDark32: 54.828 GMAC
NAFNet32:        35.106 GMAC
```

两种统计器对反卷积、池化等算子的处理不同，报告模型复杂度时必须同时注明
工具、输入尺寸和 1 MAC 对应多少 FLOPs。

### 6.4 SID 整帧计算量估算

SID Sony packed RAW 常见尺寸约为 `[1, 4, 1424, 2128]`。由于两个空间维度
都可以被 16 整除，按面积从 512 patch 线性换算：

```text
约 405.1 GMAC / 810.1 GFLOPs（仅卷积类算子）
```

这仍然是很大的整帧计算量。NAFNet32 比当前 U-Net 更省计算，但不能仅凭
35 GMAC/patch 就认定它已经适合移动端或实时 AI-ISP。

## 7. 实际部署与训练注意事项

1. **LayerNorm2d**：需要逐像素跨通道求均值和方差，某些 NPU/ISP 加速器
   可能需要算子分解或回退。
2. **AdaptiveAvgPool2d**：SCA 需要整幅特征图的全局归约，不利于严格的
   行流式处理。
3. **PixelShuffle**：理论 MAC 很低，但需要通道到空间的数据重排。
4. **门控和注意力乘法**：常规 MAC 工具可能不计入，但会产生真实带宽和
   elementwise 开销。
5. **激活峰值**：四级 encoder/decoder 仍需保存多尺度 skip，参数量不能
   代表运行时显存。
6. **残差语义**：网络实现输出 `network(inp) + inp`，训练损失目标应保持
   为干净 RAW；不需要在外部再次加输入。
7. **输入输出通道必须相同**：全局残差要求 `img_channel` 同时作为输入和
   输出通道，SID packed RAW 应设置为 4。

## 8. 使用与复现命令

直接实例化：

```python
from models.natnet_arch import NAFNet

model = NAFNet(
    img_channel=4,
    width=32,
    enc_blk_nums=(2, 2, 2, 2),
    middle_blk_num=2,
    dec_blk_nums=(2, 2, 2, 2),
)
```

统计参数量与计算量：

```bash
cd /home/zhengwu/Desktop/ye-workspace/raw_image_denoising

conda run --no-capture-output -n LED-ICCV23 \
  python tools/calculate_model_info.py \
  --model nafnet \
  --input-shape 1 4 512 512 \
  --features 32 \
  --encoder-blocks 2 2 2 2 \
  --middle-blocks 2 \
  --decoder-blocks 2 2 2 2 \
  --device cuda:1 \
  --output-json worklog/nafnet32_model_info_512.json
```

当前 `train_sid_sony.py` 仍默认构造 `UNetSeeInDark`。本次工作只新增并验证
NAFNet 网络结构，没有擅自替换正在训练的 U-Net。

## 9. 10–15 GFLOPs NAFNet-Tiny 配置

在保持 4 级 encoder/decoder 和原始 NAFBlock 不变的前提下，仅调整宽度与
block 数量：

```yaml
model: nafnet
model_width: 16
encoder_blocks: [1, 1, 1, 1]
middle_blocks: 2
decoder_blocks: [1, 1, 1, 1]
```

对应模型信息：

|指标|NAFNet32|NAFNet-Tiny|
|---|---:|---:|
|参数量|7,600,228|1,604,692|
|FP32 参数大小|28.99 MiB|6.12 MiB|
|卷积 MACs|35.042 GMAC|5.695 GMAC|
|卷积 FLOPs|70.084 GFLOPs|11.390 GFLOPs|
|THOP FLOPs|70.212 GFLOPs|11.422 GFLOPs|
|NAFBlock 数量|18|10|

完整训练配置为：

```text
configs/train_sid_sony_nafnet_tiny.yaml
```

训练命令：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py \
  --config configs/train_sid_sony_nafnet_tiny.yaml
```

训练脚本和真实 SID / held-out synthetic 评测脚本均可从训练 checkpoint 的
`args` 自动恢复 NAFNet 宽度和 block 配置。真实 SID 评测示例：

```bash
for ratio in 100 250 300; do
  conda run --no-capture-output -n LED-ICCV23 python test_denoise_sideld.py \
    --cp-dir experiments/sid_sony_nafnet_tiny/checkpoints/latest.pth \
    --testset-type sid --eval-ratio "$ratio" \
    --num-workers 2 \
    --result-json "experiments/sid_sony_nafnet_tiny/sid_x${ratio}.json"
done
```

本地短程健康检查使用 10 epoch × 128 step，共 1,280 step。训练 L1 从首
batch 的 `0.1949` 降至 epoch 10 平均 `0.0246`；20 个独立 held-out
synthetic patch 的 PSNR 最高达到 `26.62 dB`。短程 checkpoint 在真实 SID
×100 第一张上的结果为 `38.04 dB / 0.9084 SSIM`。这些数值只证明训练和
评测链路有效，不能替代完整分组或完整训练结果。

正式训练已作为用户级独立服务在 GPU 1 上启动：

```text
systemd unit: sid-nafnet-tiny.service
experiment:   experiments/sid_sony_nafnet_tiny
log:          experiments/sid_sony_nafnet_tiny/train_stdout.log
```

第 1 个完整 epoch 已跑通：1,288 step、平均 L1 为 `0.04531`，耗时约
94.2 秒，并成功写入 `latest.pth`、`metrics.jsonl` 和 `model_info.json`。
按首轮速度粗略估算，1,000 epoch 约需 26 小时，实际时间会受另一个 U-Net
训练、RAW 解码和磁盘缓存影响。

查看运行状态和日志：

```bash
systemctl --user status sid-nafnet-tiny.service --no-pager
tail -f experiments/sid_sony_nafnet_tiny/train_stdout.log
```
