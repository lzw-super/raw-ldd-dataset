# MEY-AN00 手机微调链路完整性审计

## 结论

现有实现的**工程训练闭环是完整的**：DNG metadata/CFA 读取、canonical RAW packing、动态 crop、shot noise、真实 dark residual、ratio 放大、L1、checkpoint/resume、held-out synthetic 验证和 tiled inference 都能运行；SID 初始化权重也确实被严格加载。

但旧手机实验并不是与 SID 完全对齐的正式复现，也不足以判断模型结构优劣。主要限制来自数据和噪声协议，而不是训练循环缺了一次 forward/backward：

1. 旧手机 DS 是每个 ISO 一张独立均值图；SID/PMN 使用逐像素 ISO 函数。
2. target 是单张 10 s pseudo-clean DNG，没有 10 s DS、dark-current 或 burst averaging 校正。
3. 所有 empirical residual 都来自 1/30 s；ratio 100/250 对应的模拟短曝光却是 1/10 s、1/25 s，只是近似复用。
4. 当前 28 张 qualitative noisy DNG 的曝光范围为 1/125 s 到约 2 s；其中只有 8 张严格对应 ratio 300/1/30 s，另有 5 张为 ratio 100/1/10 s，剩余 15 张与训练曝光/ratio 不一致或发生 ratio cap。
5. 没有注册配准的真实 noisy-clean test pair，所以 synthetic PSNR 只能证明模型拟合了合成器，不能测真实域差。

因此，`experiments/mey_an00_pseudoclean` 与 `experiments/mey_an00_unet` 应标记为旧的 `per_condition_mean` DS 原型实验，不能直接与新连续 DS 实验混合比较，也不能通过 `--resume` 切换协议。

## 已实现的 SID 对齐 DS

新增 `tools/fit_phone_dark_shading.py`，从 calibration-only split 生成：

```text
data/MEY_AN00/calibration/continuous_dark_shading/
├── darkshading_lowISO_k.npy
├── darkshading_lowISO_b.npy
├── darkshading_highISO_k.npy
├── darkshading_highISO_b.npy
└── metadata.json
```

模型与 SID/PMN 同构：

```text
DS(ISO, p) = k_branch(p) * ISO + b_branch(p) + BLE(ISO)
```

- `ISO <= 1600` 使用 low branch；更高 ISO 使用 high branch。
- `k/b` 在每个 canonical packed sensor pixel 上拟合。
- BLE 是每通道全局残差，并在 log2 ISO 上连续插值，所以 ISO 250 等未直接采集的档位不再使用最近邻 DS。
- calibration frame 按样本数加权；calibration、residual train、residual val 仍严格分离。
- 训练 residual、held-out synthetic 验证、qualitative inference 和真实 pair evaluator 使用同一 DS evaluator。
- `per_condition_mean` 仍保留为显式 ablation，并用于忠实复评没有新字段的旧 checkpoint。

当前 9 个 ISO 的 low branch 为 `[200,319,400,500,800,1278]`，high branch 为 `[2000,3200,8000]`。在 27 张 held-out dark frame 上，连续 DS 后每帧/通道 residual 绝对均值平均为 `0.00481 DN`，与离散均值法相同；同时多数 ISO 的 residual 标准差更低，例如 ISO 200 约从 `0.851 DN` 降至 `0.810 DN`，说明连续拟合减少了 calibration 均值图自身的有限样本噪声。

## 两个旧实验说明了什么

### `experiments/mey_an00_pseudoclean`

- NAFNet-Tiny，500 epoch、约 90k steps。
- train L1 从约 `0.0108` 降到 `0.0065`。
- 该次运行没有独立 held-out synthetic 指标记录，因此无法仅靠日志判断泛化；只能说明优化和 I/O 正常。
- checkpoint 未记录 `dark_shading_model`，按兼容规则属于旧的 `per_condition_mean` 协议。

### `experiments/mey_an00_unet`

- U-Net 的 SID checkpoint 被正确加载，500 epoch、90k steps。
- held-out synthetic PSNR 从 epoch 1 的约 `36.92 dB` 上升到 epoch 50 的约 `39.31 dB`，最终约 `39.35 dB`；50–100 epoch 后基本饱和。
- 这说明模型容量、优化器、学习率、checkpoint 与 synthetic validation 都在工作。
- 真实效果仍差，且 synthetic 指标很好，是明显的 synthetic-to-real/domain-gap 信号，不支持“训练脚本没有训练起来”或“只要换更大网络就会解决”的判断。

对已有 comparison PNG 的抽查也符合这一判断：ratio 300 的 1/30 s 样本能恢复主体结构，但在当前未做 white balance/CCM 的 packed preview 中呈明显绿色，并有暗部过平滑和细节损失；ratio 10 的长曝光样本输出几乎塌缩为平坦纹理。后者不是 U-Net 容量不足的直接证据，因为 ratio 10 从未出现在训练集合 `[100,250,300]` 中。

## 已确认正确的部分

- DNG 使用 stored pre-opcode linear mosaic，而不是 ISP RGB。
- 每帧使用自身 BlackLevel/WhiteLevel，并统一 pack 为 `[R,G1,G2,B]`。
- residual 保持 signed DN，输入只裁上界，不裁负值。
- clean crop 与 dark crop 使用相同 sensor coordinates，保留 FPN 空间位置。
- calibration dark、residual train、residual val 没有复用。
- hybrid 合成只加入 Poisson shot noise和 empirical residual，没有重复加入 NoiseProfile `O`。
- shot noise 在短曝光域采样，再与 residual 一起乘 ratio。
- spatial flip 被禁止，避免破坏 sensor-coordinate FPN。
- SID checkpoint 通过 strict state-dict load；手机 `--resume` 与 SID `--init-checkpoint` 语义分开。
- 新版 resume 会检查模型、ratio、patch、synthesis、DS 和训练 schedule，拒绝静默改变协议。

## 尚不能靠代码补齐的部分

### P0：clean target 不是 clean master

单张 10 s DNG 自身含 shot noise、10 s dark current、热像素和可能的空间 bias。当前 target 只减 metadata BlackLevel，不减 10 s DS。这会让模型学习保留 target bias，并限制真实去噪上限。

需要为每个训练场景采同设置 clean burst，完成配准后 robust average；同时采对应 ISO/mode/thermal 的 10 s dark frames，对 clean master 做独立校正。

### P0：缺少真实 paired test

当前 `raw-test/noisy` 没有 clean GT。必须采 scene-disjoint、严格配准的 short noisy / long clean burst pair，锁定 test split 后再判断 NAFNet-Tiny、U-Net 和标准 NAFNet。

### P0：测试曝光与训练分布不一致

当前 qualitative 集合按 `10 s / exposure` 得到的实际 ratio 包含约 `5、6.25、7.69、10、80、100、150、300、1250`。训练只使用 `100/250/300`，推理又把 1250 cap 到 300。

正式对比应先只选同 ISO、1/30 s 的 8 张输入做严格 ratio-300 检查。若要覆盖其余曝光，必须为这些 exposure 采 condition-matched dark residual，并把对应 ratio 加入训练；不能仅扩展 YAML ratio 而继续复用 1/30 s residual。

### P1：采集量和 condition coverage 偏少

每 ISO 的 DS calibration 只有 6–18 张，residual train/val 每档只有 2–6 张；condition key 也未显式包含 session/thermal bin。连续 DS 减少了均值估计噪声，但不能创造未采集的温度、时间和曝光分布。

## 推荐实验顺序

1. 运行连续 DS 拟合与 `validate_phone_training.py`。
2. 先用 SID checkpoint 分别重跑 NAFNet-Tiny 和 U-Net 到新输出目录；不要 resume 旧手机 checkpoint。
3. 第一轮定性只评估真实 1/30 s 输入，确认 DS 对齐能否缩小 domain gap。
4. 用旧 checkpoint 显式 `--dark-shading-model per_condition_mean` 复评同一批图，完成唯一变量为 DS 的 ablation。
5. 若连续 DS 改善有限，优先补 clean burst master、10 s dark 和真实 pair，而不是继续增加 epoch 或模型宽度。
