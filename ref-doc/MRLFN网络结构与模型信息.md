# MRLFN N=4、d=16 网络结构、模型信息与评测指令

## 1. 参考实现与当前配置

网络和损失参考以下文件：

```text
/home/zhengwu/Desktop/graduate-workspace/LED-mine/led/archs/mrlfn_arch.py
/home/zhengwu/Desktop/graduate-workspace/LED-mine/led/losses/mrlfn_loss.py
/home/zhengwu/Desktop/graduate-workspace/LED-mine/ref-doc/Pochimireddy_2026_RAW去噪网络复现说明.md
```

当前工程实现位于：

```text
models/mrlfn_arch.py
losses/mrlfn_loss.py
```

本文使用固定配置：

```yaml
model: mrlfn
feature_channels: 16  # d
num_blocks: 4         # N
model_bias: true
```

输入和输出都是四通道 2×2 Bayer packed RAW。网络保持参考实现的前向结构，
没有额外增加全局 `output + input` 图像残差。

## 2. 网络结构

```text
packed RAW [B,4,H,W]
  │
  ├─ shallow_conv: 3×3 Conv, 4 → 16
  │       │
  │       ├───────────────────────────────┐
  │       │                               │
  │       └─ 4 × mRLFB                    ├─ shallow_fusion: 1×1 Conv
  │              └─ deep_fusion: 3×3 Conv│
  │                                       │
  └──────────────── concat [B,32,H,W] ────┘
                  │
                  └─ output_conv: 1×1 Conv, 32 → 4
                         ↓
                  clean packed RAW
```

每个 mRLFB 包含三个可重参数化 3×3 单元和一个线性 1×1 卷积：

```text
x → ReparamConv3x3 → ReparamConv3x3 → ReparamConv3x3 → feature
                                                      │
                              linear(feature + x) + x ┘
```

训练态单个 `ReparamConv3x3` 为：

```text
y = ReLU(x + Conv1x1_b(Conv3x3(x + Conv1x1_a(x))))
```

其中第一个 1×1 卷积不使用 bias，以保证零填充边界也能严格融合。验证和推理前，
恒等映射、三个卷积核及 bias 会解析合并为一个普通 3×3 卷积：

```text
y = ReLU(Conv3x3_reparameterized(x))
```

`model.deploy()` 返回融合副本，不改变实时训练模型；`switch_to_deploy()` 才是原地转换。

## 3. 损失函数与 Bayer 通道

训练使用参考实现的联合目标：

```text
G = (G1 + G2) / 2
Lraw = mean(abs(pred - target))
Lchromatic = mean(abs(pred_BG - target_BG))
           + mean(abs(pred_RG - target_RG))
L = 0.6 × Lraw + 0.4 × Lchromatic
```

参考仓库损失的 `channel_order` 语义为 `[R,G1,B,G2]`，而当前工程实际 packed
顺序为 `[R,G1,G2,B]`。因此这里必须设置：

```yaml
chromatic_channel_order: [0, 1, 3, 2]
```

这只是在不同物理打包顺序之间映射通道，没有修改损失公式。MRLFN 训练 loss
直接作用于网络输出，不在 loss 前 clamp；PSNR/SSIM 评测仍将输出限制到 `[0,1]`。

## 4. 参数量与计算量

统计工具为 `tools/calculate_model_info.py`，口径如下：

- 1 次乘加记为 1 MAC，近似为 2 FLOPs；
- 统计 Conv2d、ConvTranspose2d 和 Linear；
- 参数量包含模型的全部参数；
- MRLFN checkpoint 统计前必须转换为部署态。

### 4.1 训练态与部署态对比

|图状态|参数量|FP32 参数大小|256×256 GMAC|512×512 GMAC|卷积层数|
|---|---:|---:|---:|---:|---:|
|训练态多分支|38,580|0.147 MiB|2.496|9.982|44|
|融合部署态|32,244|0.123 MiB|2.093|8.372|20|

部署态 `256×256` 输入对应约 4.186 GFLOPs，`512×512` 输入对应约
16.744 GFLOPs。随机输入验证的重参数化前后最大绝对误差约为 `7.75e-7`。

### 4.2 部署态统计命令

不加载 checkpoint、只统计结构：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python tools/calculate_model_info.py \
  --model mrlfn \
  --features 16 \
  --num-blocks 4 \
  --input-shape 1 4 512 512 \
  --device cuda:0 \
  --output-json worklog/mrlfn_n4_d16_deploy_model_info_512.json
```

统计实际 checkpoint：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python tools/calculate_model_info.py \
  --model mrlfn \
  --checkpoint experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest.pth \
  --input-shape 1 4 512 512 \
  --device cuda:0 \
  --output-json experiments/sid_sony_mrlfn_n4_d16/deploy_model_info_512.json
```

输出 JSON 的 `graph_state` 必须为 `deploy`，参数量应为 32,244。结构参数会优先
从 checkpoint 的 `args` 读取。

## 5. SID 预训练和手机微调

SID 预训练：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py \
  --config configs/train_sid_sony_mrlfn_n4_d16.yaml \
  --resume 
```

MEY-AN00 微调：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_phone.py \
  --config configs/train_mey_an00_mrlfn_n4_d16.yaml
```

训练 checkpoint 同时保存：

- `model`：多分支训练态权重，用于 `--resume` 和 SID→手机微调；
- `model_deploy`：融合部署态权重，用于模型统计和评测；
- `training_graph`、`inference_graph`：图状态元数据。

训练产生的 `model_info.json` 已在部署态副本上统计，`metrics.jsonl` 中的
`validation_graph` 也应为 `deploy`。

## 6. SID 定量评测：PSNR 与 SSIM

定量入口为 `test_denoise_sideld.py`。它使用 SID short/long 测试对，模型输出先执行
与原评测一致的照度校正，然后统计线性 packed RAW 的 PSNR、SSIM。MRLFN 会优先严格
加载 checkpoint 的 `model_deploy`；旧 checkpoint 只有 `model` 时，会先自动融合再评测。

分别评测 ×100、×250、×300：

```bash
for ratio in 100 250 300; do
  conda run --no-capture-output -n LED-ICCV23 python test_denoise_sideld.py \
    --cp-dir experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest.pth \
    --testset-type sid \
    --eval-ratio "$ratio" \
    --device cuda:0 \
    --num-workers 2 \
    --result-json "experiments/sid_sony_mrlfn_n4_d16/latest_sid_x${ratio}.json"
done
```

```bash
for ratio in 100 250 300; do
  conda run --no-capture-output -n LED-ICCV23 python test_denoise_sideld.py \
    --cp-dir experiments/sid_sony_mrlfn_n4_d16/checkpoints/best.pth \
    --testset-type sid \
    --eval-ratio "$ratio" \
    --device cuda:0 \
    --num-workers 2 \
    --result-json "experiments/sid_sony_mrlfn_n4_d16/b_sid_x${ratio}.json"
done
```
每个 JSON 包含：

```text
model: mrlfn
graph_state: deploy
model_config.feature_channels: 16
model_config.num_blocks: 4
model_config.weight_source: model_deploy
psnr / ssim
```

终端也会打印类似信息：

```text
Loaded mrlfn checkpoint with deploy graph (weights: model_deploy)
```

如需快速检查，可添加 `--max-samples 1`。这只能验证流程，不能作为正式结果。

### 6.1 SID synthetic held-out 补充评测

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python tools/evaluate_synthetic_checkpoint.py \
  --checkpoint experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest.pth \
  --device cuda:0 \
  --result-json experiments/sid_sony_mrlfn_n4_d16/synthetic_heldout.json
```

该结果是独立 scene-prefix 2 上的合成噪声指标，不能替代真实 SID short/long 指标。

### 6.2 LearningDWT（32通道、atlas）

同一入口支持`learning_dwt`，自动从checkpoint恢复结构。当前文件名含d64的
`experiments/sid_sony_learning_dwt_sym4_l3_d64_atlas/checkpoints/latest.pth`
实际是32通道sym4三层模型，1000epoch。三倍率PSNR为40.522/36.667/33.752 dB。
与传统小波的同口径指标及完整命令见 [LearningDWT网络结构与评测](LearningDWT网络结构与评测.md)。

## 7. SID 定性可视化

可视化入口为 `test-op/run_qual_compare.sh`，底层调用
`test-op/qual_denoise_compare.py`。每一行展示：

```text
含噪输入 | MRLFN 去噪结果 | 干净 GT
```

默认从每张图中心裁剪 512×512 区域，并在去噪结果上标注 PSNR。图中模型标签会显示
`mrlfn (deploy)`，终端会打印 `N=4 d=16 graph=deploy weights=model_deploy`。

三档 ratio 全部生成：

```bash
for r in 100 250 300; do
  CP_DIR=experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest.pth \
  OUT_DIR=experiments/sid_sony_mrlfn_n4_d16/qualitative \
    bash test-op/run_qual_compare.sh 0 "$r"
done
```

产物为：

```text
experiments/sid_sony_mrlfn_n4_d16/qualitative/denoise_comparison_ratio100.png
experiments/sid_sony_mrlfn_n4_d16/qualitative/denoise_comparison_ratio250.png
experiments/sid_sony_mrlfn_n4_d16/qualitative/denoise_comparison_ratio300.png
```

第三个位置参数可以控制抽样图片数，例如每档展示 8 张：

```bash
bash test-op/run_qual_compare.sh 0 300 8
```

固定相同测试图下标进行跨模型比较：

```bash
CP_DIR=experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest.pth \
OUT_DIR=experiments/sid_sony_mrlfn_n4_d16/qualitative \
EXTRA="--indices 0 10 20 30 --crop-size 512" \
  bash test-op/run_qual_compare.sh 0 300
```

若 checkpoint 是没有 `args` 的裸 MRLFN 权重，可显式指定结构：

```bash
MODEL=mrlfn \
EXTRA="--feature-channels 16 --num-blocks 4" \
CP_DIR=/path/to/bare_mrlfn_state_dict.pth \
OUT_DIR=experiments/mrlfn_bare/qualitative \
  bash test-op/run_qual_compare.sh 0 100
```

### 7.1 LearningDWT与传统小波四列对照

```bash
for r in 100 250 300; do
  CP_DIR=experiments/sid_sony_learning_dwt_sym4_l3_d64_atlas/checkpoints/latest.pth \
  OUT_DIR=experiments/sid_sony_learning_dwt_sym4_l3_d64_atlas/qualitative_baseline \
  EXTRA="--indices 0 10 20 30 --crop-size 512 --with-wavelet-baseline" \
    bash test-op/run_qual_compare.sh 0 "$r"
done
```

四列：输入 / 传统sym4 L3 BayesShrink / LearningDWT / GT。两种输出采用相同照度校正。
标签从checkpoint读取真实32通道结构，不按目录名推断。标注PSNR为整图RAW指标。

## 8. 手机微调模型的定量与定性评测

手机合成留出集定量评测：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python tools/evaluate_phone_synthetic_checkpoint.py \
  --checkpoint experiments/mey_an00_mrlfn_n4_d16_continuous_ds/checkpoints/latest.pth \
  --device cuda:0 \
  --result-json experiments/mey_an00_mrlfn_n4_d16_continuous_ds/synthetic_eval.json
```

该指标是 pseudo-clean 上的 held-out synthetic PSNR，不应写成真实手机配对 PSNR。只有准备好
经过配准并标记 `registration.status=verified` 的 noisy/clean manifest 后，才使用：

```bash
python tools/evaluate_phone_pairs.py \
  --checkpoint experiments/mey_an00_mrlfn_n4_d16_continuous_ds/checkpoints/latest.pth \
  --pair-manifest data/MEY_AN00/manifests/registered_pairs.jsonl \
  --output-json experiments/mey_an00_mrlfn_n4_d16_continuous_ds/real_pairs.json \
  --device cuda:0
```

未配对手机 noisy DNG 只做定性可视化：

```bash
python tools/qual_denoise_phone.py \
  --checkpoint experiments/mey_an00_mrlfn_n4_d16_continuous_ds/checkpoints/latest.pth \
  --input-root raw-test/noisy \
  --output-dir experiments/mey_an00_mrlfn_n4_d16_continuous_ds/qualitative_noisy \
  --device cuda:0 \
  --write-summary
```

手机三个入口都通过 `utils.phone_model.load_phone_checkpoint` 加载模型，MRLFN 同样会先使用
部署态权重；输出 JSON 或 `run_summary.json` 的 `graph_state` 应为 `deploy`。

## 9. 独立导出部署 checkpoint

训练 checkpoint 已经包含 `model_deploy`。若只需要部署权重，可以导出紧凑文件：

```bash
python tools/export_mrlfn_deploy_checkpoint.py \
  --checkpoint experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest.pth \
  --output experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest_deploy.pth
```

导出文件只用于统计和推理，不能拿来恢复多分支训练。继续训练或手机微调必须使用包含
`model` 的原训练 checkpoint。
