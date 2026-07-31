# U-Net 与标准 NAFNet 手机微调对照实验

这组实验用于判断 NAFNet-Tiny 的手机去噪效果是否主要受模型容量限制。三组模型使用相同的手机训练集、held-out synthetic 验证集、ratio、噪声合成和优化设置；只改变模型结构及其对应的 SID 初始化权重。

|实验|结构|参数量|SID 阶段|手机阶段输出目录|
|---|---|---:|---|---|
|现有基线|NAFNet-Tiny，width 16，blocks 1/1/1/1|1,604,692|已完成|`experiments/mey_an00_pseudoclean`|
|大模型 1|UNetSeeInDark，nf 32|7,760,484|已完成|`experiments/mey_an00_unet`|
|大模型 2|标准 NAFNet，width 32，blocks 2/2/2/2|7,600,228|需要先运行|`experiments/mey_an00_nafnet`|

## 1. U-Net：直接进行手机微调

配置默认读取当前已经完成的 SID 权重：

```text
experiments/sid_sony_paper_fair/checkpoints/latest.pth
```

确认该文件存在后运行：

```bash
python train_phone.py \
  --config configs/train_mey_an00_unet.yaml
```

这是权重初始化，不是断点续训。脚本通过 `init_checkpoint` 只载入模型权重，并为手机微调创建新的 Adam 优化器和余弦学习率计划。不要把 SID checkpoint 传给 `--resume`。

如果要改用仓库自带的纯 U-Net 权重 `checkpoints/sonya7s2.pth`，可在命令行覆盖：

```bash
python train_phone.py \
  --config configs/train_mey_an00_unet.yaml \
  --init-checkpoint checkpoints/sonya7s2.pth \
  --output-dir experiments/mey_an00_unet_official_init
```

两种初始化不要写入同一个输出目录。

## 2. 标准 NAFNet：先 SID 预训练，再手机微调

SID 预训练：

```bash
python train_sid_sony.py \
  --config configs/train_sid_sony_nafnet.yaml
```

完成后应得到：

```text
experiments/sid_sony_nafnet/checkpoints/latest.pth
```

再进行手机微调：

```bash
python train_phone.py \
  --config configs/train_mey_an00_nafnet.yaml
```

如果 SID 预训练中断，使用 `--resume` 恢复同一个阶段：

```bash
python train_sid_sony.py \
  --config configs/train_sid_sony_nafnet.yaml \
  --resume experiments/sid_sony_nafnet/checkpoints/latest.pth
```

手机微调中断时同理，但 checkpoint 必须来自手机阶段：

```bash
python train_phone.py \
  --config configs/train_mey_an00_nafnet.yaml \
  --resume experiments/mey_an00_nafnet/checkpoints/latest.pth
```

配置里的 `init_checkpoint` 会保留，但当显式提供 `--resume` 时，训练脚本优先恢复 `resume`，不会再次应用 SID 初始化。

## 3. 先做可选的短流程检查

若想在长训练前确认显存、数据读取、权重载入和反向传播都正常，可分别使用新的临时输出目录。短流程 checkpoint 不要再用于正式训练：

```bash
python train_phone.py \
  --config configs/train_mey_an00_unet.yaml \
  --max-steps 2 \
  --output-dir experiments/smoke_mey_an00_unet

python train_sid_sony.py \
  --config configs/train_sid_sony_nafnet.yaml \
  --epochs 1 \
  --steps-per-epoch 2 \
  --validate-steps 1 \
  --output-dir experiments/smoke_sid_sony_nafnet
```

`train_sid_sony.py` 的短流程会创建一个独立的一轮学习率计划，因此不能通过 `--resume` 把它延长为正式 1000 epoch 实验。

## 4. 使用统一口径评估

手机 held-out synthetic 复评：

```bash
python tools/evaluate_phone_synthetic_checkpoint.py \
  --checkpoint experiments/mey_an00_unet/checkpoints/latest.pth \
  --ratios 100 250 300 \
  --result-json experiments/mey_an00_unet/synthetic_val.json

python tools/evaluate_phone_synthetic_checkpoint.py \
  --checkpoint experiments/mey_an00_nafnet/checkpoints/latest.pth \
  --ratios 100 250 300 \
  --result-json experiments/mey_an00_nafnet/synthetic_val.json
```

手机真实 noisy DNG 的定性结果：

```bash
python tools/qual_denoise_phone.py \
  --checkpoint experiments/mey_an00_unet/checkpoints/latest.pth \
  --input-manifest data/MEY_AN00/manifests/noisy_qualitative.jsonl \
  --output-dir experiments/mey_an00_unet/qualitative_noisy

python tools/qual_denoise_phone.py \
  --checkpoint experiments/mey_an00_nafnet/checkpoints/latest.pth \
  --input-manifest data/MEY_AN00/manifests/noisy_qualitative.jsonl \
  --output-dir experiments/mey_an00_nafnet/qualitative_noisy
```

`synthetic_heldout_psnr` 只用于检查合成域性能；`raw-test/noisy` 没有 clean GT，因此最终判断仍应把三种模型对同一批 noisy DNG 的定性输出并排比较，不能把 synthetic PSNR 当成真实手机 paired PSNR。

## 5. 公平比较时保持不变的设置

- 不修改三组实验的手机 manifest、`val_seed`、ratios 或 dark-frame policy。
- 使用相同训练步数；默认都是 500 epoch、每张 clean DNG 每轮 2 个动态 crop。
- 分开比较 ratio 100、250、300。当前 ratio 300 的暗帧曝光严格匹配，100/250 是记录在案的 1/30 s residual 近似复用。
- 正式运行不要复用 smoke 输出目录，也不要让不同模型共用同一输出目录。
- 若开启 AMP 或改变 patch size，应对三个模型采用相同设置，并把它视作一组新的对照实验。
