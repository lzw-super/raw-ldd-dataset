# MRLFN N=4、d=16 训练与部署态验证

本实现参考：

- `/home/zhengwu/Desktop/graduate-workspace/LED-mine/led/archs/mrlfn_arch.py`
- `/home/zhengwu/Desktop/graduate-workspace/LED-mine/led/losses/mrlfn_loss.py`
- `/home/zhengwu/Desktop/graduate-workspace/LED-mine/ref-doc/Pochimireddy_2026_RAW去噪网络复现说明.md`

网络配置固定为 4 个 mRLFB（`N=4`）和 16 个特征通道（`d=16`）。训练图含可重参数化的
1×1–3×3–1×1 分支；部署图将每组分支精确融合为单个 3×3 卷积。部署态在
`1×4×256×256` 输入下为 32,244 个参数、2.093 GMAC。

## 1. SID 预训练

先检查配置中的数据路径和 GPU：

```bash
python train_sid_sony.py --config configs/train_sid_sony_mrlfn_n4_d16.yaml
```

断点续训必须使用训练 checkpoint 中的 `model`（多分支训练态），命令为：

```bash
python train_sid_sony.py \
  --config configs/train_sid_sony_mrlfn_n4_d16.yaml \
  --resume experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest.pth
```

损失与参考仓库一致：

```text
L = 0.6 * L1(pred_raw, target_raw)
  + 0.4 * (L1(pred_B-G, target_B-G) + L1(pred_R-G, target_R-G))
```

当前工程的物理打包顺序是 `[R,G1,G2,B]`，参考仓库损失参数的语义顺序是
`[R,G1,B,G2]`，所以配置使用 `chromatic_channel_order: [0,1,3,2]`。这是通道布局适配，
不是改变损失公式。MRLFN 训练 loss 不对网络输出预先 clamp；验证指标仍在 `[0,1]` 范围统计。

学习率使用参考 MRLFN 的 Adam `1e-4`，调度器沿用本工程的逐 step warmup + cosine；其余 SID
噪声合成、训练预算和验证划分沿用现有 SID 配置。

## 2. MEY-AN00 手机数据微调

配置默认从上一步的 SID `latest.pth` 加载训练态权重：

```bash
python train_phone.py --config configs/train_mey_an00_mrlfn_n4_d16.yaml
```

手机阶段沿用当前工程的 `continuous_iso_fit` calibrated DS、三档连续 ISO 条件表达、hybrid
噪声合成、Adam `1e-5` 和 warmup + cosine。要恢复手机训练仍使用 `--resume`；不要用 SID
checkpoint 作为 `--resume`，SID→手机迁移应保留配置中的 `init_checkpoint`。

## 3. checkpoint 中的训练态与部署态

MRLFN 每个训练 checkpoint 同时包含：

- `model`：训练态分支权重，用于 `--resume` 和 SID→手机微调；
- `model_deploy`：已融合的部署态权重，用于统计和所有评估；
- `training_graph` / `inference_graph`：明确记录图状态。

训练开始时的 `model_info.json` 是在部署态副本上统计的，字段
`graph_state` 应为 `deploy`。训练期间 SID 和手机验证同样先融合副本，`metrics.jsonl` 的
`validation_graph` 应为 `deploy`。融合只作用于副本，不会破坏实时训练网络或优化器状态。

如需独立、紧凑的部署 checkpoint：

```bash
python tools/export_mrlfn_deploy_checkpoint.py \
  --checkpoint experiments/mey_an00_mrlfn_n4_d16_continuous_ds/checkpoints/latest.pth \
  --output experiments/mey_an00_mrlfn_n4_d16_continuous_ds/checkpoints/latest_deploy.pth
```

## 4. 部署态模型统计与测试

统计工具对 MRLFN 会先融合；如 checkpoint 同时含两套权重，会严格加载 `model_deploy`：

```bash
python tools/calculate_model_info.py \
  --model mrlfn --features 16 --num-blocks 4 \
  --input-shape 1 4 256 256 \
  --checkpoint experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest.pth \
  --device cuda:0 \
  --output-json experiments/sid_sony_mrlfn_n4_d16/deploy_model_info_256.json
```

SID 真实 short/long 测试和合成留出集测试：

```bash
python test_denoise_sideld.py \
  --cp-dir experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest.pth \
  --device cuda:0 --testset-type sid --eval-ratio 100

python tools/evaluate_synthetic_checkpoint.py \
  --checkpoint experiments/sid_sony_mrlfn_n4_d16/checkpoints/latest.pth \
  --device cuda:0
```

手机合成留出集测试：

```bash
python tools/evaluate_phone_synthetic_checkpoint.py \
  --checkpoint experiments/mey_an00_mrlfn_n4_d16_continuous_ds/checkpoints/latest.pth \
  --device cuda:0 \
  --result-json experiments/mey_an00_mrlfn_n4_d16_continuous_ds/synthetic_eval.json
```

`test_denoise_sideld.py`、SID 合成评估、SID 定性脚本和所有 `utils.phone_model` 驱动的手机
测试入口都会优先读取 `model_deploy`；若输入旧版、只有训练态 `model` 的 MRLFN checkpoint，
则先自动融合再测试。输出 JSON 中的 `graph_state` 可用于核查实际图状态。
