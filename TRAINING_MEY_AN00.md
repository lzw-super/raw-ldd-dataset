# HONOR MEY-AN00 DNG 训练、验证与测试

本流程固定在 DNG 的 stored pre-opcode mosaic 域：按每张图自己的 `BlackLevel`/`WhiteLevel` 归一化，并打包为 canonical `[R,G1,G2,B]`。对当前 MEY-AN00 GBRG 文件，这对应 `[R_BL,G_BR,G_TL,B_TR]`。不会使用 ISP RGB、16-bit 容器范围或目录名 ISO。

当前数据分成四种用途，彼此不能混用：

|目录|当前数量|用途|可报告指标|
|---|---:|---|---|
|`raw-test/cleanframe`|90 张 10 s|pseudo-clean 训练内容源|无|
|`raw-test/biasframe-1-30`|134 张 1/30 s|DS 标定与合成暗帧残差|无|
|`raw-test/val`|10 张 10 s、独立场景|held-out synthetic 验证 target|仅 `synthetic_heldout_psnr`|
|`raw-test/noisy`|28 张混合曝光、无 GT|无参考定性去噪|无；**不计算 PSNR/SSIM**|

`val` 的 ISO 为 200/319/400/800/2000/3200/8000，均有相同最终 EXIF ISO 的 held-out dark residual，因此 ratio 300（10 s 对 1/30 s）是严格匹配的合成验证条件。它仍是单张 pseudo-clean，不是 noisy-clean pair，故任何 `synthetic_heldout_psnr` 都不能简称为“真实 PSNR”。

## 1. 建 manifest 与 Dark Shading

在仓库根目录执行：

```bash
python tools/build_phone_manifest.py \
  --raw-root raw-test \
  --output-dir data/MEY_AN00/manifests

python tools/calibrate_phone_dark_shading.py \
  --manifest data/MEY_AN00/manifests/dark_calibration.jsonl \
  --output-root data/MEY_AN00/calibration

# 用各 ISO 的独立 calibration artifact 拟合与 SID/PMN 同构的连续 DS：
# DS(ISO) = k_branch * ISO + b_branch + BLE(ISO)
python tools/fit_phone_dark_shading.py \
  --calibration-root data/MEY_AN00/calibration \
  --iso-breakpoint 1600
```

manifest 工具不移动原始 DNG，按 DNG 的 `iso_exif` 配对而非目录名。它会写出：

```text
data/MEY_AN00/manifests/
├── clean_train.jsonl             # 仅 cleanframe，90 张
├── clean_synthetic_val.jsonl     # 仅 val，10 张独立场景
├── noisy_qualitative.jsonl       # 仅 noisy，当前 28 张、无 GT
├── dark_calibration.jsonl
├── dark_train.jsonl
└── dark_val.jsonl
```

暗帧在每个 ISO 内按时间块切成 60% DS calibration、20% residual train、20% residual val；DS 帧不与 residual train/val 复用。连续模型在 packed sensor coordinates 中逐像素拟合低/高 ISO 两段 `k*ISO+b`，并对每通道 BLE 做 log2-ISO 连续插值；训练 residual、synthetic 验证和真实 DNG 推理共用同一模型。旧的逐 ISO `dark_shading_dn.npy` 仍保留，只有显式设置 `dark_shading_model=per_condition_mean` 时才作为消融使用。当前 `cleanframe` 中仍有 4 张位于 `iso 100-10` 文件夹、但 DNG EXIF ISO 为 500 的文件；这是 manifest 记录的来源信息，不需要移动文件。

## 2. 训练与周期性 held-out synthetic 验证

```bash
python train_phone.py \
  --config configs/train_mey_an00_nafnet_tiny.yaml \
  --max-steps 100
```

新配置写入 `experiments/mey_an00_pseudoclean_continuous_ds`。已有的 `experiments/mey_an00_pseudoclean` 和 `experiments/mey_an00_unet` 来自旧的离散 DS 协议，不能用 `--resume` 切换为连续 DS；需要以 SID checkpoint 作为 `--init-checkpoint` 开一个新实验。

默认配置会：

- 仅以 `clean_train.jsonl` 和 `dark_train.jsonl` 训练；`val`/`noisy` 永不混入训练。
- 每个 epoch 在 `clean_synthetic_val.jsonl` 的 10 个固定 crop 上验证；clean、dark residual 与 Poisson 噪声都由 `val_seed=10001` 固定。
- 使用 `dark_val.jsonl` 分别评估 ratio 100/250/300；三个值保存为 `synthetic_heldout_psnr_by_ratio`，完整逐 ISO 结果保存为 `synthetic_heldout_ratios`。ratio 300 严格曝光匹配，100/250 使用记录在案的 1/30 s residual 近似复用。

`run_protocol.json` 同时固定训练/验证 manifest 的路径和哈希、checkpoint、噪声合成方式、ratio 与验证随机种子。这个验证适合发现训练退化或合成域过拟合，不能用于声称真实相机去噪性能。

## 3. 独立复评 checkpoint

对保存的模型运行完全相同口径的 synthetic 验证：

```bash
python tools/evaluate_phone_synthetic_checkpoint.py \
  --checkpoint experiments/mey_an00_pseudoclean_continuous_ds/checkpoints/latest.pth \
  --ratios 100 250 300 \
  --result-json experiments/mey_an00_pseudoclean_continuous_ds/synthetic_val.json
```

输出 JSON 的 `metric_protocol` 会明确标记为 `synthetic_heldout_pseudoclean_not_real_pair_psnr`。默认评估就是 `--ratios 100 250 300 --dark-exposure-policy approximate_reuse_1_30s`；报告时必须将近似的 100/250 与严格的 ratio 300 分开解释。

## 4. `noisy` 的无参考定性去噪

`noisy` 是实际 1/30 s 输入且没有 clean GT。下列命令先减去对应 1/30 s calibrated DS、再按 `10 s / input_exposure` 自动放大输入，并使用重叠 tile 避免全图显存峰值：

```bash
python tools/qual_denoise_phone.py \
  --checkpoint experiments/mey_an00_pseudoclean_continuous_ds/checkpoints/latest.pth \
  --input-manifest data/MEY_AN00/manifests/noisy_qualitative.jsonl \
  --output-dir experiments/mey_an00_pseudoclean_continuous_ds/qualitative_noisy
```

默认每张图只产生一张 `*_comparison.png`；左侧是经同一显示缩放的输入，右侧是去噪结果。PNG 是 packed-RGB 预览，并非色彩管理的 DNG ISP 输出，也不会写出可误作评测 RAW 的 `.npy`。该命令不产生 PSNR 或 SSIM。

用 `--ratio` 可强制所有输入使用同一去噪亮度/曝光倍率（例如 `--ratio 100`）；不传时会按 `10 s / input_exposure` 自动求值。comparison 顶部会标明实际 `Noisy input (x倍率)`，便于区分不同设置的结果。

连续 DS 模型可以直接计算 EXIF ISO=250 的 DS，不再退化到 ISO 200 最近邻。对旧 checkpoint，工具仍会自动使用其旧的 `per_condition_mean` 协议；此时 `--missing-ds-policy` 才决定缺失 ISO 的行为。工具还会把不在 checkpoint 训练 ratio 集合内的输入标成 `OOD`。当前 `noisy` 已包含 1/125 s 到约 2 s 的混合曝光，其中很多样本不属于训练使用的 ratio 100/250/300，不能把这些结果直接归因于网络容量。

## 5. 未来真实 noisy-clean PSNR

补齐严格配准的 pair 后，创建 JSONL（路径可相对 manifest 文件）并锁定 scene split。例如：

```json
{"pair_id":"scene01_iso800", "scene_id":"scene01", "split":"test", "noisy_path":"paired/noisy/scene01.dng", "clean_path":"paired/clean/scene01.dng", "exposure_ratio":300.0, "registration":{"status":"verified", "method":"tripod_identity"}, "valid_mask_path":"paired/masks/scene01.npy"}
```

`valid_mask_path` 可省略；若提供，必须是 `[H,W]` 或 `[4,H,W]` 的 `.npy` bool mask。所有 clean/noisy DNG 必须已在同一 canonical RAW 坐标中严格配准；clean 的饱和像素会自动排除。然后运行：

```bash
python tools/evaluate_phone_pairs.py \
  --checkpoint experiments/mey_an00_pseudoclean_continuous_ds/checkpoints/latest.pth \
  --pair-manifest raw-test/pairs_test.jsonl \
  --output-json experiments/mey_an00_pseudoclean_continuous_ds/real_pair_test.json
```

评估器默认要求 `registration.status="verified"`，在 noisy 端减对应 calibrated DS、逐帧归一化、曝光比放大、有效/非饱和 mask 后报告每 pair、macro 以及全像素 `global_psnr`。未配准的 pair 不应绕过此检查或作为可报告的真实 PSNR。

## 当前限制

- 10 s single-frame pseudo-clean 不减 1/30 s DS：两种曝光的 dark current/热状态不同。
- 暗噪声按相同最终 EXIF ISO 匹配。训练中的 ratio 100/250 近似复用同 ISO/mode 1/30 s residual，ratio 300 才严格曝光匹配。
- 不做空间翻转；FPN residual crop 要保留传感器绝对坐标。
- 只有未来严格配准、未用于模型选择的真实 noisy-clean test pair 才能支持最终 RAW PSNR 结论。
