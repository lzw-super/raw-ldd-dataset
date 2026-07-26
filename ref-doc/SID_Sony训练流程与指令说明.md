# SID Sony RAW 去噪训练流程与指令说明

本文说明本仓库为论文 *Noise Modeling in One Hour* 补齐的 SID Sony-A7S II 训练流程。训练目标是使用 **SID 长曝光 clean RAW** 与 **LLD 真实暗帧** 在线合成噪声，训练 `UNetSeeInDark`；不将 SID short RAW 用作监督标签。

## 1. 训练数据与边界

|数据|本地位置|用途|
|---|---|---|
|SID 完整 long RAW|`/home/shared_files/dataset/SID/Sony/long`|动态裁剪 clean target|
|SID 原始 short RAW|`/home/shared_files/dataset/SID/Sony/short`|仅用于官方真实配对评估|
|LLD Sony A7S II 暗帧|`biasframe_et_1_30/ISO*/`|真实 signal-independent noise|
|PMN dark shading|`resources/SonyA7S2/`|训练、评估中去除空间偏置|

训练索引只保留 SID 文件名前缀为 `0` 的训练场景，避免与前缀 `1`/`2` 的官方测试、验证样本泄漏。当前本地索引包含：

- 161 个训练 clean 场景，每个 epoch 动态裁 8 个 patch；
- 20 个前缀 `2` 的独立 synthetic validation 场景；
- 每个 epoch 共 1,288 个 `4×512×512` packed RAW patch；
- 25 个训练 ISO；
- 260 张 LLD 暗帧（ISO100 有 20 张，其余每档 10 张）。

索引中的 ISO 直接从 SID **long RAW EXIF** 读取。不能直接相信 `Sony_train_list.txt` 的 short-pair ISO：同一 long RAW 的不同 short 图偶尔会有不同 ISO。

## 2. 端到端训练数据流

```text
SID train full-resolution long RAW
    └─ 归一化：(DN - 512) / (16383 - 512)
       └─ 每个 epoch 重新动态裁剪 packed 512×512 patch
          └─ 随机选择 ratio：100 / 250 / 300

LLD 同 ISO 暗帧
    └─ 减去 black level 512 和 PMN dark shading
       └─ 独立随机裁剪为 signed dark residual

clean target + Poisson shot noise + dark residual
    └─ 合成带噪 packed RAW
       └─ UNetSeeInDark（4 通道输入、4 通道输出）
          └─ clamp(prediction, 0, 1) 后计算 L1 loss
```

### 2.1 真实暗帧残差

对请求 ISO `g`，默认优先匹配相同 ISO 的暗帧；只有缺档时才在 `log2(ISO)` 空间选择最近档。本地 25 个训练 ISO 均为精确匹配。

使用论文公平比较所需的 PMN 模式：

\[
N_{dark}=D_{raw}-512-DS_{PMN}(g)
\]

其中 `N_dark` 保持 DN 域的正、负值，不能在加入网络前裁到零。

### 2.2 ratio-aware 噪声合成（主配置）

设 clean packed RAW 去 black level 后的 DN 为 `I`，SID 增亮倍率为 `r`，论文给出的系统增益为：

\[
K=\frac{ISO}{1000}\quad\text{DN/e$^-$}
\]

先模拟短曝光 shot noise：

\[
Z=K\cdot\mathcal P\left(\frac{I}{rK}\right)
\]

再按 SID 官方评估方式增亮：

\[
Y=r(Z+N_{dark})
\]

最终输入为 `min(Y / 15871, 1)`，仅裁上界；目标为 `clip(I / 15871, 0, 1)`。这让合成 shot noise 和真实 short RAW 经数字增益后的统计关系一致。

另有 `paper_literal` 消融模式：不显式引入 ratio。该模式必须使用独立输出目录和 checkpoint，不能与主配置混用。

## 3. 关键代码

|文件|职责|
|---|---|
|`datasets/sid_synthetic_train.py`|构建无泄漏索引、加载 clean patch、抽取训练样本|
|`noise/dark_frame_bank.py`|读取 MAT 暗帧、PMN shading 校正、ISO 匹配|
|`noise/sid_noise_synthesis.py`|DN 域 shot/dark noise 合成|
|`train_sid_sony.py`|训练、synthetic sanity validation、checkpoint 保存/恢复|
|`tools/build_sid_train_index.py`|生成可审计的训练 JSON 索引|
|`tools/validate_sid_training.py`|数据、MAT 元数据、Poisson 统计校验|
|`test_denoise_sideld.py`|官方 SID/ELD 真实数据评估入口|
|`configs/train_sid_sony.yaml`|主训练配置|

## 4. 执行顺序与指令解释

所有命令在 `raw_image_denoising` 目录执行，并使用已有的 `LED-ICCV23` 环境：PyTorch 1.13.1 + CUDA 11.7、rawpy 0.16.0。

### 4.1 生成训练索引

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python tools/build_sid_train_index.py
```

作用：扫描完整 long RAW，读取 EXIF ISO，生成：

```text
infos/SID_train_clean_raw.json       # 161 个前缀 0 训练场景
infos/SID_validation_clean_raw.json  # 20 个前缀 2 验证场景
```

索引中包含绝对路径，因此换机器或移动数据目录后应重新执行。

### 4.2 校验数据与噪声模块

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python tools/validate_sid_training.py
```

作用：

- 检查训练样本均属于前缀 `0`；
- 检查 clean ISO 与暗帧 ISO 的映射；
- 读取全部 MAT 的 `ISO`、`expo` 元数据；
- 抽取真实暗帧残差，确认包含负值；
- 检验 Poisson 合成的均值、方差是否满足理论值。

报告写入 `worklog/sid_training_asset_validation.json`。本地已验证 260 张暗帧元数据全部一致，曝光均为约 `0.0333333 s`。

### 4.3 最小冒烟训练

```bash
conda run --no-capture-output -n LED-ICCV23 python train_sid_sony.py \
  --config configs/train_sid_sony.yaml \
  --output-dir experiments/smoke_sid_sony \
  --epochs 1 --steps-per-epoch 8 --num-workers 0 \
  --validate-every 1 --validate-steps 2 --save-every 1
```

参数说明：

- `--config`：加载主配置；命令行参数会覆盖配置中的同名值；
- `--output-dir`：本次实验的日志和 checkpoint 目录；
- `--epochs 1 --steps-per-epoch 8`：只运行 8 次反传，用于确认显存、数据读取和 loss；
- `--num-workers 0`：关闭 DataLoader 子进程，便于首次排错；
- `--validate-every 1 --validate-steps 2`：epoch 结束时取 2 个合成样本做 sanity validation；
- `--save-every 1`：保存该短跑 checkpoint。

此命令只验证工程，不产生可与论文比较的指标。

### 4.4 正式 ratio-aware 训练

```bash
conda run --no-capture-output -n LED-ICCV23 python train_sid_sony.py \
  --config configs/train_sid_sony.yaml \
  --rebuild-manifest
```

主配置含义如下：

|配置|值|含义|
|---|---:|---|
|`epochs`|1000|论文主训练轮数|
|`batch_size`|1|11GB 显存下的 packed 512 patch batch|
|`num_workers`|2|并发读取 ARW/MAT，减少 GPU 等待|
|`patch_size`|512|packed RAW 域大小，对应 mosaic 1024×1024|
|`crops_per_image`|8|每张完整 long RAW 每个 epoch 动态抽取 8 个新 patch|
|`ratios`|100/250/300|均匀采样 SID 三个评估倍率|
|`synthesis`|`ratio_aware`|主噪声合成方式|
|`k_scale`|0.1|得到 `K=ISO/1000`|
|`learning_rate`|2e-4|Adam 初始学习率|
|`warmup_epochs`|5|warmup 后 cosine 衰减|
|`amp`|false|初次严格复现关闭混合精度|
|`keep_checkpoints`|3|仅保留最近三个编号 checkpoint，控制磁盘占用|

每 epoch 为 1,288 step。完整 RAW 会由有界 LRU cache 缓存；实际时间会受 GPU 竞争、rawpy 解码和磁盘读取影响。

每轮都会更新：

```text
experiments/sid_sony_raw_dynamic/
├── config.json
├── model_info.json
├── metrics.jsonl
└── checkpoints/
    ├── latest.pth
    └── epoch_XXXX.pth
```

训练启动时会根据实际 `patch_size` 自动统计一次 U-Net 信息并打印到终端。
完整结果单独保存到 `model_info.json`，同时写入 `metrics.jsonl` 的每条 epoch
记录，包括参数量、可训练参数量、FP32 权重大小、MACs 和 FLOPs。

也可以不启动训练，单独运行统计脚本：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python tools/calculate_model_info.py \
  --input-shape 1 4 512 512 \
  --device cuda:0 \
  --output-json worklog/unet_model_info.json
```

若训练进程是在加入自动统计功能之前启动的，可以在不中断训练的情况下，
把一条 `record_type=model_info` 元数据追加到已有日志：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python tools/calculate_model_info.py \
  --checkpoint experiments/sid_sony_raw_dynamic/checkpoints/latest.pth \
  --device cuda:1 \
  --output-json experiments/sid_sony_raw_dynamic/model_info.json \
  --metrics-jsonl experiments/sid_sony_raw_dynamic/metrics.jsonl
```

这里采用 `1 MAC = 1 次乘加 ≈ 2 FLOPs` 的口径，只统计卷积、反卷积和
全连接层；激活、池化、拼接、裁剪和补边等少量操作不计入，因此与采用
“1 MAC = 1 FLOP”口径的软件相比，FLOPs 数值会相差约两倍。

### 4.5 中断恢复

仅在 **完全相同的 epoch 数、steps-per-epoch、调度器设置** 下恢复：

```bash
conda run --no-capture-output -n LED-ICCV23 python train_sid_sony.py \
  --config configs/train_sid_sony.yaml \
  --resume experiments/sid_sony_raw_dynamic/checkpoints/latest.pth
```

不要把 `--steps-per-epoch 8` 之类短冒烟 run 的 checkpoint 恢复到正式 1,288 step/epoch、1,000 epoch 训练中，因为学习率调度总步数不同。

### 4.6 真实 SID 指标验证

对训练完成的 checkpoint，在三个官方倍率分别运行：

```bash
for ratio in 100 250 300; do
  conda run --no-capture-output -n LED-ICCV23 python test_denoise_sideld.py \
    --cp-dir experiments/sid_sony_raw_dynamic/checkpoints/latest.pth \
    --testset-type sid --eval-ratio "$ratio" \
    --num-workers 2 \
    --result-json "experiments/sid_sony_raw_dynamic/sid_x${ratio}.json"
done
```

参数说明：

- `--cp-dir`：待评估权重。兼容官方裸 state dict 与本训练脚本的可恢复 checkpoint；
- `--testset-type sid`：选择 Sony SID 真实 short-long 配对；
- `--eval-ratio`：选择 ×100、×250 或 ×300 分组；
- `--num-workers`：评估数据读取 worker 数；
- `--result-json`：保存 PSNR、SSIM 到 JSON；
- `--max-samples 1`：可附加用于快速检查，不能作为正式指标。

评估会执行官方流程：真实 short RAW 减 PMN dark shading、打包归一化、乘 exposure ratio、网络推理、`ELDIlluminanceCorrect` 全局亮度校正，最后计算 PSNR/SSIM。

## 5. 已完成的本地验证结果

|项目|结果|解释|
|---|---:|---|
|暗帧 MAT 元数据|260/260 通过|ISO 与文件夹一致，曝光均为 1/30 s|
|Poisson 均值相对误差|0.013%|DN 单位正确|
|Poisson 方差相对误差|0.003%|ratio-aware 方差实现正确|
|完整 1 epoch|1,288 step / 85.8 s|训练、加载、保存均正常|
|1 epoch synthetic L1|0.0550|仅用于训练健康检查|
|1 epoch SID ×100 单图|20.78 dB / 0.3649|从随机初始化训练一轮，尚未收敛|
|官方预训练权重、同一单图|42.64 dB / 0.9774|证明真实 SID 评估管线可用|

单图和 1 epoch 结果不能与论文表格直接比较。论文目标应以三个完整官方分组的平均 PSNR/SSIM 判断；若正式训练结果差距明显，应优先检查 ratio 合成、PMN dark shading、DN 单位和评估分组，而不是先修改网络结构。

## 6. 常见注意事项

1. 不要用 SID short RAW 作为主训练损失输入，否则就变成 paired supervised training。
2. 不要把 dark residual 或带噪输入的下界裁为 0；负值是有效噪声信息。
3. 训练和评估统一采用 `black=512`、`white=16383`；旧 NPZ 路径仅保留为诊断兼容模式。
4. 不要把暗帧 residual 再次减去 black level；PMN 模式中它已按 `D_raw - 512 - DS` 计算。
5. `paper_literal` 仅作消融，必须使用独立实验目录。
6. 正式训练前确认磁盘空间；checkpoint 含优化器状态，单个约 89MB。

## 7. 固定 NPZ 过拟合问题与修复

早期复现使用 1,288 个预裁 NPZ patch，并在 1,000 个 epoch 中反复循环同一批空间区域；synthetic validation 又复用了训练 manifest 的前 4 个 patch。结果是网络记住了固定 clean 内容：

- 旧日志 synthetic PSNR 约 48 dB；
- 同一 checkpoint 在前缀 `2` 的 20 个未见完整 RAW 场景上，三档 synthetic 平均仅 23.54 dB；
- 真实 SID 全量约为 ×100 29.61 dB、×250 28.78 dB、×300 27.57 dB。

修复包括：

1. clean source 改为 161 张完整 long RAW；
2. 每个 epoch 为每张图重新随机裁 8 个 patch；
3. synthetic validation 改为 20 张前缀 `2` 的独立 long RAW；
4. validation 默认覆盖 20 个不同场景；
5. 新增 `tools/evaluate_synthetic_checkpoint.py`，用于对固定 checkpoint 做独立三倍率 synthetic 评估。

使用旧 epoch 690 权重进行仅 1,280 step 的动态 RAW 诊断微调后：

|指标|旧固定 NPZ checkpoint|动态 RAW 短微调|
|---|---:|---:|
|held-out synthetic 平均 PSNR|23.54 dB|33.10 dB|
|真实 SID ×100（完整 40 张）|29.61 dB|41.19 dB / 0.9480 SSIM|
|真实 SID ×250（完整 40 张）|28.78 dB|39.11 dB / 0.9165 SSIM|
|真实 SID ×300（完整 49 张）|27.57 dB|35.81 dB / 0.8891 SSIM|

上表的真实 SID 数值已覆盖完整的 40/40/49 张配对。动态 RAW 模型只是从旧 epoch 690 权重继续进行 1,280 step 的诊断微调，不是最终论文复现模型；但独立 synthetic 与真实域指标同时显著上升，已经验证了修复方向。正式结果仍应从随机初始化运行完整动态 RAW 配置。
