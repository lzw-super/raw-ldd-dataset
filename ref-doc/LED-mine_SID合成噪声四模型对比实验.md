# LED-mine SID 合成噪声四模型对比实验

## 目的与入口

新增 `ledmine_sid_` 前缀实验，使用当前仓库的网络实现，在相邻 `../LED-mine/` 仓库的默认 LED 预训练数据和噪声协议下从头训练。没有启动正式训练，也没有生成新的精度结果。

这些是 **LED 嵌套配置**，应使用新增的 `run_led_mine.py`，不能传给原来的 `train_sid_sony.py`。入口复用 LED 的训练、噪声生成、验证、检查点和日志实现；不修改 LED 仓库。

| 网络 | 配置 | 结构 | 训练态参数量 |
|---|---|---|---:|
| Rep-NCB | `configs/ledmine_sid_repncb.yaml` | Haar 3 层；HF Rep-NCB＋PReLU＋输入残差；LL width32/depth5/Rep-NCB/concat1x1/无归一化；精修 width32、5 blocks、K2、原始输入 skip | 292,224 |
| MRLFN | `configs/ledmine_sid_mrlfn.yaml` | 当前仓库 paper S2D k4、n4、d32 | 155,632 |
| SplitterNet | `configs/ledmine_sid_splitternet.yaml` | 当前仓库单帧 packed RAW 4→4 实现 | 731,636 |
| NAFNet | `configs/ledmine_sid_nafnet.yaml` | width32，encoder/decoder 均 [2,2,2,2]，middle2 | 7,600,228 |

Rep-NCB 对应原 `...hf_cnn_depth5_repncb_prelu_residual_ll_no_norm_each5.yaml` 的网络。NAFNet 使用本仓库完整 width32 结构，不使用 LED 仓库某些配置中的 tiny 替代结构。所有网络均无预训练初始化、无蒸馏，不额外套 LED RepNR 包装。

## 统一数据与训练协议

公共配置 `configs/led_mine/base.yaml` 继承 LED 的以下配置：

- `options/base/dataset/pretrain/SID_raw_gt.yaml`
- `options/base/dataset/test/SID_SonyA7S2_val_split.yaml`
- `options/base/noise_g/noise_g_virtual.yaml`
- `options/base/pretrain/CVPR20_ELD.yaml`
- `options/base/val_and_logger.yaml`

| 项目 | 设置 |
|---|---|
| 干净训练数据 | LED 的 `datasets/ICCV23-LED/Sony_train_long_patches`；当前 1,848 个 npz，实测样本 4×512×512 |
| 合成方式 | LED `VirtualNoisyPairGenerator`，5 个虚拟相机，`ptrqc`：Poisson、Tukey-lambda、行噪声、量化噪声、颜色偏置 |
| 曝光放大比 | 每个样本随机整数 100～300（包含端点） |
| batch / workers | 单 GPU batch=1 / workers=8 |
| patch | 使用预先生成的 512×512 packed RAW patch，crop_size=null |
| 优化器 | Adam，lr=1e-4，betas=[0.9,0.999]，weight_decay=0 |
| 学习率 | HandieLR；128800 次迭代降到 5e-5，231840 次降到 1e-5 |
| 训练长度 | 257600 iterations，无 warmup |
| 损失 | 纯 L1，mean，权重 1 |
| seed | 2022 |
| 验证 / 日志 / 检查点 | 每 1288 / 200 / 25500 iterations；最后另存 latest |

训练时先按 black/white level 将 long GT 归一化，再由 LED 原始实现按随机 ratio 和相机合成 noisy 输入。新实验不是本仓库之前的 `ratio_aware` 合成流程，也不是 LED 的 MRLFN 特定复合损失训练配方；四个网络共用默认 LED 纯 L1 协议。

注意：LED 数据配置虽然包含 `use_hflip`、`use_rot`，当前 `RAWGTDataset.__getitem__` 没有执行这两个增强，本实验保持该代码的实际行为。

### 固定噪声参数

LED 默认配置引用的 `pretrained/noise_g/VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth` 在当前机器上缺失。改为复用现有实验保存的参数：

```text
../LED-mine/experiments/LED_Pretrain_MRLFN_CVPR26_Paper_Setting/noise_g.pth
SHA256: da22dbd608cfc6bc68c34d9ec9e132d20e25ebc423d9379987c5962849d2cc9d
```

此文件与当前找到的另外三个 LED 实验保存的噪声状态哈希一致；它是噪声生成器参数，不是 MRLFN 网络权重。四组均引用它；请保留文件。数据、噪声、CRF 路径均相对于 `--led-root` 解析，输出在当前仓库。启动后保存的配置是完整展开、路径绝对化的 YAML。

### RAW 通道顺序

LED 使用 `[R,G1,B,G2]`，当前仓库网络使用 `[R,G1,G2,B]`。`models/led_mine_adapter.py` 在网络输入和输出分别用 `[0,1,3,2]` 置换，保证 MRLFN 的 Bayer/S2D 语义正确，且 LED 的 GT、噪声参数、颜色处理与指标保持原顺序。

## 手动训练

在当前仓库根目录执行，四条命令分别启动四个实验。不要同时在同一块显卡启动四组。

```bash
conda run --no-capture-output -n LED-ICCV23 python run_led_mine.py --config configs/ledmine_sid_repncb.yaml
conda run --no-capture-output -n LED-ICCV23 python run_led_mine.py --config configs/ledmine_sid_mrlfn.yaml
conda run --no-capture-output -n LED-ICCV23 python run_led_mine.py --config configs/ledmine_sid_splitternet.yaml
conda run --no-capture-output -n LED-ICCV23 python run_led_mine.py --config configs/ledmine_sid_nafnet.yaml
```

可在命令前设置 `CUDA_VISIBLE_DEVICES=0`。支持 `--led-root /path/to/LED-mine`。运行前只检查路径、配置和模型构造：

```bash
conda run --no-capture-output -n LED-ICCV23 python run_led_mine.py \
  --config configs/ledmine_sid_repncb.yaml --check-only
```

输出为 `experiments/ledmine_sid_<模型>/`，TensorBoard 位于 `tb_logger/ledmine_sid_<模型>/`。断点续训在原命令后加 `--auto_resume`；LED 会搜索对应 training_states。普通重新启动会沿用 LED 的已有实验目录归档行为。

四组统一修改训练设置时优先编辑公共配置，也可以在单个 YAML 中覆盖嵌套字段；入口支持 LED 原有 `--force_yml` 等参数。如果比较不同 batch、训练长度或损失，应另建实验名，以免混淆。

## 验证与测试

使用 LED `Sony_npy` 中真实 short/long 配对，分别读取 `valid_100.txt`、`valid_250.txt`、`valid_300.txt`。这是 LED 的验证划分；训练后独立运行测试入口仍评估这三份列表，不能将它称为与模型选择完全独立的 held-out test。需要独立测试集时应显式替换配对列表并记录划分。

指标沿用 LED RAW PSNR/SSIM，`crop_border=2`、`illumination_correct=true`、`metric_in_srgb=false`。与此前本仓库不同裁边、亮度校正或 SSIM 实现下的结果不直接混排。

```bash
conda run --no-capture-output -n LED-ICCV23 python run_led_mine.py \
  --config configs/ledmine_sid_repncb.yaml --mode test \
  --checkpoint experiments/ledmine_sid_repncb/models/net_g_latest.pth
```

其他三组将上述配置名、实验目录中的 `repncb` 替换为 `mrlfn`、`splitternet`、`nafnet`。也可以指定 `models/net_g_<iteration>.pth`。测试日志保存在当前仓库的 `results/ledmine_sid_<模型>/`，记录三档指标。

权重采用 LED 的 `params` 格式，并包含 adapter 的 `network.` 前缀，不是原 `train_sid_sony.py` 的 checkpoint 格式；请用本入口测试。保存策略也沿用 LED 周期/latest 机制，不继承原入口的 best PSNR 或最低 train_l1 保存策略。

## 验证记录

使用 `LED-ICCV23` 环境（PyTorch 1.13.1+cu117）：

- 四个配置成功展开，数据和噪声文件路径存在，训练协议一致。
- 四模型 CPU 1×4×32×32 前向、L1 反向检查通过，输出形状及梯度有限。
- 部署转换前后最大绝对误差：Rep-NCB 5.96e-8，MRLFN 7.15e-7，SplitterNet/NAFNet 0。
- 实际读取一个训练 npz，确认 4×512×512、black/white level 等元数据可用；加载固定五相机噪声状态并成功合成一对训练输入/GT，结果有限。
- 三档真实验证集分别有 29 / 29 / 35 对，各读取一对成功，packed 输入与 GT 均为 4×1424×2128。
- 独立测试配置移除了训练数据与噪声生成器，仅加载指定 checkpoint 评估真实配对。

以上是接口与数值冒烟检查，不代表正式训练收敛或全分辨率显存测试。正式训练和精度比较由用户离线执行。

## CUDA 设备选择

四个 `configs/ledmine_sid_*.yaml` 均新增顶层参数：

```yaml
cuda_visible_devices: "0"
```

例如改为 `"1"`，即可选择编号 1 的 GPU。入口在导入 LED/PyTorch 前设置可见设备；进程内选中的单卡会重新编号为 `cuda:0`，公共配置中的 `num_gpu: 1` 保持不变。

优先级为 **命令行 `--cuda-visible-devices` > 已有环境变量 `CUDA_VISIBLE_DEVICES` > YAML 参数**。因此已有 shell/调度器显卡限制会优先于 YAML；需要显式覆盖时使用：

```bash
conda run --no-capture-output -n LED-ICCV23 python run_led_mine.py \
  --config configs/ledmine_sid_repncb.yaml --cuda-visible-devices 1
```

训练、测试和 `--check-only` 均采用此选择。该参数应写在四个实验 YAML 的顶层；不要通过 LED 的 `--force_yml` 修改设备可见性，因为那时 CUDA 初始化可能已完成。填写多个编号只控制可见性，不会自动启用多卡训练。
