# LLD BiasFrame_ET_1_30 下载暂存区

本目录对应百度网盘分享中的 `SonyA7S2/BiasFrame_ET_1_30/`，按
《Noise Modeling in One Hour SID Sony 训练复现指南》**24.6 方案B**
组织 ISO 档位。

## 分享来源

- 链接：<https://pan.baidu.com/s/1eLKzjOSDCR4NNLcvbyenEQ?pwd=WAXY>
- 提取码：`WAXY`
- 来源仓库：LLD <https://github.com/happycaoyue/LLD>

## 下载规则（方案B）

1. 只下载 `BiasFrame_ET_1_30`，**不下载** `FlatField_Frame_ET_1_30`。
2. 每个下面列出的 ISO 文件夹先各下 **10 张** `.mat`，并在完整编号序列中
   **均匀抽取**（开头/中间/末尾，如 0001、0200、0400…），不要只连续下
   `0001`–`0010`，以保留传感器升温带来的噪声多样性。
3. **跳过** ISO100000、ISO200000 等高于 ISO25600 的扩展档。
4. 单档单文件约 24.24 MB；25 档 × 10 张 ≈ **5.8 GB**。

## 25 个 ISO 档位文件夹

| 类别 | ISO 文件夹 | 说明 |
|---|---|---|
| 低 ISO（单独下载） | `ISO50` `ISO64` `ISO80` | 方案B：论文只评估 ISO100–25600，这三档合计仅对应 5 个 SID 训练场景；按你的决定仍单独下载 |
| 主档 | `ISO100` `ISO160` `ISO200` `ISO250` `ISO320` `ISO400` `ISO500` `ISO640` `ISO800` | SID 训练 clean RAW 实际出现 ISO（24.5 节） |
| 主档 | `ISO1000` `ISO1250` `ISO1600` `ISO2000` `ISO2500` `ISO4000` `ISO5000` `ISO6400` | 同上 |
| 主档 | `ISO8000` `ISO10000` `ISO12800` `ISO16000` `ISO25600` | 同上 |

完整 25 档：`50 64 80 100 160 200 250 320 400 500 640 800 1000 1250 1600
2000 2500 4000 5000 6400 8000 10000 12800 16000 25600`

## 文件命名约定（实测，见指南 24.2）

`NNNN_ISOXXXX.mat`，例：`0001_ISO100.mat`

每个 MAT 内部变量：

| 变量 | 形状 | dtype | 含义 |
|---|---:|---|---|
| `Inoisy_crop` | 2848×4256 | uint16 | 二维 Bayer RAW mosaic（非 4 通道 packed） |
| `ISO` | 1×1 | int32 | ISO 值 |
| `expo` | 1×1 | single | 曝光时间（秒），ET_1_30 即 1/30 s ≈ 0.0333 s |

## 下载完成后的下一步

1. 核对每个 ISO 文件夹文件数（首轮各 10 张）；
2. 用 `tools/inspect_raw_metadata.py` 读取每张 MAT 的 `ISO`、`expo`，断言与
   文件夹名一致；
3. 建立暗帧索引 `infos/LLD_SonyA7S2_dark.json`；
4. 按 `data/LLD/SonyA7S2/dark/<ISO>/` 重组（纯数字目录名），或保持本目录
   结构并在索引中记录路径。
