---
title: "Noise Modeling in One Hour：SID Sony 训练复现指南"
paper_title: "Noise Modeling in One Hour: Minimizing Preparation Efforts for Self-supervised Low-Light RAW Image Denoising"
authors:
  - "Feiran Li"
  - "Haiyang Jiang"
  - "Daisuke Iso"
year: 2025
venue: "CVPR 2025"
document_type: "reproduction-guide"
scope: "SID Sony-A7S2 training only"
source_pdf: "C:\\Users\\77436\\Desktop\\研究生文件\\课题-ai isp\\ref\\pdf-summary\\ai-isp\\Li 等 - 2025 - Noise Modeling in One Hour Minimizing Preparation Efforts for Self-supervised Low-Light RAW Image D.pdf"
review_date: "2026-07-24"
tags:
  - ai-isp
  - raw-denoising
  - noise-modeling
  - reproduction
  - sid
  - sony-a7s2
aliases:
  - "Noise Modeling in One Hour SID复现"
---

# Noise Modeling in One Hour：SID Sony 训练复现指南

关联论文笔记：[[Noise Modeling in One Hour - Minimizing Preparation Efforts for Self-supervised Low-Light RAW Image Denoising]]

## 1. 文档目标

本文档给出论文在 **SID 数据集 Sony α7S II 子集**上的训练复现方案，目标是补齐官方仓库缺失的训练部分，包括：

- SID Sony 数据下载、整理与训练/评估划分；
- LLD Sony A7S II 暗帧的获取和组织；
- PMN dark shading 资源的使用；
- Bayer RAW 打包与归一化；
- 论文提出的 shot noise 和真实暗帧噪声合成；
- `UNetSeeInDark` 训练；
- SID ×100、×250、×300 的官方评估；
- 关键单元测试、消融实验及结果核查。

本文只复现以下设置：

```text
数据集：SID
相机：Sony α7S II
CFA：Bayer
输入：4通道packed RAW
输出：4通道clean packed RAW
网络：UNetSeeInDark，nf=32
训练方式：clean RAW + 合成噪声，自监督/合成数据训练
评估倍率：×100、×250、×300
```

不包含 ELD、LRID、手机 IMX686、多传感器训练及其他基线复现。

## 2. 复现边界与证据等级

官方仓库目前只提供预训练模型的推理和评估代码，未提供训练数据集类、噪声合成器、优化器配置及训练入口。因此无法保证仅依据公开材料复现出逐比特一致的训练过程。

本文使用以下标记区分证据来源：

| 标记        | 含义                                  |
| --------- | ----------------------------------- |
| `[论文明确]`  | 论文正文、公式、图表直接给出                      |
| `[官方代码]`  | CVPR 2025 官方仓库能够直接确认                |
| `[PMN参考]` | 论文声明沿用 PMN，PMN 代码可作为实现参考，但不等于本文训练源码 |
| `[复现选择]`  | 论文未说明，为形成可执行实现而做出的选择                |
| `[待消融]`   | 存在两种合理解释，应通过实验判断                    |

### 2.1 论文明确给出的训练信息

- SID/ELD 使用 LLD 的 Sony A7S II 标定数据；
- 暗帧约为 **24 个 ISO、每 ISO 约400张**；
- 使用 SID 论文中的 U-Net 作为所有方法统一的 denoising backbone；
- clean image 和 dark frame 随机裁剪为 \(512\times512\) patch；
- 随机水平、垂直翻转；
- Adam 优化器；
- L1 loss；
- 主训练1000 epochs；
- 论文同时提到400 epochs finetuning，但没有说明该阶段是否用于本文纯合成训练结果；
- 对 SID/ELD 使用：

\[
K(\mathrm{ISO})=\frac{\mathrm{ISO}}{100}\times0.1
=\frac{\mathrm{ISO}}{1000};
\]

- 训练和测试均使用 PMN 提供的 dark shading，以保证比较公平；
- 不使用 HBNR（High-Bit Noise Recovery）。

### 2.2 官方代码能够确认的信息

- 网络是4通道输入、4通道输出的 `UNetSeeInDark`；
- 基础通道数 `nf=32`；
- Sony RAW 使用 black level 512、white level 16383；
- Bayer RAW 打包为4通道；
- 真实短曝光图在减去 ISO 相关 dark shading 后归一化；
- 短曝光输入乘以 exposure ratio；
- 输入上限裁剪为1，但保留小于0的值；
- clean target 裁剪到 \([0,1]\)；
- SID 分别在 ×100、×250、×300 下评估；
- 网络输出经过 `ELDIlluminanceCorrect` 全局亮度校正后计算 PSNR/SSIM。

### 2.3 公开材料没有说明的信息

- SID训练时 ×100、×250、×300 的采样概率；
- shot noise 合成时是否显式加入 exposure ratio；
- 每个 epoch 的 patch 数量；
- batch size；
- Adam learning rate 和 betas；
- learning-rate scheduler；
- gradient clipping、AMP、weight decay；
- 400 epochs finetuning 的具体对象；
- LLD暗帧的精确24档 ISO 列表及训练采样顺序；
- 暗帧曝光时间是否参与匹配；
- 训练时使用自行平均的 dark shading，还是直接用 PMN dark shading 从暗帧中去除偏置；
- 模型选择使用 SID 验证集还是 ELD 验证集。

这些项目必须写入配置文件和实验日志，不能隐藏在代码默认值中。

## 3. 目标结果

论文表1中本文方法在 SID Sony 上的结果为：

|倍率|PSNR|SSIM|
|---:|---:|---:|
|×100|43.69|0.9618|
|×250|41.43|0.9486|
|×300|38.06|0.9356|
|平均|40.85|0.9478|

建议把复现成功分成三个层级：

### 3.1 工程复现

- 能下载并解析 SID；
- 能读取 LLD 暗帧；
- 能在线合成 noisy–clean pair；
- loss 正常下降；
- 能加载训练权重运行官方评估；
- 三个 ratio 均能输出合理图像。

### 3.2 方法复现

- `真实暗帧直接采样` 明显优于 `Poisson-only`；
- `减去dark shading的暗帧` 优于 `直接加入未居中暗帧`；
- \(K\) 扩大两倍后性能变化较小；
- 暗帧数量从400降到10时退化较小。

### 3.3 数值复现

- 三个 ratio 的平均 PSNR 接近40.85 dB；
- 建议以 \(\pm0.2\) dB 作为较严格目标；
- 若差距大于0.5 dB，应优先检查 ratio、dark shading、RAW单位和评估协议，而不是先改网络。

由于作者没有发布训练代码和完整超参数，未达到逐数值复现不能直接判定方法实现错误。

## 4. 所需代码和数据

## 4.1 CVPR 2025 官方仓库

仓库：

<https://github.com/SonyResearch/raw_image_denoising>

```bash
git clone https://github.com/SonyResearch/raw_image_denoising.git
cd raw_image_denoising
pip install -r requirements.txt
```

需要复用的文件：

```text
models/ELD_models.py          # UNetSeeInDark
datasets/real_dataset.py      # SID官方评估预处理
datasets/process.py           # RAW/ISP辅助函数
utils/utils.py                # brightness correction和PMN metric
utils/imgproc.py              # RAW转sRGB可视化
get_dataset_infos.py          # 元数据解析参考
resources/SID_Sony_paired.txt # 官方SID评估配对列表
test_denoise_sideld.py        # 官方评估入口
```

注意：`test_denoise_sideld.py` 当前写成：

```python
parser.parse_args(args=[])
```

这会忽略命令行参数。若需要从命令行指定 checkpoint、ratio 和设备，应改成：

```python
parser.parse_args()
```

## 4.2 Python环境

官方 `requirements.txt` 主要版本为：

```text
torch==2.1.0+cu118
torchvision==0.16.0+cu118
rawpy==0.22.0
ExifRead==3.0.0
numpy==2.2.3
scipy==1.15.2
opencv_python==4.10.0.84
imageio==2.34.2
scikit-image==0.18.3
tqdm==4.66.4
```

其中较老的 `scikit-image` 与较新的 NumPy 可能出现环境兼容问题。推荐：

- 首先尝试官方依赖；
- 若安装失败，使用 `numpy<2` 与较新的 `scikit-image`；
- 把最终 `pip freeze` 保存到实验目录；
- PSNR/SSIM库版本变化可能带来轻微数值差异。

## 4.3 SID Sony 数据集

官方直接下载：

<https://storage.googleapis.com/isl-datasets/SID/Sony.zip>

数据约25 GB。解压后应整理成：

```text
data/
└── SID/
    └── Sony/
        ├── long/
        │   ├── 00001_00_10s.ARW
        │   ├── ...
        │   ├── 100xx_00_10s.ARW
        │   └── 200xx_00_10s.ARW
        └── short/
            ├── 00001_00_0.1s.ARW
            ├── 00001_01_0.1s.ARW
            ├── ...
            └── 200xx_xx_0.033s.ARW
```

SID 文件名前缀表示划分：

|首字符|划分|本复现中的用途|
|---:|---|---|
|`0`|train|长曝光clean RAW用于合成训练|
|`1`|test|官方真实pair评估|
|`2`|validation|验证或官方evaltest评估|

同一场景的前5个字符可作为 scene ID。多个 short RAW 可以对应同一个 long RAW。

### 4.3.1 SID训练中真正使用哪些图

本文是噪声合成训练，核心训练数据应为：

```text
SID Sony train split中的long RAW
+ LLD Sony A7S2 dark frames
→ 在线合成noisy RAW
```

不应把 SID train split 中的真实 short RAW 作为网络输入进行监督训练，否则训练设置会变成真实 paired training，而不再是论文的 self-supervised/noise-synthesis 设置。

训练 short RAW 可以用于：

- 读取场景真实 ratio 分布；
- 核查 long/short ISO 是否一致；
- 合成噪声与真实噪声的统计比较；
- 非论文主结果的 domain-gap 诊断。

不能用于本文主模型的损失计算。

## 4.4 LLD Sony A7S II 暗帧

论文使用 LLD：

> Physics-Guided ISO-Dependent Sensor Noise Modeling for Extreme Low-Light Photography

LLD仓库：

<https://github.com/happycaoyue/LLD>

LLD数据下载：

<https://pan.baidu.com/s/1eLKzjOSDCR4NNLcvbyenEQ?pwd=WAXY>

提取码：`WAXY`

需要从完整 LLD 数据中取得：

```text
相机：Sony A7S II
类型：bias/dark frames
数量：约400张/ISO
ISO：论文称24档
```

下载后建议重新组织为：

```text
data/
└── LLD/
    └── SonyA7S2/
        └── dark/
            ├── 100/
            │   ├── *.ARW
            │   └── ...
            ├── 125/
            ├── 160/
            ├── ...
            └── 25600/
```

不要只相信文件夹名字。首次建立索引时应读取每张RAW的EXIF并检查：

- camera model；
- ISO；
- exposure time；
- RAW尺寸；
- black level；
- white level；
- CFA pattern。

### 4.4.1 如果拿不到 LLD 原始暗帧

官方提供的 pre-computed dark shading 只包含均值/拟合结果，不能替代每一张独立暗帧，因为本文的核心是直接采样真实 signal-independent noise。

因此：

- 只有 `darkshading_*.npy/pkl`：可以做真实SID推理，但不能完整训练本文方法；
- 有少量同型号相机暗帧：可以进行方法复现，但不是论文数据复现；
- 用高斯噪声替代暗帧：只能作为 P-G baseline；
- 用 SID 的黑暗图像伪装暗帧：会混入场景信号，不建议作为主实验。

## 4.5 PMN dark shading资源

本文说明训练和测试始终使用 PMN dark shading 以保证公平比较。

官方仓库提供下载入口：

<https://drive.google.com/drive/folders/18YUiNsSH-YR9L5KJZGP23-DsOvJC0JWz?usp=sharing>

下载后目录应包含：

```text
resources/
└── SonyA7S2/
    ├── darkshading_BLE.pkl
    ├── darkshading_highISO_k.npy
    ├── darkshading_highISO_b.npy
    ├── darkshading_lowISO_k.npy
    └── darkshading_lowISO_b.npy
```

官方代码中的 dark shading 为：

\[
DS(\mathrm{ISO})=
\begin{cases}
k_{\mathrm{low}}\cdot\mathrm{ISO}
+b_{\mathrm{low}}+\mathrm{BLE}[\mathrm{ISO}],&
\mathrm{ISO}\le1600,\\
k_{\mathrm{high}}\cdot\mathrm{ISO}
+b_{\mathrm{high}}+\mathrm{BLE}[\mathrm{ISO}],&
\mathrm{ISO}>1600.
\end{cases}
\]

这里的 dark shading 是 black level 512 之外的附加空间偏置。官方评估顺序是：

```python
lr_raw = lr_raw - dark_shading
lr_norm = (lr_raw - 512) / (16383 - 512)
```

因此不能把 PMN dark shading 当作已经包含完整 black level 的图。

## 5. 推荐工程目录

在官方仓库中补充：

```text
raw_image_denoising/
├── configs/
│   └── train_sid_sony.yaml
├── datasets/
│   ├── real_dataset.py
│   └── sid_synthetic_train.py
├── noise/
│   ├── dark_frame_bank.py
│   └── sid_noise_synthesis.py
├── tools/
│   ├── build_sid_index.py
│   ├── build_dark_index.py
│   ├── inspect_raw_metadata.py
│   ├── validate_dark_frames.py
│   └── compare_synthetic_real_noise.py
├── models/
│   └── ELD_models.py
├── train_sid_sony.py
├── test_denoise_sideld.py
├── infos/
│   ├── SID_train_clean.json
│   ├── SID_evaltest.info
│   └── LLD_SonyA7S2_dark.json
├── resources/
│   └── SonyA7S2/
├── checkpoints/
└── experiments/
```

推荐使用 JSON/CSV 保存新建的训练索引，以便人工核查。官方 `.info` 为 pickle，不易审计。

## 6. SID索引生成

## 6.1 不建议直接照搬官方训练索引分组

官方 `get_dataset_infos.py` 的 `get_SID_info()` 可作为参考，但训练 short 文件的分组代码应重新检查。更安全的方式是：

1. 扫描 `long/*.ARW`；
2. 只选择文件名首字符为 `0` 的训练图；
3. 使用文件名前5个字符作为 scene ID；
4. 在 `short/` 中查找相同 scene ID 的全部文件；
5. 分别读取 long 和每个 short 的真实 EXIF；
6. 计算实际 ratio；
7. 对 ISO、尺寸和CFA做断言。

推荐的索引结构：

```json
{
  "scene_id": "00001",
  "long": "/data/SID/Sony/long/00001_00_10s.ARW",
  "long_iso": 250,
  "long_exposure": 10.0,
  "shorts": [
    {
      "path": "/data/SID/Sony/short/00001_00_0.1s.ARW",
      "iso": 250,
      "exposure": 0.1,
      "ratio": 100.0
    }
  ],
  "wb": [],
  "ccm": []
}
```

ratio应使用元数据计算：

\[
r=\frac{t_{\mathrm{long}}}{t_{\mathrm{short}}}.
\]

建议保存浮点值，并额外映射到最近的 SID 类别：

```python
ratio_class = min([100, 250, 300], key=lambda x: abs(x - ratio))
```

同时检查误差：

```python
assert abs(ratio - ratio_class) / ratio_class < 0.05
```

## 6.2 防止数据泄漏

训练 clean image 只能来自前缀 `0`。

官方评估应使用：

```text
resources/SID_Sony_paired.txt
```

不要把该文件中的 long RAW 加入训练 clean pool。生成索引后至少执行：

```python
assert set(train_scene_ids).isdisjoint(eval_scene_ids)
```

## 7. RAW数据表示

## 7.1 Sony A7S II数值范围

按官方代码：

```python
BLACK_LEVEL = 512
WHITE_LEVEL = 16383
DYNAMIC_RANGE = WHITE_LEVEL - BLACK_LEVEL  # 15871
```

归一化：

\[
x_{\mathrm{norm}}
=\frac{x_{\mathrm{DN}}-512}{16383-512}.
\]

clean target：

```python
target = clip(target, 0, 1)
```

noisy input：

```python
input = minimum(input, 1)
```

不要在送入网络前把 input 的下界裁成0。官方评估设置 `clip_low=False`，dark-shading校正后的负值是噪声分布的一部分。

## 7.2 Bayer打包

官方打包方式：

```python
packed = np.stack(
    [
        raw[0::2, 0::2],
        raw[0::2, 1::2],
        raw[1::2, 0::2],
        raw[1::2, 1::2],
    ],
    axis=-1,
)
```

输出形状：

```text
原始mosaic：H × W
packed RAW：H/2 × W/2 × 4
PyTorch：4 × H/2 × W/2
```

训练 patch 应采用 **packed RAW 域 \(512\times512\)**，对应 mosaic 域 \(1024\times1024\)。依据是：

- 官方网络输入为4通道；
- PMN 的 Sony A7S II 配置使用 `patch_size: 512`；
- 论文说明 clean image 和 dark frame 均裁成512 patch。

该解释不是论文逐字说明，需在实验配置中记录。

## 7.3 数据增强

论文只明确使用：

- horizontal flip；
- vertical flip。

不建议在主复现中加入：

- 90°旋转；
- 任意角度旋转；
- resize；
- JPEG压缩；
- RGB颜色增强；
- 随机gamma。

在packed RAW上做水平、垂直翻转不会改变四个CFA子通道的语义。若在mosaic上翻转，则需要处理CFA相位变化。

## 8. DarkFrameBank实现

## 8.1 暗帧索引

`DarkFrameBank` 至少提供：

```python
class DarkFrameBank:
    def available_isos(self) -> list[int]: ...
    def map_iso(self, requested_iso: int) -> int: ...
    def sample_path(self, iso: int) -> str: ...
    def get_dark_shading(self, iso: int) -> np.ndarray: ...
    def sample_residual_patch(
        self,
        iso: int,
        top: int,
        left: int,
        patch_size: int,
    ) -> np.ndarray: ...
```

## 8.2 ISO匹配

默认规则：

1. 优先选择完全相同 ISO；
2. 缺少该 ISO 时，在 \(\log_2(\mathrm{ISO})\) 空间找最近档；
3. 记录 requested ISO 与 matched ISO；
4. 不允许静默跨越已知DCG/读出模式切换点；
5. 统计每个 epoch 的匹配分布。

```python
distance = abs(np.log2(dark_iso / requested_iso))
```

论文没有公布24档 ISO 的完整列表，因此代码不能把列表硬编码成未经验证的值。应从 LLD 暗帧 EXIF 自动建立 `available_isos`。

## 8.3 两种dark-shading模式

### 模式A：paper-fair，主复现默认

使用 PMN dark shading：

\[
N_{\mathrm{dark,DN}}
=D_{\mathrm{raw}}
-512
-DS_{\mathrm{PMN}}(\mathrm{ISO}).
\]

这与论文“训练和测试均使用PMN dark shading”的表述最一致。

### 模式B：method-pure

使用同ISO暗帧均值：

\[
\overline D_g=\frac1M\sum_{j=1}^{M}D_{g,j},
\]

\[
N_{\mathrm{dark,DN}}=D_{g,j}-\overline D_g.
\]

该模式更贴近论文方法图中“用收集到的暗帧平均得到dark shading”，但不一定与论文表1公平比较设置一致。

两种模式必须分开报告，不能混合。

## 8.4 暗帧裁剪

推荐先把整张暗帧处理为packed 4通道残差，再从packed域随机裁剪 \(512\times512\)。

clean patch与dark patch的坐标独立采样：

```text
clean patch：保留场景结构
dark patch：作为独立真实噪声样本
```

暗帧不能先裁到0–1；必须保留负残差。

## 9. Shot noise实现

## 9.1 论文公式

论文将 clean image \(I\) 的shot-noisy版本写为：

\[
Y_{\mathrm{shot}}
=K\cdot\mathcal P\left(\frac{I}{K}\right),
\]

并有：

\[
\mathbb E[Y_{\mathrm{shot}}]=I,\qquad
\operatorname{Var}(Y_{\mathrm{shot}})=KI.
\]

对 SID/ELD：

\[
K_{\mathrm{DN/e^-}}=\frac{\mathrm{ISO}}{1000}.
\]

建议始终在 DN 域调用 Poisson，避免把 DN/e⁻ 的 \(K\) 直接误用到归一化图像。

```python
k_dn = iso / 1000.0
target_dn = target_norm * (WHITE_LEVEL - BLACK_LEVEL)
shot_dn = k_dn * torch.poisson(torch.clamp(target_dn / k_dn, min=0))
```

若要在归一化域实现：

\[
K_{\mathrm{norm}}
=\frac{K_{\mathrm{DN}}}{16383-512}.
\]

## 9.2 SID exposure ratio的关键歧义

官方评估把真实短曝光RAW乘以 \(r\in\{100,250,300\}\)，而论文噪声合成公式只写了 \(K\)，没有说明数字增亮倍率如何进入训练合成。

因此至少有两种实现。

### 实现A：ratio-aware，主复现推荐

令 \(I_{\mathrm{DN}}\) 为长曝光clean target在去black level后的DN值，短曝光期望信号为：

\[
I_{\mathrm{short}}=\frac{I_{\mathrm{DN}}}{r}.
\]

先在短曝光域生成shot noise：

\[
Z_{\mathrm{shot}}
=K_g\cdot\mathcal P
\left(\frac{I_{\mathrm{DN}}}{rK_g}\right).
\]

加入同ISO暗帧残差，然后按SID评估方式增亮：

\[
Y_{\mathrm{DN}}
=r\left(Z_{\mathrm{shot}}+N_{\mathrm{dark,DN}}\right).
\]

最终：

\[
Y_{\mathrm{norm}}
=\frac{Y_{\mathrm{DN}}}{16383-512},
\qquad
T_{\mathrm{norm}}
=\frac{I_{\mathrm{DN}}}{16383-512}.
\]

等价写法是：

\[
K_{\mathrm{eff}}=rK_g,
\]

\[
Y_{\mathrm{shot}}
=K_{\mathrm{eff}}
\mathcal P\left(\frac{I_{\mathrm{DN}}}{K_{\mathrm{eff}}}\right),
\]

\[
Y_{\mathrm{DN}}
=Y_{\mathrm{shot}}+rN_{\mathrm{dark,DN}}.
\]

该实现与真实短曝光乘ratio后的方差最一致：

\[
\operatorname{Var}(Y_{\mathrm{shot}})
=rK_gI_{\mathrm{DN}},
\]

\[
\operatorname{Var}(rN_{\mathrm{dark}})
=r^2\operatorname{Var}(N_{\mathrm{dark}}).
\]

### 实现B：paper-literal，必须消融

完全按论文图和公式：

\[
Y_{\mathrm{DN}}
=K_g\mathcal P
\left(\frac{I_{\mathrm{DN}}}{K_g}\right)
+N_{\mathrm{dark,DN}}.
\]

不显式使用 ratio。

该版本更接近论文公式字面表达，但可能低估SID数字增亮后的噪声强度。必须作为消融运行，判断作者是否在数据管线的其他位置隐式处理了ratio。

### 推荐结论

主复现使用 ratio-aware；同时运行 paper-literal。若 paper-literal 明显更接近论文数值，应进一步检查：

- 作者是否把 effective ISO/K 预先乘了ratio；
- clean target是否先做了曝光缩放；
- 网络是否在低亮度域训练、输出后再乘ratio；
- PMN训练代码中的 `ori` 设置。

## 9.3 Ratio采样

论文没有说明 ratio 的训练采样方式。

主复现推荐：

```python
ratio = random.choice([100, 250, 300])
```

即三类均匀采样。原因是论文最终对三类取算术平均，均匀训练可以避免×300样本被低频类别压制。

需要额外运行：

|设置|含义|
|---|---|
|`uniform_class`|100/250/300各1/3，主复现|
|`empirical_sid`|按SID训练short文件真实分布采样|
|`per_scene_available`|只从该clean scene实际存在的ratio中采样|
|`fixed_300`|只训练最困难×300，诊断极端噪声|

## 10. 完整噪声合成伪代码

推荐在DN域实现：

```python
def synthesize_sid_ratio_aware(
    clean_norm,          # [4,H,W], [0,1]
    dark_residual_dn,    # [4,H,W], signed DN
    iso,
    ratio,
    dynamic_range=15871.0,
):
    # clean target, black level already removed
    clean_dn = torch.clamp(clean_norm, 0.0, 1.0) * dynamic_range

    # paper's hypothesized system gain
    k_dn = float(iso) / 1000.0

    # short-exposure expected signal
    poisson_rate = torch.clamp(clean_dn / (ratio * k_dn), min=0.0)
    shot_short_dn = k_dn * torch.poisson(poisson_rate)

    # align short exposure to long-exposure brightness
    noisy_dn = ratio * (shot_short_dn + dark_residual_dn)

    noisy_norm = noisy_dn / dynamic_range

    # Match official evaluation: preserve negative values,
    # clip only the high end.
    noisy_norm = torch.clamp_max(noisy_norm, 1.0)
    target_norm = torch.clamp(clean_norm, 0.0, 1.0)

    return noisy_norm, target_norm
```

### 10.1 不要犯的单位错误

错误写法：

```python
torch.poisson(clean_norm / k_dn) * k_dn
```

这里 `clean_norm` 是0–1，`k_dn` 是DN/e⁻，单位不一致。

正确方式：

- 要么把 clean 转回DN；
- 要么把 \(K\) 除以 dynamic range 转成归一化单位。

### 10.2 暗帧不能重复减black level

PMN dark shading模式下：

```python
dark_residual_dn = dark_raw - BLACK_LEVEL - pmn_dark_shading
```

暗帧均值模式下：

```python
dark_residual_dn = dark_raw - mean_dark_raw
```

第二种写法已经去掉完整均值，不能再减512。

## 11. 训练Dataset实现

`SIDSyntheticTrainDataset.__getitem__()` 推荐流程：

1. 根据索引选择一个 SID train long RAW；
2. 读取 scene ISO；
3. 将 long RAW 按 black/white level 归一化；
4. 打包为4通道；
5. 随机裁剪一个 \(512\times512\) clean patch；
6. 从 \(\{100,250,300\}\) 选择 ratio；
7. 将 scene ISO 映射到暗帧 ISO；
8. 随机选择一张同ISO暗帧；
9. 计算或读取该ISO dark shading；
10. 随机裁剪独立的 \(512\times512\) dark residual patch；
11. 合成shot noise；
12. 加入dark residual；
13. 执行ratio增亮；
14. 对input只裁上界，对target裁到0–1；
15. 对input/target同步执行水平、垂直翻转；
16. 返回input、target和调试元数据。

返回结构：

```python
{
    "lr": noisy_patch,           # [4,512,512]
    "hr": clean_patch,           # [4,512,512]
    "iso": iso,
    "matched_dark_iso": dark_iso,
    "ratio": ratio,
    "clean_path": clean_path,
    "dark_path": dark_path,
}
```

### 11.1 一个epoch包含多少patch

论文没有说明。

可参考 PMN 的 Sony 配置，每张全分辨率图取8个patch。推荐：

```text
crop_per_image = 8
每个epoch patch数 = clean image数 × 8
```

两种实现均可：

- Dataset每次返回8个patch，DataLoader batch size=1；
- Dataset把每个 `(image, crop_id)` 当成独立样本，batch size=8。

第二种更容易实现多进程加载。

## 12. 网络实现

直接复用：

```text
models/ELD_models.py::UNetSeeInDark
```

网络结构：

|阶段|通道数|操作|
|---|---:|---|
|Encoder 1|32|2×Conv3×3 + MaxPool|
|Encoder 2|64|2×Conv3×3 + MaxPool|
|Encoder 3|128|2×Conv3×3 + MaxPool|
|Encoder 4|256|2×Conv3×3 + MaxPool|
|Bottleneck|512|2×Conv3×3|
|Decoder 4|256|TransposedConv + skip + 2×Conv|
|Decoder 3|128|TransposedConv + skip + 2×Conv|
|Decoder 2|64|TransposedConv + skip + 2×Conv|
|Decoder 1|32|TransposedConv + skip + 2×Conv|
|Output|4|Conv1×1|

特点：

- 输入4通道；
- 输出4通道；
- LeakyReLU slope=0.2；
- 无BatchNorm；
- 无残差输出，直接预测clean RAW；
- 下采样4次，因此输入尺寸应能被16整除。

### 12.1 官方padding实现的边界行为

官方代码：

```python
mod_pad_h = ds_scale - x.shape[2] % ds_scale
mod_pad_w = ds_scale - x.shape[3] % ds_scale
```

当尺寸已经能被16整除时，它仍会额外padding 16像素。为了与官方 checkpoint 行为一致，主复现应先保留该实现。

更常见的正确写法为：

```python
mod_pad_h = (ds_scale - h % ds_scale) % ds_scale
mod_pad_w = (ds_scale - w % ds_scale) % ds_scale
```

但修改后会改变边界行为。若修正，应作为独立实验，不要与主复现混用。

## 13. 训练超参数

## 13.1 论文明确值

```yaml
patch_size: 512
epochs: 1000
optimizer: Adam
loss: L1
horizontal_flip: true
vertical_flip: true
```

## 13.2 推荐补全值

以下数值主要参考 PMN 的 Sony A7S II 配置，不是本文明确公布：

```yaml
seed: 1
model:
  name: UNetSeeInDark
  in_channels: 4
  out_channels: 4
  nf: 32

data:
  black_level: 512
  white_level: 16383
  patch_size: 512
  crops_per_image: 8
  ratios: [100, 250, 300]
  ratio_sampling: uniform_class
  preserve_negative_input: true
  clip_input_high: true
  clip_target: true

noise:
  synthesis: ratio_aware
  k_mode: hypothesized_narrow
  k_scale: 0.1
  dark_shading: pmn
  use_hbnr: false
  iso_matching: exact_then_nearest_log2

train:
  epochs: 1000
  batch_size: 1
  learning_rate: 2.0e-4
  betas: [0.9, 0.999]
  weight_decay: 0
  scheduler: warmup_cosine
  amp: false
  grad_clip: null
  loss: l1
  clamp_prediction_for_loss: true
  save_every: 10
  validate_every: 10
```

建议第一轮关闭 AMP，以减少不同GPU和Poisson采样精度带来的差异。跑通后再开启。

### 13.3 400 epochs finetuning如何处理

本文“1000 epochs with 400 epochs for finetuning”表述不够明确。对本文纯合成 Ours 主结果，推荐：

- 主模型只训练1000 epochs；
- 不使用真实SID short–long pair微调；
- 不把额外400 epochs加入主结果。

若用真实pair微调400 epochs，训练设置会从 self-supervised 变成 hybrid，应单独命名为：

```text
Ours-synthetic-pretrain + SID-real-finetune
```

不能与论文表中的 self-supervised Ours 混淆。

## 14. 训练循环

推荐实现：

```python
model = UNetSeeInDark(in_nc=4, out_nc=4, nf=32).cuda()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=0.0,
)
criterion = torch.nn.L1Loss()

for epoch in range(1, 1001):
    model.train()

    for batch in train_loader:
        noisy = batch["lr"].cuda(non_blocking=True)
        clean = batch["hr"].cuda(non_blocking=True)

        pred = model(noisy)

        # PMN trainer uses clipped prediction for L1.
        # Paper only states L1, so expose this as a config.
        pred_for_loss = pred.clamp(0, 1)
        loss = criterion(pred_for_loss, clean)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    scheduler.step()

    if epoch % validate_every == 0:
        validate_sid_ratios(model, ratios=[100, 250, 300])

    if epoch % save_every == 0:
        save_checkpoint(...)
```

checkpoint至少保存：

```python
{
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "config": resolved_config,
    "rng_state": ...,
    "git_commit": ...,
}
```

最终导出给官方评估脚本的文件应只包含：

```python
model.state_dict()
```

因为官方代码直接：

```python
model.load_state_dict(torch.load(cp_dir), strict=True)
```

## 15. 评估复现

## 15.1 生成SID评估索引

按官方README：

```bash
python get_dataset_infos.py \
  --dstname SID \
  --root_dir /data/SID/Sony \
  --mode evaltest
```

应生成：

```text
infos/SID_evaltest.info
```

评估使用：

```text
resources/SID_Sony_paired.txt
```

官方评估列表按ratio分为：

```text
×100：40组
×250：40组
×300：其余组
```

不要随机重排 `.info` 后再使用官方 `evaltest_remap()`，因为它按位置切分ratio。

## 15.2 官方真实输入处理

对每个pair：

```python
lr_raw = short_raw - PMN_dark_shading(iso)
lr = pack_and_normalize(lr_raw, black=512, white=16383)
hr = pack_and_normalize(long_raw, black=512, white=16383)
lr = lr * ratio
lr = clamp_max(lr, 1)
hr = clamp(hr, 0, 1)
```

网络：

```python
pred = model(lr)
```

亮度校正：

\[
\alpha
=\frac{\langle pred,gt\rangle}
{\langle pred,pred\rangle+\epsilon},
\]

\[
pred_{\mathrm{corrected}}=\alpha pred.
\]

饱和GT像素被排除在比例估计之外。

然后：

```python
pred = pred.clamp(0, 1)
gt = gt.clamp(0, 1)
```

在4通道RAW域计算PSNR和SSIM。

## 15.3 运行三个ratio

```bash
python test_denoise_sideld.py \
  --cp_dir checkpoints/sonya7s2_reproduced.pth \
  --testset_type sid \
  --eval_ratio 100

python test_denoise_sideld.py \
  --cp_dir checkpoints/sonya7s2_reproduced.pth \
  --testset_type sid \
  --eval_ratio 250

python test_denoise_sideld.py \
  --cp_dir checkpoints/sonya7s2_reproduced.pth \
  --testset_type sid \
  --eval_ratio 300
```

运行前需将官方脚本的 `parse_args(args=[])` 改为 `parse_args()`。

## 16. 必须实现的单元测试

## 16.1 RAW打包测试

- packed四通道位置与官方 `SIDEvalDataset.pack_raw()` 完全一致；
- unpack(pack(raw)) 能恢复原图；
- crop尺寸为 `[4,512,512]`；
- 不发生RGB/RGGB通道重排。

## 16.2 Shot noise均值测试

固定clean patch，重复采样至少1000次：

\[
\left|\operatorname{mean}(Y)-I\right|
\]

应接近0。

ratio-aware版本在数字增亮后同样应满足：

\[
\mathbb E[Y]=I.
\]

## 16.3 Shot noise方差测试

对不同亮度bin拟合：

\[
\operatorname{Var}(Y|I)\approx rKI.
\]

斜率应接近 \(rK\)。如果接近 \(K\) 而不是 \(rK\)，说明ratio没有进入shot noise。

## 16.4 Dark residual均值测试

每个 ISO 随机抽取大量暗帧patch：

\[
\operatorname{mean}(N_{\mathrm{dark}})\approx0.
\]

还要分别检查RGGB四通道均值，防止某个CFA通道存在残余偏置。

## 16.5 负值保留测试

dark-shading校正后应允许：

```python
(noisy_input < 0).any() == True
```

如果所有输入都被裁到非负，训练分布与官方评估不一致。

## 16.6 ISO匹配测试

对每个 SID train clean ISO：

- 输出匹配的dark ISO；
- 统计距离；
- 所有nearest匹配写入日志；
- 不允许无提示回退到固定ISO。

## 16.7 数据泄漏测试

训练long scene ID与官方evaltest scene ID交集必须为空。

## 17. 必须运行的消融实验

## 17.1 \(K\) 鲁棒性

|设置|公式|
|---|---|
|Narrow| \(K=\mathrm{ISO}/100\times0.1\) |
|Broad| \(K=\mathrm{ISO}/100\times0.2\) |
|Calibrated|如能取得LLD标定值，则使用真实值|

目标是验证 Narrow 与 Broad 的SID平均PSNR差距是否小于约0.1 dB。

## 17.2 暗帧数量

每 ISO 分别使用：

```text
400、280、160、40、10
```

论文表5结果显示 SID 在400张和10张之间约下降0.05 dB：

```text
400 frames：40.85 dB
10 frames：40.80 dB
```

抽取子集时固定随机种子，并保证每次训练使用同一子集。

## 17.3 Ratio实现

至少比较：

```text
A：ratio-aware
B：paper-literal
C：只放大shot noise，不放大dark residual
D：只放大dark residual，不调整shot noise
```

该消融是补齐未公开训练代码时最关键的实验。

## 17.4 Dark shading

比较：

```text
PMN dark shading
LLD dark-frame mean
不减dark shading
```

若“不减dark shading”出现颜色或边缘偏置，符合论文分析。

## 17.5 HBNR

主结果：

```text
HBNR = false
```

可选消融：

```text
HBNR = true
```

但HBNR实现需要噪声分布拟合，不属于本文简化方案的必要部分。

## 17.6 输入下界裁剪

比较：

```text
保留负值
clip到[0,1]
```

官方评估保留负值，因此主复现应采用前者。

## 18. 诊断与排错顺序

### 18.1 ×100尚可、×300明显很差

优先检查：

1. ratio是否进入shot noise；
2. dark residual是否乘ratio；
3. ratio训练采样是否过度偏向×100；
4. ×300输入是否提前裁剪到0；
5. 训练时是否错误使用同一固定噪声强度。

### 18.2 所有ratio都有整体颜色偏移

优先检查：

1. dark frame是否减去dark shading；
2. black level是否重复减；
3. PMN dark shading是否被误当作含512的完整偏置；
4. RGGB通道顺序；
5. 训练与评估是否使用不同归一化。

### 18.3 loss下降但真实SID性能低

优先检查：

1. 是否错误地在归一化域直接使用DN单位的 \(K\)；
2. LLD暗帧是否来自Sony A7S II；
3. ISO是否正确匹配；
4. synthetic noise与真实short noise的方差—信号曲线；
5. 输入负值是否保留；
6. 官方brightness correction和PMN metric是否一致；
7. 训练是否误用了paper-literal而真实评估需要ratio-aware。

### 18.4 PSNR比论文低约0.1–0.3 dB

可能来自：

- 未公开的LR、scheduler、batch size；
- LLD/PMN暗帧版本不同；
- dark shading版本不同；
- 随机暗帧子集；
- padding边界实现；
- scikit-image版本；
- epoch定义不同。

这一级差距应先多跑种子，不建议立刻修改核心噪声模型。

### 18.5 PSNR比论文低1 dB以上

通常不是随机种子问题，应检查数据、ratio、单位、dark shading和评估协议。

## 19. 推荐实施顺序

### 阶段1：只打通官方评估

1. 下载SID；
2. 下载官方pretrained checkpoint；
3. 下载PMN dark shading；
4. 生成 `SID_evaltest.info`；
5. 复现官方 ×100、×250、×300 数值。

如果这一步不能复现，暂时不要实现训练。

### 阶段2：验证数据和噪声模块

1. 建立SID train clean索引；
2. 建立LLD暗帧索引；
3. 验证black/white level；
4. 验证packed RAW；
5. 验证dark residual零均值；
6. 验证Poisson均值和方差；
7. 保存若干synthetic/real residual对比图。

### 阶段3：小规模过拟合

使用：

```text
2张clean RAW
每ISO 10张暗帧
固定ratio=100
固定ISO
固定patch
```

检查网络能否把训练patch过拟合到高PSNR。若不能，先修复网络、loss或数据尺度。

### 阶段4：单ratio训练

分别训练：

```text
ratio=100
ratio=250
ratio=300
```

用于验证噪声强度和评估管线。

### 阶段5：统一多ratio模型

使用100/250/300均匀采样，训练1000 epochs，得到论文主设置。

### 阶段6：消融和数值对齐

按以下优先级：

1. ratio-aware vs paper-literal；
2. PMN DS vs dark mean；
3. Narrow K vs Broad K；
4. 400 vs 10 dark frames；
5. prediction clamp；
6. scheduler和batch size；
7. padding修正。

## 20. 推荐训练命令设计

```bash
python tools/build_sid_index.py \
  --sid-root /data/SID/Sony \
  --split train \
  --output infos/SID_train_clean.json

python tools/build_dark_index.py \
  --dark-root /data/LLD/SonyA7S2/dark \
  --output infos/LLD_SonyA7S2_dark.json

python tools/validate_dark_frames.py \
  --index infos/LLD_SonyA7S2_dark.json \
  --resources resources/SonyA7S2

python train_sid_sony.py \
  --config configs/train_sid_sony.yaml
```

消融：

```bash
python train_sid_sony.py \
  --config configs/train_sid_sony.yaml \
  noise.synthesis=paper_literal \
  experiment.name=sid_paper_literal

python train_sid_sony.py \
  --config configs/train_sid_sony.yaml \
  noise.k_scale=0.2 \
  experiment.name=sid_broad_k

python train_sid_sony.py \
  --config configs/train_sid_sony.yaml \
  noise.dark_frames_per_iso=10 \
  experiment.name=sid_dark10
```

## 21. 实验日志必须记录

每次训练至少保存：

- SID下载版本和文件数量；
- train/eval scene ID；
- LLD暗帧来源；
- 每个ISO暗帧数量；
- available ISO列表；
- SID ISO到暗帧 ISO的匹配表；
- dark-shading模式和资源hash；
- \(K\) 公式；
- ratio合成公式；
- ratio采样分布；
- RAW black/white level；
- patch定义在mosaic域还是packed域；
- optimizer、LR、scheduler、batch size；
- epoch内patch数；
- random seed；
- code commit；
- Python/CUDA/PyTorch/rawpy版本；
- ×100、×250、×300逐项指标；
- 是否使用brightness correction；
- 是否使用HBNR；
- 是否保留输入负值。

没有这些记录，即使得到接近论文的PSNR，也很难证明训练流程可重复。

## 22. 最终推荐的“主复现配置”

综合论文描述、官方评估代码和物理一致性，建议把以下配置作为第一版主复现：

```text
训练clean：SID Sony train split的long RAW
真实short：不参与训练loss
暗帧：LLD Sony A7S II，全部可用暗帧
ISO匹配：相同ISO优先，缺失时log2最近邻
dark shading：PMN资源
RAW范围：BL=512，WL=16383
patch：packed RAW 512×512
ratio：100/250/300均匀采样
噪声公式：ratio-aware
K：ISO/1000，DN/e-
HBNR：关闭
输入裁剪：只裁上界1，保留负值
target：裁到[0,1]
增强：水平/垂直翻转
网络：官方UNetSeeInDark，nf=32
loss：L1
optimizer：Adam
LR：2e-4，作为PMN参考值
scheduler：warmup cosine，作为PMN参考值
epochs：1000
真实pair微调：不使用
评估：官方SIDEvalDataset + PMN dark shading
指标：官方brightness correction后的RAW PSNR/SSIM
```

同时必须运行 `paper-literal` 合成版本。只有比较这两个版本，才能判断论文未公开的 ratio 处理细节是否是复现差距的主要来源。

## 23. 复现结论

这篇论文的方法可以根据公开信息实现，但只能称为 **best-evidence reproduction**，不能声称是作者训练代码的精确重建。

最关键的实现点不是U-Net，而是：

1. 在正确RAW单位下生成Poisson shot noise；
2. 使用与ISO对应的真实暗帧；
3. 正确减去dark shading和black level；
4. 不对暗帧做统计分布拟合或HBNR；
5. 训练和评估保持相同的负值、归一化与brightness correction协议；
6. 明确处理 SID exposure ratio；
7. 把所有未公开参数作为配置和消融，而不是隐藏假设。

如果主复现结果与论文相差较大，优先验证 `ratio-aware vs paper-literal`，其次检查 PMN dark shading 和 DN/归一化单位。这三项比调整网络结构更可能解释差异。

## 24. LLD选择性下载与MAT实测

### 24.1 目录作用判断

LLD Sony A7S II 数据中：

```text
SonyA7S2/
├── BiasFrame_ET_1_30/
└── FlatField_Frame_ET_1_30/
    ├── LUX45/
    ├── LUX65/
    ├── LUX85/
    ├── LUX100/
    ├── LUX123/
    └── LUX145/
```

对 `Noise Modeling in One Hour` 的 SID Sony 主训练：

| 目录                        | 是否需要 | 原因                                   |
| ------------------------- | ---- | ------------------------------------ |
| `BiasFrame_ET_1_30`       | 需要   | 论文直接采样真实暗帧作为signal-independent noise |
| `FlatField_Frame_ET_1_30` | 不需要  | 用于受控照明下的系统增益/噪声参数标定，而本文主方法明确跳过该标定    |
| 不同`LUX*`目录                | 不需要  | 只属于flat-field标定                      |
| ISO高于25600的BiasFrame      | 不需要  | 本文SID/ELD实验范围为ISO 100–25600          |

因此选择性下载时，应优先：

```text
只下载 BiasFrame_ET_1_30
只下载所需ISO文件夹
每个ISO先下载10张MAT
```

不要下载 `FlatField_Frame_ET_1_30`，除非计划额外复现论文表3中的 calibrated-\(K\) 对照，或自行完成 photon-transfer/variance–mean 标定。

### 24.2 两个实测MAT文件

本次实际读取：

```text
F:\baidudownload_sub\0001_ISO100.mat
来源：BiasFrame_ET_1_30/ISO100

F:\baidudownload_sub\0002_ISO100.mat
来源：FlatField_Frame_ET_1_30/LUX100/ISO100
```

两个文件均为 MATLAB 5.0 MAT-file，大小均为24,242,488 bytes，内部变量完全一致：

|变量|形状|dtype|含义|
|---|---:|---|---|
|`Inoisy_crop`|2848×4256|uint16|二维Bayer RAW mosaic|
|`ISO`|1×1|int32|ISO值|
|`expo`|1×1|single|曝光时间，单位秒|

两者均有：

```text
ISO = 100
expo = 0.033333335 s ≈ 1/30 s
```

`Inoisy_crop` 的尺寸与PMN Sony A7S II配置中的 `H=2848, W=4256` 完全一致。它不是4通道packed RAW，而是原始二维Bayer mosaic，需要按四个CFA相位打包。

MAT文件没有保存：

- black level；
- white level；
- CFA名称/方向；
- white balance；
- CCM；
- LUX；
- 温度；
- 相机序列号。

这些信息需要从数据集说明、路径或Sony A7S II配置中补充。LUX只编码在FlatField目录名中。

### 24.3 BiasFrame实测统计

`BiasFrame_ET_1_30/ISO100/0001_ISO100.mat`：

|统计量|数值，DN|
|---|---:|
|均值|513.5358|
|标准差|1.0177|
|中位数|514|
|最小值|502|
|最大值|608|
|1%分位数|511|
|99%分位数|516|

四个Bayer相位：

|相位|均值|标准差|中位数|
|---|---:|---:|---:|
|00|513.5478|1.0144|514|
|01|513.5255|1.0192|514|
|10|513.5247|1.0202|514|
|11|513.5454|1.0168|514|

四个相位均值几乎相同，且整体紧贴Sony black level 512，证明这是有效暗帧。

尾部分布还保留了真实异常像素：

```text
像素总数：12,121,088
像素值 ≥ 518：2,360个
像素值 ≥ 520：254个
最大值608：1个明显离群/热像素
```

这些非高斯尾部和异常像素正是本文直接采样暗帧、而不只拟合高斯分布的价值之一。

单帧均值比512高约1.54 DN，说明暗帧中仍包含black-level/dark-shading偏置。训练前不能直接把原始 `Inoisy_crop` 加到clean RAW上，必须先减去：

- PMN ISO相关dark shading和black level；或
- 同ISO多张暗帧的完整均值图。

### 24.4 FlatField实测统计

`FlatField_Frame_ET_1_30/LUX100/ISO100/0002_ISO100.mat`：

|统计量|数值，DN|
|---|---:|
|均值|587.8852|
|标准差|29.2840|
|中位数|580|
|最小值|530|
|最大值|660|

四个Bayer相位：

|相位|均值|标准差|中位数|
|---|---:|---:|---:|
|00|556.9702|6.3861|558|
|01|614.6986|14.0985|616|
|10|614.9836|14.0915|616|
|11|564.8885|7.3642|566|

FlatField与BiasFrame相减后的平均有效信号约为：

```text
全图：74.35 DN
00相位：43.42 DN
01相位：101.17 DN
10相位：101.46 DN
11相位：51.34 DN
```

两个绿色相位响应接近，另外两个相位响应不同，符合受光Bayer mosaic的特征。该文件包含：

- 入射光信号；
- photon shot noise；
- Bayer通道响应差异；
- 平场空间不均匀；
- 可能的光源/镜头渐晕。

因此FlatField不能作为暗帧加入训练。它适合：

- 拟合variance–mean曲线；
- 标定真实system gain \(K\)；
- 分析PRNU/平场不均匀；
- 复现calibrated-\(K\)消融。

本文主方法使用hypothesized \(K\)，所以可以完全跳过。

### 24.5 SID训练clean RAW实际ISO分布

对PMN整理的 `SID_train.info` 实际解析得到：

```text
训练long RAW场景数：161
实际出现ISO数：25
```

|ISO|场景数|ISO|场景数|ISO|场景数|
|---:|---:|---:|---:|---:|---:|
|50|1|64|1|80|3|
|100|2|160|9|200|13|
|250|28|320|17|400|7|
|500|16|640|4|800|6|
|1000|7|1250|7|1600|6|
|2000|3|2500|1|4000|3|
|5000|1|6400|1|8000|7|
|10000|8|12800|3|16000|1|
|25600|6||||

对应short RAW数量的ratio分布为：

```text
×100：587
×250：407
×300：711
```

该25档是PMN版SID训练索引中实际出现的clean ISO，不等于论文所称LLD的24档暗帧列表。论文没有公开24档的完整名单。

### 24.6 最推荐的选择性下载方案

#### 方案A：接近论文、控制容量，首选

下载：

```text
BiasFrame_ET_1_30/
└── ISO 100–25600范围内的全部可用ISO文件夹
    └── 每个ISO先下载均匀分布在采集序列中的10张MAT
```

跳过：

```text
全部FlatField
ISO100000
ISO200000
以及其他高于ISO25600的扩展ISO
```

论文表5显示，每ISO从400张减少到10张时，SID结果约从40.85 dB降到40.80 dB，因此优先覆盖ISO范围，比在少数ISO下载400张更划算。

按单文件24,242,488 bytes估算：

| ISO档数 | 每ISO帧数 |       约占用 |
| ----: | -----: | --------: |
|    24 |     10 |   5.82 GB |
|    24 |     40 |  23.27 GB |
|    24 |    100 |  58.18 GB |
|    24 |    400 | 232.73 GB |
|    25 |     10 |   6.06 GB |
|    25 |     40 |  24.24 GB |
|    25 |    400 | 242.42 GB |

建议先下载10张/ISO。训练跑通且确实需要冲击论文PSNR时，再扩展到40张/ISO；没有必要一开始下载400张/ISO。

#### 方案B：严格按照SID训练索引取ISO

若云盘中对应文件夹均存在，只下载：

```text
ISO50
ISO64
ISO80
ISO100
ISO160
ISO200
ISO250
ISO320
ISO400
ISO500
ISO640
ISO800
ISO1000
ISO1250
ISO1600
ISO2000
ISO2500
ISO4000
ISO5000
ISO6400
ISO8000
ISO10000
ISO12800
ISO16000
ISO25600
```

每档先选10张。若每档约400张，建议在编号/采集时间上均匀抽取，例如靠近序列开头、中间和末尾的文件，而不是连续下载 `0001`–`0010`。这样更可能保留传感器升温和时间变化带来的噪声多样性。

注意：论文只评估ISO 100–25600。ISO50/64/80合计只对应5个训练场景，可选择：

- 下载这三档；
- 删除这5个clean场景；
- 或映射到最近的ISO100暗帧。

为了节约容量，映射到ISO100是合理的首轮复现选择，但必须记录。

#### 方案C：更小的快速可行性实验

以下13档覆盖PMN索引中约137/161，即85.1%的训练clean场景：

```text
ISO160
ISO200
ISO250
ISO320
ISO400
ISO500
ISO800
ISO1000
ISO1250
ISO1600
ISO8000
ISO10000
ISO25600
```

每档10张约3.15 GB。

此方案适合验证训练代码和方法趋势，但不适合声称完整复现论文表1，因为ISO覆盖不完整。

### 24.7 下载优先级结论

推荐实际执行顺序：

1. 保留已下载的 `BiasFrame ISO100/0001_ISO100.mat` 用作解析测试；
2. 不再下载任何FlatField；
3. 在 `BiasFrame_ET_1_30` 中，为ISO 100–25600的每个可用ISO先各选10张，并在完整编号序列中均匀取样；
4. 跳过ISO100000、ISO200000及更高扩展ISO；
5. 首轮训练成功后，再把每ISO从10张扩到40张；
6. 只有做 calibrated-\(K\) 对照时才下载FlatField，且无需下载所有LUX，可先选2–3个亮度和少数ISO验证variance–mean拟合。

## 25. 来源定位

### 本地论文

`C:\Users\77436\Desktop\研究生文件\课题-ai isp\ref\pdf-summary\ai-isp\Li 等 - 2025 - Noise Modeling in One Hour Minimizing Preparation Efforts for Self-supervised Low-Light RAW Image D.pdf`

### 论文关键位置

- Figure 2：暗帧采集、dark shading、shot noise与训练合成总流程；
- Section 3.1：RAW image formation model；
- Equation 5：Poisson shot noise生成；
- Equation 6：\(K=\mathrm{QE}\times\mathrm{AG}\)；
- Equation 7：shot noise均值和方差；
- Section 3.2：hypothesized system gain；
- Section 3.3：直接采样dark frame及不使用HBNR；
- Section 4.1：SID/ELD/LLD数据、24 ISO×约400暗帧、训练设置、\(K\)公式和PMN dark shading；
- Table 1：SID ×100/×250/×300主结果；
- Table 3：\(K\)鲁棒性；
- Table 5：每ISO暗帧数量消融；
- Table 7：HBNR消融。

### 在线资源

- 论文：<https://arxiv.org/abs/2505.00045>
- 官方代码：<https://github.com/SonyResearch/raw_image_denoising>
- SID代码与说明：<https://github.com/cchen156/learning-to-see-in-the-dark>
- SID Sony数据：<https://storage.googleapis.com/isl-datasets/SID/Sony.zip>
- LLD代码与数据说明：<https://github.com/happycaoyue/LLD>
- LLD数据：<https://pan.baidu.com/s/1eLKzjOSDCR4NNLcvbyenEQ?pwd=WAXY>
- PMN代码：<https://github.com/megvii-research/PMN/tree/TPAMI>
