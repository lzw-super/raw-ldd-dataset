# 图像恢复清晰度相关 Loss 调研与 RAW 去噪实验建议

调研日期：2026-09-10。场景：SID Sony、4 通道 packed RAW、MRLFN 去噪，希望保留物体轮廓与细纹理，允许适量残留噪声。

本文基于原始论文、作者代码与官方文档，并核对当前仓库实现。论文结论与针对本项目的实验建议分别标注；建议权重尚未经过本项目验证。本次交付为调研文档，文中候选 loss 和模型选择方案不代表已经接入训练代码。

## 1. 针对当前需求的选择

建议优先做两个独立实验：**原始像素目标 + 梯度匹配**、**原始像素目标 + 小波高频子带匹配**。前者成本低、便于定位边缘改善；后者可以延续现有小波框架，控制不同尺度和方向的细节。随后再试多尺度 Laplacian 或 MS-SSIM；如果真实 RGB 观感仍不满意，再引入弱感知损失。

这些是结合 RAW 数据形式与工程成本给出的优先级，并非论文已经证明的 SID/MRLFN 排名。没有一种 loss 能保证恢复观测中已经丢失的真实细节。

| 类别 | 直接约束 | 对清晰度的预期作用 | 主要代价或失败表现 | 本项目优先级 |
|---|---|---|---|---|
| 梯度 / Sobel / Scharr 匹配 | 空间一阶导数与 GT 的差异 | 轮廓位置、边缘过渡与局部对比度 | 噪声、错位也会产生梯度误差 | 第一批 |
| 小波 LH/HL/HH 匹配 | 多尺度、分方向高频系数 | 细线、纹理和不同尺度细节 | 高频 GT 噪声、DWT 位移敏感性 | 第一批 |
| Laplacian / 金字塔边缘损失 | 带通残差或二阶响应 | 局部锐利度、多尺度边缘 | 振铃、光晕，对噪声敏感 | 第二批 |
| SSIM / MS-SSIM | 局部亮度、对比度、结构相似性 | 改善局部结构观感 | 不是专门锐度指标，可能放松色彩保真 | 第二批 |
| FFT / Focal Frequency Loss | 频域复数误差、难恢复频率 | 补充空间损失遗漏的频率误差 | 高频噪声、边界与相位处理 | 后续补充 |
| VGG / LPIPS 感知损失 | RGB 预训练网络特征 | 更符合观看时的纹理与结构感受 | RGB 域转换、算力、细节真实性 | 后续弱权重微调 |
| GAN / 对抗损失 | 输出与真实图像分布的可区分性 | 更强自然纹理感 | 假纹理、结构偏移、训练不稳定 | 不作为首轮方案 |
| Charbonnier | 平滑的稳健像素误差 | 适合作为基础重建目标 | 本身没有显式细节约束 | 可做控制组 |
| 已有 LL-only 小波损失 | 多尺度低频近似 | 亮度与大尺度结构一致性 | 没有直接监督细节子带 | 保留作对照 |

上表作用与风险是下文方法机制的工程判断。代表性证据包括 [SPSR 梯度指导](https://openaccess.thecvf.com/content_CVPR_2020/html/Ma_Structure-Preserving_Super_Resolution_With_Gradient_Guidance_CVPR_2020_paper.html)、[MPRNet 作者代码](https://github.com/swz30/MPRNet)、[图像恢复损失比较](https://research.nvidia.com/publication/2017-03_loss-functions-image-restoration-neural-networks)；跨 RGB 超分、去模糊任务的效果不能直接视作 RAW 去噪收益。

## 2. 为什么像素误差较低，图像仍可能模糊

对确定性估计器，逐像素 L2 的条件风险最优解是条件均值，L1 则是条件中位数。观测无法确定某段纹理时，像素目标可能倾向保守估计。**不能把 L1 的模糊解释为“L1 求平均”，也不能认为换 L2 会更锐。** Zhao 等比较了 L1、L2、SSIM、MS-SSIM 与混合目标，说明基础目标本身也需要实证比较。[原论文](https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf)

感知质量与像素保真存在一般性的权衡，但这不表示每次提高观感都必须降低 PSNR，也不表示本模型已经达到最优边界。Blau 与 Michaeli 的理论讨论的是分布意义的感知质量与失真，不等同于某一张图的锐度。[CVPR 2018 原论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)

本项目应同时排查以下可检验因素，避免把所有模糊都归因于 loss：

- GT 自身是否存在长曝光运动模糊、失焦或残留噪声。
- 合成噪声与真实短曝光噪声、暗场校正是否匹配。
- 模型容量、S2D 空间处理、训练预算是否限制细节恢复。
- 展示时的去马赛克、缩放、降噪、锐化是否一致。
- 配对中的微小错位是否让边缘监督相互冲突。当前合成训练输入来自同一 clean 图，天然配准；真实验证仍可能存在曝光间运动。

“允许残留噪声”意味着可以接受另一种质量折中，**不意味着应该训练模型复制输入噪声**。噪声与细节都含高频，必须用 GT 或可靠结构作参照。

## 3. 常用候选损失

以下令预测为 P，GT 为 T，所有 L1 默认对 batch、通道和空间取平均。算子、边界、尺度和归一化必须固定后再比较权重。

### 3.1 梯度匹配：首选低成本方案

一个直接可用的定义为：

```text
L_grad = 0.5 × [mean(|Dx(P) - Dx(T)|) + mean(|Dy(P) - Dy(T)|)]
L_total = L_base + λ_grad × L_grad
```

Dx、Dy 可以使用相邻差分、Sobel 或 Scharr。SPSR 通过梯度图与梯度空间监督约束恢复结构，提供了梯度指导的代表性证据；上式是本项目建议的简单有符号梯度匹配，不是对 SPSR 全部目标的原样复现。[SPSR，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Ma_Structure-Preserving_Super_Resolution_With_Gradient_Guidance_CVPR_2020_paper.html)

**项目建议：**先对 4 个 RAW 通道分别应用 Sobel，并将标准 Sobel 核除以 8。匹配有符号的水平/垂直导数，比只匹配梯度幅度更能保留方向和极性。需要抗噪时，可增加一个轻度高斯平滑后的梯度尺度；过强平滑则会遗漏细线。

为了体现“更重视物体清晰度”，后续可用 GT 梯度构造软权重 w，在结构区域提高监督比重：

```text
L_grad_masked = sum(w × gradient_error) / (sum(w) + ε)
```

w 由 GT 的平滑结构生成并停止梯度；不要由 noisy 输入直接生成，也不要让模型通过改变预测来降低自身权重。这是建议的消融变体，不能当作已有验证结果。全图像素项继续约束平坦区域。

### 3.2 Laplacian / 多尺度金字塔损失

```text
R_s(X) = X_s - Upsample(LowpassDownsample(X_s))
L_lap = sum_s α_s × mean(ρ(R_s(P) - R_s(T)))，sum_s α_s = 1
```

ρ 可选绝对值或 Charbonnier。这里 R_s 是金字塔带通残差；它和直接对图像卷积一个离散二阶 Laplacian 核不是完全相同的实现。

MPRNet 官方去模糊代码采用 Charbonnier 加 `0.05 × EdgeLoss`，EdgeLoss 由高斯滤波、下采样、上采样后的残差构成。需要区分：其官方去噪训练脚本使用的是 Charbonnier，不能把去模糊设置说成已验证的去噪配置。[EdgeLoss 实现](https://raw.githubusercontent.com/swz30/MPRNet/main/Deblurring/losses.py)、[去模糊训练](https://raw.githubusercontent.com/swz30/MPRNet/main/Deblurring/train.py)、[去噪训练](https://raw.githubusercontent.com/swz30/MPRNet/main/Denoising/train.py)

**项目建议：**作为梯度项的替代实验先单独测试，暂不叠加。对 RAW 采用通道独立滤波，核随通道数和设备适配，避免照抄作者实现中的 RGB 三通道假设。观察亮暗边界是否出现过冲、光晕与双边缘。

### 3.3 小波高频损失：与当前 LL 实验衔接

现有 [WaveletLLLoss](../losses/wavelet_ll_loss.py) 递归取 LL，只计算各层低频近似的 L1；它没有计算 LH、HL、HH 的监督。LL 仍包含部分结构信息，但无法直接约束被细节分支承载的边缘和纹理。二维 DWT 的近似/细节子带划分可见 [PyWavelets 官方文档](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html)。

建议另行定义高频辅助项：

```text
L_HF = sum_j α_j × (1/3) × sum_b mean(|W_(j,b)(P) - W_(j,b)(T)|)
b ∈ {LH, HL, HH}，sum_j α_j = 1
L_total = L_base + λ_HF × L_HF
```

不同库对 LH/HL 与水平/垂直的命名可能不同，实现时应核对轴约定。这里是匹配真实系数，而非把高频系数推大，也不是对预测做传统硬阈值或软阈值去噪。

CVPR 2024 的 wavelet-domain loss 工作在生成式超分中使用子带保真约束及高频判别，说明子带目标能控制尺度、方向相关的误差与伪影；不能据此推断仅加 RAW 高频 L1 就能获得同样收益。[原论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Korkmaz_Training_Generative_Image_Super-Resolution_Models_by_Wavelet-Domain_Losses_Enables_Better_CVPR_2024_paper.pdf)

**项目建议：**先 Haar、1–2 层、各层等权；验证方向有效后再比较 db2、sym4、coif1、bior2.2。高频子带先全部启用，若 HH 对角子带明显携带噪点，再单独测试降低其权重。较长滤波器不自动意味着更清晰；边界、位移敏感性、子带尺度均影响结果。bior2.2 为双正交体系，跨基函数时尤其不能直接假设各子带能量尺度相同。

LL 与 HF 可联合，但建议先做 LL-only、HF-only、LL+HF 三组对照。已有 LL 每层系数尺度未归一化，且层数增加会改变损失量级；不要直接沿用 LL 的权重数值作为 HF 的等效强度。

### 3.4 SSIM / MS-SSIM 结构损失

```text
L_struct = 1 - MS-SSIM(P, T)
L_total = L_base + λ_struct × L_struct
```

MS-SSIM 在多个尺度比较局部结构。Zhao 等研究了将它与 L1 结合的目标，但论文 MIX 的 L1 带有与窗口相关的加权，不能将任意 `L1 + MS-SSIM` 宣称为严格复现；也不宜照搬论文的混合系数。[原论文](https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf)

**项目建议：**它适合补充局部结构监督，但不应被当作直接锐度保证。RAW 上逐通道计算属于本项目适配方案，其观看意义弱于固定 ISP 后的 RGB/亮度域。固定 `data_range=1`、窗口、尺度和边界；小 patch 应减少尺度。保留像素项以约束亮度和色彩，并记录训练中是否发生硬截断：硬 clamp 的饱和区梯度会为零。

### 3.5 频域损失与 Focal Frequency Loss（FFL）

```text
ΔF = FFT(P) - FFT(T)
L_FFL = mean(w × |ΔF|²)
```

FFL 动态提高难恢复频率的相对权重，既不是固定高通，也不是只比较频谱幅度。作者实现用正交归一化 FFT，依据频域误差计算归一化权重并 detach。[ICCV 2021 原论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.pdf)、[官方实现](https://raw.githubusercontent.com/EndlessSora/focal-frequency-loss/master/focal_frequency_loss/focal_frequency_loss.py)

**项目建议：**可作为后续补充，尤其关注条纹或特定频率误差，但收益需单独验证。只匹配幅度会放松空间位置约束；保留复数差异和像素目标。依据 Parseval 等式，完整正交 FFT 下、不加权的复数 L2 与空间 L2 等价，单纯“换到频域算 L2”并没有额外强调细节。FFT 的周期边界可能引入 patch 边缘效应，裁块或加窗必须两侧一致并记录。FFL 也可能集中关注难消除的噪声频率。

### 3.6 VGG 特征与 LPIPS 感知损失

```text
L_perc = sum_l β_l × mean(|φ_l(ISP(P)) - φ_l(ISP(T))|)
L_total = L_base + λ_perc × L_perc
```

φ 是冻结的预训练特征网络。Johnson 等以特征距离代替纯逐像素监督，在 RGB 超分中获得更符合视觉的结果；这为感知目标提供依据，但不是 RAW 去噪或真实细节逐点恢复的保证。[ECCV 2016 作者页面](https://cs.stanford.edu/people/jcjohns/eccv16/)

LPIPS 可用作评估，也可反向传播为损失；官方默认接口接收 RGB、N×3×H×W、数值范围 [-1,1]，距离越低越相似。[LPIPS 作者代码与使用说明](https://github.com/richzhang/PerceptualSimilarity)

**项目建议：**不要把 `[R,G1,G2,B]` 当成 RGB，或随意删除一个通道后直接喂给 VGG。应先构造固定、可微的 ISP，对 P/T 使用相同白平衡、色彩矩阵、去马赛克与色调映射；不能在训练分支经过 NumPy、uint8 或不可微图像处理。当前仓库用于展示的 `rggb_to_srgb` 不能直接视为可微训练 ISP。

初期可以先把 LPIPS 仅作为固定 ISP 后的验证指标，确认观看偏好确实被反映，再考虑浅层 VGG 特征辅助微调。冻结特征网络参数时仍须保留预测分支的计算图；GT 特征可以 no_grad。深层语义特征可能容忍微小位置改变，不宜作为唯一目标。

### 3.7 对抗损失：更强纹理感，较低真实性约束

GAN 使输出更接近真实图像的分布。ESRGAN 联合对抗、像素及感知目标，并研究激活前特征的感知损失，以改善纹理与亮度一致性。[ESRGAN 原论文](https://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.html)

**项目建议：**“允许保留少量噪声”和“接受生成不存在的纹理”是两个不同要求。文字、栅格和重复纹理的结构真实性仍然重要，因此首轮不引入 GAN。若以后采用，应在保真模型上弱权重微调，建立错纹理、光晕与彩色伪影的专门检查；不能用更高的锐度分数代替真实性检查。

### 3.8 Charbonnier、TV 与容易误用的“锐度损失”

Charbonnier 的典型形式为 `mean(sqrt((P-T)² + ε²))`，是平滑稳健的像素惩罚。它可以替代 L1 做控制组，但不会自动增加空间结构监督。[MPRNet 实现](https://raw.githubusercontent.com/swz30/MPRNet/main/Deblurring/losses.py)

以下判断直接来自目标函数的优化方向：

- 最小化输出的 TV / 梯度范数，倾向减少空间变化，不能作为保锐的主要手段。
- 梯度**匹配**是最小化 `|∇P-∇T|`，与最小化 `|∇P|` 不同。
- 最大化 Laplacian 方差、梯度能量或小波高频能量，会同时奖励噪点、振铃和过锐边缘，不建议作为独立训练目标。
- 不建议把 noisy 输入混入 GT 来“保细节”；这会把部分输入噪声也变成正确答案。
- 仅将全部 L1 乘以一个较小常数，并未改变它偏好的最优解；在优化器、梯度裁剪等影响下训练轨迹可能改变，但这不是明确的清晰度约束。

## 4. 本仓库的落地约束

核对的代码与配置：

- [训练入口](../train_sid_sony.py)：合成噪声训练，真实配对验证；`best.pth` 依据 `real_psnr`。
- [MRLFN 基础 loss](../losses/mrlfn_loss.py)：RAW 重建项加通道色差项。
- [指定基线配置](../configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml)：当前 `0.6 × L_raw + 0.4 × L_chromatic`，**不是纯 L1**。
- [已有小波 LL loss](../losses/wavelet_ll_loss.py)：4 通道独立、各层 LL L1 平均、symmetric 边界、固定低通核、未做阈值。
- [实际测试入口](../test_denoise_sideld.py)：亮度校正、截断后计算 PMN 指标。

如果观察来自“纯 L1”训练，实验需要单列纯 L1 基线：MRLFN 设置 `raw_loss_weight=1`、`chromatic_loss_weight=0`、`wavelet_loss_weight=0`。所有对比保留同一模型、数据、随机种子、采样、步数和学习率，不要同时改模型宽度或 batch。

RAW 通道独立梯度对应同色采样格点上的变化；直接对 Bayer mosaic 求梯度会混入 CFA 色彩交替，不宜未经处理就当作物体边缘。packed 域的 1 像素尺度还对应 mosaic 域的 2 像素，写实验记录时需要说明。

真实配对中的 GT 不是绝对无噪，长曝光还可能模糊。高频监督越强，越应检查 GT 高频是否可靠。饱和和错位区域可单列指标或用预先定义的掩码，但不能为了提高分数临时剔除困难样本。

## 5. 建议实验顺序与权重

以下为本项目建议，未运行，权重不是论文普适值。

### 5.1 第一轮：验证最直接的方向

| 实验 | 目标 | 控制要求 |
|---|---|---|
| B0 | 纯 RAW L1 | 对应用户所观察的纯 L1 模型 |
| B1 | 当前 0.6 RAW + 0.4 chromatic | 与指定配置直接对齐 |
| G | B1 + 梯度匹配 | Sobel/8，4 通道平均 |
| W | B1 + 小波 HF | Haar，1–2 层，LH/HL/HH 等权 |
| LL | B1 + 已有 LL | 保留已有 3 层、0.1 权重作低频对照 |

如果 G 或 W 有收益，再各加一组 B0+同一辅助项，确认不是 chromatic 项造成的交互。第一轮不要同时叠加多个锐度目标。

Sobel/8 梯度项可从 `λ_grad ∈ {0.02, 0.05, 0.1}` 小范围开始；数字仅适用于上文“mean + 核归一化”的定义。HF、Laplacian、MS-SSIM 与感知损失的尺度不同，不共用这些数值。

更可比的权重选择方式，是在固定训练 batch 上记录各项对预测 P 的梯度范数：

```text
r = λ_aux × ||∂L_aux/∂P||₂ / (||∂L_base/∂P||₂ + ε)
```

初始可让 r 大致落在 5%–20% 作为保守搜索起点，再用验证结果决定是否增加。这是工程启发式，不是质量保证；参数梯度还受模型 Jacobian 影响。记录 loss 原值、加权值和梯度范数，避免只看数值大小猜测影响。

### 5.2 第二轮：必要时补充结构或感知目标

先比较 `B1 + Laplacian` 和 `B1 + MS-SSIM`；如果梯度方案已足够，则无需增加复杂度。若主要问题是纹理观感而非轮廓，再试弱感知损失；频域误差突出时再加入 FFL 对照。

可以从同一个保真 checkpoint 微调来快速筛选，但所有候选与一个“继续原目标训练”的控制组必须使用相同起点和额外步数。不要将多训练过的候选与未继续训练的基线直接比较。确定候选后，再进行相同完整预算的从头训练和多随机种子复核。

## 6. 评价：同时观察细节、噪声和真实性

仅保存最高 PSNR 模型可能错过偏清晰的模型。建议保留现有 PSNR 选优，后续额外保留由固定细节指标选出的候选，最后在验证集上选择折中；这一建议尚未修改当前保存逻辑。

| 维度 | 建议记录 | 解释与限制 |
|---|---|---|
| 总体保真 | RAW PSNR、SSIM，分 ×100/250/300 | 继续与原测试协议一致 |
| 边缘准确性 | GT 边缘区域的梯度 MAE | 越低越好；需固定算子与掩码 |
| 边缘清晰程度 | 固定边缘的剖面、过渡宽度、过冲 | 同时看锐度和光晕；场景需配准 |
| 纹理保真 | 小波 HF 误差、细节 ROI 放大图 | 不能只看预测高频能量 |
| 平坦区残留 | 固定平坦 ROI 的误差标准差、均值 | 基于 P-T；仍含 GT 噪声和配准误差 |
| RGB 观感 | 固定 ISP 后 LPIPS 与盲评 | LPIPS 域、网络版本、输入范围保持一致 |
| 伪影 | 文字、栅格、毛发、暗部、强边缘 | 检查假纹理、拉链、彩噪与振铃 |

平坦区域应由 GT/固定 ROI 选择，不能由各模型自行选择“最平”的位置。梯度或 HF 指标是 GT 一致性指标，不是独立感知真值；训练和选择都用同一指标时尤其需要固定裁剪图复核。

展示预测、输入、GT 时使用一致曝光、白平衡、ISP、缩放与锐化参数，以 100% 像素裁剪观察。随机噪声不要通过显示缩小被平均掉。

当前评估对预测做依赖 GT 的亮度校正。保留该协议用于历史可比，同时建议补充不做该校正的结果，以区分曝光偏差和实际细节差异；两类分数分栏报告，不混合排序。

在验证集上画“边缘误差—平坦区残留误差”散点图，寻找二者不能同时被其他模型改善的候选，即 Pareto 前沿。允许的 PSNR 下降或噪声增加应根据实际观看偏好确定，不预设所有场景统一阈值。**验证集用于选权重和 checkpoint，测试集用于最后报告，不拿测试集反复调参。**

## 7. PSNR 之外，如何评价清晰度与噪声的平衡

**存在比纯像素误差更贴近结构或观看感受的指标，但不存在一个在所有 RAW 去噪场景下都能自动给出理想折中的通用分数。** 对当前有真实 GT 的 SID 验证，建议优先采用全参考指标；用一个感知指标排序，再用平坦区残留误差与边缘检查限制不希望出现的变化。

推荐的第一版评估组合是：**RGB DISTS ↓（或 LPIPS ↓）+ RGB/亮度 FSIM ↑ + 平坦区残留误差 ↓**，继续报告 RAW PSNR/SSIM。算力有限时，可先用 MS-SSIM 与 GMSD 替代较重的特征网络。这里是针对本项目的选择建议，不是这些指标在 SID 上已经验证的排名。

### 7.1 候选指标总览

“全参考”表示需要与预测对应的 GT；“无参考”表示只输入预测。↑ 越高越好，↓ 越低越好。以下敏感性与局限是依据方法设计作出的工程判断，不能保证每个样本上的排序。

| 指标 | 参考需求 / 方向 | 如何同时反映模糊与噪声 | 本项目用途与局限 |
|---|---|---|---|
| SSIM / MS-SSIM | 全参考，↑ | 比较局部结构、对比度等统计；模糊和噪声都会改变这些关系 | MS-SSIM 可作低成本总体补充；仍可能偏好较平滑结果 |
| FSIM / FSIMc | 全参考，↑ | 相位一致性与梯度幅值匹配，关注结构而非仅看像素误差 | 候选结构主指标；FSIMc 补充颜色，平坦区仍需单测 |
| GMSD | 全参考，↓ | 梯度相似图的空间标准差，反映结构受损的非均匀性 | 计算较轻；不是简单梯度误差，也不是锐度越高越好 |
| VIF | 全参考，↑ | 比较失真图与参考图之间可保留的信息 | 可作信息保真补充；可能奖励对比增强，不宜独立选优 |
| LPIPS | 全参考，↓ | 比较学习校准的深层特征距离 | 已有较广泛感知评估用途；细小错位、纹理变化的惩罚与像素指标不同 |
| DISTS | 全参考，↓ | 联合深层结构相似与纹理统计相似 | 值得用于清晰度/纹理观感；容忍纹理重采样，不保证微结构真实性 |
| NIQE / BRISQUE | 无参考，↓ | 自然图像统计偏离反映多类质量退化 | 无 GT 图像的辅助观察；RAW、极暗图、特殊纹理存在域偏差 |
| MANIQA | 无参考，通常 ↑，以所用权重/接口为准 | 学习预测人类质量评分，综合多维特征 | 无 GT 的后续补充；依赖训练数据库与 checkpoint，不代替配对保真 |
| GT 边缘梯度误差 + 平坦区残留 | 全参考，均 ↓ | 将细节准确性与噪声残留分开测量 | 最易解释和调节偏好；属于项目自定义诊断，不是标准 IQA 分数 |

原始定义与实现依据：[SSIM/MS-SSIM 作者页面](https://www.live.ece.utexas.edu/research/quality/ssim/)、[FSIM](https://web.comp.polyu.edu.hk/cslzhang/IQA/FSIM/FSIM.htm)、[GMSD](https://web.comp.polyu.edu.hk/cslzhang/IQA/GMSD/GMSD.htm)、[VIF](https://live.ece.utexas.edu/research/Quality/VIF.htm)、[LPIPS](https://github.com/richzhang/PerceptualSimilarity)、[DISTS](https://github.com/dingkeyan93/DISTS)、[NIQE/BRISQUE](https://live.ece.utexas.edu/research/Quality/nrqa.htm)、[MANIQA 原论文](https://arxiv.org/abs/2204.08958)。

### 7.2 更值得优先试的指标

#### FSIM / FSIMc：结构保留是否接近 GT

FSIM 用相位一致性表征局部结构显著性，以梯度幅值补充对比信息，再加权汇总。FSIM 主要比较亮度，FSIMc 还考虑色度。[作者说明与代码](https://web.comp.polyu.edu.hk/cslzhang/IQA/FSIM/FSIM.htm)

其优势在于目标是“特征与 GT 相似”，加噪产生更多边缘并不会天然被视为更好。不过结构加权可能弱化平坦暗区在总分中的影响，因此建议与平坦区残留指标成对报告。它适合回答轮廓、细线是否保留，不能独立裁决噪声是否可接受。

#### GMSD：轻量的结构质量补充

令 gP、gT 为梯度幅值，其核心形式是：

```text
GMS(p) = [2 × gP(p) × gT(p) + C] / [gP(p)² + gT(p)² + C]
GMSD = std_p(GMS(p))
```

这是梯度**相似度图**的标准差，不是输出图像梯度的标准差；小值更好。完整算法还涉及预处理、梯度算子和稳定常数，应使用核对过的实现。[GMSD 原始说明](https://web.comp.polyu.edu.hk/cslzhang/IQA/GMSD/GMSD.htm)

项目应用时不要把它改写成 `1-mean(GMS)` 后继续叫 GMSD。仅依赖标准差汇总也有退化情形：相似度空间分布近乎恒定，不代表绝对质量必然很高。因此它适合作辅助，仍需像素/感知指标与样图复核。

#### DISTS：结构与纹理观感的候选主指标

DISTS 联合深层特征的结构关系和纹理统计，并有意容忍纹理重采样。[DISTS 原论文](https://arxiv.org/abs/2004.07728)

这使它值得用于比较“纹理更自然、略有残噪”和“更平滑”的模型，但也意味着相似的草地纹理不一定在每个位置都正确。对于文字、栅格、细线，不应仅因 DISTS 更低就认定真实细节恢复更好。官方接口接收 RGB [0,1]，低值更好。[DISTS 官方实现](https://github.com/dingkeyan93/DISTS)

LPIPS 则可作为平行的感知距离，两者不要机械地视为可互换分数。若训练使用其中之一，可用另一个与结构指标作交叉检查，减轻仅针对训练目标排名带来的偏差。

#### VIF：信息保真，不是限定在 0–1 的“准确率”

VIF 从信息论和自然场景统计角度衡量参考信息的保留。作者特别指出，无附加噪声的线性对比增强可能得到大于 1 的分数。[VIF 作者说明](https://live.ece.utexas.edu/research/Quality/VIF.htm)

因此，VIF 高于基线时仍应查看对比度、过锐光晕和曝光，不能把它解释为“恢复了超过 100% 的真实细节”。若使用 VIFp 等变体，要明确版本，不与另一实现直接混用。

### 7.3 无参考指标适合哪些情况

NIQE 不依赖人为评分的失真图训练，BRISQUE 则使用自然场景统计预测质量；二者都属于无参考评价。[LIVE 官方说明](https://live.ece.utexas.edu/research/Quality/nrqa.htm)

在有 GT 的 SID 验证中，它们不宜优先于全参考方法。低 NIQE/BRISQUE 不等于更忠实于该场景：不同降噪、锐化、色调映射可能改变自然统计。对于没有长曝光 GT 的新增真实拍摄，可以在固定 ISP 后辅助比较，并做人工盲评。

MANIQA 等学习型无参考方法综合图像特征预测质量，可作为另一类候选，但必须固定模型权重和输入协议。[MANIQA 原论文](https://arxiv.org/abs/2204.08958)

这些方法对“看起来质量好”的预测不能取代“真实物体结构是否正确”的检查。也不建议用 FID 一类数据集分布指标为单张配对去噪结果选优。

### 7.4 显式控制清晰度与噪声：可解释的双指标

下面是建议新增的**项目诊断指标**，不是已发表的统一质量标准。

从 GT 或预先标注产生固定边缘掩码 E 与多个平坦 ROI F_k；所有候选使用同一组区域。RAW 上按通道独立计算，RGB 上另报亮度/色度结果，不混为一个单位。

```text
e = P - T
D_edge = sum(E × (|Dx(P)-Dx(T)| + |Dy(P)-Dy(T)|)) / (2 × sum(E) + ε)
μ_k = mean(e within F_k)
σ_k = sqrt(mean((e - μ_k)² within F_k))
RMS_k = sqrt(mean(e² within F_k))
N_flat = mean_k(RMS_k)
```

- D_edge 越低，表示与 GT 的边缘导数更一致，同时约束过度平滑与错误高频；它不是纯物理锐度。
- σ_k 表示局部误差波动，μ_k 表示局部偏差。只看 σ_k 会漏掉恒定偏色/亮度误差，所以选优时建议同时报告 μ_k，并使用 N_flat 作残留误差限制。
- N_flat 包含预测误差、GT 噪声及配准误差，不能称为传感器噪声标准差的无偏估计。
- 某个样本没有足够的平坦区域时标记缺失并报告有效覆盖，不填 0；多 ROI 等权还是按像素加权应固定并注明。

第一种选优方式是保留 `(D_edge, N_flat)` 的 Pareto 前沿，并从中比较 RGB DISTS/LPIPS 与固定裁剪。第二种是“可接受残留上限内，优先感知质量”：

```text
合格候选：N_flat(model) <= N_flat(baseline) + δ_noise
最终候选：在合格者中，优先 DISTS 更低且边缘/伪影检查通过的模型
```

δ_noise 的单位与图像域、数值范围一致。它表示你愿意接受的额外残留误差，需要通过验证集观看确定；不是通用推荐常数。如果基线误差接近 0，采用加性容差比相对百分比更稳定。该方案比把“噪点越多越锐”奖励成高分更符合需求。

### 7.5 如果确实需要一个数保存 best checkpoint

最简单的候选是 `best_dists.pth`，同时用平坦区残留上限和伪影规则过滤；现有 `best.pth` 继续按 PSNR 保存，便于比较。

如果实验管理必须提供单个复合分数，可以定义一个**自定义分数**，但不能把它称为公认的清晰度指标：

```text
z_m = (m - a_m) / (b_m - a_m + ε)
Q = 1 - [w_p × z_DISTS + w_e × z_D_edge + w_n × z_N_flat]
w_p + w_e + w_n = 1，所有 w >= 0；Q 越大越好
```

a_m、b_m 是事先冻结的标定区间，例如来自固定校准模型和失真探针的分位数；不要每加入一个候选就重新 min-max，否则旧模型分数也会改变。Q 不必落在 [0,1]，不建议静默截断。权重表达观看偏好，可用一小批固定样本的盲评标定；不要直接相加 DISTS、PSNR、噪声标准差等不同尺度的原始值。

首轮更推荐约束筛选和 Pareto 展示，因为它们能说明“提高了多少细节，增加了多少残留”，比未标定的 Q 更容易解释。若没有候选满足限制，应报告无合格候选，而不是自动放宽标准。

### 7.6 RAW 输入协议与指标可信度检查

1. **分清两条评价路径。** RAW 保真及通道独立诊断直接计算；DISTS、LPIPS 和自然图像无参考指标使用固定 ISP 得到的 RGB。RAW 通道平均 SSIM 可以记录，但不能把它等同于标准 RGB 感知分数。
2. **核对输入范围。** 官方 DISTS 使用 RGB [0,1]；官方 LPIPS 通常使用 RGB [-1,1]；原始 FSIM MATLAB 接口采用 0–255，其他封装可能有不同约定。明确库、版本、权重、颜色转换、归一化和分数方向，不因都叫同一指标就默认接口相同。[DISTS 接口](https://raw.githubusercontent.com/dingkeyan93/DISTS/master/README.md)、[LPIPS 接口](https://github.com/richzhang/PerceptualSimilarity)、[FSIM 接口](https://web.comp.polyu.edu.hk/cslzhang/IQA/FSIM/FSIM.htm)
3. **细节评估不任意缩小图像。** 下采样会掩盖小噪点和细线损失。固定全图预处理，再补充固定原分辨率 ROI；分别报告全图和 ROI 分数，不要把 ROI 均值当成等价全图分数。某些指标自身带下采样，保留标准实现并记录。
4. **先做失真探针。** 对同一 GT 生成递增模糊、递增加噪、不同强度锐化，以及相近残留下的不同模糊版本；观察指标是否能发现单独的模糊与噪声，以及是否被强锐化误导。跨失真类型谁更好，以固定条件盲评校准，不预设所有指标必然一致。
5. **报告分布而非仅平均。** 先逐图得到指标，再按 ×100/250/300 分组；同时报中位数与差值分布。反复短曝光属于同一场景，若计算置信区间应按场景重采样，避免把相关配对当成独立场景。跨倍率可额外报告倍率等权平均，与全部配对加权平均区分。
6. **完整记录比较条件。** 保留第 6 节的亮度校正/未校正两套协议说明。ISP 参数必须对预测和 GT 一致，不能针对各模型调锐化以提高指标。

工具方面可参考 [IQA-PyTorch 官方项目](https://github.com/chaofengc/IQA-PyTorch)及其[指标模型卡](https://github.com/chaofengc/IQA-PyTorch/blob/main/docs/ModelCard.md)；统一封装能降低接入成本，但仍需核对与作者实现的数值约定。**以上为补充调研与建议，目前没有安装新指标依赖，也没有修改训练验证及 best 权重保存逻辑。**

## 8. 参考资料索引

1. [Zhao et al., Loss Functions for Image Restoration with Neural Networks, IEEE TCI 2017](https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf)：像素与结构损失比较。
2. [Blau & Michaeli, The Perception-Distortion Tradeoff, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)：感知—失真权衡。
3. [Ma et al., Structure-Preserving Super Resolution with Gradient Guidance, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Ma_Structure-Preserving_Super_Resolution_With_Gradient_Guidance_CVPR_2020_paper.html)：梯度结构指导。
4. [Zamir et al., Multi-Stage Progressive Image Restoration, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.pdf)：恢复网络与作者实现中的 Charbonnier/边缘目标。
5. [PyWavelets 二维 DWT 文档](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html)：低频近似、高频细节子带约定。
6. [Korkmaz et al., Training Generative Image Super-Resolution Models by Wavelet-Domain Losses Enables Better Control of Artifacts, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Korkmaz_Training_Generative_Image_Super-Resolution_Models_by_Wavelet-Domain_Losses_Enables_Better_CVPR_2024_paper.pdf)：小波域保真与高频约束。
7. [Jiang et al., Focal Frequency Loss for Image Reconstruction and Synthesis, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.pdf)：动态频域误差加权。
8. [Johnson et al., Perceptual Losses for Real-Time Style Transfer and Super-Resolution, ECCV 2016](https://cs.stanford.edu/people/jcjohns/eccv16/)：预训练特征感知损失。
9. [Zhang et al., The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018](https://richzhang.github.io/PerceptualSimilarity/)：LPIPS 感知距离。
10. [Wang et al., ESRGAN, ECCV Workshops 2018](https://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.html)：感知与对抗目标的纹理恢复取舍。

11. [Wang et al., SSIM / MS-SSIM 作者资料](https://www.live.ece.utexas.edu/research/quality/ssim/)：结构相似性评价。
12. [Zhang et al., FSIM, IEEE TIP 2011](https://web.comp.polyu.edu.hk/cslzhang/IQA/FSIM/FSIM.htm)：相位一致性与梯度特征。
13. [Xue et al., GMSD, IEEE TIP 2014](https://web.comp.polyu.edu.hk/cslzhang/IQA/GMSD/GMSD.htm)：梯度相似度的标准差汇总。
14. [Sheikh & Bovik, VIF 作者说明](https://live.ece.utexas.edu/research/Quality/VIF.htm)：信息保真框架。
15. [Ding et al., Image Quality Assessment: Unifying Structure and Texture Similarity](https://arxiv.org/abs/2004.07728)：DISTS 结构与纹理质量指标。
16. [LIVE 无参考质量评价资料](https://live.ece.utexas.edu/research/Quality/nrqa.htm)：NIQE 与 BRISQUE。
17. [Yang et al., MANIQA, 2022](https://arxiv.org/abs/2204.08958)：学习型无参考质量评价。
