# 可学习 sym4 小波阈值 RAW 去噪

本实现延续初稿的“LL 递归分解 → 十子带平铺 → CNN 输出同 shape 阈值 → 系数收缩 → IWT”。
默认三层 sym4，固定分析/合成滤波器，只学习阈值网络，不学习小波基本身。
已接入 `train_sid_sony.py`、checkpoint 工厂、真实配对验证和原 SID 测试入口。

## 最新实测结论（1000 epoch）

当前配置已由用户调整为32通道、batch32、warmup10；文件名/输出目录中的d64是历史名称。
`atlas + smooth` 的1000 epoch / 81,000步权重在本地129对SID索引上，
相对传统sym4 L3 BayesShrink的平均PSNR提高：x100 +2.325、x250 +1.491、x300 +1.488 dB；
SSIM也均提高。两者使用相同的GT亮度校正协议。
这修正了200步短训的局限性判断；短训与长训还存在宽度、atlas/bandwise差异，不能只归因于epoch数。
完整结果与命令见 `ref-doc/LearningDWT网络结构与评测.md`。

## 1. 变换、shape 和边界

输入 PyTorch 张量 `[N,4,H,W]`，四个通道是 `[R,G1,G2,B]`，小波变换不跨通道。
CNN 可利用四个 CFA 通道的关联。每一级仅分解上一级 LL：

| 终端子带 | 张数 | 每张空间尺寸 |
|---|---:|---|
| LH1 / HL1 / HH1 | 3 | H/2 × W/2 |
| LH2 / HL2 / HH2 | 3 | H/4 × W/4 |
| LH3 / HL3 / HH3 | 3 | H/8 × W/8 |
| LL3 | 1 | H/8 × W/8 |

面积之和为原图面积；采用 `[LL,LH;HL,HH]` 递归左上角平铺，无 resize 或插值。
阈值图为 `[N,4,H,W]`，其位置对应**系数位置**，不对应原图同坐标的像素。

sym4 必须配合 critically sampled `periodization` 才能保持这个紧密布局；
已有传统 BayesShrink 默认使用 `symmetric`，会产生额外边界系数。
不能将后者的系数强行裁切成 H×W 再宣称可逆。
参考 [PyWavelets 系数拼接说明](https://pywavelets.readthedocs.io/en/latest/ref/dwt-coefficient-handling.html)。

`dwt_pad_input: true` 将非 8 倍数输入在右/下 replicate 补齐；IWT 后裁回原尺寸。
辅助阈值图保留补齐后尺寸。SID 整图 packed 尺寸 1424×2128 可被 8 整除，无需补齐。
`PeriodizedWavelet` 全程采用 PyTorch 固定卷积、转置卷积和周期折叠，可微；
系数与 `pywt.wavedec2(..., mode='periodization')` 对齐，方向顺序由代码及测试明确。
Haar 辅助函数保留用于兼容初稿测试，正式配置均为 sym4、3 层。

## 2. 阈值网络的改进和合理性

提供四个独立配置，不改变用户的阈值收缩主路径：

| 配置后缀 | CNN 上下文 | LL 收缩 | 用途 |
|---|---|---|---|
| bandwise | 共享网络分别处理十子带 | 开启、有限上限 | 逐子带候选，待充分训练对照 |
| atlas | 整张系数拼图卷积 | 开启、有限上限 | 当前已完成长训模型 |
| no_ll | 同 bandwise | 关闭 | 检验 LL 引起的亮度/偏色及收益 |
| bandwise_soft | 同 bandwise | 开启、有限上限 | 初稿软阈值收缩对照 |

整图 atlas 卷积会将不相邻的物理区域和不同方向/尺度当作邻居。
bandwise 消除此人工边界，但不等于已经实现跨尺度对齐融合。
三层尺度归一化仍使用 `s_j=2**j`，CNN 输入包含 `Z/s_j` 和 `abs(Z)/s_j`
共 8 个通道，使网络同时可利用符号、通道相关性和幅值。
bandwise 在第一层特征上加 10×width 的可学习子带 embedding，区别不同层级和方向；
atlas 禁用这个逐子带 embedding，保留原有输出子带偏置。

阈值 CNN 为 4 个 3×3 卷积：8→32→32→32→4（当前配置），前三层 ReLU。
不使用 BN 或全局池化，当前32通道参数量 bandwise=22,348、atlas=22,028；早期64通道分别为81,516、80,876。
十个子带共享权重，输出再拼回完整系数图；按子带串行推理降低峰值特征工作集。
CNN 最后一层小非零初始化，使第一步也能向前层传播梯度。

高频阈值：`T_j = s_j * softplus(logits + band_bias)`。
LL 默认采用有上限的阈值：`T_LL = s_J * cap * sigmoid(logits + band_bias)`。
默认 cap=0.01、初始 LL 阈值=0.001、高频初始阈值=0.01（归一化系数单位）。
偏置初始化分别使用 inverse-softplus / inverse-sigmoid，确保切换参数化不会改变初始量级。
关闭 LL 时阈值严格为零，LL 系数原样通过。

保留原版软阈值 `dwt_shrink_mode: soft` 及等效缩放：

```
S(Z,T) = sign(Z) * max(abs(Z)-T, 0)
       = Z * max(1-T/abs(Z), 0)       # Z != 0
filtered = leak * Z + (1-leak) * S(Z,T)
```

改进配置默认 `dwt_shrink_mode: smooth`：

```
S_smooth(Z,T) = Z * abs(Z) / (abs(Z)+T)
filtered = leak*Z + (1-leak)*S_smooth(Z,T)
```

T 此时是半增益幅值：abs(Z)=T 时系数保留一半；仍可解释为逐系数阈值尺度，
但不再是传统软阈值的硬清零边界。系数不放大、不翻转符号；T=0 为恒等，零系数保持零。
非零系数即使 abs(Z)<T，仍可对阈值反传梯度；避免原 soft 的零梯度区间。
这不是“证明更优”的理论保证，须通过同预算训练比较。

默认 leak=0；不截断模型最终输出，重建损失直接回传。
DWT/IWT 与阈值网络在 AMP 环境也采用 FP32，避免长滤波器逆变换及小阈值的精度损失。

**限制与消融必要性：**

- LL 承载亮度及低频结构，正系数收缩有变暗倾向。阈值上限只限制它，不保证无偏；
  所以必须同时看未经 GT 亮度校正的 PSNR/SSIM、各 CFA 均值偏差和 no_ll 对照。
- 纯收缩不能增加系数幅值或翻转符号；无法完全恢复被噪声抵消的细节。
- `2**j` 是幅度缩放而非噪声标准差。正交 DWT 不使白噪声方差随层数成倍增加。
- 输入幅值和子带身份为阈值学习提供信息，但尚无显式曝光/ISO、噪声尺度归一化或空间对齐的跨尺度引导。
- crop训练的周期边界占比远高于整图，边界分布差异也可能影响泛化；本次未做有效区域loss裁剪消融。
- 参数少不等于计算量小：在256×256 packed输入下 32通道CNN约1.434 GMAC（旧64通道约5.285 GMAC）；
  现有 profiler 未统计 functional DWT/IWT、拼接和阈值运算，model_info 已注明。
- sym4 的支撑长于 Haar，不能沿用初稿 Haar 的40像素 halo 保证；当前正式评估使用整图，
  未提供未经验证的切片等价性或 FPGA 性能结论。

## 3. 正式训练

从项目根目录运行，沿用用户更新后的d32 MRLFN训练设置：batch=32、patch=256、Adam、
LR=0.0002、warmup=10、epochs=1000、seed=2026，保持原重建损失：

```
L = 0.6 * RAW_L1 + 0.4 * chromatic_L1
```

不默认添加小波/高频/梯度辅助损失。训练只用0前缀长曝光+原有暗帧噪声合成。
当前用户配置使用 `Sony_val_list.txt`，每50 epoch验证前40对；
如需全量验证，显式设置 `validate_steps: -1`。latest/best/最近epoch保存逻辑均沿用现有实现。

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_sym4_l3_d64_bandwise.yaml
```

其他配置：

```
configs/train_sid_sony_learning_dwt_sym4_l3_d64_atlas.yaml
configs/train_sid_sony_learning_dwt_sym4_l3_d64_no_ll.yaml
configs/train_sid_sony_learning_dwt_sym4_l3_d64_bandwise_soft.yaml
```

各自有独立输出目录。保持相同seed、训练预算及验证集，以匹配预算训练各组才能判断各项结构改进；
目前确认完成的长训是用户提供的32通道atlas组，其他组需独立对照。
结构参数全部以 `dwt_` 为前缀，例如 `dwt_width`、`dwt_depth`、`dwt_wavelet`、
`dwt_levels`、`dwt_context`、`dwt_shrink_ll`、`dwt_ll_max_threshold`。
CLI 使用连字符形式如 `--dwt-width`，布尔项可用 `--no-dwt-shrink-ll`。
模型结构变化应新建实验；不要将不同结构的旧checkpoint强行用于resume。

## 4. 评估与传统算法对照

原脚本可自动从checkpoint args恢复结构：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python test_denoise_sideld.py \
  --cp-dir experiments/sid_sony_learning_dwt_sym4_l3_d64_bandwise/checkpoints/best.pth \
  --device cuda:0 --eval-ratio 100 --result-json test_res/learning_dwt_x100.json
```

新增同pair比较入口：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python test_compare_learning_wavelet_sid.py \
  --checkpoint experiments/sid_sony_learning_dwt_sym4_l3_d64_bandwise/checkpoints/best.pth \
  --output-dir test_res/learning_dwt_full_comparison
```

默认三种倍率全部配对；`--max-samples 5`仅取各倍率前5对，`--visualize-samples 3`保存图。
同时输出 input、原 symmetric BayesShrink、periodization BayesShrink、同seed未训练网络、
训练后网络的逐图及平均指标。两个传统算法都固定sym4三层且只递归LL。
指标包括原始输出和原脚本的 GT illuminance 校正后输出，以及四通道均值偏差。
PNG 展示未经GT校正的 input / 原传统算法 / learned / GT 和局部。
BayesShrink CPU与CNN GPU耗时不应被解释为同硬件速度排名。

本地 `SID_evaltest.info` 混有1/2前缀场景；全部129对结果是本地索引口径，
不等于纯官方test split。正式训练若使用2前缀验证选best，再在混合索引评估时含有验证场景，
不能据此声称独立测试泛化；论文测试应另固定仅1前缀的评估索引。

## 5. 本次验证

`tests/test_learning_sym4.py` 覆盖 PyWavelets 系数一致性、sym4逆变换/能量/梯度、
阈值非负与LL上限、网络梯度、任意尺寸补齐、关闭LL和checkpoint加载。
初稿Haar测试继续保留，不能将其结论直接当作sym4边界测试。

小规模实际训练：200步、batch=4、每epoch50步、4epoch、warmup1epoch；
使用原SID真实长曝光与暗帧合成，关闭验证，评估最后一步而不按测试结果选模型。
实验目录 `experiments/learning_dwt_sym4_l3_pilot200`，不是正式配置的输出目录。
这是800个随机训练crop的初步检查，无法替代充分训练、多seed和完整独立测试。
原软阈值结果及图：`test_res/learning_dwt_sym4_l3_pilot200_compare/`。
诊断首张x100纹理图发现9个高频子带约99.8%–100%系数清零，阈值远大于系数幅值。
仅在推理中关闭LL，PSNR从31.91变成32.19，仍远低于传统算法40.24；
说明这张图的主要损失来自高频过度收缩，并不只是LL变暗。
这是事后单图诊断，不是经重训的no_ll消融实验。

因此新增smooth连续收缩，并从相同seed以相同200步预算重新训练，
目录 `experiments/learning_dwt_sym4_l3_smooth_pilot200`。
连续收缩短训指标及图：`test_res/learning_dwt_sym4_l3_smooth_pilot200_compare/`。
这些是历史探索记录，不能代替上文最新1000 epoch模型的实测结论。

复现200步试验（使用新目录）：

```bash
conda run --no-capture-output -n LED-ICCV23 python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_sym4_l3_d64_bandwise.yaml \
  --epochs 4 --max-steps 200 --steps-per-epoch 50 --batch-size 4 \
  --num-workers 2 --warmup-epochs 1 --validate-steps 0 \
  --output-dir experiments/my_learning_dwt_pilot200
```

先检查学习曲线及阈值分布，再决定正式训练预算。高频阈值若长期远高于系数幅值，
单纯延长训练不保证能解决问题；下一步可单独研究从输入估计噪声尺度并归一化阈值，
以及合成噪声与真实short噪声的幅度分布是否一致。此处不声称这些后续方案已经验证。
