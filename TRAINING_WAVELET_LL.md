# SID 多层 LL 小波辅助损失对比实验

保持基线 `configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml` 的模型、
数据、随机种子、优化器、训练预算及真实配对验证设置，仅增加辅助损失和独立输出目录。
基线默认 `wavelet_loss_weight=0`，不会构建小波损失或改变原目标。

令 P_0 为模型原始输出，T_0 为 GT；递归计算 P_j=LL(P_{j-1})、T_j=LL(T_{j-1})：

```
L_wavelet = (1 / J) * sum_j mean(abs(P_j - T_j))
L_total = 0.6 * L_raw + 0.4 * L_chromatic + wavelet_loss_weight * L_wavelet
```

这里使用每一层 LL（默认 LL_1、LL_2、LL_3），逐层 L1 后取平均。
4 个 packed RAW 通道独立分解，采用固定低通滤波器、两倍下采样和 symmetric 边界延拓。
计算标准 DWT 系数，不额外按层归一化；二维 LL 对常数的增益每层为 2，
因此改变分解层数时辅助损失尺度也会变化。默认 0.1 是对比实验起点，并非已验证的最优权重。
分解全程使用可微 PyTorch 运算；PyWavelets 仅提供固定滤波器系数。
这里借鉴小波多尺度表示，不做硬/软阈值处理，也不监督 LH/HL/HH。

滤波器和边界约定参考：
https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html

## 配置与运行

已提供 5 份配置，均为 3 层、权重 0.1：

| 小波基 | 配置文件（位于 configs/） |
|---|---|
| haar | train_sid_sony_mrlfn_paper_s2d_k4_n4_d32_wavelet_ll_l3_haar_w010.yaml |
| db2 | train_sid_sony_mrlfn_paper_s2d_k4_n4_d32_wavelet_ll_l3_db2_w010.yaml |
| sym4 | train_sid_sony_mrlfn_paper_s2d_k4_n4_d32_wavelet_ll_l3_sym4_w010.yaml |
| coif1 | train_sid_sony_mrlfn_paper_s2d_k4_n4_d32_wavelet_ll_l3_coif1_w010.yaml |
| bior2.2 | train_sid_sony_mrlfn_paper_s2d_k4_n4_d32_wavelet_ll_l3_bior2_2_w010.yaml |

```bash
python train_sid_sony.py --config configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml
python train_sid_sony.py --config configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32_wavelet_ll_l3_haar_w010.yaml
```

YAML 参数：

```yaml
wavelet_basis: haar  # haar / db2 / sym4 / coif1 / bior2.2
wavelet_levels: 3
wavelet_loss_weight: 0.1  # 0 禁用
```

对应 CLI 为 `--wavelet-basis`、`--wavelet-levels`、`--wavelet-loss-weight`。
覆盖参数另做实验时同时指定独立 `--output-dir`，避免混入其他实验的日志和权重。
对比实验从相同初始化条件开始。日志记录未加权 `train_wavelet_ll_l1`、
加权 `train_wavelet_weighted` 和总 `train_loss`。`model_info.json` 与 checkpoint args
保存小波设置。最佳权重仍由真实配对验证的 `real_psnr` 选择。

依赖 `PyWavelets>=1.4,<2`，已加入 requirements.txt。
