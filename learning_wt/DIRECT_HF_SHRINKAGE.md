# 高频无动态除法收缩对照

基准为 `configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5.yaml`。新增同名前缀的 `_soft.yaml`、`_firm.yaml`、`_pwl.yaml`，仅改变 dwt_shrink_mode 和输出目录。沿用当前基准文件的实际 LL/精修深度，不按文件名猜测结构。三组从头联合训练，固定 Haar、LL 与精修网络结构不变。

设 `R(z,t)=ReLU(z-t)-ReLU(-z-t)`，每组参数对应一个高频子带和一个 RAW 通道，共 9×4 组。

| 模式 | 公式 | 可训练收缩参数 |
|---|---|---:|
| soft | R(z,T) | 36 |
| firm | k R(z,T1) - (k-1) R(z,T2) | 72 |
| pwl | m0 z + (m1-m0)R(z,tau1) + (m2-m1)R(z,tau2) + (1-m2)R(z,tau3) | 216 |

这是参考代码的奇对称等价写法，不需要显式 Abs/Sign；直接输出恢复系数而非预测 gain。Firm/PWL 要求 band_channel 和 leak=0。

## 参数化与初始化

Firm/PWL 遵循用户文件 `tmp/div-sub-code.txt`：

- Firm：p1=0、p2=1；T1=softplus(p1)，delta=softplus(p2)+1e-4，T2=T1+delta，k=T2/delta。初始归一化 T1≈0.693147，delta≈1.313362，T2≈2.006509。T2>T1，超过 T2 时回到 identity。
- PWL：p_tau、p_slope 均为零；三个正间隔为 softplus(p_tau)+1e-4，累积为有序折点；三个斜率 sigmoid(p_slope)，初始均为 0.5，最后斜率固定为 1。归一化初始折点约 0.693247、1.386494、2.079742。最后一段是斜率为 1 的直线，不一定是 y=z；本实现不擅自改变参考公式。
- Soft：参考文件未提供该类的初始化，因此沿用基准 dwt_init_threshold（0.01）以及 inverse-softplus 初始化。

基准启用 normalize_bands，所有折点/阈值乘对应层 2^level，折线斜率保持不变。这等价于在归一化系数域应用参考公式，再恢复尺度；实际输入 z 无需在线除以 scale。

参考 Firm/PWL 的初始阈值比基准 smooth 的 0.01 大很多。Firm 初始可能将多数高频压为零，远小于折点的样本也可能令某些折点梯度为零；PWL 在折点以下初始约为 0.5z。这不是拟合已有 smooth 权重的初始化。当前按用户参考代码保留，若后续做微调需要另外设计数据校准初始化，不能直接把效果差异都归因于函数形式。

## 部署

部署时预计算并保存 Firm 的 T1/T2/k、PWL 的折点与斜率差；soft 保存其阈值。softplus/sigmoid/cumsum 和 Firm 中的参数除法全部离线执行。高频数据路径仅含 ReLU、加减和常数乘法，无 Div、Abs、Sign；训练时 Firm 参数换算仍有除法，与输入无关。

继续使用逐层 LL 分解、高频查表处理与卷积形式 Haar 部署。完整模型仍可能有 LL 输入归一化的常数 /8，不能把“高频无 Div”描述为“全模型无 Div”。导出脚本的 --simplify-static 同样可用。训练态、部署态 checkpoint 均支持加载；旧 smooth checkpoint 与新 Firm/PWL 参数结构不兼容。

未启动完整训练。测试覆盖参考公式等价性、梯度、部署/权重加载以及单独高频部署 ONNX 无 Div/Softplus/Sigmoid/Abs/Sign。
