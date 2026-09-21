# LL 中间卷积替换为 PlainUSR RepMBConv

配置：`configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repmbconv.yaml`。

基于固定 Haar w32 配置，仅增加 `dwt_ll_block_type: repmbconv` 并改变输出目录。默认值 `conv3x3` 保留所有旧实验结构。不是 kernel_finetune：本实验从头联合训练所有 CNN，Haar 仍固定。

LL 网络：`Conv3×3 4→32 → ReLU → RepMBConv 32→32 → ReLU → RepMBConv 32→32 → ReLU → Conv3×3 32→4`，之后继续与原 LL 拼接并通过原来的 1×1 融合。LL 深度 4 指四个逻辑位置，两个中间位置展开为 RepMBConv 多层训练结构。没有改动阈值网络或后续 Rep-NCB 精修网络。

参考官方 [PlainUSR_train_arch.py](https://github.com/icandle/PlainUSR/blob/d7ff11afe3d12a68c12375a7959d3aa8a3a50eda/2024_PlainUSR_ACCV/archs/PlainUSR_train_arch.py) 的 `MBConv` 和 `ASR`，MIT 许可证保留在 `models/licenses/PlainUSR-MIT.txt`。

单个模块：1×1 扩张 32→64，稠密 3×3 64→64，经 ASR 缩放后与扩张特征相加，1×1 压缩 64→32，再加模块输入。ASR 使用输入无关的可学习常量、两层无偏置 Linear（64→16→64）、SiLU 和 Sigmoid；因此可以在部署时折叠。实现保留官方常量 0.1 和 Linear 权重全 1 的初始化。官方注释所说“初始 0.5”与该初始化实际数值不符；本实现遵循代码数值。保留 LL 原有 ReLU，而不是引入 PlainUSR 外层 Block 的 LeakyReLU、注意力或网络结构。

边界采用前置偏置填充，融合包含 ASR、内外残差以及所有偏置。`deploy()` 自动复制并融合 LL RepMBConv 和精修 Rep-NCB；每个 RepMBConv 成为普通 3×3 卷积，ReLU 保留。训练态和部署态 checkpoint 均支持评测加载。旧 LL 标准卷积 checkpoint 不能直接 strict-load 到该训练结构。

已检查非零随机偏置、小尺寸边界的融合等价性、端到端梯度和两种 checkpoint 的恢复。训练仍由用户离线执行。

## 新增对照

- `_repncb_w32_ll_repncb.yaml`：`dwt_ll_block_type: repncb`，LL 深度仍为 4，中间两个位置使用现有 Rep-NCB（含 PReLU，因此不再额外接 ReLU）。首尾仍为标准 4→32、32→4 卷积，首层后保留 ReLU。
- `_repncb_w32_ll_repmbconv_depth5.yaml`：沿用 RepMBConv 配置，只将 `dwt_ll_depth` 设为 5，中间增加为三个 `RepMBConv + ReLU`，首尾标准卷积不变。

两个配置使用各自独立的输出目录，其他设置沿用原 RepMBConv 配置。LL Rep-NCB 和 RepMBConv 均支持训练态/部署态权重加载与自动融合。
