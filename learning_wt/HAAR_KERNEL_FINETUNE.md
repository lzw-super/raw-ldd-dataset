# 仅微调 Haar 分析/重建核

配置：`configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_kernel_finetune.yaml`。

`dwt_trainable_haar: true` 启用可训练的 2×2 分析核和重建核。`train_sid_sony.py` 在此模式冻结除 `wavelet.transform.analysis`、`wavelet.transform.synthesis` 外的全部参数，优化器只接收这 32 个参数。梯度仍经过冻结的 CNN 回传，不使用 no_grad 或 detach。

分析使用 stride=2 的 Conv2d，重建使用 stride=2 的 ConvTranspose2d，均无偏置。四个核按 LL/LH/HL/HH 排列，初值分别是以下矩阵乘 1/2：

```
LL: [1, 1; 1, 1]    LH: [1,-1; 1,-1]
HL: [1, 1;-1,-1]    HH: [1,-1;-1, 1]
```

分析和重建各有 4×2×2=16 个独立参数，四个 RAW 通道、三个层级共享核，不混合 RAW 通道。仅 LL 递归分解，atlas 排列和归一化沿用基准。初始化等价于现有 Haar 实现；训练后两端不强制正交、零均值或严格互逆，因此应理解为 Haar 初始化的可学习滤波器组。当前实验不加入额外正交/重建约束，使用原始最终输出损失优化。

默认加载 Haar w32 基准 `checkpoints/best.pth`，学习率降至 `1e-5`，使用独立输出目录，其他训练参数沿用基准。可以通过 `--init-checkpoint` 指定另一个结构相同的 Haar 基准训练 checkpoint。初始化只允许缺少新增的两个核参数，所有 CNN 权重严格匹配。必须指定 init_checkpoint 或 resume，避免冻结随机 CNN。`--resume` 应使用本实验 checkpoint，以恢复滤波器和优化器；旧基准只用于 init_checkpoint。

```
conda run --no-capture-output -n LED-ICCV23 python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_kernel_finetune.yaml
```

验证和测试使用相同可学习核，checkpoint 同时保存核与 CNN。继续保存 best PSNR、best_train_l1、latest 等权重。未启动完整微调，是否提升真实 SID PSNR 需与原始初始化 checkpoint 对照评测。

## 通道/层级共享对照

在 `_kernel_finetune.yaml` 文件名后缀前分别追加以下名称，三个配置保留相同 Haar 基准 init_checkpoint、学习率与训练设置，并使用独立输出目录：

| 后缀 | dwt_haar_share_channels | dwt_haar_share_levels | 可训练参数 |
|---|---|---|---:|
| 原共享基准 | true | true | 32 |
| `_channel_unshared` | false | true | 128 |
| `_level_unshared` | true | false | 96 |
| `_channel_level_unshared` | false | false | 384 |

每套分析/重建滤波器独立初始化为 Haar。通道不共享时采用 groups=4，各 RAW 通道单独处理，不混合通道；层级不共享时第一级处理原输入，第二/三级递归处理上一级 LL，重建时逆序使用相应层级的重建核。两端依旧独立学习。共享基准保持旧权重形状兼容；不同共享模式的微调 checkpoint 不能直接互相 resume，应从原始固定 Haar 基准初始化。
