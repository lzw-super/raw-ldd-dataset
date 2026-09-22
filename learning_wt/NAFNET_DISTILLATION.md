# NAFNet 输出蒸馏

配置：`configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_distill_nafnet.yaml`。

- 教师来自 `configs/train_sid_sony_nafnet.yaml` 的输出目录 `experiments/sid_sony_nafnet_new/checkpoints/best.pth`，通过 checkpoint args 自动还原结构。
- 学生使用 Haar w32 LL-RepNCB 实验的 `best.pth` 初始化训练态权重，所有学生参数正常训练，固定 Haar 不改变。
- 学习率设为 1e-5，`distill_loss_weight: 1.0`；其余训练设置沿用学生配置，使用独立输出目录。

每个 batch 只合成一次噪声，教师和学生接收完全相同的 noisy packed RAW。教师保持 eval 且 requires_grad=False，前向使用 no_grad。蒸馏损失为两者未裁剪输出的逐像素平均绝对误差，不使用 GT 亮度校正。

`loss = 原有 GT 监督 loss + distill_loss_weight × mean(abs(student(noisy) - teacher(noisy)))`。

默认基准的 GT 监督为 0.6 RAW L1 + 0.4 chromatic L1；其他辅助损失若配置启用仍会保留。日志记录 `train_distill_l1` 和 `train_distill_weighted`，`train_l1` 仍表示对 GT 的原始 L1。验证仅评测学生，best.pth 仍由学生真实验证 PSNR 选择，best_train_l1.pth 仍按学生对 GT 的训练 L1 选择。

checkpoint 仅保存学生模型和优化器，教师路径及蒸馏参数随 args 保存。继续蒸馏请使用同一配置加 `--resume`，并保留教师 checkpoint 文件。测试蒸馏后的学生不需要加载教师。普通配置默认蒸馏权重为 0，不加载教师。

运行：

```bash
conda run --no-capture-output -n LED-ICCV23 python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_distill_nafnet.yaml
```

没有启动完整训练。已用真实师生 best.pth 验证结构加载、教师无梯度与学生蒸馏梯度；batch 显存容量需要实际训练时确认。
