# 高频子带/通道固定阈值对照

配置：`configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf.yaml`。

在原 LL-RepNCB 配置上增加 `dwt_threshold_mode: band_channel`，独立输出目录后缀 `_static_hf`。原默认 `cnn` 不变。新组没有阈值 CNN 或原 band_bias，只学习 9×4 个 hf_logits。三个层级的 LH/HL/HH，各四个 RAW 通道独立；同一子带同一通道的所有像素、所有输入图像共用阈值。

阈值 `T = softplus(hf_logits)`；为保留基准的 normalize_bands 语义，启用归一化时实际系数域阈值再乘 `2**level`。初始化的归一化阈值为 `dwt_init_threshold`（默认 0.01），三层实际阈值分别为 0.08、0.04、0.02。表格顺序是 LH3/HL3/HH3、LH2/HL2/HH2、LH1/HL1/HH1，列是原 packed RAW 四通道。

高频沿用 `dwt_shrink_mode: smooth`：`z_filtered = z * abs(z)/(abs(z)+T)`（实现保留原数值稳定处理）。只改变阈值来源，不改变缩放公式。LL 不使用这组阈值，继续由 LLRestorationCNN 恢复；LL 网络和后续精修网络均联合训练。dwt_width/dwt_depth 等阈值 CNN 参数在此模式不生效，但保留配置字段以便与基准比较。

`model.deploy()` 在副本上预计算 softplus 和尺度换算结果为 `wavelet.fixed_hf_thresholds` buffer，删除 hf_logits；测试脚本加载训练 checkpoint 时自动转换，加载 model_deploy 时直接使用阈值表，不再动态预测阈值或计算 softplus。训练 checkpoint 保留 hf_logits 以支持续训。新组应从头训练，不能直接 strict-load 原动态阈值 CNN checkpoint。

```bash
conda run --no-capture-output -n LED-ICCV23 python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf.yaml
```

## 逐层查表部署（无需 atlas）

默认部署 forward 已改为按层处理：

1. 第一级 DWT(I) 得到 LL1 和三组 HF1，按阈值表缩放 HF1。
2. 对原始 LL1 分解得到 LL2 和 HF2，按表缩放 HF2。
3. 对原始 LL2 分解得到 LL3 和 HF3，按表缩放 HF3。
4. LL 网络恢复 LL3，然后结合对应已缩放高频，从第三级向第一级 IWT 重建。

不再拼接十个子图，不生成全分辨率阈值图；只有每个子带的四通道阈值广播。阈值表已包含 softplus 与子带 scale，仍按深层到浅层排列，查表时映射当前层。smooth 缩放中的 `abs(z)/(abs(z)+T)` 仍依赖输入系数，必须在线计算，离线固化的是 T 而不是整张缩放系数图。

两个 static_hf 配置（包括 depth5）不需要修改或重训。原 best.pth/latest.pth 可直接通过评测模型工厂加载：有 model_deploy 时使用已保存阈值表，只有训练权重时在加载阶段计算一次；训练保存的 model_deploy 已包含该表。没有更改 checkpoint 键或表顺序。默认训练路径保留 atlas，`return_aux=True` 显式调试也保留 atlas 输出。

初始 LL 的归一化与恢复尺度仍保留，这是 LL CNN 的输入/输出语义，不属于高频阈值查表；不会重复做高频 scale 换算。四子带输入的 LL 实验仍使用未缩放的最深层原始四个子带，与原实现一致。尚未给出速度或显存改善的量化结论。
