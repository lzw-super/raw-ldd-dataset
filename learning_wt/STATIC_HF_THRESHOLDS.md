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
