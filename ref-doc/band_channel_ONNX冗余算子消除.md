# 静态高频 ONNX 冗余算子消除

导出开关：`--simplify-static --tag conv2x2_clean`。验证设置 `--ort-opt-level disabled`，与启用简化前的 PyTorch 模型比较。

移除 leak=0 的加零/乘零、乘一、固定尺寸恒等裁剪和四通道 LL 的恒等 Slice。只有固定阈值均有限且不小于 FP32 tiny 时才移除分母 clamp；否则保留。非零 leak 不做该恒等简化。LL 的 /8、×8 和非恒等缩放保留。

## sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5

- 文件：`experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5/onnx/learning_dwt_repncb_best_deploy_1x4x360x640_conv2x2_clean.onnx`
- 节点总数（含 Constant）：173 → 85。
- 最大绝对误差：8.34465027e-07，uniform/signed_raw/zeros/tiny 均通过。
- 算子统计：`{'Constant': 7, 'Conv': 19, 'Split': 3, 'Abs': 9, 'Add': 9, 'Div': 10, 'Mul': 10, 'Relu': 1, 'PRelu': 8, 'Concat': 5, 'ConvTranspose': 4}`

## sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf

- 文件：`experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf/onnx/learning_dwt_repncb_best_deploy_1x4x360x640_conv2x2_clean.onnx`
- 节点总数（含 Constant）：169 → 81。
- 最大绝对误差：8.34465027e-07，uniform/signed_raw/zeros/tiny 均通过。
- 算子统计：`{'Constant': 7, 'Conv': 17, 'Split': 3, 'Abs': 9, 'Add': 9, 'Div': 10, 'Mul': 10, 'Relu': 1, 'PRelu': 6, 'Concat': 5, 'ConvTranspose': 4}`

新文件未覆盖旧 conv2x2 ONNX。shape 为 FP32 [1,4,360,640]，opset 17；没有手机 NPU 速度数据。FP16/INT8 转换后的数值需单独验证。


## Soft Threshold 权重导出 clean ONNX

现有导出脚本支持 `dwt_shrink_mode: soft`，自动加载 checkpoint 中的模型配置、冻结阈值并融合 Rep-NCB。无需手工修改权重或切换模型结构。

```bash
python tools/export_sid_onnx.py \
  experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft/checkpoints/best.pth \
  --height 360 --width 640 \
  --tag conv2x2_clean \
  --simplify-static \
  --ort-opt-level disabled
```

该命令使用已安装 torch、onnx、onnxruntime 的当前 Python 环境（本次实际使用的版本见旁边 JSON 报告）。固定 FP32 NCHW `[1,4,360,640]`，opset 17；更换输入尺寸需要重新导出。

- ONNX：`experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft/onnx/learning_dwt_repncb_best_deploy_1x4x360x640_conv2x2_clean.onnx`
- 验证报告：`experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft/onnx/learning_dwt_repncb_best_deploy_1x4x360x640_conv2x2_clean.json`
- 节点总数（含 Constant）：103。
- 算子统计：`{'Constant': 7, 'Conv': 19, 'Split': 3, 'Sub': 27, 'Relu': 19, 'Neg': 9, 'Div': 1, 'PRelu': 8, 'Concat': 5, 'Mul': 1, 'ConvTranspose': 4}`。
- 四类输入（uniform、signed_raw、zeros、tiny）全部通过，最大绝对误差 7.74860382e-07。验证关闭 ORT 图优化，与简化前的 PyTorch 输出比较。

高频处理为 `ReLU(z-T)-ReLU(-z-T)`，无动态 Div、Abs、Sign、Softplus；阈值已离线包含层级 scale。Haar DWT/IWT 和 S2D/D2S 继续使用固定 2×2 Conv/ConvTranspose。完整图仍保留 LL 归一化的常数除法，其节点为 `/wavelet/Div`，不能据此认为高频仍有动态除法。手机 NPU 实测与量化验证尚未进行。
