# band_channel 固定 2×2 卷积部署导出

输入 FP32 NCHW [1,4,360,640]，opset 17，使用各实验 best.pth 的 checkpoint args 恢复实际结构。

固定 Haar DWT/S2D 使用 stride=2、无偏置的稠密 2×2 Conv；IWT/D2S 使用对应 ConvTranspose。核无训练参数、无分组卷积，RAW 通道顺序保持不变。阈值表包含 softplus 和 scale，逐层处理高频，没有 atlas 拼装。固定常量由代码重建，因此保持旧 checkpoint 兼容。

同名 conv2x2 导出文件为本次产物；旧的无后缀导出文件未覆盖。

## sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5

- ONNX：`experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5/onnx/learning_dwt_repncb_best_deploy_1x4x360x640_conv2x2.onnx`
- 精度报告：`experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5/onnx/learning_dwt_repncb_best_deploy_1x4x360x640_conv2x2.json`
- 最大绝对误差：8.34465027e-07（uniform、signed RAW、零输入）。
- 算子计数：`{'Constant': 54, 'Conv': 19, 'Split': 3, 'Abs': 9, 'Add': 18, 'Clip': 9, 'Div': 10, 'Mul': 28, 'Relu': 1, 'PRelu': 8, 'Slice': 5, 'Concat': 5, 'ConvTranspose': 4}`

## sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf

- ONNX：`experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf/onnx/learning_dwt_repncb_best_deploy_1x4x360x640_conv2x2.onnx`
- 精度报告：`experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf/onnx/learning_dwt_repncb_best_deploy_1x4x360x640_conv2x2.json`
- 最大绝对误差：8.34465027e-07（uniform、signed RAW、零输入）。
- 算子计数：`{'Constant': 54, 'Conv': 17, 'Split': 3, 'Abs': 9, 'Add': 18, 'Clip': 9, 'Div': 10, 'Mul': 28, 'Relu': 1, 'PRelu': 6, 'Slice': 5, 'Concat': 5, 'ConvTranspose': 4}`

## 复现

```bash
python tools/export_sid_onnx.py \
  experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf/checkpoints/best.pth \
  experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5/checkpoints/best.pth \
  --height 360 --width 640 --tag conv2x2
```

CPUExecutionProvider 验证通过不等于手机 NPU 全部算子可执行。当前仍有 Split/Concat、PReLU、Abs/Div/Clip 等算子；ConvTranspose 支持、图分区及 FP16/INT8 精度需在目标 NPU SDK 核实。尚未测试手机速度，不作提速承诺。其他输入分辨率需重新导出。
