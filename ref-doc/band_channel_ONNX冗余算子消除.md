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
