# 4K packed RAW 五模型 W8A8 NPU测速（1088×1920）

日期：2026-10-10。仅比较部署速度，未进行训练或INT8画质评估。

## 1. 输入与测试条件

- 输入逻辑shape：**NCHW `[1,4,1088,1920]`**，对应用户指定的HWC `1088×1920×4`。原始4K Bayer RAW 3840×2160在packed域高度1080补至1088，相当于原始RAW高度补至2176。
- 相比此前 `[1,4,544,960]`，宽高均翻倍，像素数为4倍。外部补边、裁剪、ISP不计入模型耗时。
- 设备：Samsung Galaxy S24，日志核对SM-S921U1 / SM8650 / Hexagon V75；同型号云设备不保证同一台物理手机。
- SDK qai-hub 0.56.0，QAIRT 2.50.0.260828221209；QNN DLC，NPU，权重/激活均8bit，IO量化。本次编译产物IO类型为uint8，不代表所有中间累加器也为8bit。
- 平台随机校准，仅测速；每模型3轮，每轮100个样本，均值/P50/P95统计全部300个样本，不剔除首个较慢样本；不含云端排队、上传、编译时间。
- 本次独立输出目录为 `qaihub_4k_1088_20261010`。此前 `qaihub_4k_20261010` 是高度1080的未完成记录，不纳入结果。

## 2. 权重与导出核对

沿用此前1080p测试的权重快照。Rep-NCB HF为已训练epoch 50权重，Dual-DW3为已训练epoch 1000权重；单DW3残差仍为seed=2026的未训练HF替换模块，LL与精修来自原基线。MRLFN与SplitterNet使用之前同一权重。因此同一模型的分辨率比较保持权重一致，跨模型结果不代表精度对比。Dual是两个DW＋ReLU分支相减，没有输入恒等残差。

| 模型 | clean ONNX节点 | 部署参数量 | FP32核对最大绝对误差 |
|---|---:|---:|---:|
| Rep-NCB3×3＋PReLU＋输入残差 | 55 | 90,368 | 1.67e-06 |
| Dual-DW3＋ReLU | 61 | 87,128 | 1.91e-06 |
| DW3×3＋ReLU＋输入残差 | 49 | 86,768 | 1.91e-06 |
| MRLFN | 49 | 130,672 | 1.79e-06 |
| SplitterNet clean | 487 | 731,636 | 8.34e-07 |

各模型六种输入的PyTorch/ORT核对均通过；导出数值核对不等价于INT8精度验证。Haar模型使用2×2卷积/转置卷积实现小波和S2D/D2S，Rep-NCB转为部署卷积，ONNX无Div/Abs/Sign。SplitterNet采用固定shape clean导出；网络边界卷积所需Pad并不等于可删除的输入尺寸对齐Pad。

## 3. NPU实测结果

| 模型 | 4K均值 / ms | P50 / ms | P95 / ms | 此前1080p均值 / ms | 耗时倍数 |
|---|---:|---:|---:|---:|---:|
| Rep-NCB3×3＋PReLU＋输入残差 | **10.1631** | 10.0545 | 11.0592 | 1.8111 | 5.61× |
| Dual-DW3＋ReLU | **10.6990** | 10.5500 | 11.4482 | 1.9079 | 5.61× |
| DW3×3＋ReLU＋输入残差 | **9.4048** | 9.2845 | 10.0460 | 1.6509 | 5.70× |
| MRLFN | **9.3688** | 9.3125 | 9.8705 | 2.2318 | 4.20× |
| SplitterNet clean | **21.1247** | 21.1560 | 21.7645 | 4.4801 | 4.72× |

| 模型 | 第一轮均值 / ms | 第二轮均值 / ms | 第三轮均值 / ms |
|---|---:|---:|---:|
| Rep-NCB3×3＋PReLU＋输入残差 | 9.8612 | 10.2595 | 10.3686 |
| Dual-DW3＋ReLU | 10.6501 | 10.6115 | 10.8353 |
| DW3×3＋ReLU＋输入残差 | 9.4518 | 9.4067 | 9.3560 |
| MRLFN | 9.4990 | 9.2710 | 9.3363 |
| SplitterNet clean | 21.0368 | 21.2777 | 21.0595 |

本次MRLFN与单DW3残差的均值只相差约0.4%，结合轮间波动，不应宣称稳定的速度优劣。三种Haar模型的耗时增长约5.6～5.7倍，MRLFN约4.2倍，SplitterNet约4.7倍。

面积增长4倍不要求延迟严格增长4倍：固定调度成本、编译后的融合/分块、缓存和访存压力都会影响比例。本表只报告实测比例，不把这些机制当成本次已被独立测量的原因。

## 4. 复现与证据

部署配置位于 `configs/deploy/*_4k.yaml`，其height=1088、width=1920；ONNX位于 `experiments/npu_4k_1088/onnx/`。

```bash
PYTHONPATH=/tmp/sid-onnxsim python tools/export_sid_npu_ablation.py \
  configs/deploy/haar_hf_repncb_prelu_residual_4k.yaml \
  configs/deploy/haar_hf_dw3_dual_relu_4k.yaml \
  configs/deploy/haar_hf_cnn_dw3_residual_4k.yaml \
  configs/deploy/mrlfn_4k.yaml configs/deploy/splitternet_4k.yaml \
  --output-dir experiments/npu_4k_1088/onnx

read -r -s -p "AI Hub API token: " QAI_HUB_API_TOKEN
export QAI_HUB_API_TOKEN
PYTHONPATH=/tmp/sid-qai-hub python tools/benchmark_sid_qaihub.py \
  --watch --height 1088 --width 1920 \
  --sources-json ref-doc/qaihub_4k_1088_20261010/sources.json \
  --output-dir ref-doc/qaihub_4k_1088_20261010
unset QAI_HUB_API_TOKEN
python tools/summarize_sid_qaihub.py \
  --output-dir ref-doc/qaihub_4k_1088_20261010
```

导出环境：PyTorch 2.1.2、ONNX 1.16.2、ORT 1.18、onnxsim 0.4.36；`/tmp`下依赖目录仅为本机环境，其他机器需安装对应依赖。重导出后应重新生成sources.json的source_sha256，且新独立测量应使用新的结果目录；现有目录会恢复已提交任务。

| 模型 | 编译任务 | 三轮测速任务 |
|---|---|---|
| Rep-NCB3×3＋PReLU＋输入残差 | [jpx07r48p](https://workbench.aihub.qualcomm.com/jobs/jpx07r48p) | [jp0o19m0p](https://workbench.aihub.qualcomm.com/jobs/jp0o19m0p) / [jp8j3req5](https://workbench.aihub.qualcomm.com/jobs/jp8j3req5) / [j5q471leg](https://workbench.aihub.qualcomm.com/jobs/j5q471leg) |
| Dual-DW3＋ReLU | [jgn19qnjp](https://workbench.aihub.qualcomm.com/jobs/jgn19qnjp) | [jglw08y2p](https://workbench.aihub.qualcomm.com/jobs/jglw08y2p) / [j56o3m8n5](https://workbench.aihub.qualcomm.com/jobs/j56o3m8n5) / [jp3o47zmp](https://workbench.aihub.qualcomm.com/jobs/jp3o47zmp) |
| DW3×3＋ReLU＋输入残差 | [jprx4d0kp](https://workbench.aihub.qualcomm.com/jobs/jprx4d0kp) | [jgod1wl15](https://workbench.aihub.qualcomm.com/jobs/jgod1wl15) / [jpv21mlzg](https://workbench.aihub.qualcomm.com/jobs/jpv21mlzg) / [jgj30yr1p](https://workbench.aihub.qualcomm.com/jobs/jgj30yr1p) |
| MRLFN | [jp2o7dw6g](https://workbench.aihub.qualcomm.com/jobs/jp2o7dw6g) | [jpe6rx78g](https://workbench.aihub.qualcomm.com/jobs/jpe6rx78g) / [jgzzxyl4g](https://workbench.aihub.qualcomm.com/jobs/jgzzxyl4g) / [j5wydzl4g](https://workbench.aihub.qualcomm.com/jobs/j5wydzl4g) |
| SplitterNet clean | [jpy842x0g](https://workbench.aihub.qualcomm.com/jobs/jpy842x0g) | [jp1od1nn5](https://workbench.aihub.qualcomm.com/jobs/jp1od1nn5) / [jgd6r4d6p](https://workbench.aihub.qualcomm.com/jobs/jgd6r4d6p) / [j57ojneng](https://workbench.aihub.qualcomm.com/jobs/j57ojneng) |

编译后输入输出规格：[compiled_specs.json](qaihub_4k_1088_20261010/compiled_specs.json)。

原始记录：[summary.json](qaihub_4k_1088_20261010/summary.json)、[jobs.json](qaihub_4k_1088_20261010/jobs.json)、[sources.json](qaihub_4k_1088_20261010/sources.json)、[export_reports.json](qaihub_4k_1088_20261010/export_reports.json)、[checkpoint_provenance.json](qaihub_4k_1088_20261010/checkpoint_provenance.json)、[execution_audit.json](qaihub_4k_1088_20261010/execution_audit.json)。每模型子目录保存DLC、profile JSON及编译/运行日志。
