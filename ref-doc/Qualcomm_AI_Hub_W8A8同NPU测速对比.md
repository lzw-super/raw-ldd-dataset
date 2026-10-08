# Qualcomm AI Hub：五种 SID 去噪模型 W8A8 同款 NPU 测速

测速任务日期：2026-10-07（UTC）；报告整理：2026-10-08（北京时间）。只比较部署推理速度，不比较 PSNR、SSIM 或量化后的图像效果。

## 1. 测试口径

五种模型均加载现有 `best.pth`，导出部署图，再由 Qualcomm AI Hub Workbench 使用 QAIRT 量化并编译为 QNN DLC。统一选择 **Samsung Galaxy S24 / Snapdragon 8 Gen 3（SM8650）/ Hexagon V75 NPU**，输入统一为 **NCHW `[1,4,360,640]` packed RAW**。这是 packed 图像尺寸，不是 Bayer mosaic 尺寸。

指定的是同一个设备型号与 NPU 架构；云平台可能调度该型号的不同物理手机，不能保证 15 个任务都在同一台物理设备上执行。

| 项目 | 本次设置 |
|---|---|
| AI Hub SDK | `qai-hub==0.56.0` |
| QAIRT | `2.50`，产物元数据为 `2.50.0.260828221209` |
| 目标运行时 | QNN DLC / HTP NPU，`--compute_unit npu` |
| 源 ONNX | FP32，静态输入，opset 17；平台编译器可能继续升级/优化图 |
| 权重 / 激活 | W8A8：`--quantize_full_type int8` |
| 边界 I/O | `--quantize_io`；实际 QNN 接口为 `uint8`，带 scale/zero-point |
| 校准数据 | 平台生成的随机输入；仅作速度评估 |
| 重复次数 | 每模型 3 个独立测速任务，每任务最多 100 次推理 |
| 单任务时间上限 | 600 秒，不是单次推理耗时 |
| 功耗/频率策略 | 没有显式覆盖 QNN 功耗档位，统一采用平台默认策略 |
| PSNR / SSIM | 未测，不作为本报告结论 |

本次使用编译阶段的 `--quantize_full_type int8` 快速量化流程，没有使用基于真实 SID 数据的 AIMET 精度校准。官方明确提供随机校准进行性能评估的流程：[量化与性能测试说明](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html#benchmarking-quantized-model-performance)。随机校准产物适合本次速度对比，不代表已经获得可交付的去噪精度。

**INT8 的准确含义：** 权重与激活按 8 位定点量化；QNN 可用无符号 8 位编码表示激活，因此接口中的 `uint8` 不等于 FP32/FP16 回退。不能把 W8A8 理解成偏置、累加器、shape/index 等所有数据都必须是有符号 int8。本次没有逐个证明编译器内部临时张量的位宽，证据是实际量化命令、产物接口和端侧执行记录。

## 2. 实测速度

**五个模型的 W8A8 编译和三轮 NPU 测速均成功。** 每个模型共 300 个原始延迟样本。

| 排名 | 模型 | 全部样本均值 / ms | P50 / ms | P95 / ms | 等效吞吐 / 次每秒 |
|---|---|---:|---:|---:|---:|
| 1 | MRLFN | 0.7723 | 0.7485 | 0.9800 | 1294.8 |
| 2 | Haar 无 LL 归一化 | 1.1976 | 1.1840 | 1.2340 | 835.0 |
| 3 | SplitterNet | 2.6029 | 2.5795 | 2.7191 | 384.2 |
| 4 | UNet | 2.8483 | 2.8020 | 3.0411 | 351.1 |
| 5 | NAFNet | 47.3201 | 47.2795 | 47.9130 | 21.1 |

三轮平台报告值（`estimated_inference_time`）：

| 模型 | 第 1 轮 / ms | 第 2 轮 / ms | 第 3 轮 / ms | 三轮报告值均值 / ms |
|---|---:|---:|---:|---:|
| MRLFN | 0.739 | 0.742 | 0.743 | 0.7413 |
| Haar 无 LL 归一化 | 1.168 | 1.163 | 1.173 | 1.1680 |
| SplitterNet | 2.538 | 2.528 | 2.483 | 2.5163 |
| UNet | 2.770 | 2.790 | 2.780 | 2.7800 |
| NAFNet | 47.368 | 46.709 | 46.567 | 46.8813 |

按全部样本均值计算，Haar 的耗时是 MRLFN 的 **1.55 倍**；相比 NAFNet、UNet、SplitterNet，Haar 的速度分别为 **39.51×、2.38×、2.17×**。这说明在该输入尺寸和 NPU 上，原有 MRLFN 仍具有更低部署延迟，Haar 方案也明显快于其余三个模型。此结论仅涉及速度，不代表画质排序。

NAFNet 在本次图与编译设置下的延迟明显更高。检查其[第一轮逐算子记录](qaihub_w8a8_20261007/nafnet/profile_1.json)，周期数最高的条目集中在首级编码器和末级解码器的 `norm1/norm2` 路径，例如 `/decoders.3/decoders.3.1/norm1/Add_1`。这些条目均标记为 NPU。由此可把归一化路径作为后续优化的检查重点；编译后节点名可能代表融合计算，不能仅凭 `Add_1` 名称断定单个加法就是瓶颈，本次也没有修改 NAFNet 来单独验证这一原因。

统计说明：

- JSON 中时间单位为微秒，本报告统一除以 1000 转成毫秒。
- 主表按三轮全部 `all_inference_times` 的均值排序；不删除首个较慢样本、不人为去除异常值。
- P50/P95 对三轮样本合并后做线性插值分位数；每轮数量以原始文件为准。
- 平台另给出 `estimated_inference_time`。它与原始样本均值不是同一指标，因此单列三轮值，不混称为平均延迟。
- `1000 / 平均毫秒数` 仅为模型连续执行的等效吞吐率，不是相机系统端到端帧率。
- 不计上传、云端排队、量化编译、首次加载，也不计应用层 RAW 打包、浮点转定点、输出反量化、ISP 与显示。端侧运行时的推理开销按平台的测量口径计入。
- 未做长期热稳定性或功耗评估；本表是上述短时、固定形状条件下的测速结果。

## 3. 模型与权重对应关系

训练配置用于确定所比较的实验，实际导出加载检查点中的模型结构与权重。尤其 `train_sid_sony.yaml` 对应的是 **UNet**；NAFNet 和 UNet 的权重分别位于带 `_new` 的实验目录。

### Haar + LL Rep-NCB + 精修（无 LL 归一化）

- 配置：[train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm.yaml](../configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm.yaml)
- 权重：[best.pth](../experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm/checkpoints/best.pth)
- 量化前部署图：[learning_dwt_repncb_best_deploy_1x4x360x640_qai_w8a8_source.onnx](../experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm/onnx/learning_dwt_repncb_best_deploy_1x4x360x640_qai_w8a8_source.onnx)
- W8A8 产物：[model.dlc](qaihub_w8a8_20261007/haar_ll_no_norm/model.dlc)

### MRLFN S2D-k4 / n4 / d32

- 配置：[train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml](../configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml)
- 权重：[best.pth](../experiments/sid_sony_mrlfn_paper_s2d_k4_n4_d32/checkpoints/best.pth)
- 量化前部署图：[mrlfn_best_deploy_1x4x360x640_qai_w8a8_source.onnx](../experiments/sid_sony_mrlfn_paper_s2d_k4_n4_d32/onnx/mrlfn_best_deploy_1x4x360x640_qai_w8a8_source.onnx)
- W8A8 产物：[model.dlc](qaihub_w8a8_20261007/mrlfn/model.dlc)

### NAFNet

- 配置：[train_sid_sony_nafnet.yaml](../configs/train_sid_sony_nafnet.yaml)
- 权重：[best.pth](../experiments/sid_sony_nafnet_new/checkpoints/best.pth)
- 量化前部署图：[nafnet_best_deploy_1x4x360x640_qai_w8a8_source.onnx](../experiments/sid_sony_nafnet_new/onnx/nafnet_best_deploy_1x4x360x640_qai_w8a8_source.onnx)
- W8A8 产物：[model.dlc](qaihub_w8a8_20261007/nafnet/model.dlc)

### UNet（train_sid_sony.yaml）

- 配置：[train_sid_sony.yaml](../configs/train_sid_sony.yaml)
- 权重：[best.pth](../experiments/sid_sony_paper_new/checkpoints/best.pth)
- 量化前部署图：[unet_best_deploy_1x4x360x640_qai_w8a8_source.onnx](../experiments/sid_sony_paper_new/onnx/unet_best_deploy_1x4x360x640_qai_w8a8_source.onnx)
- W8A8 产物：[model.dlc](qaihub_w8a8_20261007/unet/model.dlc)

### SplitterNet

- 配置：[train_sid_sony_splitternet.yaml](../configs/train_sid_sony_splitternet.yaml)
- 权重：[best.pth](../experiments/sid_sony_splitternet/checkpoints/best.pth)
- 量化前部署图：[splitternet_best_deploy_1x4x360x640_qai_w8a8_source.onnx](../experiments/sid_sony_splitternet/onnx/splitternet_best_deploy_1x4x360x640_qai_w8a8_source.onnx)
- W8A8 产物：[model.dlc](qaihub_w8a8_20261007/splitternet/model.dlc)

Haar 路径使用已融合的 Rep-NCB、离线静态 soft threshold、卷积形式的 Haar/S2D/D2S，以及 `ll_normalize=False`；导出时启用 `--simplify-static`。MRLFN 使用部署态权重。NAFNet、UNet、SplitterNet 保留各自原有模型结构，交给相同 QAIRT 流程优化，没有为了测速裁剪网络。

所有源 ONNX 均通过导出脚本中的 PyTorch/ONNX Runtime 一致性检查（uniform、signed_raw、zeros、tiny）。这是**量化前导出正确性检查**，不构成量化精度评测。配置、检查点、ONNX 的 SHA256、模型元数据及误差记录见 [sources.json](qaihub_w8a8_20261007/sources.json)；DLC 哈希和速度统计见 [summary.json](qaihub_w8a8_20261007/summary.json)。

## 4. W8A8 与 NPU 执行的核实

五份编译日志均实际调用 QAIRT quantizer，并含有：

```text
--weights_bitwidth 8 --act_bitwidth 8
```

量化模型的实际输入/输出 dtype、scale、zero-point、QAIRT 版本已保存到 [compiled_specs.json](qaihub_w8a8_20261007/compiled_specs.json)。本次下载的 **DLC 是量化后产物**，文件名带 `qai_w8a8_source.onnx` 的 ONNX 是量化前输入文件，不能仅凭文件名当成 INT8 模型。

| 模型 | 各轮 NPU profiler 条目数 | 各轮 CPU / GPU 条目数 | 实际迭代数 |
|---|---|---|---|
| MRLFN | 49 / 49 / 49 | 0 / 0 / 0 | 100 / 100 / 100 |
| Haar 无 LL 归一化 | 111 / 111 / 111 | 0 / 0 / 0 | 100 / 100 / 100 |
| SplitterNet | 492 / 492 / 492 | 0 / 0 / 0 | 100 / 100 / 100 |
| UNet | 54 / 54 / 54 | 0 / 0 / 0 | 100 / 100 / 100 |
| NAFNet | 494 / 494 / 494 | 0 / 0 / 0 | 100 / 100 / 100 |

所有 15 份端侧日志均确认 `SM-S921U1 / SM8650 / Android 14`，并加载 `libQnnHtp.so`，版本 `2.50.0.260828221209`。没有在 profiler 条目中发现 CPU/GPU 算子回退。提取证据见 [execution_audit.json](qaihub_w8a8_20261007/execution_audit.json)。

`execution_detail` 统计的是 profiler 的条目，可能包含 Input、融合节点或内部节点，不能直接当成训练网络层数。NPU 条目不意味着 CPU 完全不参与调度或内存管理。编译日志里的 `QNN_CPU` 是主机侧校准执行，不能拿它判断手机端发生了 CPU fallback。

## 5. 云端任务与本地结果

| 模型 | 编译任务 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---|---|---|---|
| Haar 无 LL 归一化 | [jp1oevz85](https://workbench.aihub.qualcomm.com/jobs/jp1oevz85) | [j5q43m6ng](https://workbench.aihub.qualcomm.com/jobs/j5q43m6ng) | [jglw31vjp](https://workbench.aihub.qualcomm.com/jobs/jglw31vjp) | [j56ondy65](https://workbench.aihub.qualcomm.com/jobs/j56ondy65) |
| MRLFN | [jp0olv09p](https://workbench.aihub.qualcomm.com/jobs/jp0olv09p) | [j5wyq1w3g](https://workbench.aihub.qualcomm.com/jobs/j5wyq1w3g) | [jp1oev285](https://workbench.aihub.qualcomm.com/jobs/jp1oev285) | [jgd6oznrp](https://workbench.aihub.qualcomm.com/jobs/jgd6oznrp) |
| NAFNet | [jp4ev9l8g](https://workbench.aihub.qualcomm.com/jobs/jp4ev9l8g) | [jpy867jlg](https://workbench.aihub.qualcomm.com/jobs/jpy867jlg) | [jgk639qn5](https://workbench.aihub.qualcomm.com/jobs/jgk639qn5) | [j56ondny5](https://workbench.aihub.qualcomm.com/jobs/j56ondny5) |
| UNet | [jp8jz4qk5](https://workbench.aihub.qualcomm.com/jobs/jp8jz4qk5) | [jpx0ydn3p](https://workbench.aihub.qualcomm.com/jobs/jpx0ydn3p) | [jgn137lkp](https://workbench.aihub.qualcomm.com/jobs/jgn137lkp) | [jprxen80p](https://workbench.aihub.qualcomm.com/jobs/jprxen80p) |
| SplitterNet | [jgk639ew5](https://workbench.aihub.qualcomm.com/jobs/jgk639ew5) | [jgd6ozqzp](https://workbench.aihub.qualcomm.com/jobs/jgd6ozqzp) | [j57ox7l9g](https://workbench.aihub.qualcomm.com/jobs/j57ox7l9g) | [jp4ev9d1g](https://workbench.aihub.qualcomm.com/jobs/jp4ev9d1g) |

每个模型目录保存 `model.dlc`、`compile_logs/`、三轮 `profile_N.json` 以及 `profile_logs_N/`。完整任务状态及原始选项见 [jobs.json](qaihub_w8a8_20261007/jobs.json)。云端链接需要登录具有相应访问权限的 AI Hub 账号。

## 6. 复现命令

以下命令均从仓库根目录执行。PyTorch/ONNX 导出使用项目现有环境；云端工具建议使用单独环境安装 `qai-hub==0.56.0`，避免其依赖与训练环境冲突。

### 6.1 重新导出五个模型

```bash

python tools/export_sid_onnx.py \
  experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm/checkpoints/best.pth \
  --height 360 --width 640 --ort-opt-level disabled \
  --simplify-static --tag qai_w8a8_source

python tools/export_sid_onnx.py \
  experiments/sid_sony_mrlfn_paper_s2d_k4_n4_d32/checkpoints/best.pth \
  experiments/sid_sony_nafnet_new/checkpoints/best.pth \
  experiments/sid_sony_paper_new/checkpoints/best.pth \
  experiments/sid_sony_splitternet/checkpoints/best.pth \
  --height 360 --width 640 --ort-opt-level disabled \
  --tag qai_w8a8_source
```

### 6.2 W8A8 编译与 NPU 测速

脚本：[tools/benchmark_sid_qaihub.py](../tools/benchmark_sid_qaihub.py)。认证从环境变量读取，文档和仓库均不保存 token：

```bash
python -m pip install 'qai-hub==0.56.0'
read -r -s -p 'AI Hub API token: ' QAI_HUB_API_TOKEN
export QAI_HUB_API_TOKEN
python tools/benchmark_sid_qaihub.py --watch
unset QAI_HUB_API_TOKEN
```

默认读取本次 `jobs.json`，已有任务只恢复状态和下载结果，不会重复提交已记录的任务。需要独立重测时指定新的输出目录，例如：

```bash
python tools/benchmark_sid_qaihub.py --watch \
  --output-dir ref-doc/qaihub_w8a8_repeat
```

此命令也需要处于已设置认证环境变量的 shell。脚本统一提交以下选项：

```text
# compile
--target_runtime qnn_dlc --quantize_full_type int8 --quantize_io --compute_unit npu --qairt_version 2.50

# inference with profile=True, inputs=None
--compute_unit npu --qairt_version 2.50 --max_profiler_iterations 100 --max_profiler_time 600
```

SDK 0.56 使用 `submit_inference_job(..., profile=True, inputs=None)` 获取性能数据。这里不提交 SID 图像推理或精度评分任务。

### 6.3 本地重新汇总

下面的汇总不访问网络，也不需要 token：

```bash
python tools/summarize_sid_qaihub.py
# 独立重测对应：
python tools/summarize_sid_qaihub.py --output-dir ref-doc/qaihub_w8a8_repeat
```

输出更新对应目录的 `summary.json`。模型源文件和权重改变后应使用新的输出目录，避免沿用旧任务结果。

## 7. 官方流程参考

- [Qualcomm AI Hub / Workbench 入门](https://aihub.qualcomm.com/get-started#workbench)
- [量化与随机校准性能测试](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html)
- [AI Hub Python API、编译与运行选项](https://workbench.aihub.qualcomm.com/docs/hub/api.html)


## 计算量与 NPU 速度差异补充

三轮逐层 profile 的模块汇总、Haar 最慢的18个条目以及优化建议见[Haar / MRLFN / SplitterNet NPU耗时与计算量差异分析](Haar_MRLFN_SplitterNet_NPU耗时与计算量差异分析.md)。其中的百分比是逐层插桩测量的周期占比，不应直接换算为整网延迟占比。
