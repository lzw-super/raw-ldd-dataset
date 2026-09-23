# 四个 SID 模型的 ONNX 导出与本机推理速度

测试日期：2026-09-22。输入和输出均为固定 FP32 NCHW `[1,4,360,640]`，即宽 640、高 360 的四通道 packed RAW；对应未打包 Bayer 空间尺寸 1280×720，并非 1280×720×4。

## ONNX 导出

脚本：[tools/export_sid_onnx.py](../tools/export_sid_onnx.py)。通过统一 checkpoint factory 加载模型：小波模型及 MRLFN 优先读取 model_deploy，必要时融合训练态分支；NAFNet、UNet 使用原生 eval 网络。opset=17，固定 batch=1 和空间尺寸，开启常量折叠，不含 RAW 文件读取、噪声合成、亮度校正或 ISP。

导出环境：PyTorch 2.1.2，ONNX 1.16.2；导出时使用 ONNX Runtime 1.19.2 CPU 进行检查。各模型目录 onnx/ 内同名 JSON 保存导出版本、模型元信息、SHA256 和数值验证结果。

```bash
python tools/export_sid_onnx.py \
  experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb/checkpoints/latest.pth \
  experiments/sid_sony_mrlfn_paper_s2d_k4_n4_d32/checkpoints/latest.pth \
  experiments/sid_sony_nafnet_new/checkpoints/latest.pth \
  experiments/sid_sony_paper_fair/checkpoints/latest.pth
```

## 测试环境与计时口径

- CPU：双路 Intel Xeon Gold 6530，合计 64 物理核 / 128 逻辑 CPU。每个 ORT session 仅用 intra_op=4、inter_op=1，未绑定 CPU/NUMA；不是全部 CPU 核心吞吐测试。
- GPU：GPU 0，NVIDIA GeForce RTX 4090 24 GB；驱动 575.64.05。PyTorch CUDA 11.8、cuDNN 8.7。
- 统一使用 onnxruntime-gpu 1.18.0 测 CPU/CUDA，ORT_ENABLE_ALL；CUDA TF32=0，cuDNN 算法搜索 HEURISTIC，不使用 TensorRT、FP16、量化或 CUDA Graph。
- CPU：预热 5 次，串行计时 20 次 session.run。GPU：每种方式预热 10 次，串行计时 50 次。随机 FP32 输入固定种子 2026，初始化、模型加载、随机数据生成和预热不计时。
- GPU 显存 I/O：输入、输出 OrtValue 预分配在 CUDA 显存，通过 I/O binding 执行。每次计时前后 torch.cuda.synchronize，包含 Python/ORT 调度及同步开销，不含主机输入/输出拷贝；并非 CUDA kernel-only 时间。
- GPU 主机 I/O：NumPy 输入 session.run，返回 NumPy 输出，包括 H2D/D2H 及输出分配。CUDA provider 优先，允许 ORT 对不支持的节点回退到 CPU；不宣称整个图所有节点均在 GPU 上。
- FPS = 1000 / 平均毫秒，表示 batch=1 串行推理吞吐，不是异步流水线峰值。

## 实测结果

| 模型 | CPU 平均 ms | CPU FPS | GPU 显存 I/O 平均 ms | GPU FPS | GPU 主机 I/O 平均 ms |
|---|---:|---:|---:|---:|---:|
| Haar + LL-RepNCB | 88.309 | 11.32 | 3.524 | 283.74 | 4.830 |
| MRLFN d32 | 67.785 | 14.75 | 4.697 | 212.90 | 2.708 |
| NAFNet | 1110.043 | 0.90 | 29.589 | 33.80 | 38.688 |
| UNet (paper_fair) | 342.867 | 2.92 | 6.092 | 164.14 | 7.653 |

| 模型 | CPU 中位 / P95 ms | GPU 显存 I/O 中位 / P95 ms | CUDA 与 CPU 最大绝对差 |
|---|---:|---:|---:|
| Haar + LL-RepNCB | 85.600 / 105.741 | 3.132 / 5.440 | 6.56e-07 |
| MRLFN d32 | 66.665 / 78.303 | 5.017 / 5.060 | 7.15e-07 |
| NAFNet | 1133.529 / 1213.716 | 26.430 / 39.982 | 8.34e-07 |
| UNet (paper_fair) | 338.480 / 378.257 | 7.591 / 7.691 | 6.56e-07 |

所有 CUDA 输出均通过与 CPU 输出的数值比较（atol=1e-4、rtol=1e-3）。

### 负载限制

测试前 GPU 0 已占用约 17 GB 显存，GPU 1 正在高负载运行，未停止任何现有任务。本次是共享机器负载下的一轮测试，存在调度、频率和其他进程干扰。MRLFN 主机 I/O 比显存 I/O 更快，是两段独立计时期间状态变化的观测结果，不能解释为加入拷贝能提高性能。不要用这次结果推断独占 GPU 的精确性能排序；空闲时可重复同一脚本复测。

## 文件与复现

测速脚本：[tools/benchmark_sid_onnx.py](../tools/benchmark_sid_onnx.py)。原始统计及前后 nvidia-smi 快照：[onnx_benchmark_results.json](onnx_benchmark_results.json)。

2026-09-23 环境恢复：当前默认 Python 为 `/opt/conda/bin/python`（Python 3.10.8，PyTorch 2.1.2 + CUDA 11.8 / cuDNN 8.7）。依赖已改为安装到当前 Python，不再依赖 `/tmp/sid-ort-gpu` 临时目录。GPU 版 ONNX Runtime 同时支持 CPU 和 CUDA，无需同时安装 CPU 版。

```bash
python -m pip install onnx==1.16.2 onnxruntime-gpu==1.18.0 'numpy<2'
python tools/export_sid_onnx.py --help
python tools/benchmark_sid_onnx.py
```

如果容器重建或切换 Python/conda 环境，需要在对应解释器下重新安装；用 `which python` 核对。早期统计使用的版本仍以上面的历史记录为准。

导出参数必须是 checkpoint 文件，不是实验目录。例如：

```bash
python tools/export_sid_onnx.py \
  experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5/checkpoints/latest.pth
```

测速脚本先 import torch 加载 CUDA/cuDNN 依赖；如果 CUDA provider 未能启用会报错，不会将静默回退的 CPU 结果标为 GPU。

ONNX 文件：

- [sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb](../experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb/onnx/learning_dwt_repncb_latest_deploy_1x4x360x640.onnx)
- [sid_sony_mrlfn_paper_s2d_k4_n4_d32](../experiments/sid_sony_mrlfn_paper_s2d_k4_n4_d32/onnx/mrlfn_latest_deploy_1x4x360x640.onnx)
- [sid_sony_nafnet_new](../experiments/sid_sony_nafnet_new/onnx/nafnet_latest_deploy_1x4x360x640.onnx)
- [sid_sony_paper_fair](../experiments/sid_sony_paper_fair/onnx/unet_latest_deploy_1x4x360x640.onnx)

## 2026-09-23：static_hf_depth5 导出验证偏差

`Parity failed` 出现在 ONNX 结构检查成功后的 ORT 数值验证，而非导出阶段。固定尺寸检查产生的 TracerWarning 不是这次异常的原因。

对同一个 ONNX、同一个带负值输入：新 session 单独执行可通过；session 先执行正值输入后，再执行 signed_raw，在 ORT 1.18.0 CPU 的 ORT_ENABLE_ALL 下最大差约 0.001463、平均差 1.46e-5；ORT_DISABLE_ALL、BASIC、EXTENDED 最大差约 1.02e-7。当前证据定位到 ALL 级运行时优化相关的跨调用偏差，尚未确定具体优化 pass，不能认定为普通浮点舍入误差。

导出脚本新增 `--ort-opt-level disabled|basic|extended|all`，默认 extended。未放宽比较容差，也未修改权重。重新导出后 uniform/signed_raw/zeros 均通过，报告记录 `validation_ort_opt_level`。

**该设置属于运行时 session，不会嵌入 ONNX 文件。** 使用 ORT 1.18.0 CPU 部署此 static_hf_depth5 模型时，应同步设置：

```python
options = onnxruntime.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
session = onnxruntime.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
```

旧测速脚本仍显式使用 ALL，不能据此保证新模型的多帧正确性；测试该新模型时应设置相同的 EXTENDED，再重新计时。更换 ORT 版本、GPU 或其他推理框架后，应测试连续不同输入，不只是单次前向。
