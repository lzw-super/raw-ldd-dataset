# 2K输入：Haar A/C、SplitterNet、U-Net、NAFNet W8A8 NPU补测

日期：2026-10-10。五个模型均为本次重新导出、编译和实测；每项3轮×100次，共1500个样本。本次未训练；部署速度为新测，SID效果另汇总用户指定的历史评测记录（见第5节）。

## 1. 输入和测试条件

- 本次2K按2560×1440 Bayer RAW定义，packed后为HWC `720×1280×4`，NCHW **`[1,4,720,1280]`**，batch=1。720与1280均整除16，遵照用户确认不再补到736。
- 设备Samsung Galaxy S24；运行日志核对SM-S921U1 / SM8650 / Hexagon V75，QAIRT 2.50.0.260828221209，qai-hub 0.56.0。指定同型号NPU，不保证同一台物理手机。
- QNN DLC，权重和激活均8bit，IO量化。平台随机校准，仅测速，不代表量化画质验证。
- 汇总全部300次的均值、P50、P95，不剔除较慢样本；不含上传、云端排队、编译、应用端补边/裁剪、ISP时间。模型内部必要操作包含在内。
- 所有运行条目均为NPU，已核对编译产物输入/输出shape和W8A8编译日志。

## 2. 模型定义与导出

Haar A与C使用同一已训练Soft阈值、LL无归一化checkpoint，阈值在部署前离线固化。收缩均为 `ReLU(z-t)-ReLU(-z-t)`，只是调度布局不同：

- **Haar A**：三级、每级三个高频子带分别按4通道处理，共9组收缩。
- **Haar C**：每级三个高频子带作为连续12通道一起处理，共3组收缩；每个子带、通道的阈值仍各自独立，不是共享阈值。

两者不是Dual-DW3，也没有在高频部分新增CNN。小波/反小波及S2D/D2S采用固定卷积实现，Rep-NCB已融合；Haar导出图无Div/Abs/Sign。两种表达各自通过与原checkpoint的六类输入FP32核对，构成同函数的部署表达对照。

SplitterNet、U-Net、NAFNet采用已有训练权重，复制独立快照并记录SHA256；未改变网络宽度、深度或重新初始化。导出脚本仅补充允许NAFNet通过通用模型工厂导出。

**U-Net内部补边**：`UNetSeeInDark.check_img_size`使用 `16-size%16`，即整除16时仍补16。因此本次内部特征输入为 **736×1296**，最后裁回720×1280。clean导出可将Pad折入卷积参数，但不会消除额外计算区域，不能由图中无独立Pad推断没有补边。这是原模型行为，本次未修改。NAFNet按其层数要求补边，当前四级编码器要求16倍数，本输入无需额外对齐。

| 模型 | clean ONNX节点 | 部署参数量 | 六类FP32检查最大绝对误差 |
|---|---:|---:|---:|
| Haar A：逐子带＋双ReLU | 94 | 86,408 | 1.67e-06 |
| Haar C：12通道合并＋双ReLU | 58 | 86,408 | 1.67e-06 |
| SplitterNet clean | 487 | 731,636 | 9.54e-07 |
| U-Net | 50 | 7,760,484 | 2.26e-06 |
| NAFNet | 668 | 7,600,228 | 2.64e-05 |

六类输入包括uniform、signed_raw、zeros、tiny、border_ramp和corner_impulses；全部通过。上述是FP32导出一致性，不是INT8效果评估。

## 3. 2K NPU实测

| 模型 | 均值 / ms | P50 / ms | P95 / ms | 第一轮均值 | 第二轮均值 | 第三轮均值 |
|---|---:|---:|---:|---:|---:|---:|
| Haar A：逐子带＋双ReLU | **5.4560** | 5.3970 | 5.7028 | 5.4754 | 5.4110 | 5.4816 |
| Haar C：12通道合并＋双ReLU | **3.4750** | 3.4230 | 3.7090 | 3.4785 | 3.4741 | 3.4725 |
| SplitterNet clean | **8.0197** | 7.9095 | 8.7311 | 8.0121 | 8.0150 | 8.0322 |
| U-Net | **13.9222** | 13.8730 | 14.5254 | 13.7857 | 14.0781 | 13.9028 |
| NAFNet | **182.4021** | 182.3770 | 184.2768 | 181.2058 | 182.3106 | 183.6898 |

本次Haar C相对Haar A节省 **1.9810 ms**，延迟降低 **36.31%**（负值表示更慢），速度倍率为 **1.570×**。布局变化会影响编译、融合及特征遍历；本次数据直接证明的是两个部署产物的耗时差，不单独证明某一种访存或缓存机制。

NAFNet本次三轮均值为181.2058、182.3106、183.6898 ms，较高延迟在三轮均出现。已核对各轮执行条目均为NPU，不是CPU回退造成；这个结果针对当前完整NAFNet、ONNX表达和QNN编译产物，不能推断其他融合或定制部署方式也会同样慢。

## 4. 与此前同shape三模型对照

以下沿用前一次补测，未在本次重复测速；输入、设备型号、量化和统计口径一致，物理设备和测试批次可能不同。

| 模型 | 同shape均值 / ms |
|---|---:|
| Rep-NCB3×3＋PReLU＋输入残差 | 3.4471 |
| DW3×3＋ReLU＋输入残差 | 4.2237 |
| MRLFN | 4.8669 |

以上是部署产物速度对比，不能据此推断去噪精度优劣；此前单DW3为未训练HF替换测速模块，其他模型权重来源见各报告。小幅差距需结合逐轮波动看待。

### 4.1 Rep-NCB3×3＋PReLU＋输入残差：参数量与计算量

统计对象是前表中 **3.4471 ms** 对应的完整两阶段去噪模型，包含Haar分解/重建、三级高频CNN、LL恢复和精修网络，不只是高频Rep-NCB模块。输入固定为 **`[1,4,720,1280]`**，batch=1；使用实际测速的clean ONNX逐节点统计，权重来源与SHA256见[统计原始记录](qaihub_2k_five_extra_20261010/repncb_2k_complexity.json)。

| 指标 | 数值 |
|---|---:|
| 部署态参数（Rep-NCB融合后） | **90,368（0.090368 M）** |
| 部署卷积计算量，包含固定变换卷积 | **14.0136 GMAC** |
| 按1 MAC＝2 FLOPs换算的卷积计算量 | **28.0272 GFLOPs** |
| 加上bias、激活及逐元素加法的总算术估算 | **28.1723 GOp** |

参数量包含卷积权重、bias和PReLU参数，不包含Haar、S2D/D2S固定核等buffer，也不包含优化器状态。固定核虽不是可学习参数，其卷积计算仍计入计算量。本节仅展示融合后的部署态参数与计算量，不统计训练分支或反向传播。

| 模块 | 部署态参数 | 部署GMAC | 总算术GOp |
|---|---:|---:|---:|
| 三级高频Rep-NCB＋PReLU＋输入残差 | 3,960 | 0.391910 | 0.798336 |
| LL3恢复CNN及concat融合 | 30,216 | 0.431770 | 0.868723 |
| 精修CNN（不含固定S2D/D2S） | 56,192 | 12.858163 | 25.841664 |
| 三级Haar分解与逆变换 | 0 | 0.154829 | 0.309658 |
| 两次S2D及一次D2S | 0 | 0.176947 | 0.353894 |
| **合计** | **90,368** | **14.013619** | **28.172275** |

**实际结构核对：**

- 高频：每级12通道，三级输入空间尺寸依次为360×640、180×320、90×160；各含一个融合后的普通12→12、3×3卷积、PReLU及输入相加。这里的Rep-NCB部署卷积是dense卷积，**不是Depthwise卷积**。
- LL3：输入4×90×160，4→32 stem＋ReLU，3个32→32 Rep-NCB＋PReLU，32→4输出卷积；CNN输出与原LL拼接，经8→4的1×1卷积融合，不进行LL归一化。
- 精修：S2D k=2，stem 16→32，**5个**32→32 Rep-NCB＋PReLU，head 32→16，与S2D后的原输入拼接为32通道，再经1×1卷积32→16和D2S输出。块数按该训练权重和实际导出图核对，不沿用早期4块版本。

精修CNN占整网卷积MAC约 **91.75%**，高频CNN约 **2.80%**。这是算术量占比，不等于NPU耗时占比。

**计数口径：**Conv按输出元素数×每个输出的核元素数；ConvTranspose按输入元素数×每个输入投射的核元素数。1 MAC计2次算术操作；每个bias加法计1 Op，PReLU按2 Op/元素，ReLU和残差Add按1 Op/元素。Split/Concat/Slice按0算术Op，仍可能产生实际访存和调度开销。

固定Haar与S2D/D2S按导出的稠密卷积形状计数，包括核中的零项和单位项，不假设NPU能将其省掉。这是可复现的图级算法量，不是芯片实际指令数。28.0272 GFLOPs仅是卷积MAC的常用换算；28.1723 GOp还含比较/激活等标量操作。W8A8执行时应理解为整数算术规模，而非实际浮点运算数量。

计算量复现（无需NPU或重新训练）：

```bash
python - <<'PY'
from tools.profile_hf_cnn_onnx import count
r = count('experiments/npu_resolution_sweep_20261010/onnx/haar_hf_repncb_prelu_residual_sweep_2k.onnx')
print('GMAC:', r['conv_macs'] / 1e9)
print('Conv GFLOPs (2/MAC):', 2 * r['conv_macs'] / 1e9)
print('Total GOp:', r['arithmetic_ops'] / 1e9)
PY
```

### 4.2 全部模型的部署态参数量与计算量

统一外部输入 **`[1,4,720,1280]`**、batch=1。参数来自严格加载测速权重后的部署模型；计算量来自本报告各模型及前次MRLFN/Rep-NCB测速实际使用的 **clean ONNX**，已核对文件SHA256。全部为部署态，不使用训练分支参数量。

| 模型 | 部署网络参数 | 冻结学习阈值 | 部署学习系数合计 | GMAC | 卷积GFLOPs（2/MAC） | 总算术GOp |
|---|---:|---:|---:|---:|---:|---:|
| Rep-NCB3×3＋PReLU＋输入残差 | 90,368 | 0 | **90,368** | **14.0136** | 28.0272 | 28.1723 |
| MRLFN | 130,672 | 0 | **130,672** | **29.9631** | 59.9261 | 60.2173 |
| Haar A：逐子带＋双ReLU | 86,408 | 36 | **86,444** | **13.6217** | 27.2434 | 27.3957 |
| Haar C：12通道合并＋双ReLU | 86,408 | 36 | **86,444** | **13.6217** | 27.2434 | 27.3957 |
| SplitterNet | 731,636 | 0 | **731,636** | **15.1300** | 30.2601 | 30.6255 |
| U-Net | 7,760,484 | 0 | **7,760,484** | **176.0589** | 352.1179 | 352.9200 |
| NAFNet | 7,600,228 | 0 | **7,600,228** | **123.1930** | 246.3860 | 253.2715 |

**参数量说明：**

- “部署网络参数”包含融合后卷积权重、bias、PReLU、NAFNet的归一化仿射系数及残差缩放系数等；固定Haar、S2D/D2S核和shape/index常量不计为学习参数。MRLFN与Rep-NCB均已融合，不再计入训练态多分支。
- Haar A/C的网络参数均为86,408，另有36个由训练得到、部署前固化为buffer的阈值，因此部署需保留的学习系数合计为 **86,444**。正负阈值常量的复制不作为新的独立学习系数；两种布局共享同一checkpoint。
- 参数量是模型结构中的标量系数数量，不等于ONNX全部initializer元素数或DLC文件字节数。常量折叠、相同常量合并和量化scale/zero-point会改变存储，W8A8也不意味着bias等所有数据都占1字节。

**计算量说明：**

- MAC与固定卷积计数沿用4.1节；包括原图实际内部尺寸，故U-Net按736×1296内部补边后的各层shape统计，输出仍为720×1280。NAFNet本输入无需额外对齐补边。
- 总GOp额外计入bias、激活、残差、归约、池化及NAFNet归一化展开的逐元素算术。PReLU/LeakyReLU按2 Op/元素，Sigmoid按4，Pow/Sqrt按1；ReduceMean/GlobalAveragePool按输入元素数，ReduceMax按输入减输出元素数，MaxPool按输出元素数×(核面积−1)估算。
- Split/Concat/Pad/Reshape/Transpose/DepthToSpace等数据组织操作按0算术Op，不代表其NPU执行免费；量化/反量化、编译器实际融合和硬件指令不在FP32源图算法量中。所有G单位均为10⁹。

Haar A/C的理论参数量和算术量相同，但实测耗时为5.4560与3.4750 ms，说明此处优化的是执行组织而不是数学运算数量。NAFNet的卷积MAC低于U-Net，当前NPU耗时却明显更高；其部署图包含额外归一化等操作，单用MAC不能解释耗时，也不能据此将全部差距归因于某一个算子。

原始逐算子统计及参数核对：[deployment_complexity.json](qaihub_2k_five_extra_20261010/deployment_complexity.json)。统计工具遇到未支持的算子会报错，不将其默认为0。

复现全部部署态统计（无需训练或调用NPU平台）：

```bash
python tools/profile_sid_deploy_onnx.py \
  --export-reports \
    ref-doc/qaihub_2k_five_extra_20261010/export_reports.json \
    ref-doc/qaihub_resolution_sweep_20261010/2k/export_reports.json \
  --models \
    haar_soft_band_relu_2k_extra haar_soft_level_relu_2k_extra \
    splitternet_2k_extra unet_2k_extra nafnet_2k_extra \
    mrlfn_sweep_2k haar_hf_repncb_prelu_residual_sweep_2k \
  --output ref-doc/qaihub_2k_five_extra_20261010/deployment_complexity.json
```

## 5. SID测试集效果指标：三档及平均值

本节汇总用户指定实验目录中已有的SID测试JSON，不重新运行评测。PSNR单位为dB，SSIM无单位；每档列为 **PSNR / SSIM**。平均值分别对×100、×250、×300三个已记录指标取等权算术平均，使用原始完整精度计算后再显示。JSON未提供逐图结果或各档样本数，因此这不是按样本数加权的全测试集指标，也不是先平均MSE再换算PSNR。

| 模型 | ×100：PSNR / SSIM | ×250：PSNR / SSIM | ×300：PSNR / SSIM | 三档平均：PSNR / SSIM |
|---|---:|---:|---:|---:|
| Rep-NCB3×3＋PReLU＋输入残差 | 42.3603 / 0.955061 | 40.2856 / 0.938663 | 36.8173 / 0.918790 | **39.8211 / 0.937504** |
| Haar A / Haar C（同一原始模型） | 42.2033 / 0.954607 | 40.1790 / 0.938181 | 36.6985 / 0.917435 | **39.6936 / 0.936741** |
| MRLFN | 42.0722 / 0.952986 | 39.8432 / 0.928255 | 36.3171 / 0.904491 | **39.4108 / 0.928578** |
| SplitterNet | 42.1595 / 0.954074 | 39.9656 / 0.929100 | 36.4395 / 0.904354 | **39.5215 / 0.929176** |
| U-Net | 43.5173 / 0.960072 | 41.2084 / 0.943617 | 37.7399 / 0.925654 | **40.8219 / 0.943114** |
| NAFNet | 43.8630 / 0.960288 | 41.4557 / 0.944946 | 37.8631 / 0.924257 | **41.0606 / 0.943164** |

### 5.1 指标与测速结果的对应范围

- 所有记录均为 `testset_type=sid`，档位分别为100/250/300，`max_samples=null`；同模型三档记录的checkpoint路径和模型配置一致。
- Haar A/C是同一原始Soft阈值模型的两种部署表达，故此处**共用一组原始模型效果记录**，不声称分别测得两份W8A8效果。两者的量化数值仍可能因编译图不同而不同。

PSNR和SSIM的数值均直接取JSON顶层字段。原始汇总、完整精度均值与18个源文件的SHA256保存在 [sid_quality_metrics.json](qaihub_2k_five_extra_20261010/sid_quality_metrics.json)。

### 5.2 原始记录来源

| 模型 | 权重记录 | ×100 | ×250 | ×300 |
|---|---|---|---|---|
| Rep-NCB3×3＋PReLU＋输入残差 | `best.pth` | [b_sid_x100.json](../experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth5_repncb_prelu_residual_ll_no_norm_each5/b_sid_x100.json) | [b_sid_x250.json](../experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth5_repncb_prelu_residual_ll_no_norm_each5/b_sid_x250.json) | [b_sid_x300.json](../experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth5_repncb_prelu_residual_ll_no_norm_each5/b_sid_x300.json) |
| Haar A / Haar C（同一原始模型） | `best.pth` | [b_sid_x100.json](../experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm/b_sid_x100.json) | [b_sid_x250.json](../experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm/b_sid_x250.json) | [b_sid_x300.json](../experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm/b_sid_x300.json) |
| MRLFN | `best.pth` | [sid_x100.json](../experiments/sid_sony_mrlfn_paper_s2d_k4_n4_d32/sid_x100.json) | [sid_x250.json](../experiments/sid_sony_mrlfn_paper_s2d_k4_n4_d32/sid_x250.json) | [sid_x300.json](../experiments/sid_sony_mrlfn_paper_s2d_k4_n4_d32/sid_x300.json) |
| SplitterNet | `latest.pth` | [latest_sid_x100.json](../experiments/sid_sony_splitternet/latest_sid_x100.json) | [latest_sid_x250.json](../experiments/sid_sony_splitternet/latest_sid_x250.json) | [latest_sid_x300.json](../experiments/sid_sony_splitternet/latest_sid_x300.json) |
| U-Net | `latest.pth` | [latest_sid_x100.json](../experiments/sid_sony_paper_new/latest_sid_x100.json) | [latest_sid_x250.json](../experiments/sid_sony_paper_new/latest_sid_x250.json) | [latest_sid_x300.json](../experiments/sid_sony_paper_new/latest_sid_x300.json) |
| NAFNet | `latest.pth` | [latest_sid_x100.json](../experiments/sid_sony_nafnet_new/latest_sid_x100.json) | [latest_sid_x250.json](../experiments/sid_sony_nafnet_new/latest_sid_x250.json) | [latest_sid_x300.json](../experiments/sid_sony_nafnet_new/latest_sid_x300.json) |

复算三档平均值时，对每个模型三个JSON的 `psnr` 与 `ssim` 分别求和除以3；不要对表中已经四舍五入的值再次计算。

## 6. 复现与记录

五份配置位于 `configs/deploy/*_2k_extra.yaml`，ONNX位于 `experiments/npu_2k_five_extra/onnx/`，权重快照位于 `experiments/npu_2k_five_extra/inputs/`。

```bash
PYTHONPATH=/tmp/sid-onnxsim python tools/export_sid_npu_ablation.py \
  configs/deploy/haar_soft_band_relu_2k_extra.yaml \
  configs/deploy/haar_soft_level_relu_2k_extra.yaml \
  configs/deploy/splitternet_2k_extra.yaml \
  configs/deploy/unet_2k_extra.yaml configs/deploy/nafnet_2k_extra.yaml \
  --output-dir experiments/npu_2k_five_extra/onnx

read -r -s -p "AI Hub API token: " QAI_HUB_API_TOKEN
export QAI_HUB_API_TOKEN
PYTHONPATH=/tmp/sid-qai-hub python tools/benchmark_sid_qaihub.py \
  --watch --height 720 --width 1280 \
  --sources-json ref-doc/qaihub_2k_five_extra_20261010/sources.json \
  --output-dir ref-doc/qaihub_2k_five_extra_20261010
unset QAI_HUB_API_TOKEN
python tools/summarize_sid_qaihub.py \
  --output-dir ref-doc/qaihub_2k_five_extra_20261010
```

`/tmp`依赖目录为本机环境：PyTorch 2.1.2、ONNX 1.16.2、ORT 1.18、onnxsim 0.4.36，其他机器需安装对应依赖。重新导出后应更新sources.json中的source_sha256；新一轮独立测量应使用新结果目录，已有目录会恢复历史任务。

| 模型 | 编译任务 | 三轮测速任务 |
|---|---|---|
| Haar A：逐子带＋双ReLU | [jpv21mxmg](https://workbench.aihub.qualcomm.com/jobs/jpv21mxmg) | [j5wydr66g](https://workbench.aihub.qualcomm.com/jobs/j5wydr66g) / [jp1odmz25](https://workbench.aihub.qualcomm.com/jobs/jp1odmz25) / [jgd6rm1ep](https://workbench.aihub.qualcomm.com/jobs/jgd6rm1ep) |
| Haar C：12通道合并＋双ReLU | [jgj30y48p](https://workbench.aihub.qualcomm.com/jobs/jgj30y48p) | [jp4ex2rvg](https://workbench.aihub.qualcomm.com/jobs/jp4ex2rvg) / [jpx07zo1p](https://workbench.aihub.qualcomm.com/jobs/jpx07zo1p) / [j5m9wlxwg](https://workbench.aihub.qualcomm.com/jobs/j5m9wlxwg) |
| SplitterNet clean | [jpe6rx30g](https://workbench.aihub.qualcomm.com/jobs/jpe6rx30g) | [jp0o1x06p](https://workbench.aihub.qualcomm.com/jobs/jp0o1x06p) / [jp8j3kyx5](https://workbench.aihub.qualcomm.com/jobs/jp8j3kyx5) / [jgk6lkx25](https://workbench.aihub.qualcomm.com/jobs/jgk6lkx25) |
| U-Net | [jgzzxyk6g](https://workbench.aihub.qualcomm.com/jobs/jgzzxyk6g) | [jgn19wvrp](https://workbench.aihub.qualcomm.com/jobs/jgn19wvrp) / [jprx4739p](https://workbench.aihub.qualcomm.com/jobs/jprx4739p) / [jp2o7zy4g](https://workbench.aihub.qualcomm.com/jobs/jp2o7zy4g) |
| NAFNet | [jg9o32evg](https://workbench.aihub.qualcomm.com/jobs/jg9o32evg) | [jglw0qm8p](https://workbench.aihub.qualcomm.com/jobs/jglw0qm8p) / [j56o30405](https://workbench.aihub.qualcomm.com/jobs/j56o30405) / [jp3o4r0lp](https://workbench.aihub.qualcomm.com/jobs/jp3o4r0lp) |

原始证据：[summary.json](qaihub_2k_five_extra_20261010/summary.json)、[jobs.json](qaihub_2k_five_extra_20261010/jobs.json)、[sources.json](qaihub_2k_five_extra_20261010/sources.json)、[export_reports.json](qaihub_2k_five_extra_20261010/export_reports.json)、[checkpoint_provenance.json](qaihub_2k_five_extra_20261010/checkpoint_provenance.json)、[compiled_specs.json](qaihub_2k_five_extra_20261010/compiled_specs.json)、[execution_audit.json](qaihub_2k_five_extra_20261010/execution_audit.json)。各模型子目录包含DLC、profile JSON及编译/设备运行日志。

关联报告：[三模型720p/1080p/2K补测](三模型720p_1080p_2K_W8A8_NPU补测.md)。
