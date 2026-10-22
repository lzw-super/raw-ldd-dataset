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
| Rep-NCB3×3＋PReLU＋输入残差（完整LL） | 3.4471 | **3.4584** | +0.33% | 3.3715 | 3.8512 | 3.3980 / 3.4132 / 3.5639 |
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

## 7. LL bypass两组消融：2K部署参数、计算量与NPU实测

日期：2026-10-10。本节只补充速度和部署复杂度，**不新增效果指标**。输入为 `[1,4,720,1280]`，batch=1；仍为Galaxy S24 / SM-S921U1 / SM8650 / Hexagon V75，QAIRT 2.50.0.260828221209、W8A8目标量化、量化IO。平台随机校准，每组3轮×100次；均值/P50/P95使用全部300个样本，不含外部预处理、上传和排队时间。所有profile执行条目均为NPU。

### 7.1 权重与结构

| 实验 | 训练配置 | 本次best权重epoch |
|---|---|---:|
| LL bypass＋Rep-NCB高频CNN | [train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth5_repncb_prelu_residual_ll_no_norm_ablation_ll_bypass.yaml](../configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth5_repncb_prelu_residual_ll_no_norm_ablation_ll_bypass.yaml) | 600 |
| LL bypass＋BayesShrink Soft | [train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth5_repncb_prelu_residual_ll_no_norm_ablation_ll_bypass_bayes_soft.yaml](../configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_hf_cnn_depth5_repncb_prelu_residual_ll_no_norm_ablation_ll_bypass_bayes_soft.yaml) | 500 |

两组均复制独立best权重快照，避免后续训练覆盖测速来源；哈希见[checkpoint_provenance.json](qaihub_2k_ll_bypass_20261010/checkpoint_provenance.json)。

- **LL bypass**：三级Haar的最终LL直接通过，删除LL恢复CNN及concat1×1融合；三级高频仍为各自独立的12→12 Rep-NCB＋PReLU＋输入残差，精修仍为32通道、5块。
- **LL bypass＋BayesShrink Soft**：LL同样直接通过，高频改为无学习参数的BayesShrink；噪声方差由每图每RAW通道的HH1非零绝对值中位数估计，各子带依据自身二阶矩计算阈值并作Soft收缩。只有后续精修CNN含学习参数。阈值依赖输入，不能像静态band_channel阈值一样预先存表。

### 7.2 部署态复杂度

| 模型 | 部署参数量 | 卷积GMAC | 卷积GFLOPs（2/MAC） | 总算术GOp说明 |
|---|---:|---:|---:|---|
| LL bypass＋Rep-NCB高频CNN | **60,152** | **13.5818** | 27.1637 | 27.3036 |
| LL bypass＋BayesShrink Soft | **56,192** | **13.1899** | 26.3799 | 可计数部分26.5479；另含TopK选择成本，完整总量未定价 |

本节参数全部为融合后的部署参数，不计训练多分支。固定Haar/S2D/D2S核不计为学习参数，但仍按导出的稠密卷积形状计入MAC，包含零核系数。BayesShrink没有可学习阈值参数，其方差、阈值、中位数都是输入相关计算。

相对第4.1节完整LL模型的90,368参数、14.0136 GMAC，LL bypass删除30,216个LL参数和0.4317696 GMAC；BayesShrink再删除3,960个高频CNN参数和0.3919104 GMAC，但新增统计与中位数选择，不能据此直接推断其更快。

BayesShrink的TopK输入为 `[1,4,230400]`，每通道取最大的115201个值，以支持任意非零数量的两个中位数秩。TopK的比较/选择成本依赖后端算法，因此JSON将完整 `arithmetic_ops` 记为null，单列 `known_arithmetic_ops` 和 `unpriced_selection`，没有把这部分误计为0。可计数GOp包含卷积、bias、逐元素运算、均值/最大值/求和归约、比较与Where等；Cast和Gather只记录数据操作，不估算硬件转换或搬运成本。

逐算子及参数记录：[deployment_complexity.json](qaihub_2k_ll_bypass_20261010/deployment_complexity.json)。

### 7.3 NPU实测结果

| 模型 | 均值 / ms | P50 / ms | P95 / ms | 第1轮均值 | 第2轮均值 | 第3轮均值 |
|---|---:|---:|---:|---:|---:|---:|
| LL bypass＋Rep-NCB高频CNN | **4.3075** | 4.2000 | 4.8310 | 4.3468 | 4.2916 | 4.2841 |
| LL bypass＋BayesShrink Soft | **158.8691** | 158.7795 | 160.6080 | 158.9218 | 158.7670 | 158.9185 |

此前完整LL模型的同尺寸均值为 **3.4471 ms**，是前次测量且使用不同实验权重，并未在本次重测。下面百分比以该值为基准，正值为更慢：

- LL bypass＋Rep-NCB高频CNN：+24.96%。
- LL bypass＋BayesShrink Soft：+4508.74%。

BayesShrink的直接瓶颈证据：三轮TopK占各自已记录算子周期之和的比例为 75.80% / 75.46% / 75.51%。这是profile算子周期占比，不是端到端墙钟时间占比；该结果对应保留非零中位数语义的TopK导出实现，不应推广为所有BayesShrink实现的速度。所有执行条目均标为NPU，没有观察到CPU/GPU回退。

参数和MAC减少不一定带来NPU提速。此处比较的是实际编译产物，图融合、布局及分块选择都可能随结构改变；未单独测量这些因素，不能把耗时差全部归因于某一个被删除的模块。

### 7.4 BayesShrink导出适配与核对

原生 `torch.quantile` 无法导出到ONNX opset17，错误记录见 [bayes_native_export_failure.log](qaihub_2k_ll_bypass_20261010/bayes_native_export_failure.log)。新增仅用于固定shape导出的 [HaarBayesExport](../models/haar_bayes_export.py)：将非零中位数写为TopK＋非零计数＋两个秩的Gather均值，保留零噪声旁路及每子带Bayes阈值计算；没有冻结阈值、没有替换成方差近似，也没有改变训练实现。

设非零数为m：在按绝对值降序排列的序列中取 `floor((m-1)/2)` 与 `floor(m/2)` 两个位置求平均；m=0时取零。两秩均不超过N/2，故只需前N/2+1个值。这个表达与原非零中位数数学等价。ONNX统计表达保留原FP64运算；QNN转换日志说明Cast在转换时解释，因此不声称NPU保持FP64执行，也不声称统计/索引算子全部以8bit执行。W8A8是编译的权重/激活量化目标，不是所有标量指令的位宽。

新增7项测试核对全零、单元素、奇偶非零数量、正负数，以及ONNX在不同零掩码输入下重新计算中位数。两个完整模型各自六类输入的PyTorch/ORT核对均通过：

- LL bypass＋Rep-NCB高频CNN：节点数52→44，最大绝对误差2.26e-06。
- LL bypass＋BayesShrink Soft：节点数263→218，最大绝对误差1.67e-06。

这些只验证导出表达的一致性，不是W8A8效果评估。本次按要求未新增PSNR/SSIM记录。

### 7.5 复现与证据

```bash
PYTHONPATH=/tmp/sid-onnxsim python tools/export_sid_npu_ablation.py \
  configs/deploy/haar_ll_bypass_2k.yaml \
  configs/deploy/haar_ll_bypass_bayes_soft_2k.yaml \
  --output-dir experiments/npu_2k_ll_bypass/onnx

python tools/profile_sid_deploy_onnx.py \
  --export-reports \
    ref-doc/qaihub_2k_ll_bypass_20261010/ll_bypass/export_reports.json \
    ref-doc/qaihub_2k_ll_bypass_20261010/bayes_soft/export_reports.json \
  --models haar_ll_bypass_2k haar_ll_bypass_bayes_soft_2k \
  --allow-selection \
  --output ref-doc/qaihub_2k_ll_bypass_20261010/deployment_complexity.json

PYTHONPATH=. python -m pytest -q tests/test_haar_bayes_export.py

read -r -s -p "AI Hub API token: " QAI_HUB_API_TOKEN
export QAI_HUB_API_TOKEN
for variant in ll_bypass bayes_soft; do
  PYTHONPATH=/tmp/sid-qai-hub python tools/benchmark_sid_qaihub.py \
    --watch --height 720 --width 1280 \
    --sources-json "ref-doc/qaihub_2k_ll_bypass_20261010/$variant/sources.json" \
    --output-dir "ref-doc/qaihub_2k_ll_bypass_20261010/$variant"
done
unset QAI_HUB_API_TOKEN
```

`/tmp`依赖目录及已有任务恢复规则同第6节；重导出后须更新sources.json的SHA256，新独立测速使用新结果目录。

| 模型 | 编译任务 | 三轮测速任务 |
|---|---|---|
| LL bypass＋Rep-NCB高频CNN | [jpe62jo1g](https://workbench.aihub.qualcomm.com/jobs/jpe62jo1g) | [jpx0mw61p](https://workbench.aihub.qualcomm.com/jobs/jpx0mw61p) / [jgn1xjmrp](https://workbench.aihub.qualcomm.com/jobs/jgn1xjmrp) / [jp2oj294g](https://workbench.aihub.qualcomm.com/jobs/jp2oj294g) |
| LL bypass＋BayesShrink Soft | [j5m94j6wg](https://workbench.aihub.qualcomm.com/jobs/j5m94j6wg) | [jpv24jwkg](https://workbench.aihub.qualcomm.com/jobs/jpv24jwkg) / [jpe62jvog](https://workbench.aihub.qualcomm.com/jobs/jpe62jvog) / [jgzzw17og](https://workbench.aihub.qualcomm.com/jobs/jgzzw17og) |

ll_bypass：[summary.json](qaihub_2k_ll_bypass_20261010/ll_bypass/summary.json)、[jobs.json](qaihub_2k_ll_bypass_20261010/ll_bypass/jobs.json)、[sources.json](qaihub_2k_ll_bypass_20261010/ll_bypass/sources.json)、[export_reports.json](qaihub_2k_ll_bypass_20261010/ll_bypass/export_reports.json)、[compiled_specs.json](qaihub_2k_ll_bypass_20261010/ll_bypass/compiled_specs.json)。

bayes_soft：[summary.json](qaihub_2k_ll_bypass_20261010/bayes_soft/summary.json)、[jobs.json](qaihub_2k_ll_bypass_20261010/bayes_soft/jobs.json)、[sources.json](qaihub_2k_ll_bypass_20261010/bayes_soft/sources.json)、[export_reports.json](qaihub_2k_ll_bypass_20261010/bayes_soft/export_reports.json)、[compiled_specs.json](qaihub_2k_ll_bypass_20261010/bayes_soft/compiled_specs.json)。

[执行核对记录](qaihub_2k_ll_bypass_20261010/execution_audit.json)。各模型目录保存DLC、profile和编译/运行日志。ONNX位于 `experiments/npu_2k_ll_bypass/onnx/`。

## 8. 八模型2K独立复测（七模型＋Rep-NCB补测）

提交日期：2026-10-10；完成日期：2026-10-11（北京时间）。为验证重复性，本次复用此前相同的W8A8编译产物，首批重新提交21个profile任务，再补测Rep-NCB完整LL模型3轮，共24个任务，每模型3轮×100次。输入为 `[1,4,720,1280]`；设备型号为Samsung Galaxy S24 / SM-S921U1 / SM8650，QAIRT 2.50.0.260828221209，NPU执行、量化IO。平台不保证同一台实体手机。没有重新校准或训练，也未新增效果指标。

DLC SHA256逐一与上次结果一致；首批2100次及补测300次、共2400次样本纳入统计，不剔除极值，均值/P50/P95来自逐次测量，包含模型内部操作，不含外部上传、排队或预处理。各轮执行条目均标为NPU。

| 模型 | 上次均值/ms | 本次均值/ms | 变化 | P50/ms | P95/ms | 三轮均值/ms |
|---|---:|---:|---:|---:|---:|---|
| Rep-NCB3×3＋PReLU＋输入残差（完整LL） | 3.4471 | **3.4584** | +0.33% | 3.3715 | 3.8512 | 3.3980 / 3.4132 / 3.5639 |
| DW3×3＋ReLU＋输入残差 | 4.2237 | **4.1760** | -1.13% | 4.0005 | 4.9098 | 4.2325 / 4.1833 / 4.1121 |
| MRLFN | 4.8669 | **4.8992** | +0.66% | 4.7605 | 5.5550 | 4.9162 / 4.9307 / 4.8508 |
| SplitterNet | 8.0197 | **8.0159** | -0.05% | 7.9030 | 8.6992 | 8.0134 / 7.9968 / 8.0375 |
| U-Net | 13.9222 | **13.8950** | -0.20% | 13.7960 | 14.6698 | 14.0142 / 14.0037 / 13.6672 |
| NAFNet | 182.4021 | **182.0083** | -0.22% | 181.8970 | 183.7070 | 181.1409 / 182.4620 / 182.4219 |
| LL bypass＋Rep-NCB高频CNN | 4.3075 | **4.3347** | +0.63% | 4.2425 | 4.7930 | 4.3369 / 4.2924 / 4.3749 |
| LL bypass＋Bayes Soft | 158.8691 | **160.4279** | +0.98% | 159.6420 | 164.4140 | 158.9978 / 159.1098 / 163.1761 |

变化为相对上次均值，正值表示更慢。参数量与计算量沿用前文：本次DLC一致，因此不重复计数。短时延模型的固定开销、设备负载及频率状态可能导致相对较大的波动；这些因素没有被单独控制或测量，因此不将轮次差异直接归因于网络结构变化。Bayes Soft仍包含精确非零中位数的TopK导出路径，不能只按卷积MAC预测耗时。

本次八个模型的均值相对上次变化为−1.13%～+0.98%，速度排序保持一致，支持此前结果的可重复性。Bayes Soft第三轮为163.1761 ms，高于前两轮约159 ms，仍完整计入统计；不能根据本次未测量的温度、负载等信息断定其原因。

### 8.1 复测任务与证据

| 模型 | 三轮新测速任务 |
|---|---|
| DW3×3＋ReLU＋输入残差 | [j56ovllv5](https://workbench.aihub.qualcomm.com/jobs/j56ovllv5) / [jpv24xx7g](https://workbench.aihub.qualcomm.com/jobs/jpv24xx7g) / [jg9o8eeqg](https://workbench.aihub.qualcomm.com/jobs/jg9o8eeqg) |
| MRLFN | [j5q4zqq7g](https://workbench.aihub.qualcomm.com/jobs/j5q4zqq7g) / [jp3ox00xp](https://workbench.aihub.qualcomm.com/jobs/jp3ox00xp) / [jgj3onn7p](https://workbench.aihub.qualcomm.com/jobs/jgj3onn7p) |
| SplitterNet | [jgd6811lp](https://workbench.aihub.qualcomm.com/jobs/jgd6811lp) / [j57okryrg](https://workbench.aihub.qualcomm.com/jobs/j57okryrg) / [jgn1ov6mp](https://workbench.aihub.qualcomm.com/jobs/jgn1ov6mp) |
| U-Net | [j56or4y75](https://workbench.aihub.qualcomm.com/jobs/j56or4y75) / [jp3ox0jzp](https://workbench.aihub.qualcomm.com/jobs/jp3ox0jzp) / [jgodo62d5](https://workbench.aihub.qualcomm.com/jobs/jgodo62d5) |
| NAFNet | [jg9oknlvg](https://workbench.aihub.qualcomm.com/jobs/jg9oknlvg) / [jp4emrnlg](https://workbench.aihub.qualcomm.com/jobs/jp4emrnlg) / [jpx03o99p](https://workbench.aihub.qualcomm.com/jobs/jpx03o99p) |
| LL bypass＋Rep-NCB高频CNN | [jp2o4yxmg](https://workbench.aihub.qualcomm.com/jobs/jp2o4yxmg) / [jp0od04ep](https://workbench.aihub.qualcomm.com/jobs/jp0od04ep) / [jp8j6y285](https://workbench.aihub.qualcomm.com/jobs/jp8j6y285) |
| LL bypass＋Bayes Soft | [j56or4275](https://workbench.aihub.qualcomm.com/jobs/j56or4275) / [jgj3ond8p](https://workbench.aihub.qualcomm.com/jobs/jgj3ond8p) / [jpe68mo0g](https://workbench.aihub.qualcomm.com/jobs/jpe68mo0g) |

原始记录：[summary.json](qaihub_2k_seven_retest_20261010/summary.json)、[previous_results.json](qaihub_2k_seven_retest_20261010/previous_results.json)、[sources.json](qaihub_2k_seven_retest_20261010/sources.json)、[jobs.json](qaihub_2k_seven_retest_20261010/jobs.json)、[execution_audit.json](qaihub_2k_seven_retest_20261010/execution_audit.json)。各模型子目录包含DLC、逐轮profile和设备运行日志。

复测命令（`QAI_HUB_API_TOKEN`由环境提供；使用新的结果目录才会提交新任务）：

```bash
PYTHONPATH=/tmp/sid-qai-hub python tools/benchmark_sid_qaihub.py \
  --watch --height 720 --width 1280 --rounds 3 \
  --sources-json ref-doc/qaihub_2k_seven_retest_20261010/sources.json \
  --output-dir ref-doc/qaihub_2k_seven_retest_20261010

python tools/summarize_sid_qaihub.py \
  --output-dir ref-doc/qaihub_2k_seven_retest_20261010
```

### 8.2 Rep-NCB完整LL模型补测

日期：2026-10-11。补测模型为 **Rep-NCB3×3＋PReLU＋输入残差（保留LL恢复网络）**，区别于表中的LL bypass消融模型。输入、设备型号、量化与profile设置同第8节，沿用原编译任务 `j56o3m9v5`，DLC哈希与原结果一致；全部300次样本及三轮NPU执行记录已核对。

本次均值 **3.4584 ms**，上次为 **3.4471 ms**，变化 **+0.33%**。参数量与计算量仍为前文的90,368和14.0136 GMAC。

三轮新测速任务：[j56or22v5](https://workbench.aihub.qualcomm.com/jobs/j56or22v5) / [jpv2eqq7g](https://workbench.aihub.qualcomm.com/jobs/jpv2eqq7g) / [j5wy8wwzg](https://workbench.aihub.qualcomm.com/jobs/j5wy8wwzg)。

补测证据：[summary.json](qaihub_2k_repncb_retest_20261011/summary.json)、[previous_results.json](qaihub_2k_repncb_retest_20261011/previous_results.json)、[sources.json](qaihub_2k_repncb_retest_20261011/sources.json)、[jobs.json](qaihub_2k_repncb_retest_20261011/jobs.json)、[execution_audit.json](qaihub_2k_repncb_retest_20261011/execution_audit.json)。

## 9. 为什么LL bypass反而更慢：配置、量化和算子核对

本节为受控实验前的分析；后续同权重、同校准样本的验证结果见第10节。

分析日期：2026-10-11。结论是：**目前没有发现模型导出错误、W8A8/W8A16混用或CPU回退；但两者并不是“相同权重、相同量化编码，只删除LL模块”的严格受控比较。现有证据支持不同训练权重/激活分布与图结构共同影响后端执行开销，尚不能唯一归因于LL结构或量化编码。**

### 9.1 哪些条件已对齐，哪些没有

| 核对项 | 结果 |
|---|---|
| 输入 | 两者均为NCHW `[1,4,720,1280]`，没有外部736行补齐差异 |
| 硬件和运行时 | Galaxy S24、SM-S921U1、SM8650、QAIRT 2.50.0.260828221209；平台不保证同一台实体手机 |
| 编译和profile选项 | 字符串完全一致：W8A8目标、量化IO、NPU、3×100次 |
| 实际量化调用 | 两份日志均包含 `--weights_bitwidth 8 --act_bitwidth 8` |
| 模型IO类型 | 两者均为uint8；W8A8表示8bit目标，不应理解为所有边界都使用有符号int8 |
| 执行位置 | 所有profile执行条目均为NPU，没有观测到CPU/GPU回退 |
| 结构设置 | 导出metadata除 `learning_dwt.ll_mode` 的 `ll_only` / `bypass` 外一致 |
| 共同ONNX节点 | 44个共同节点的算子类型、属性、输出shape逐一一致；完整模型额外11个LL相关节点 |
| 权重 | 不同训练实验的checkpoint；高频、精修权重也不同，并非只移除同一模型的LL网络 |
| 校准数据 | 同为平台随机校准，但没有保存相同校准样本的哈希，不能证明校准样本完全一致 |
| 量化编码 | 并非完全一致，输出scale已直接证实不同；尚未取得并逐张量核对DLC内部编码和实际kernel选择 |

完整模型输出scale为 `0.002601688727736473`，bypass为 `0.002944938838481903`，输出zero point均为0。输入scale分别为 `0.003921568393707275` 与 `0.003921566065400839`，差异极小。输出scale不同本身是不同权重/输出分布下可能出现的正常结果，**不能单凭这一点断定量化出错或造成25%的耗时差**。

编译日志里的 `[QNN_CPU]` 出现在离线校准阶段，不是手机推理CPU回退的证据；运行归属应看profile执行记录。量化的位宽目标和逐张量编码是不同层面的概念，参见[Qualcomm量化说明](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html)及[编译API参数](https://workbench.aihub.qualcomm.com/docs/hub/api.html)。当前记录不足以证明每个内部算子均使用完全相同的8bit kernel。

### 9.2 删除LL只节省了很小一部分计算

| 项目 | 完整LL | LL bypass | 变化 |
|---|---:|---:|---:|
| 部署参数 | 90,368 | 60,152 | −33.44% |
| 卷积GMAC | 14.0136 | 13.5818 | −3.08% |
| ONNX节点 | 55 | 44 | 删除11个LL相关节点 |
| 复测均值/ms | 3.4584 | 4.3347 | **+25.34%（+0.8764 ms）** |

LL网络工作在三级分解后的 `90×160`，面积仅为原packed输入的1/64；删除它可以显著减少参数，却只省约3.08%的卷积MAC。删除它不会省掉全分辨率的Haar变换/重建、精修S2D/D2S和主要数据搬运。

三轮profile中，完整模型所有 `ll_restorer` 条目的平均周期合计仅约0.1333百万。这个值只是工具归属到这些节点的周期，不能直接当作其独立墙钟时间，但说明没有证据支持“LL本来占了很大一块推理时间”。

### 9.3 差异实际集中在哪里

下表使用各模型三轮对应节点的 `execution_cycles` 均值，单位为百万周期。它们是工具记录的算子归属，不可直接相加换算端到端毫秒或耗时占比。

| profile节点/区域 | 完整LL | bypass | bypass/完整 |
|---|---:|---:|---:|
| `/refiner/d2s/ConvTranspose` | 1.8804 | 3.4705 | **1.85×** |
| 高频第一级 `processors.0/.../PRelu` | 0.1677 | 0.6381 | **3.81×** |
| 精修第一块 `blocks.0/.../PRelu` | 0.2103 | 0.4786 | **2.28×** |
| 第一级高频拼接 `/wavelet/Concat` | 0.6784 | 1.0163 | **1.50×** |
| 输出布局转换条目 `output_0_0231` | 1.7343 | 1.2571 | 0.72× |
| 被移除的全部LL条目 | 0.1333 | 0 | — |

对应Rep-NCB卷积节点记录为0周期，而其PReLU条目记录非零周期，因此**不能将上表PReLU条目解释为单独激活的成本，也不能将0周期解释为卷积没有执行**；它可能包含融合或后端归属到激活节点的处理，现有profile没有提供更细的kernel映射。

特别是末尾D2S：两者都是权重shape `[16,4,2,2]` 的固定ConvTranspose，ONNX核值完全一致，输出均为 `[1,4,720,1280]`，但记录周期明显不同。这排除了“bypass把该层通道或空间尺寸意外放大”的解释。它提示量化重缩放、后端融合/布局/分块或调度上下文值得检查，但当前没有内部编码和kernel trace，不能断言是哪一个因素。

### 9.4 PReLU参数确实发生了明显变化，但因果尚未验证

| 层 | 完整LL斜率范围 | bypass斜率范围 |
|---|---|---|
| 高频第一级 | 0.1622～0.2862 | **−0.1230～0.8827** |
| 高频第二级 | 0.2766～0.7003 | 0.8742～1.0489 |
| 高频第三级 | 0.2251～0.6400 | **−0.0652～1.2141** |
| 精修第一块 | 0.0624～0.3693 | **−0.1295～0.6774** |

完整模型所有PReLU斜率都在0～1内；bypass多个层包含负值或大于1的值。第一层高频和精修第一块的这种差异，与上表较大的周期差同时出现，是值得验证的线索。负斜率是无约束PReLU可以学到的合法参数，不等于模型实现错误。

不过，bypass精修第2～4块也存在负斜率，其记录周期仍与完整模型接近。因此不能简单断言“只要出现负PReLU就慢”，更不能在没有实测前声称约束斜率就能恢复原速度。权重变化也会改变中间激活范围，即使校准输入完全相同，各模型的量化编码仍可能不同。

### 9.5 下一步如何严格区分结构与量化因素

建议按下面顺序做受控验证，本节仅分析现有证据，未执行新的模型修改或校准实验：

1. **同权重结构消融**：取完整模型同一checkpoint，仅在导出图中绕过LL恢复，保留所有HF、PReLU、精修权重。与原完整图使用同一份固定校准数据、相同工具版本编译。若仍明显变慢，更支持图拓扑/激活编码/后端执行变化；若不慢，现有独立训练权重就是重要混杂因素。由于LL旁路改变了下游数值，即使这样也不是完全固定下游量化编码的比较。
2. **固定校准样本**：显式上传并复用同一批packed RAW校准数据，保存数据哈希。排除不同随机样本的影响，但不要强迫两个不同分布模型使用不合适的同一scale并将其作为正式部署方案。
3. **PReLU单因素诊断**：在bypass副本中只替换指定层的斜率，或约束到0～1后重新校准，先观察上述两个热点是否变化。此操作会改变函数，只能作为测速诊断；若用于正式模型，需要训练或微调及效果复核。
4. **D2S路径诊断**：检查输入/输出量化编码与compiled kernel；必要时对比原生DepthToSpace与当前固定ConvTranspose，或单独抽取D2S测试，判断额外开销是否随重缩放/布局改变。不能预设原生算子一定更快。

所以，对用户问题更准确的回答是：**宏观测试条件已对齐，也没有发现LL bypass实现多做计算；但训练权重、逐张量量化编码和编译后的具体执行路径没有完全受控对齐。当前是“两个实际部署产物”的公平速度比较，还不足以得出“删除LL结构本身必然变慢”的因果结论。** 单纯再次profile相同DLC可以证明差距可重复，无法消除这些混杂因素。

核对结果、逐轮周期和PReLU参数范围保存在 [comparison.json](qaihub_ll_bypass_analysis_20261011/comparison.json)。原始profile、编译日志及量化IO specs可通过第7、8节证据链接追溯。

## 10. 同权重、同校准数据的LL旁路受控测速

日期：2026-10-11。按照用户要求，两组均使用完整 **Rep-NCB3×3＋PReLU＋输入残差** 模型的同一份部署权重。完整组保留LL恢复；bypass组仅移除 `wavelet.ll_restorer.*` 的15个参数张量，其余30个部署张量严格加载且逐值相同。没有训练或微调，没有使用独立训练的bypass权重。

原始权重为 `experiments/npu_4k/inputs/haar_hf_repncb_prelu_residual_snapshot.pth`（epoch 50）。两份新checkpoint的哈希不同，因为bypass删除了LL权重和修改了结构元数据；共同权重完全一致。两份导出ONNX的44个共同节点属性及45对常量输入逐一核对通过。

### 10.1 测试控制

- 输入 `[1,4,720,1280]`；Galaxy S24 / SM-S921U1 / SM8650，QAIRT 2.50.0.260828221209。平台不保证同一台实体手机。
- 两组均重新编译为W8A8目标、量化IO，未复用旧DLC；编译日志均确认权重/激活8bit。所有profile执行条目为NPU。
- 显式共用校准数据集 `d9w5r63k9`：固定seed=20261011，3个 `[1,4,720,1280]` float32均匀随机[0,1)样本。数据只上传一次，并将同一个Dataset对象对应ID用于两组编译；文件和各样本哈希见provenance。此数据用于控制速度实验，不作为真实RAW效果校准结论。
- 各3轮×100次，全部600次样本纳入统计，无极值剔除；不含上传/排队/外部预处理。
- 相同校准输入不意味着相同激活量化编码：LL旁路会改变下游数值，编码由各自模型校准得到。此次同时控制权重和校准样本，但没有强制共享内部量化scale。

### 10.2 结果

| 同权重模型 | 参数量 | 卷积GMAC | 均值/ms | P50/ms | P95/ms | 三轮均值/ms |
|---|---:|---:|---:|---:|---:|---|
| 完整LL | 90,368 | 14.0136 | **3.7927** | 3.7330 | 4.0430 | 3.7997 / 3.7799 / 3.7984 |
| LL bypass | 60,152 | 13.5818 | **4.3768** | 4.2740 | 4.8853 | 4.3735 / 4.3805 / 4.3762 |

本次同权重bypass相对完整模型耗时变化为 **+15.40%**，绝对变化 **+0.5841 ms**（正值更慢）。此前独立训练权重对比为4.3347 vs 3.4584 ms、bypass慢25.34%；它与本节不是同一种受控比较。

在共同权重和校准样本一致后，bypass仍存在额外耗时，支持LL旁路造成的图拓扑/激活分布及后端执行变化值得进一步检查；但尚未固定内部量化编码，因此不能区分纯拓扑和量化路径的贡献。

### 10.3 对应热点的周期记录

| profile条目 | 完整LL/百万周期 | bypass/百万周期 | 比值 |
|---|---:|---:|---:|
| `/refiner/d2s/ConvTranspose` | 1.9405 | 3.3934 | 1.75× |
| `/wavelet/processors.0/block/activation/PRelu` | 0.1623 | 0.5816 | 3.58× |
| `/refiner/blocks/blocks.0/activation/PRelu` | 0.2113 | 0.5116 | 2.42× |
| `/wavelet/Concat` | 0.6714 | 0.9993 | 1.49× |

周期为三轮对应条目均值，不等于独立算子墙钟时间；PReLU条目可能包含融合卷积的归属，不能解释为单独激活耗时。未新增PSNR/SSIM：直接旁路LL会改变模型输出，本次仅用于速度消融。

**对第9节推测的更新**：即使PReLU斜率及其余共同权重完全相同，bypass仍慢15.40%，且此前的几个热点仍存在。因此，负PReLU斜率或独立训练权重差异并不是该慢速现象出现的必要条件，不能将其视为已证明的根因。可以确认LL旁路这一改动及其后续校准/编译结果会在本次流程中产生额外耗时；尚不能将纯图结构、内部量化编码和后端kernel选择的贡献分开。

本次两组输入编码完全一致（scale `0.003921568393707275`、zero point 0）；输出编码仍有差异：完整组scale `0.0036837689112871885`、zero point 28，bypass为 `0.0038416380994021893`、zero point 27。这符合旁路LL后输出分布改变的情况，并不说明位宽未对齐。下一步若要继续定位，需要检查DLC内部编码和编译后的kernel/布局，或对D2S路径做进一步受控实验。

新完整组3.7927 ms与旧完整DLC的3.4584 ms也不同；此次重新校准、重新编译，且设备实体未固定，因此不能把这部分差额全部归于校准。对LL消融应使用本节两组新编译结果配对比较。

### 10.4 复现与原始证据

部署配置：[完整组](../configs/deploy/haar_same_weights_full_2k.yaml)、[bypass组](../configs/deploy/haar_same_weights_bypass_2k.yaml)。[权重准备脚本](qaihub_ll_same_weights_20261011/prepare_models.py)在仓库根目录运行；只生成独立副本，不修改原训练权重。

| 模型 | 新编译任务 | 三轮测速任务 |
|---|---|---|
| full | [jgzz82yxg](https://workbench.aihub.qualcomm.com/jobs/jgzz82yxg) | [jp1o72175](https://workbench.aihub.qualcomm.com/jobs/jp1o72175) / [jgd68n4zp](https://workbench.aihub.qualcomm.com/jobs/jgd68n4zp) / [j57ok2n9g](https://workbench.aihub.qualcomm.com/jobs/j57ok2n9g) |
| bypass | [j5wy8wzmg](https://workbench.aihub.qualcomm.com/jobs/j5wy8wzmg) | [jp4emn41g](https://workbench.aihub.qualcomm.com/jobs/jp4emn41g) / [jpx039rlp](https://workbench.aihub.qualcomm.com/jobs/jpx039rlp) / [j5m9oek9g](https://workbench.aihub.qualcomm.com/jobs/j5m9oek9g) |

记录：[summary.json](qaihub_ll_same_weights_20261011/summary.json)、[provenance.json](qaihub_ll_same_weights_20261011/provenance.json)、[sources.json](qaihub_ll_same_weights_20261011/sources.json)、[jobs.json](qaihub_ll_same_weights_20261011/jobs.json)、[export_reports.json](qaihub_ll_same_weights_20261011/export_reports.json)、[onnx_weight_audit.json](qaihub_ll_same_weights_20261011/onnx_weight_audit.json)、[compiled_specs.json](qaihub_ll_same_weights_20261011/compiled_specs.json)、[execution_audit.json](qaihub_ll_same_weights_20261011/execution_audit.json)、[mean_operator_cycles.json](qaihub_ll_same_weights_20261011/mean_operator_cycles.json)。ONNX和权重副本位于 `experiments/npu_ll_same_weights_20261011/`。

测速命令（环境变量提供token；已有目录恢复已有任务）：

```bash
PYTHONPATH=/tmp/sid-qai-hub python tools/benchmark_sid_qaihub.py \
  --watch --height 720 --width 1280 --rounds 3 \
  --sources-json ref-doc/qaihub_ll_same_weights_20261011/sources.json \
  --output-dir ref-doc/qaihub_ll_same_weights_20261011
```

## 11. LL旁路慢速的进一步定位：相同浮点子图仍出现周期差

日期：2026-10-11。基于第10节同权重、同校准实验继续分析。本节没有新增云端测速或修改训练模型；新增的是本地中间张量核对。

### 11.1 “少一个CNN应更快”的前提

如果删除LL网络后，其余算子仍使用相同kernel、相同数据布局、相同量化转换和相同内存调度，那么删除它确实应减少工作量。但重新编译后的整体执行方案未必保持不变：

```text
完整图耗时 = 共同部分在方案A下的耗时 + LL分支的增量成本
旁路图耗时 = 共同部分在方案B下的耗时
```

这是解释成本变化的概念分解，不假设实际分支完全串行。此次旁路移除了约0.4318 GMAC，只减少总卷积MAC的3.08%，而共同部分仍有13.5818 GMAC。只要共同部分出现更大的额外搬运/转换/执行开销，就可能抵消删除LL的收益。实际总耗时为3.7927→4.3768 ms，多约0.5841 ms。

LL处理分辨率为90×160，原图为720×1280；LL面积仅为原图的1/64。参数少30,216个并不等于全图耗时能减少三分之一。

### 11.2 新验证：高频网络的浮点输入输出没有变化

为两个原始浮点ONNX临时增加中间输出，用CPU ONNX Runtime关闭图优化运行同一份校准数据的3个样本，对对应张量逐值比较。临时诊断图未用于NPU测速，没有改变部署产物。

| 位置 | 三个校准样本的full/bypass比较 |
|---|---|
| 第一级高频Concat输出 | 全部逐值完全相同 |
| 第一级高频Conv输出 | 全部逐值完全相同 |
| 第一级高频PReLU输出 | 全部逐值完全相同 |
| 第一级高频残差Add输出 | 全部逐值完全相同 |
| 第二、三级高频残差Add输出 | 全部逐值完全相同 |
| 小波最终重建、精修第一块、最终输出 | 不同，符合LL处理改变后的预期 |

因此，对于高频部分，“删了LL导致高频输入更难处理”“PReLU数值变了”均不是这次测试中的解释。高频子图的形状、权重、浮点输入输出相同，但其NPU profile归属周期仍不同。这是应优先检查整图编译上下文而非CNN数学计算量的重要证据。

**边界**：这里只核对浮点源图；它不能证明两份DLC中高频张量的量化编码也相同。后端量化与图重写可能结合整个图处理。

数据见 [intermediate_float_audit.json](qaihub_ll_same_weights_20261011/intermediate_float_audit.json)，包含shape、范围、逐样本哈希和最大差值。

### 11.3 最值得怀疑的两类机制

**机制A：量化重缩放和图边界变化（待验证）。** 一个张量按 `x ≈ s(q−z)` 表示。如果连接算子需要将不同输入转换到共同编码，可能产生如下整数重缩放：

```text
q_out = round((s_in / s_out) × (q_in − z_in)) + z_out
```

这里的比例是编译期常数，不代表网络又出现逐像素浮点除法。问题是转换是否需要额外遍历、是否能融合、能否只搬运而不用改变数值。

LL恢复CNN提供一个独立的输出张量；直接旁路则将DWT的LL输出接入重建拼接。这样的图边界变化可能影响Concat/重建路径的编码协调与融合选择。该机制可以解释为何模型位宽一致，搬运或转换成本仍不同；但目前没有DLC内部逐张量编码，尚未证实后端具体在哪条边插入了转换。

末尾D2S尤其值得检查：数学上它是元素重排，固定2×2 ConvTranspose只是表达方式。如果输入输出编码相同，可以设想纯重排；如果编码不同，则可能还需重缩放，或被当作通用反卷积处理。两组D2S固定权重完全相同，但记录周期1.9405→3.3934百万，约1.75倍；这是已观测事实，而“多了重缩放”仍是待验证解释，不能仅由模型最终output scale推断D2S内部编码。

**机制B：分块、布局、缓冲区和算子调度变化（待验证）。** 删除节点会改变拓扑及张量使用关系。后端可能重新选择分块尺寸、缓冲区复用、融合或数据布局；共同节点的ONNX属性相同并不能保证设备kernel相同。第一级高频Concat和CNN位于LL恢复之前，其浮点张量完全一样仍出现周期差，使这种整图相关机制更值得优先调查。

数据遍历成本不能忽略：第一级高频有 `12×360×640=2,764,800` 个元素，INT8下约2.64 MiB；一次额外完整读写至少约5.27 MiB。D2S输入和输出各有3,686,400个元素，INT8下约3.52 MiB，完整读写约7.03 MiB。这只是张量有效数据量，不包含通道填充、缓存、布局中间缓冲和真实DRAM流量，也不能直接换算成耗时。相比90×160处的LL CNN，多出几次大特征遍历完全可能比省下低分辨率计算更贵。

Qualcomm文档给出了HTP张量布局和分块优化机制，但这只是上述解释的技术背景，不是本模型已经发生某个具体优化的证据：[Tensors and Memory Layout](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/tensors_and_memory_layout.html)、[HTP分块优化示例](https://docs.qualcomm.com/nav/home/common_optimization_utility_funcs_usage_examples.html?product=924033590759186372)。

### 11.4 哪些结论现在还不能下

- 不能断言是负PReLU、CPU回退或W8A16混入：共同PReLU权重已相同，记录均为NPU，编译目标同为W8A8。
- 不能断言“LL网络帮助缓存预热所以更快”：现有数据没有证明这种具体机制。
- 不能把PReLU条目的周期全部解释成激活耗时：对应卷积条目为0周期，存在融合/归属问题。
- 不能把算子周期差直接加总换算成0.5841 ms：Workbench逐层分析是额外一次运行，和多轮端到端计时不是同一个采样过程。[官方测量说明](https://workbench.aihub.qualcomm.com/docs/hub/howitworks-inference.html)
- 不能用 `estimated_inference_peak_memory` 判断是否发生HTP VTCM溢出：官方说明通用内存统计针对应用进程，HTP内存需看HTP Analysis。当前保存的JSON没有完整的kernel、VTCM和DDR spill/fill明细。
- 不能将运行日志里的 `No registered op ... crouton_to_vtcm` 警告直接认定为根因；两组日志均有类似信息，任务均成功，且没有与具体热点建立映射。

### 11.5 下一步最有区分力的验证

1. **检查两份图的HTP Analysis与内部编码**：重点对比D2S、第一层高频Concat/Conv/PReLU的kernel、输入输出scale、布局转换、VTCM/DDR读写。此项最可能直接解释“同一个算子为何变慢”。当前本地记录不足以完成这一层归因。
2. **做D2S表达方式对照**：同权重、同校准，将固定ConvTranspose替换为原生DepthToSpace，两个模型都测。若差距显著收敛，说明末尾表达方式/布局/量化路径贡献较大；不保证原生算子必然更快。
3. **抽取相同高频子图单独编译测速**：两组该子图的函数与输入已证实相同，使用同一个子图产物可建立独立基线。若整图差距仍只出现在原集成环境，支持整图编译/调度影响。独立子图边界会改变优化，不能把独立耗时直接相加预测整网耗时。
4. **增加明确的LL输出边界进行诊断**：例如固定恒等1×1卷积，但必须检查它是否被编译器消掉；若保留后更快，只能说明图表达/优化策略发生变化，不能将“多加算子更快”当成普遍规律。

当前最准确的解释是：**LL bypass确实减少了数学计算，但部署后的共同部分执行效率下降，抵消了省掉LL CNN的收益。证据已将问题收窄到整图量化/编译/执行上下文；具体是编码转换还是布局、分块和内存调度，仍需要更底层的HTP记录或针对性对照确认。**
