# 传统小波 SID 去噪基线

默认使用 3 层 sym4，4 个 packed RAW 通道独立做二维 DWT，symmetric 边界。
最细层 HH 的 median(abs(HH))/0.67449 估计各通道噪声标准差（排除精确零值），
各层 LH/HL/HH 使用 BayesShrink 软阈值：

```
signal_std = sqrt(max(mean(detail**2) - sigma**2, 0))
threshold = sigma**2 / signal_std
output_detail = sign(detail) * max(abs(detail) - threshold, 0)
```

信号方差非正时将该细节子带清零；保留最深层 LL，逆变换恢复图像。
算法仅使用 noisy RAW，无训练、无 GT 阈值拟合。噪声模型为每通道一个高斯噪声尺度，
未专门建模 RAW 的信号依赖散粒噪声或条纹。
算法参考：[scikit-image BayesShrink](https://scikit-image.org/docs/stable/auto_examples/filters/plot_denoise_wavelet.html)。

运行所有倍率：

```bash
python test_denoise_wavelet_sid.py --levels 3 --wavelet sym4 --output-dir test_res/my_wavelet_run
```

`--levels` 控制层数；`--wavelet` 支持 PyWavelets 正交离散小波，如 haar、db2、sym4、coif1。
该 BayesShrink 实现的噪声尺度假设要求正交变换，拒绝 bior2.2 等双正交基。
可用 `--ratios 100` 单独测一个倍率，`--max-samples 2` 快速检查。
`--visualize-samples 3` 为每倍率前 3 对保存整图缩略图和中心 256x256 RGB 局部对照。
每次实验使用新的输出目录，防止混合结果。

## 评估口径

直接复用 `SIDEvalDataset`，与 `test_denoise_sideld.py` 使用完全相同的
`infos/SID_evaltest.info`、分组、暗场校正、曝光倍率缩放和输入截断方式。
当前本地索引共 129 对：x100=40、x250=40、x300=49。
其中包含 1 前缀与 2 前缀场景，故这是本地测试脚本索引的结果，不能标为纯官方 test split。

输出 `results.json`（分倍率均值）、`per_image.jsonl`（逐图结果、源路径、实际倍率、纯算法耗时）和对比 PNG。
四组指标都按 `PMN_metric` 在 packed RAW 上计算，先逐图再平均：

- input：输入截断到 [0,1]。
- input_corrected：输入经 GT 亮度校正再截断。
- wavelet：算法输出直接截断。
- wavelet_corrected：算法输出经 ELDIlluminanceCorrect 再截断，与原测试脚本一致。

GT 只用于指标和亮度校正，未经校正的 wavelet 结果体现可独立使用的算法输出。
PNG 展示 input / 未经 GT 校正的 wavelet / GT，使用原测试的白平衡、CCM、gamma=3。
定量数据是 RAW PSNR/SSIM，不是显示图上的 RGB 指标。

## 2026-09-14 实测结果

| 倍率 | 对数 | 输入 PSNR/SSIM | 小波 PSNR/SSIM | 小波+亮度校正 PSNR/SSIM | 算法秒/图 |
|---|---:|---|---|---|---|
| ×100 | 40 | 29.350 / 0.5259 | 37.895 / 0.8668 | 38.196 / 0.8714 | 0.707 |
| ×250 | 40 | 23.995 / 0.3346 | 34.876 / 0.7950 | 35.177 / 0.7982 | 0.637 |
| ×300 | 49 | 22.483 / 0.2757 | 32.007 / 0.7302 | 32.264 / 0.7320 | 0.671 |

完整图文报告：`test_res/wavelet_sym4_l3_20260914_report/report.html`。
逐图数据及各倍率指标在同目录的 `per_image.jsonl` 和 `results.json`。
