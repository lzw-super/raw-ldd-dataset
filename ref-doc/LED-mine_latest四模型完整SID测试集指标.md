# LED latest 四模型：完整 SID 测试集指标（2026-10-23）

测试源：`../../Datasets/SID-sub/Sony_test_list.txt`，共 598 对。100×/250×/300×：182/180/236 对。

LED 本地 `valid_100/250/300.txt` 的 93 对全部在该测试列表中，且用于这些模型的训练期间验证。此前 `evaluation_latest/` 结果只覆盖这 93 对，不是完整测试集结果。本次使用 latest，没有按测试结果选择权重。

指标：LED RAW PSNR/SSIM、亮度校正开启、裁边 2 像素、完整图像 FP32 推理。平均列为三档等权平均；各实验 JSON 另含全体图像等权平均。

| 模型 | 100× PSNR / SSIM | 250× PSNR / SSIM | 300× PSNR / SSIM | 三档平均 PSNR / SSIM |
|---|---|---|---|---|
| repncb | 40.9766 / 0.958973 | 37.0561 / 0.910038 | 35.9569 / 0.930506 | 37.9965 / 0.933172 |
| mrlfn | 40.8222 / 0.958648 | 36.7231 / 0.902520 | 35.7029 / 0.926095 | 37.7494 / 0.929088 |
| splitternet | 42.0679 / 0.959560 | 35.5189 / 0.893044 | 35.7258 / 0.925275 | 37.7709 / 0.925960 |
| nafnet | 43.1625 / 0.964663 | 35.7978 / 0.906242 | 36.1448 / 0.927374 | 38.3683 / 0.932760 |

结果位于各自 `experiments/ledmine_sid_<模型>/evaluation_latest_sid_test/`，包括完整 JSON、指标汇总、原始日志和三档配对列表。

复现：

```bash
for model in repncb mrlfn splitternet nafnet; do
  conda run --no-capture-output -n LED-ICCV23 python evaluate_led_mine.py \
    --config "configs/ledmine_sid_${model}.yaml" \
    --checkpoint "experiments/ledmine_sid_${model}/models/net_g_latest.pth" \
    --cuda-visible-devices 0 --split sid-test
done
```
