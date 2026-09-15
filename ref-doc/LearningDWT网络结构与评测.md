# LearningDWT 32通道 atlas：结构与SID评测

## 1. 已核实的模型

checkpoint：`experiments/sid_sony_learning_dwt_sym4_l3_d64_atlas/checkpoints/latest.pth`。
虽然路径保留d64，checkpoint实际args及权重为：**width32、depth4、atlas、sym4、3层、smooth**。
已训练1000 epoch / 81,000步，可训练参数22,028。使用latest作为本次指定评估权重；
它记录的验证best_epoch=800，本报告未用best.pth替换用户指定的latest。

```
[N,4,H,W] packed RAW
 → 固定sym4 DWT，三级只递归分解LL
 → 10个终端子带平铺为[N,4,H,W]（periodization边界）
 → 按层归一化 + signed/abs拼接 [N,8,H,W]
 → Conv3×3 8→32→32→32→4（前三层ReLU）
 → 子带/CFA bias，正阈值与LL阈值上限
 → Z_hat = Z * abs(Z)/(abs(Z)+T)
 → sym4 IWT → restored packed RAW
```

CNN预测的是系数位置阈值，不是原图像素坐标阈值。小波滤波器固定，无需训练。
LL也参与连续收缩，cap=0.01；输出不做训练前截断。
当前atlas网络没有band embedding；bandwise配置是另一项实验，不能从当前结果断定哪一种更好。

## 2. 1000 epoch定量结果

| 倍率 | 对数 | 传统 PSNR / SSIM | 学习版 PSNR / SSIM | ΔPSNR | ΔSSIM |
|---|---:|---|---|---:|---:|
| ×100 | 40 | 38.196 / 0.8714 | 40.522 / 0.9134 | +2.325 | +0.0420 |
| ×250 | 40 | 35.177 / 0.7982 | 36.667 / 0.8327 | +1.491 | +0.0344 |
| ×300 | 49 | 32.264 / 0.7320 | 33.752 / 0.7894 | +1.488 | +0.0573 |

口径：双方输出均做ELDIlluminanceCorrect、clamp至[0,1]、逐图PMN_metric，再取算术平均。
传统算法为固定sym4、3层、symmetric边界的BayesShrink。
学习版使用periodization以保持严格同shape平铺；当前收益是两套完整方法的比较，
不是只改变“阈值是否学习”的严格单因素消融。可用同边界BayesShrink补充对照。

学习版数值来自用户已生成的sid_x100/250/300.json，本次读取并核验checkpoint元数据；
传统数值来自此前全量实测。本次检查了此前逐图记录与当前索引的全部129组short/long路径一致。
未重复运行用户已经完成的全量定量评估。汇总含checkpoint SHA256和源文件路径：
`experiments/sid_sony_learning_dwt_sym4_l3_d64_atlas/comparison_vs_wavelet.json`。

结论：**在当前本地测试索引和评测协议下，训练后的模型确实优于传统基线。**
前述200步、64通道bandwise探索不能否定这个结果；预算和结构均不同。
要分离atlas模式、通道数、LL与连续收缩的贡献，需要匹配训练预算进行各项消融。
本地索引混有1/2前缀，含训练流程使用的验证场景；独立官方测试结论需要额外按1前缀统计。

## 3. 训练指令

沿用用户已更新的32通道配置；不因文件名中d64而改回64通道：

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py \
  --config configs/train_sid_sony_learning_dwt_sym4_l3_d64_atlas.yaml
```

当前batch32、patch256、warmup10、epoch1000，验证每50epoch前40对。

## 4. 定量评测（对应MRLFN文档第6节）

```bash
for ratio in 100 250 300; do
  conda run --no-capture-output -n LED-ICCV23 python test_denoise_sideld.py \
    --cp-dir experiments/sid_sony_learning_dwt_sym4_l3_d64_atlas/checkpoints/latest.pth \
    --testset-type sid --eval-ratio "$ratio" --device cuda:0 --num-workers 2 \
    --result-json "experiments/sid_sony_learning_dwt_sym4_l3_d64_atlas/sid_x${ratio}.json"
done
```

该模型应显示`learning_dwt / native`，无需MRLFN部署重参数化。
实际宽度见JSON的`model_config.learning_dwt.width`，不是通用的`feature_channels`字段。

同一进程逐pair比较原传统算法、periodization传统算法和学习模型，并记录校正前后指标：

```bash
conda run --no-capture-output -n LED-ICCV23 python test_compare_learning_wavelet_sid.py \
  --checkpoint experiments/sid_sony_learning_dwt_sym4_l3_d64_atlas/checkpoints/latest.pth \
  --output-dir test_res/learning_dwt_atlas32_full_comparison
```

## 5. 定性对比（对应MRLFN文档第7节）

已扩展原定性入口，新增`--with-wavelet-baseline`，四列依次为：
**输入 | 传统sym4 BayesShrink | 学习版 | GT**。
两种去噪输出都使用与定量一致的亮度校正，标注是整张RAW的PSNR，展示的是同位置512×512 RGB裁剪。
固定各倍率下标0、10、20、30，避免挑选不同图片造成视觉误判。

```bash
for r in 100 250 300; do
  CP_DIR=experiments/sid_sony_learning_dwt_sym4_l3_d64_atlas/checkpoints/latest.pth \
  OUT_DIR=experiments/sid_sony_learning_dwt_sym4_l3_d64_atlas/qualitative_baseline \
  EXTRA="--indices 0 10 20 30 --crop-size 512 --with-wavelet-baseline" \
    bash test-op/run_qual_compare.sh 0 "$r"
done
```

不加`--with-wavelet-baseline`则保持原三列输入/学习版/GT展示。
同目录图文报告为`report.html`，PNG名为`denoise_comparison_ratio100/250/300.png`。

## 6. 实现与验证

详细设计与历史短训记录见`learning_wt/README.md`。
38项模型、factory、真实验证、传统基线回归测试在base环境通过；LED-ICCV23环境的11项unittest通过，且该环境完成实际checkpoint推理和可视化。
当前已完成长训模型的读取和三种倍率的固定样本可视化；不将早期短训负面结果延伸为对方法的否定。
