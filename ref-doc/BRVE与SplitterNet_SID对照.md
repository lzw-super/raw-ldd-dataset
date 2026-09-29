# BRVE 与 SplitterNet 的 SID packed RAW 结构对照

## 来源与复现范围

依据本地论文：

- `ref-pdf/Zhang 等 - 2024 - Binarized Low-light Raw Video Enhancement.pdf`
- `ref-pdf/Flepp 等 - 2024 - Real-World Mobile Image Denoising Dataset with Efficient Baselines.pdf`

官方代码版本：

- [BRVE](https://github.com/ying-fu/BRVE/tree/9a79224b7ccdb5105b47ac887ed7ce272d828e66)，读取 `configs/BRVE_LLRVD.py`、`mmedit/models/backbones/bnnvd.py`、`DABC.py`。
- [SplitterNet](https://github.com/rflepp/SplitterNet-Efficient-Mobile-Denoising-Models-CVPR2024/tree/ce3598a1b340a73a3e0c8eef0f44766309130afc)，原 Efficient_Mobile_Denoising_Models URL 当前重定向到此仓库；读取 `models/SplitterNet.py` 和官方 H5/参考输出。

这是网络结构的 SID 对照适配，不是复现两篇论文的完整数据、优化器、损失、训练阶段或部署指标。公共接口为 N×4×H×W，模型独立，不包含小波、LL 或 Rep-NCB。

## BRVE：单帧适配

实现位于 `models/brve_vendor/`，移植官方 BNNVD/DABC，去掉 MMEditing/mmcv 注册和权重加载依赖；原网络计算、二值前向及梯度代理保留。autograd Function 改用类方法 apply，避免实例化弃用警告。

结构参数沿用 LLRVD 官方配置：mid_channels=24，feat_extract_blocks=3，num_unets=1，unet_n_feat=[24,48,96]，unet_n_block=[1,3,3]，stage1_n_feat=[24,48,48,48]，Raw2Raw。包括初步 BNN U-Net、时空移位编码器及融合后的 BNN U-Net。参数量 4,508,584。

当前训练数据没有邻近视频帧，因此 `BRVESingleFrame` 将同一噪声图重复三次作为序列，调用官方 forward，取中间帧输出。不会为三帧额外独立生成噪声，避免人为提供多次观测。该适配保留网络结构，但无法测量真实时序信息带来的收益；名称使用 `brve_single_frame`。`.core` 保留原 BNNVD 的 N×T×4×H×W 接口供后续视频实验。

输入边界 replicate pad 至 8 的倍数，最小 128×128，以支持深层空间移位；输出裁回原尺寸。官方 spatial_shift 索引行为原样保留，未自行改写算法。训练/测试均使用该单帧适配。

DABC 在普通 PyTorch 上仍通过浮点卷积模拟二值运算；并未实现手机的 bit-packed XNOR/popcount 内核。参数数量、普通 PyTorch MACs 或延迟不能直接视作论文 1-bit 部署成本。

## SplitterNet：4 通道 RAW 适配

`models/paper_denoisers.py` 为官方 Keras 无 LayerNorm、宽度 32、四级结构的 PyTorch 移植：每级每条分支按通道一分为二，独立降采样，分支数依次为 2/4/8/16；16 个中间块含简化通道注意力、空间注意力及残差；解码配对拼接、转置卷积上采样、加 encoder skip；末端全局残差。

保留 reflect padding、LeakyReLU(0.3)、无 LN、Glorot 初始化。特别匹配 TensorFlow stride=2/kernel=3 SAME 转置卷积对齐：使用 PyTorch padding=0 后裁掉右/下多余部分，而非直接套 padding=1/output_padding=1。

RGB 原版参数量 731,059，与论文一致。SID 适配只把首尾 RGB 的 3 通道改为 packed RAW 的 4 通道，参数量 731,636。非整除尺寸 pad 到 16 的倍数，最小 32×32，以便最深层 reflect padding 有效。

用官方 `SplitterNet_MIDD_model.h5` 全量映射卷积权重，运行仓库 `splitternet_midd_reference.npz` 的原 TensorFlow 测试输入，最大绝对误差 **3.5762787e-07**。验证命令：

```bash
python tools/verify_splitternet_reference.py /path/to/official/SplitterNet-repository
```

这个 H5 仅用于验证 RGB 移植，新的 RAW 实验不加载该权重。

## 训练设置与命令

两份配置从用户指定的 `...static_hf_depth5_soft.yaml` 复制所有非 dwt/refine 训练字段，仅改变模型和输出目录：包括 batch_size=32、patch_size=256、1000 epochs、Adam lr=0.0002、warmup=10、数据合成、ratio、验证及保存周期。损失明确统一为基准的 0.6 RAW L1 + 0.4 chromatic L1，而不是 BRVE 原 Charbonnier 或 SplitterNet 原 MSE。

```bash
conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py --config configs/train_sid_sony_brve_single_frame.yaml

conda run --no-capture-output -n LED-ICCV23 \
  python train_sid_sony.py --config configs/train_sid_sony_splitternet.yaml
```

新增模型已接入统一工厂、训练脚本、定量/定性评测 CLI，可直接用新 checkpoint 自动识别模型。无额外的 deploy 融合函数，普通保存权重即可测试；best PSNR、best_train_l1、latest 逻辑不变。复用了原训练管线，不需要安装 mmcv/MMEditing/TensorFlow/Keras 来训练。

## 验证与限制

验证覆盖训练配置一致性、不同尺寸的前向/反向、所有参数梯度有限、checkpoint 保存/重载；另在 LED-ICCV23 做 CUDA batch=1、256×256 前向与反向。没有启动完整训练，也没有确认原 batch=32 的显存容量。BRVE 重复三帧会增加中间激活，完整 SID 验证也可能占用较大显存，应根据硬件调整 batch 或后续引入有重叠的分块评测并单独说明评测差异。

授权文件已保留：BRVE MIT 位于 `models/licenses/BRVE-MIT.txt`；SplitterNet 仓库 CC BY-NC-SA 4.0 位于 `models/licenses/SplitterNet-CC-BY-NC-SA-4.0.md`。本地 SplitterNet 为上述网络的 PyTorch/RAW 改编，BRVE 为去依赖及单帧包装的改编。
