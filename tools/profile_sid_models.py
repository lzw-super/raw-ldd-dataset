"""Operator-level inference arithmetic and live-storage profiling (not training).
Counts ATen operations, including functional wavelets/shrinkage. Aliases/views share
storage; weak references track retained skips until the last tensor alias dies.
"""
import argparse, collections, gc, json, math, sys, weakref
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
import train_sid_sony as training
from utils.model_deployment import prepare_model_for_inference

PREFIX='configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb'
DEFAULTS=['configs/train_sid_sony_nafnet.yaml','configs/train_sid_sony.yaml',
'configs/train_sid_sony_mrlfn_paper_s2d_k4_n4_d32.yaml','configs/train_sid_sony_splitternet.yaml',
'configs/train_sid_sony_brve_single_frame.yaml',PREFIX+'_static_hf_depth5_soft.yaml']


def tensors(x):return [v for v in tree_flatten(x)[0] if isinstance(v,torch.Tensor)]
def storage(t):return t.untyped_storage()._cdata


class Census(TorchDispatchMode):
    def __init__(self,model):
        super().__init__()
        self.excluded={storage(t) for t in list(model.parameters())+list(model.buffers())}
        self.refs={};self.live={};self.current=0;self.peak=0;self.peak_op='';self.scope=[];self.origins={};self.peak_storages=[];self.peak_scope='' 
        self.ops=collections.Counter();self.calls=collections.Counter();self.unknown=collections.Counter()
        self.macs=0;self.binary_macs=0;self.in_binary=False
    def track(self,t):
        key=storage(t)
        if key in self.excluded or id(t) in self.refs:return
        size=t.untyped_storage().nbytes();ident=id(t)
        if key not in self.live:
            self.live[key]=[0,size];self.current+=size
            self.origins[key]={"shape":list(t.shape),"bytes":size,"module":self.scope[-1] if self.scope else "input_or_root"}
        self.live[key][0]+=1
        def release(_):
            self.refs.pop(ident,None)
            self.live[key][0]-=1
            if self.live[key][0]==0:
                self.current-=self.live[key][1];del self.live[key];self.origins.pop(key,None)
        self.refs[ident]=weakref.ref(t,release)
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};ins=tensors((args,kwargs))
        for t in ins:self.track(t)
        out=func(*args,**kwargs);outs=tensors(out)
        for t in outs:self.track(t)
        name=str(func).split('.')[1];self.calls[name]+=1
        if self.current>self.peak:
            self.peak=self.current;self.peak_op=str(func)
            self.peak_scope=self.scope[-1] if self.scope else "root"
            self.peak_storages=[dict(self.origins[k]) for k in self.live]
        n=sum(t.numel() for t in outs);cost=0
        if name in ('convolution','_convolution','conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d'):
            x,w=args[:2]
            transpose=args[6] if name in ("convolution","_convolution") else name.startswith("conv_transpose")
            if transpose:
                mac=x.numel()*w.shape[1]*math.prod(w.shape[2:])
            else:mac=outs[0].numel()*w.shape[1]*math.prod(w.shape[2:])
            if self.in_binary:self.binary_macs+=mac
            self.macs+=mac;cost=2*mac+(outs[0].numel() if (args[2] if len(args)>2 else kwargs.get("bias")) is not None else 0)
        elif name in ('mm','bmm','addmm'):
            a,b=(args[-2],args[-1]);mac=outs[0].numel()*a.shape[-1];self.macs+=mac;cost=2*mac+(n if name=='addmm' else 0)
        elif name in ('sum','mean','amax','amin','max','min'):
            cost=ins[0].numel()+(n if name=='mean' else 0)
        elif name in ('var','var_mean','std'):
            cost=4*ins[0].numel()+n
        elif name in ('avg_pool2d','adaptive_avg_pool2d','_adaptive_avg_pool2d'):
            cost=ins[0].numel()+n
        elif name in ('max_pool2d_with_indices','max_pool2d'):cost=outs[0].numel()*(math.prod(args[1]) if isinstance(args[1],(list,tuple)) else args[1]**2)
        elif name in ('upsample_bilinear2d',):cost=7*n
        elif name in ('native_layer_norm',):cost=8*ins[0].numel()
        elif name in ('sigmoid',):cost=4*n
        elif name in ('softplus',):cost=4*n
        elif name in ('silu','gelu'):cost=5*n
        elif name in ('_prelu_kernel','prelu','leaky_relu','leaky_relu_'):cost=2*n
        elif name in ('add','add_','sub','sub_','mul','mul_','div','div_','pow','abs','neg','sqrt','rsqrt','exp','log','sign','relu','relu_','clamp','clamp_min','clamp_max','threshold','gt','ge','lt','le','eq','ne','where','maximum','minimum','reciprocal','tanh','floor','ceil','round'):
            cost=n
        elif name in ('view_as','pad','chunk','repeat','expand_as','flatten','view','_unsafe_view','reshape','permute','transpose','t','slice','select','split','split_with_sizes','unbind','unsqueeze','squeeze','expand','detach','alias','as_strided','cat','stack','clone','contiguous','copy_','_to_copy','empty','empty_like','empty_strided','zeros','zeros_like','ones','ones_like','full','full_like','new_zeros','new_empty','new_ones','new_full','fill_','zero_','constant_pad_nd','replication_pad2d','reflection_pad2d','roll','pixel_shuffle','pixel_unshuffle','index','index_select','lift_fresh','lift_fresh_copy','slice_scatter','select_scatter'):
            pass
        else:self.unknown[name]+=1
        self.ops[name]+=int(cost)
        return out


def profile(config,args):
    sys.argv=['train_sid_sony.py','--config',config];cfg=training.parse_args()
    torch.manual_seed(2026)
    model,arch=training.build_model(cfg);model,graph=prepare_model_for_inference(model)
    model=model.to(args.device).eval();x=torch.rand(args.batch,4,args.height,args.width,device=args.device)
    # Specialize clean static shrink in the same way as the ONNX exporter.
    w=getattr(model,'wavelet',None)
    if w is not None and hasattr(w,'fixed_hf_thresholds'):
        w.export_simplify_static=True
        t=w.fixed_hf_thresholds;w.export_safe_thresholds=bool(torch.isfinite(t).all() and (t>=torch.finfo(t.dtype).tiny).all())
    with torch.inference_mode():
        y=model(x);del y
        if args.device.startswith('cuda'):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
        else:baseline=None
        c=Census(model);c.track(x)
        handles=[]
        for name,module in model.named_modules():
            handles.append(module.register_forward_pre_hook(lambda m,a,name=name: c.scope.append(name or "root")))
            handles.append(module.register_forward_hook(lambda m,a,o: (c.scope.pop(), None)[1]))
        for module in model.modules():
            if module.__class__.__name__ == "HardBinaryConv":
                handles.append(module.register_forward_pre_hook(lambda m,a: setattr(c,"in_binary",True)))
                handles.append(module.register_forward_hook(lambda m,a,o: setattr(c,"in_binary",False)))
        with c:y=model(x)
        for handle in handles:handle.remove()
        cuda_peak=torch.cuda.max_memory_allocated() if baseline is not None else None
        assert y.shape==x.shape and torch.isfinite(y).all()
    result=dict(config=config,model=cfg.model,graph=graph,architecture=arch,shape=list(x.shape),parameters=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()),
        arithmetic_ops=sum(c.ops.values()),conv_matmul_macs=c.macs,binary_conv_macs=c.binary_macs,live_feature_peak_bytes=c.peak,live_peak_operator=c.peak_op,live_peak_module=c.peak_scope,peak_storages=c.peak_storages,
        cuda_baseline_bytes=baseline,cuda_peak_allocated_bytes=cuda_peak,cuda_increment_bytes=None if baseline is None else cuda_peak-baseline,
        operator_arithmetic=dict(c.ops),operator_calls=dict(c.calls),unclassified_ops=dict(c.unknown))
    del model,x,y,c;gc.collect()
    if args.device.startswith('cuda'):torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--configs',nargs='+',default=DEFAULTS);p.add_argument('--height',type=int,default=360);p.add_argument('--width',type=int,default=640);p.add_argument('--batch',type=int,default=1);p.add_argument('--device',default='cuda:0');p.add_argument('--output',default='ref-doc/model_complexity/results.json');p.add_argument('--report',default='ref-doc/各模型推理计算量与峰值特征内存.md');args=p.parse_args()
    torch.set_num_threads(4);results=[];out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    for cfg in args.configs:
        r=profile(cfg,args);results.append(r);out.write_text(json.dumps(dict(torch=torch.__version__,device=args.device,device_name=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',dtype='float32',weight_source='config initialization; no trained checkpoints loaded',results=results),indent=2,ensure_ascii=False)+'\n')
        print(cfg,r['arithmetic_ops']/1e9,r['live_feature_peak_bytes']/2**20,'MiB',r['unclassified_ops'],flush=True)
    write_report(results,args)


def write_report(results,args):
    lines=["# 各模型推理计算量与峰值特征内存", "", "## 统计范围", "",
        f"统一输入 FP32 NCHW [{args.batch},4,{args.height},{args.width}]，eval + inference_mode，设备 {args.device}（{torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU'}）。PyTorch {torch.__version__}。",
        "本文模型唯一指定为 `configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft.yaml`，与五个外部网络对比。",
        "从当前 YAML 构建网络，不加载训练 checkpoint；结构参数以配置实际值为准。重参数化网络先融合为部署态；static_hf 使用 2×2 卷积形式 Haar/S2D/D2S、离线阈值及 clean 收缩路径。训练图的多分支不重复计入。",
        "", "## 汇总", "",
        "计算量不是仅统计卷积的 THOP 数值：通过 TorchDispatch 捕获 functional 和模块调用。GMAC 只列卷积/矩阵乘，GOp 列全部已归类的算术估算；乘加计 2 Op。",
        "", "| 模型/实验 | 参数 M | 卷积/矩阵 GMAC | 总算术 GOp | 同时存活特征峰值 MiB | CUDA 分配峰值 MiB |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        label=Path(r['config']).stem.removeprefix('train_sid_sony_') or 'unet'
        if label=='train_sid_sony':label='unet'
        if r['config']==PREFIX+'_static_hf_depth5_soft.yaml':label='本文模型（静态 Soft，depth5）'
        gpu='—' if r['cuda_peak_allocated_bytes'] is None else f"{r['cuda_peak_allocated_bytes']/2**20:.2f}"
        lines.append(f"| {label} | {r['parameters']/1e6:.4f} | {r['conv_matmul_macs']/1e9:.4f} | {r['arithmetic_ops']/1e9:.4f} | {r['live_feature_peak_bytes']/2**20:.2f} | {gpu} |")
    lines += ["", "## 计算量的具体口径", "",
        "- 普通/分组/深度卷积：输出元素数 × 每个输出的 kernel 输入元素数为 MAC；乘加按 2 Op，bias 按额外一次加法。转置卷积按输入元素数 × 每输入的输出 kernel 元素数，采用名义稠密计数，包含边界填充对应的运算，不估计硬件跳零优化。",
        "- 小波部署 Conv/ConvTranspose、固定重排卷积按实际稠密执行形式计算：即使权重含零或 ±0.5，也不假设 NPU 识别其稀疏性。动态 Haar 的逐元素加减乘则直接记录。",
        "- 阈值路径的 Abs/Add/Div/Mul、Softplus、Sigmoid、ReLU、比较、Pow、Sqrt 等均计入；Div/Exp/Sqrt 各视作一次标量原语，绝不表示它们与加法的硬件耗时相同。Sigmoid/Softplus 按约 4 Op，SiLU/GELU 按约 5 Op，PReLU/LeakyReLU 按 2 Op。",
        "- sum/max 归约按输入元素数近似；mean 额外计输出除法；variance/std 按约 4×输入元素数+输出元素数；layernorm 按约 8×输入元素数；双线性插值约 7 Op/输出。这些是明确约定的算法估算，不是硬件指令计数。",
        "- Split/Concat、Slice/View/Transpose、复制、pad、roll、PixelShuffle 等数据搬运计 0 算术 Op，但其产生和保留的 storage 仍计入内存。因此 GOp 不能单独预测移动端速度。",
        "- JSON 保留逐算子调用数、算术估算和 unclassified_ops。未归类的算子不会悄悄归入已统计总量；加入新网络后应检查这个字段。",
        "", "## 峰值内存的具体口径", "",
        "目标是 max_t Σ bytes(在时刻 t 仍存活的不同特征 storage)，而不是把全网络各层输出大小累加。后者是累计生成量/访存规模，不是峰值。",
        "脚本为每个 tensor 设置弱引用，记录最后一个已跟踪别名释放的时刻；按底层 storage 去重，view/slice 不重复计数，但一个小 view 保留的大 storage 按整体容量计算。每个算子执行后同时统计其输入与输出，因此 U-Net encoder 跳接、SplitterNet 多分支、BRVE 的序列/缓存特征都会持续计入，直到代码实际释放它们。包含输入和输出，排除模型参数及预注册 buffer；运行时临时张量（包括 BRVE 的权重二值化临时副本）仍计入，因此更准确地说是可见非持久张量峰值。无 backward/梯度/优化器状态。",
        "这是当前 PyTorch eager 代码可见的特征存储峰值，不是模型无关的最小理论内存：Python 局部变量、列表可能比最后使用点保留更久；编译器可重排或复用，故 NPU 编译后的峰值可能不同。未计入单个 ATen 算子内部不可见的 workspace、CUDA context、驱动和 allocator 保留缓存。",
        "CUDA 列是预热后 reset_peak_memory_stats 测得的 max_memory_allocated，含权重/buffer/输入和临时分配，可反映部分后端 workspace；它不是 reserved memory，也不是手机 NPU 内存。JSON 另含 baseline、增量和参数/buffer 字节数。",
        "", "## BRVE 及比较限制", ""]
    for r in results:
        if r['model']=='brve_single_frame':
            lines.append(f"BRVE 共 {r['conv_matmul_macs']/1e9:.3f} GMAC，其中 HardBinaryConv 的名义二值卷积为 {r['binary_conv_macs']/1e9:.3f} G binary-MAC，其余浮点卷积/矩阵乘为 {(r['conv_matmul_macs']-r['binary_conv_macs'])/1e9:.3f} GMAC。这里统计的是实际 PyTorch 浮点模拟路径；不能声称它与论文 bit-packed XNOR/popcount 的 BOP/FLOP 或内存相同，也不直接用除以 64 当作实测。")
    lines += ["BRVE 是三次重复同一输入的单帧适配，SplitterNet 是 RGB→4通道 RAW 适配；具体结构来源见 BRVE与SplitterNet_SID对照.md。较高运算量或内存不等于原论文视频基准表现。",
        "", "## 峰值位置与分类完整性", "", "| 配置 | 达峰算子 | 未归类调用 |", "|---|---|---|"]
    for r in results:lines.append(f"| `{r['config']}` | `{r['live_peak_operator']}` | `{r['unclassified_ops']}` |")
    lines += ["", "峰值位置是已有特征加上新输出共同达到峰值的位置，不等于该算子单独分配了全部峰值。当前所有模型的未归类集合应为空。",
        "", "## 复现", "", "```bash", "python tools/profile_sid_models.py --height 360 --width 640 --batch 1 --device cuda:0", "```", "",
        "使用 --configs 可传入任意已支持模型的 YAML 列表；--output 指定 JSON、--report 指定报告。没有 CUDA 时可 --device cpu，仍统计算术和特征生命周期，CUDA 列为空。需要 PyTorch 的 TorchDispatchMode（本次使用 2.1.2）。该脚本只做推理，不启动训练。",
        "", f"原始结果：`{args.output}`。单元测试验证了卷积计数、输入输出同时存活、跨层保留和 view 别名去重。",
        "", "## 对当前结果的解读", "",
        "NAFNet/U-Net 的大尺寸多通道特征与跳接提高了峰值；SplitterNet 虽然卷积量较低，但分支/skip 保留及 padding 会增加内存。所提网络主要在低分辨率工作，静态高频版本进一步移除阈值 CNN 与 atlas。",
        "本文模型使用静态 Soft 收缩，消除了高频动态 Div；不同算子的单次硬件成本不同，GOp 不能直接换算耗时。计算量表用于统一结构比较，不能代替目标 NPU 的端到端延迟与编译后内存统计。"]
    explanations={
        "nafnet":"最后一级（最高分辨率）decoder 的第二个 NAFBlock、第二次 LayerNorm。最高分辨率 skip、block 残差输入与归一化临时张量同时存活；内部 padding 后高度为 368。",
        "unet":"最高分辨率 decoder 的 conv9_2。encoder skip、上采样拼接结果以及 conv9_1/conv9_2 的输出同时保留；当前实现内部补到 368×656。",
        "mrlfn":"第二个 mRLFB 的 linear 卷积。浅层特征、前一块输出、当前块残差和卷积输出同时存活。",
        "splitternet":"最后一级 decoder：up.3.0 转置卷积之后的 LeakyReLU。其裁剪视图仍保留 369×641 的底层 storage，同时保留 stem skip 和较低分辨率分支列表。",
        "brve_single_frame":"stage2 U-Net 的最后一次上采样 up21，二值卷积输入 sign 阶段。三帧折叠为 batch=3，融合输入、前面 stage 特征、上采样特征及二值化临时张量同时存活。",
        "learning_dwt_repncb":"精修网络第二个 Rep-NCB 的 PReLU；不是 LL 小波深层。原始 RAW、小波恢复图、S2D 结果，以及 stem/前一块/当前卷积/当前激活四份 32 通道特征同时存活。"
    }
    lines += ["", "## 峰值出现阶段（实测定位）", "", "| 模型 | 特征峰值 MiB | 具体阶段与原因 |", "|---|---:|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['live_feature_peak_bytes']/2**20:.4f} | {explanations.get(r['model'],'参见模块路径')} |")
    lines += ["", "本文模型峰值可直接复核：4 × (1×32×180×320×4 bytes) + 3 × (1×4×360×640×4 bytes) = 40,550,400 bytes = **38.671875 MiB**。三份后者分别对应原输入、小波恢复图和相同元素数的 S2D 输出；四份前者是精修 stem、前块激活、当前卷积和当前激活。该明细针对本次默认输入与模型配置。", ""]
    lines += ["", "## 峰值阶段与同时存活张量明细", "", "下表重新运行并记录模块调用栈与达峰瞬间的底层 storage。各行相加就是对应模型的特征峰值；同形状、同来源的独立存储合并为一行，view 不重复统计。来源为存储首次被观察到的位置，不一定是最后使用者。CUDA 总峰值的阶段未由此推断：该字段与可见特征峰值是两个不同口径。", ""]
    for r in results:
        lines += ["### "+r["model"]+" — "+Path(r["config"]).stem, "", f"峰值 **{r['live_feature_peak_bytes']/2**20:.4f} MiB**；模块 `{r.get('live_peak_module','unrecorded')}`；算子 `{r['live_peak_operator']}`。", "", "| 首次观察模块 | 张量形状 | 独立存储数量 | 合计 MiB |", "|---|---|---:|---:|"]
        groups=collections.Counter()
        for v in r.get("peak_storages",[]):groups[(v["module"],tuple(v["shape"]),v["bytes"])]+=1
        for (module,shape,size),count in sorted(groups.items(),key=lambda kv: -kv[0][2]*kv[1]):
            lines.append(f"| `{module}` | `{shape}` | {count} | {size*count/2**20:.4f} |")
        lines.append("")
    report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
