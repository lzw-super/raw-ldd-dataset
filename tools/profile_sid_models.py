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
PREFIX+'_static_hf_depth5_soft_ll_no_norm.yaml']


def tensors(x):return [v for v in tree_flatten(x)[0] if isinstance(v,torch.Tensor)]
def storage(t):return t.untyped_storage()._cdata


class Census(TorchDispatchMode):
    def __init__(self,model):
        super().__init__()
        self.excluded={storage(t) for t in list(model.parameters())+list(model.buffers())}
        self.refs={};self.live={};self.current=0;self.peak=0;self.peak_op='';self.scope=[];self.origins={};self.peak_storages=[];self.peak_scope='' 
        self.ops=collections.Counter();self.calls=collections.Counter();self.unknown=collections.Counter()
        self.macs=0;self.binary_macs=0;self.in_binary=False
        self.buffer_names={storage(v):k for k,v in model.named_buffers()}
        self.categories=collections.Counter();self.module_ops=collections.defaultdict(collections.Counter)
        self.fixed_events=[];self.conv_bias_ops=0
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
        n=sum(t.numel() for t in outs);cost=0;category="non_convolution"
        if name in ('convolution','_convolution','conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d'):
            x,w=args[:2]
            transpose=args[6] if name in ("convolution","_convolution") else name.startswith("conv_transpose")
            if transpose:
                mac=x.numel()*w.shape[1]*math.prod(w.shape[2:])
            else:mac=outs[0].numel()*w.shape[1]*math.prod(w.shape[2:])
            if self.in_binary:self.binary_macs+=mac
            self.macs+=mac
            bias_ops=outs[0].numel() if (args[2] if len(args)>2 else kwargs.get("bias")) is not None else 0
            self.conv_bias_ops+=bias_ops;cost=2*mac+bias_ops
            buffer_name=self.buffer_names.get(storage(w), "")
            category=("fixed_haar" if buffer_name=="wavelet.transform.weight" else
                      "fixed_space_depth" if buffer_name in ("refiner.s2d.weight", "refiner.d2s.weight") else "learned_convolution")
            if category.startswith("fixed_"):
                self.fixed_events.append(dict(category=category,operator=name,input_shape=list(x.shape),output_shape=list(outs[0].shape),weight_shape=list(w.shape),macs=int(mac),ops=int(cost)))
        elif name in ('mm','bmm','addmm'):
            a,b=(args[-2],args[-1]);mac=outs[0].numel()*a.shape[-1];self.macs+=mac;cost=2*mac+(n if name=='addmm' else 0);category='matrix'
        elif name in ('sum','mean','amax','amin','max','min'):
            cost=ins[0].numel() if name=='mean' else ins[0].numel()-outs[0].numel()
        elif name in ('var','var_mean','std'):
            cost=4*ins[0].numel()+n
        elif name in ('avg_pool2d','adaptive_avg_pool2d','_adaptive_avg_pool2d'):
            cost=ins[0].numel()
        elif name in ('max_pool2d_with_indices','max_pool2d'):cost=outs[0].numel()*(math.prod(args[1]) if isinstance(args[1],(list,tuple)) else args[1]**2)-outs[0].numel()
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
        self.categories[category]+=int(cost)
        self.module_ops[self.scope[-1] if self.scope else "root"][name]+=int(cost)
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
        arithmetic_categories=dict(c.categories),convolution_bias_ops=c.conv_bias_ops,
        module_arithmetic={k:dict(v) for k,v in c.module_ops.items()},fixed_operator_events=c.fixed_events,
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
    t=args.batch*4*args.height*args.width
    specs={
        'nafnet':(25,'16T + 8T + T','最高分辨率 NAFBlock：64 通道展开特征 + 32 通道块内残差 + RAW 全局残差'),
        'unet':(16,'16T','最高分辨率 decoder 的 64 通道拼接结果；对应 skip 已包含在拼接中'),
        'mrlfn':(7,'T + (1+2)×2T','RAW 旁路 + mRLFB 当前特征 + 两条块内残差连接（按用户连接位置口径）'),
        'splitternet':(17,'8T + 8T + T','最后 decoder 上采样特征 + stem skip + RAW 全局残差'),
        'learning_dwt_repncb':(3,'2T + T','半分辨率32通道精修主干 + 原RAW或等元素数S2D旁路')}
    lines=['# 各模型推理计算量与需保留的最大特征图','',
        '## 本次更新与统计口径','',
        '本文模型使用 `configs/train_sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm.yaml`，移除 BRVE。计算量已按新配置重新执行统计。',
        '本表将原来的“同时存活特征峰值”替换为用户指定的**结构性特征保留指标**：当前计算路径的一份工作特征，加上尚未消费的跨层/残差连接；按阶段检查取最大值。输入输出缓冲不双计，拼接所包含的 skip 不再另计，已消费连接视为可释放。',
        '**MRLFN 按用户指定的两条残差连接分别预留位置，因此采用 7T。** 两条连接在 PyTorch 中实际引用同一块输入 x，可共享存储；此项是连接位置预算，不是 storage 去重后的内存下界。它与其他模型一起用于指定规则下的结构比较，不能当作统一最优调度算法求出的实际显存峰值。',
        '不计参数、算子内部临时张量、卷积输入输出双缓冲、allocator、backend workspace、padding/crop 的额外存储。MiB 列仅把结构元素数按 FP32 换算，不能用来保证设备实际可运行内存。',
        '',f'统一逻辑输入 [{args.batch},4,{args.height},{args.width}]；T={t:,} 个元素，FP32 时 1T={t*4/2**20:.6f} MiB。',
        '', '## 比较结果','',
        '| 模型 | 参数 M | 卷积/矩阵 GMAC | 总算术 GOp | 需保留的最大特征图 | FP32 等效 MiB |',
        '|---|---:|---:|---:|---|---:|']
    for r in results:
        mult,formula,stage=specs[r['model']]
        r['retained_feature_metric']={'T_elements':t,'multiple':mult,'elements':mult*t,'fp32_bytes':mult*t*4,'formula':formula,'stage':stage,'basis':'manual architecture accounting; ignores padding and kernel temporaries; MRLFN counts residual connection slots'}
        label='本文模型（静态 Soft，depth5，LL无归一化）' if r['model']=='learning_dwt_repncb' else r['model']
        lines.append(f"| {label} | {r['parameters']/1e6:.4f} | {r['conv_matmul_macs']/1e9:.4f} | {r['arithmetic_ops']/1e9:.4f} | {formula} = **{mult}T** | {mult*t*4/2**20:.4f} |")
    lines+=['','## 各网络的逐阶段推导','',
        '以下尺寸均用原始逻辑 H、W 表示；宽度按本次配置为32。局部特征的T倍数公式是 C/(4s²)，其中 s 是相对packed RAW的空间降采样倍数。',
        '', '### 本文模型：3T','',
        '精修路径 S2D(k=2) 的16通道特征为 T；stem 后32通道特征为 (H/2)(W/2)32=2T。保存原RAW的旁路为 T，也可以保留S2D后的16通道旁路，不能两者重复计数。串联的部署态Rep-NCB不需要保留每个历史块输出，故主干阶段为 2T+T=3T。head输出为T，末端与旁路拼接为2T，不超过3T。',
        '小波阶段：缓存HF1、HF2、HF3分别为3T/4、3T/16、3T/64，合计63T/64；LL3为T/64，32通道LL工作特征为T/8。即使保守同时预留原RAW T、全部HF 63T/64、当前LL工作T/8和LL拼接旁路T/64，总计136T/64=2.125T，也小于3T。推理完成重建后不再额外保留已消费高频。',
        'dwt_ll_normalize=false 去掉LL输入/输出尺度变换，不改变张量形状，因此本项3T不变；算术量则按该新配置重新统计。',
        '', '### MRLFN：7T（按指定连接位置口径）','',
        'mosaic域S2D(k=4)对应packed RAW的空间降低2倍：16通道为T，主干32通道为2T。按要求为RAW旁路预留T、mRLFB当前工作特征预留2T、两个块内残差连接各预留2T，得到T+3×2T=7T。',
        '代码 `feature=linear(feature+x); return feature+x` 中的两个x是同一个张量；若按实际共享storage，两条残差不必各占一份。这里保留7T以遵循用户要求，不再把该值称为PyTorch实测峰值。末端两个32通道特征拼接为4T，仍小于此7T预算。',
        '', '### NAFNet：25T','',
        '四级encoder的skip依次为8T、4T、2T、T；瓶颈512通道、空间1/16，特征为0.5T。NAFBlock会将通道扩大2倍，因此最上层64通道工作特征为16T，同时块内残差32通道为8T，全局RAW残差为T，总计25T。',
        'encoder第i层（i=0..3）的预算是 T + Σ(j<i)8T/2^j + 16T/2^i + 8T/2^i，分别25T、21T、19T、18T；瓶颈为T+15T+1T+0.5T=17.5T。decoder消费相应skip后再运行NAFBlock，最高分辨率阶段仍为25T。',
        '加法融合时一旦skip被消费，就不应在随后的block中继续重复计数。LayerNorm内部的均值、方差及临时图不计入新指标；它们正是旧实测峰值更大的原因之一。',
        '', '### U-Net：16T','',
        '四级encoder skip为8T、4T、2T、T，瓶颈特征为0.5T，因此瓶颈预算15.5T。decoder四级拼接工作图依次为2T、4T、8T、16T，对应尚未消费的更浅skip之和为14T、12T、8T、0，每一级均为16T。',
        '例如最后一级把8T上采样特征和8T的conv1 skip拼接为16T；这时不能再把该8T skip加一遍。该U-Net没有原RAW到输出的全局残差，所以没有额外T。',
        '', '### SplitterNet：17T','',
        '各encoder级全部分支总特征量依次为8T、4T、2T、T；深层16分支合计0.5T。保留全部encoder skip为15T，加全局RAW T、深层工作0.5T，以及单个深层分支注意力残差0.03125T，为约16.53125T。',
        '采用逐分支处理、用完skip立即释放的结构预算。最后一级decoder转置卷积产生32通道全分辨率工作特征8T，此时stem skip还需8T，RAW全局残差还需T，因此17T。两者相加后stem skip可释放。中间层全部分支要计总量，不能只统计一条分支的大小；若实现同时物化多个额外输出缓冲，会超过这里的预算。',
        '', '## 新旧指标为什么不能直接混用','',
        '旧“同时存活特征峰值”是TorchDispatch跟踪当前Python执行、按storage去重的实测结果；它包含多份输入/输出、中间算子张量和由于变量/列表未释放而存活的特征。新指标描述结构性的工作特征与连接预留，不是把旧数值换个标题。',
        '本次JSON仍保留 live_feature_peak_bytes、CUDA峰值及peak_storages供工程诊断，新增 retained_feature_metric 存放新指标。报告主表只展示新指标。参数、浮点精度、TensorRT/ONNX/NPU调度和padding均会影响实际内存；不得把本表当作实际峰值显存。',
        '', '## 计算量口径与复现','',
        '部署态算术统计包含functional小波、阈值收缩、激活、归约和重排卷积，不只统计CNN。MAC按乘加2 Op，Div/Exp/Sqrt按一个标量原语；归约、方差、LayerNorm、插值及组合激活采用脚本内注明的近似计数规则。数据搬运算0算术Op但可产生内存。它是统一算法量估算，不是硬件指令数或速度预测。',
        '计算量使用实际padding后尺寸，而新特征指标按H×W逻辑尺寸计算，以保持T公式清晰；两者的尺寸口径差异需在引用结果时保留。',
        '```bash','python tools/profile_sid_models.py --height 360 --width 640 --batch 1 --device cuda:0','```','',
        '默认列表为NAFNet、U-Net、MRLFN、SplitterNet、本文LL无归一化Soft模型。没有训练模型。新特征指标是基于这些固定配置的人工结构推导，不支持仅凭更换网络宽度而自动沿用本表常数；修改结构后应重新推导。',
        f'原始算术、运行时内存与结构指标结果：`{args.output}`。']
    lines+=ops_report(results,args)
    report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n')
    output=Path(args.output)
    if output.exists():
        payload=json.loads(output.read_text());payload['results']=results;output.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')

def ops_report(results,args):
    """Disjoint operator accounting; all values below come from the measured trace."""
    labels={'learning_dwt_repncb':'本文模型（LL无归一化 Soft）','nafnet':'NAFNet','unet':'U-Net','mrlfn':'MRLFN','splitternet':'SplitterNet'}
    lines=['','## Ops 重新统计与非卷积操作明细','',
        '本节由统计脚本在一次完整推理中记录 ATen 运算生成。使用配置初始化权重，不加载训练权重；执行部署重参数化、静态阈值离线固化和 clean 阈值路径，不进行训练或反向传播。一般稠密运算量与训练所得权重数值无关。',
        '', '### 统一计数规则','',
        '- 1 MAC = 2 Ops（一次乘法、一次累加）；1 GOp = 10⁹ Ops，1 MOp = 10⁶ Ops。这里也计入比较、取负、非线性等标量原语，因此不是只计浮点乘加的 FLOPs。',
        '- 普通卷积：MAC = N×Hout×Wout×Cout×(Cin/groups)×Kh×Kw；转置卷积：MAC = N×Hin×Win×Cin×(Cout/groups)×Kh×Kw。bias 另加输出元素数。转置卷积按输入散射的完整核计数，不扣边界裁掉的贡献。',
        '- 固定核也按实际稠密卷积计数，包含零权重乘加；不假定 NPU 自动跳过零权重。Haar 与 S2D/D2S 的固定卷积已从普通网络卷积分列，不能再次加到总量中。',
        '- 逐元素 add/sub/mul/div/pow/neg/abs/sqrt/ReLU：每输出元素 1 Op；PReLU、LeakyReLU：每元素按一次比较及一次乘法，共 2 Ops。pow(2) 视为一次乘法。Sigmoid 以取负、exp、加一、倒数合计 4 Ops/元素。复杂原语按算法计数，不表示其硬件成本与加法相等。',
        '- 归约输入元素数 A、输出元素数 B：sum/max 为 A−B Ops；mean 为 (A−B) 次加法加 B 次除法，即 A Ops。全局平均池化同理。k×k 最大池化为每输出 k²−1 次比较。本次修正了旧版归约和最大池化的近似多计。',
        '- LayerNorm 在当前 NAFNet 实现中展开统计：对 NCHW 特征，令 A=NCHW、B=NHW，两次 mean 共 2A，两次中心化减法 2A，平方 A，方差加 eps 与 sqrt 共 2B，除法 A，仿射乘加 2A，总计 8A+2B。不会再额外叠加一个 LayerNorm 估算值。',
        '- cat/split/chunk/view/slice、padding、PixelShuffle/Unshuffle、拷贝等记 0 算术 Ops；它们仍可能占用带宽与运行时间。',
        '', '### 五个模型总量拆分','',
        '各列互斥：总量 = 普通网络卷积/矩阵 + 固定 Haar + 固定 S2D/D2S + 非卷积。前三列包含各自 bias（若有）。',
        '', '| 模型 | 普通卷积/矩阵 GOp | 固定 Haar GOp | 固定 S2D/D2S GOp | 非卷积 GOp | 总 GOp |',
        '|---|---:|---:|---:|---:|---:|']
    for r in results:
        c=r['arithmetic_categories'];assert sum(c.values())==r['arithmetic_ops']
        lines.append(f"| {labels.get(r['model'],r['model'])} | {(c.get('learned_convolution',0)+c.get('matrix',0))/1e9:.9f} | {c.get('fixed_haar',0)/1e9:.9f} | {c.get('fixed_space_depth',0)/1e9:.9f} | {c.get('non_convolution',0)/1e9:.9f} | {r['arithmetic_ops']/1e9:.9f} |")
    lines+=['','### 各模型非卷积明细','',
        '下表是实际执行操作的聚合（单位 MOp），包括模块内部 functional 操作和跨层加法。卷积 bias 已计入前表卷积列，不计入本表。NAFNet 的归一化、门控和注意力已分解到对应标量操作；SplitterNet 的注意力统计同理。',
        '', '| 操作 | '+' | '.join(labels.get(r['model'],r['model']) for r in results)+' |',
        '|---|'+ '---:|'*len(results)]
    excluded={'convolution','_convolution','conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d','mm','bmm','addmm'}
    names=sorted({k for r in results for k,v in r['operator_arithmetic'].items() if v and k not in excluded})
    for name in names:
        lines.append('| '+name+' | '+' | '.join(f"{r['operator_arithmetic'].get(name,0)/1e6:.6f}" for r in results)+' |')
    lines.append('| 合计 | '+' | '.join(f"{r['arithmetic_categories'].get('non_convolution',0)/1e6:.6f}" for r in results)+' |')
    for r in results:
        assert sum(v for k,v in r['operator_arithmetic'].items() if k not in excluded)==r['arithmetic_categories'].get('non_convolution',0)
    ours=next((r for r in results if r['model']=='learning_dwt_repncb'),None)
    if ours:
        t=args.batch*4*args.height*args.width
        lines+=['','### 本文模型：小波、阈值与重排的逐项计算','',
            '以下公式针对当前三层 Haar、4 通道输入、leak=0、LL不归一化配置。设 T=N×4×H×W；三层仅递归分解上一级 LL，层输入元素数依次为 T、T/4、T/16。当前尺寸可被8整除，无额外小波 padding。',
            '', '**Haar 正/逆变换。** 每层部署 DWT 是权重 [16,4,2,2]、stride=2 的稠密 Conv2d，输出元素数等于该层输入 Tₗ，故为 2×Tₗ×4×2×2=32Tₗ Ops。IDWT 为相同权重的 ConvTranspose2d，输入也是 Tₗ 个元素，每个输入向4个通道的2×2区域贡献，故同样为32Tₗ Ops。正逆三层合计64T×(1+1/4+1/16)=84T。',
            '', '| 固定操作（实际执行顺序） | 输入形状 | 输出形状 | Ops | MOp |',
            '|---|---|---|---:|---:|']
        for event in ours['fixed_operator_events']:
            label=('Haar '+('IDWT' if 'transpose' in event['operator'] else 'DWT')) if event['category']=='fixed_haar' else ('D2S' if 'transpose' in event['operator'] else 'S2D')
            lines.append(f"| {label} | {event['input_shape']} | {event['output_shape']} | {event['ops']:,} | {event['ops']/1e6:.6f} |")
        lines+=['',
            '固定 Haar 的上述数字是本次部署图口径。如果使用直接非零公式实现，每个2×2块的4个输出各需4乘3加，即28 Ops/块，每层正或逆约7Tₗ，三层正逆约18.375T；复用中间加减还可更少。这个理论算法数字不替代当前稠密部署卷积的84T，也不叠加到总量。',
            '', '**高频 Soft 阈值收缩。** 实际 clean 路径为 `relu(z−t) − relu(−z−t)`。每个系数有3次减法、1次取负、2次ReLU，共6 Ops，无 Div/Abs/Sign。每层3个高频子带合计元素数为3T/4ˡ（l=1,2,3）；阈值为分子带分通道的广播常量，广播不增加额外算术。',
            '', '| 分解层 | 高频元素数（3个子带×4通道） | 收缩公式 | Ops | MOp |',
            '|---|---:|---|---:|---:|']
        for level in range(1,4):
            n=3*t//4**level
            lines.append(f'| L{level} | {n:,} | 6×{n:,} | {6*n:,} | {6*n/1e6:.6f} |')
        hf=3*t*sum(1/4**level for level in range(1,4));shrink=6*hf
        measured=sum(ours['module_arithmetic'].get('wavelet',{}).values())-ours['arithmetic_categories']['fixed_haar']
        assert measured==shrink, (measured,shrink)
        lines+=['',f'高频合计63T/64={int(hf):,}个元素，收缩共378T/64=**{int(shrink):,} Ops（{shrink/1e6:.6f} MOp）**。36个阈值的 softplus 和子带 scale 已在部署转换时离线计算，推理为0 Ops；不再构建阈值 CNN 或十子带 atlas，也不在线计算 gain。当前 LL 无归一化，LL 尺度乘除为0 Ops。',
            '', '**S2D/D2S。** 精修主路径与 RAW 旁路各一次 S2D，末端一次 D2S，共3次。当前2×2稠密固定卷积每次32T Ops，合计96T；它们是重排的部署实现成本。若用真正 PixelShuffle/Unshuffle，则为0算术Ops，但存在数据搬运，本表不混用这两种实现。',
            '', '**其余操作。** LL恢复及精修 CNN 的卷积、融合1×1卷积已归入普通卷积列；PReLU及LL输出处ReLU已归入非卷积列。cat/split/select/view 等均为0算术Ops。',
            '', '| 本文模型非卷积来源 | Ops | MOp |', '|---|---:|---:|']
        for name,predicate in [('高频 Soft 收缩',lambda k:k=='wavelet'),('LL恢复网络激活',lambda k:k.startswith('wavelet.ll_restorer')),('精修网络激活',lambda k:k.startswith('refiner'))]:
            # Fixed convolutions are excluded by operator name, regardless of module scope.
            value=sum(v for k,ops in ours['module_arithmetic'].items() if predicate(k) for op,v in ops.items() if op not in excluded)
            lines.append(f'| {name} | {value:,} | {value/1e6:.6f} |')
    unknown={r['model']:r['unclassified_ops'] for r in results if r['unclassified_ops']}
    lines+=['',f'本次未分类算子：`{unknown}`。JSON 额外保存 `arithmetic_categories`、`module_arithmetic` 和 `fixed_operator_events`，可逐模块/逐固定操作复核；各分类之和已校验等于总 Ops。归约计数修正会使部分基线总量较旧版略有下降，模型结构没有改变。']
    return lines


if __name__=='__main__':main()
