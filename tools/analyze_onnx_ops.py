#!/usr/bin/env python3
"""Analyze operator composition of the exported deploy ONNX model.

Outputs:
  1. op type counts
  2. per-op details (attrs, input/output shapes, weights)
  3. conv/linear FLOPs breakdown
  4. NPU/INT8 risk flags (layout-movement ops, unsupported activations, etc.)
"""
import sys
from collections import Counter, defaultdict

import onnx
from onnx import shape_inference, TensorProto

ONNX_PATH = sys.argv[1]

model = onnx.load(ONNX_PATH)
graph = model.graph

print("=" * 78)
print("MODEL:", ONNX_PATH)
print("IR version:", model.ir_version, " opset:", [(o.domain, o.version) for o in model.opset_import])

# ---------- value name -> type/shape ----------
inferred = shape_inference.infer_shapes(model)
vi_map = {}
for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
    t = vi.type.tensor_type
    dims = []
    for d in t.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else d.dim_param or "?")
    vi_map[vi.name] = (t.elem_type, dims)

def shape_of(name):
    return vi_map.get(name, (None, None))[1]

def dtype_str(et):
    return TensorProto.DataType.Name(et) if et else "?"

init_names = {i.name for i in graph.initializer}
init_map = {i.name: i for i in graph.initializer}

# ---------- op stats ----------
op_count = Counter()
op_flops = Counter()          # MAC-heavy ops
op_risk = defaultdict(list)  # risky ops detail list

total_conv_macs = 0.0
total_linear_macs = 0.0
total_macs = 0.0

CONV_LIKE = {"Conv", "ConvTranspose", "MatMul", "Gemm"}

for idx, node in enumerate(graph.node):
    op = node.op_type
    op_count[op] += 1

    oshape = shape_of(node.output[0]) if node.output else None
    ishape = [shape_of(i) for i in node.input if i]

    # FLOPs (MACs) for conv-like ops
    if op in ("Conv", "ConvTranspose"):
        wname = node.input[1]
        w = init_map.get(wname)
        if w is not None:
            oc, ic, kh, kw = list(w.dims)
            cout = oshape[1] if oshape else None
            hout = oshape[2] if oshape else None
            wout = oshape[3] if oshape else None
            groups_attr = next((a.i for a in node.attribute if a.name == "group"), 1)
            per_group_ic = ic
            if cout and hout and wout:
                macs = cout * hout * wout * per_group_ic * kh * kw / groups_attr
                total_conv_macs += macs
                op_flops[op] += macs
    elif op in ("MatMul", "Gemm"):
        macs = 0
        a_shape = ishape[0] if ishape else None
        b_name = node.input[1]
        b = init_map.get(b_name)
        if a_shape and b is not None:
            m = 1
            for d in a_shape[:-1]:
                m *= d if isinstance(d, int) else 1
            k = a_shape[-1] if isinstance(a_shape[-1], int) else 0
            n = b.dims[-1]
            macs = m * k * n
        total_linear_macs += macs
        op_flops[op] += macs
        total_macs += macs

    # ---- NPU / INT8 risk tagging ----
    risk = None
    if op in ("Transpose", "DepthToSpace", "SpaceToDepth", "Reshape", "Squeeze", "Unsqueeze",
              "Slice", "Gather", "Split", "Tile", "Expand", "Pad", "Scatter", "Roll"):
        risk = "layout/data movement (bandwidth-bound, often falls back or serializes on NPU)"
    elif op in ("Mul", "Div", "Add", "Sub", "Pow", "Sqrt", "Reciprocal", "Exp", "Log",
                "Erf", "Sin", "Cos", "ATan", "Round", "Sign", "Not", "And", "Or", "Xor",
                "Less", "Greater", "Equal", "Where", "Min", "Max", "Mod", "LeakyRelu", "Clip",
                "HardSwish", "Mish", "RandomNormalLike", "Softplus"):
        risk = "elementwise/activation (per-op activation-quant overhead or unsupported fallback)"
    elif op in ("Sigmoid",):
        risk = "Sigmoid: exp-based, many NPUs lack HW support -> LUT or float fallback"
    elif op in ("LayerNormalization", "InstanceNormalization", "BatchNormalization",
                "MeanVarianceNormalization", "Softmax", "ReduceMean", "ReduceL2", "ReduceLogSumExp"):
        risk = "reduction/norm: cross-channel reductions often unsupported in INT8, fallback to float"
    elif op in ("Resize", "Upsample"):
        mode = next((a.s for a in node.attribute if a.name == "mode"), b"")
        risk = f"Resize (mode={mode.decode()}): interpolation often slow/fallback"
    elif op in ("Concat", "Flatten", "Identity", "Dropout", "Cast", "Constant"):
        risk = None if op == "Identity" else "pure data plumbing (copy overhead on NPU DMA)"
    if risk:
        op_risk[op].append((idx, node.name, ishape, oshape, risk))

    # print full node list on request
    if len(sys.argv) > 2 and sys.argv[2] == "-v":
        attrs = {a.name: (list(a.ints) if a.ints else a.i if a.type == onnx.AttributeProto.INT
                          else a.f if a.type == onnx.AttributeProto.FLOAT
                          else a.s.decode() if a.type == onnx.AttributeProto.STRING else "...")
                 for a in node.attribute}
        print(f"[{idx:4d}] {op:20s} name={node.name!r} in={[shape_of(i) for i in node.input if i and i not in init_names]} "
              f"out={oshape} attrs={attrs}")

total_macs = total_conv_macs + total_linear_macs
print()
print("=" * 78)
print("OP TYPE COUNTS (sorted by count):")
for op, c in op_count.most_common():
    flops = op_flops.get(op, 0)
    flops_s = f" MACs={flops/1e6:.1f}M" if flops else ""
    print(f"  {op:22s} x{c:3d}{flops_s}")

print()
def numel(dims):
    n = 1
    for d in dims:
        n *= d
    return n

print(f"Total initializers: {len(graph.initializer)}  params bytes≈{sum(numel(list(i.dims))*4 for i in graph.initializer if i.data_type==TensorProto.FLOAT)/1e6:.2f} MB (fp32)")
print(f"Total Conv MACs:  {total_conv_macs/1e9:.3f} G  (~{2*total_conv_macs/1e9:.2f} GFLOPs)")
print(f"Total MatMul MACs:{total_linear_macs/1e6:.1f} M")

print()
print("=" * 78)
print("INPUTS:")
for i in graph.input:
    print("  ", i.name, shape_of(i.name), dtype_str(vi_map.get(i.name, (None,))[0]))
print("OUTPUTS:")
for o in graph.output:
    print("  ", o.name, shape_of(o.name), dtype_str(vi_map.get(o.name, (None,))[0]))

print()
print("=" * 78)
print("NPU/INT8 RISK FLAGS (grouped):")
if not op_risk:
    print("  none")
for op, items in sorted(op_risk.items(), key=lambda kv: -len(kv[1])):
    print(f"\n  -- {op}  x{len(items)}")
    for idx, name, ish, osh, risk in items[:8]:
        print(f"     node[{idx}] {name!r} {ish} -> {osh}")
        print(f"       risk: {risk}")
    if len(items) > 8:
        print(f"     ... and {len(items)-8} more")

# ---------- conv attribute distribution ----------
print()
print("=" * 78)
print("CONV KERNEL/STRIDE/PAD/GROUP DISTRIBUTION:")
ksp = Counter()
for node in graph.node:
    if node.op_type in ("Conv", "ConvTranspose"):
        w = init_map.get(node.input[1])
        attrs = {}
        for a in node.attribute:
            if a.name in ("kernel_shape", "strides", "pads", "dilations", "group"):
                attrs[a.name] = tuple(a.ints) if a.ints else a.i
        ksp[(tuple(w.dims[2:]) if w else None,
             attrs.get("strides"), attrs.get("pads"), attrs.get("group"), attrs.get("dilations"))] += 1
for (k, s, p, g, d), c in ksp.most_common():
    print(f"  k={k} s={s} p={p} g={g} d={d}  x{c}")
