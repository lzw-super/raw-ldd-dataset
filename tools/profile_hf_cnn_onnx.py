"""Count arithmetic on actual, simplified static ONNX graphs used for HF benchmarking."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import onnx
from onnx import helper


def count(path):
    model=onnx.shape_inference.infer_shapes(onnx.load(str(path)), strict_mode=True, data_prop=True)
    shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in list(model.graph.input)+list(model.graph.output)+list(model.graph.value_info)}
    shapes.update({v.name:list(v.dims) for v in model.graph.initializer})
    events=[];ops=Counter();macs=0
    for node in model.graph.node:
        op=node.op_type
        attributes={a.name:helper.get_attribute_value(a) for a in node.attribute}
        output_shape=shapes[node.output[0]]
        assert all(d>0 for d in output_shape), (node.name,output_shape)
        elements=math.prod(output_shape);cost=0;mac=0
        if op in ('Conv','ConvTranspose'):
            weight=shapes[node.input[1]]
            # weight[1] already includes the division by groups.
            if op=='Conv':mac=elements*weight[1]*math.prod(weight[2:])
            else:mac=math.prod(shapes[node.input[0]])*weight[1]*math.prod(weight[2:])
            cost=2*mac+(elements if len(node.input)>2 and node.input[2] else 0)
        elif op in ('PRelu','LeakyRelu'):cost=2*elements
        elif op in ('Relu','Sub','Add','Neg','Min','Max','Mul','Div','Abs','Pow','Sqrt'):cost=elements
        elif op=='Sigmoid':cost=4*elements  # negate, exp, add, reciprocal
        elif op in ('ReduceMean','GlobalAveragePool'):
            # Each reduced group: N-1 additions and one division.
            cost=math.prod(shapes[node.input[0]])
        elif op=='ReduceMax':cost=math.prod(shapes[node.input[0]])-elements
        elif op=='MaxPool':cost=elements*(math.prod(attributes['kernel_shape'])-1)
        elif op not in ('Split','Concat','Slice','Reshape','Transpose','Identity','Constant',
                        'Pad','DepthToSpace','SpaceToDepth'):
            raise ValueError(f'Unclassified operator: {op}')
        ops[op]+=cost;macs+=mac
        events.append(dict(name=node.name,operator=op,output_shape=output_shape,ops=cost,macs=mac,
                           group=attributes.get('group',1) if op in ('Conv','ConvTranspose') else None))
    hf=[e for e in events if '/processors.' in e['name']]
    return dict(onnx=str(path),node_count=len(events),arithmetic_ops=sum(ops.values()),conv_macs=macs,
                operator_ops=dict(ops),hf_cnn_ops=sum(e['ops'] for e in hf),hf_cnn_macs=sum(e['macs'] for e in hf),events=events)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path('ref-doc/qaihub_hf_cnn_20261008'))
    args=parser.parse_args();root=args.results_dir
    sources=json.loads((root/'sources.json').read_text());exports=json.loads((root/'export_reports.json').read_text())
    # The unchanged deployed LL/refiner has 86,408 parameters, checked by subtracting
    # the individually counted HF modules in every newly exported model.
    bases={r['deployed_parameter_count']-r['hf_parameter_count'] for r in exports.values()}
    assert len(bases)==1
    base=bases.pop();results={}
    expected_hf={'dw1':(72,907200,3628800),'dw3':(360,8164800,18144000),
                 'dw3_pw1':(828,19051200,40824000),'dw1_residual':(72,907200,4536000)}
    for label,record in sources.items():
        row=count(record['source'])
        if label in exports:
            report=exports[label];variant=report['settings']['hf_cnn_variant']
            params=report['deployed_parameter_count'];hf_params=report['hf_parameter_count'];thresholds=0
            assert (hf_params,row['hf_cnn_macs'],row['hf_cnn_ops'])==expected_hf[variant]
        else:
            params=base;hf_params=0;thresholds=36
        row.update(deployed_nn_parameters=params,hf_cnn_parameters=hf_params,
                   frozen_learned_thresholds=thresholds,deployed_learned_coefficients=params+thresholds)
        results[label]=row
        print(label,params,'parameters +',thresholds,'thresholds',row['conv_macs']/1e9,'GMAC',row['arithmetic_ops']/1e9,'GOp')
    (root/'complexity.json').write_text(json.dumps(dict(shape=[1,4,360,640],base_cnn_parameters=base,
                    mac_to_ops=2,counts_include_bias_activation_and_fixed_dense_transform=True,
                    results=results),indent=2)+'\n')


if __name__=='__main__':main()
