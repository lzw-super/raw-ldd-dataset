"""Export and validate fixed-shape deployment-only ablations from YAML configs."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from models.haar_soft_export import HaarSoftExport
from models.haar_hf_cnn import HaarHFCNNExport
from utils.model_factory import build_denoiser_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configs', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/npu_priority1_ablation/onnx'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for config_path in args.configs:
        config = yaml.safe_load(config_path.read_text())
        model, metadata = build_denoiser_from_checkpoint(config['checkpoint'], 'cpu')
        model.eval()
        h, w = config['height'], config['width']
        shape = (1, 4, h, w)
        ramp = torch.linspace(-0.2, 1.2, h*w).reshape(1,1,h,w).expand(1,4,h,w).contiguous()
        impulse = torch.zeros(shape)
        impulse[..., 0, 0] = 1
        impulse[..., -1, -1] = -0.2
        inputs = {'uniform': torch.rand(shape), 'signed_raw': torch.randn(shape)*0.1,
                  'zeros': torch.zeros(shape), 'tiny': torch.full(shape,1e-30),
                  'border_ramp': ramp, 'corner_impulses': impulse}
        # CNN variants intentionally change the function. Compare ONNX to their
        # own untrained PyTorch network, never claim parity with soft shrinkage.
        reference_kind = 'original_checkpoint'
        if config['kind'] == 'haar_hf_cnn':
            model.wavelet = HaarHFCNNExport(model.wavelet, config['hf_cnn_variant'], config['seed'])
            reference_kind = 'untrained_cnn_variant'
        with torch.no_grad():
            expected = {k:model(x).numpy() for k,x in inputs.items()}
        if config['kind'] == 'haar':
            model.wavelet = HaarSoftExport(model.wavelet, config['hf_layout'], config['hf_formula'])
        elif config['kind'] not in ('splitternet', 'haar_hf_cnn'):
            raise ValueError(config['kind'])
        path = args.output_dir / (config['name']+'.onnx')
        with torch.no_grad():
            torch.onnx.export(model, inputs['uniform'], str(path), opset_version=17,
                              input_names=['noisy_raw'], output_names=['denoised_raw'],
                              do_constant_folding=True, dynamic_axes=None)
        graph = onnx.load(str(path))
        before = dict(Counter(n.op_type for n in graph.graph.node))
        simplifier_version = None
        if config['simplify']:
            from onnxsim import simplify, __version__ as simplifier_version
            # Numeric validation below covers random, signed and boundary-sensitive data.
            graph, ok = simplify(graph, check_n=0)
            if not ok:
                raise RuntimeError('onnxsim validation failed')
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, str(path))
        after = dict(Counter(n.op_type for n in graph.graph.node))
        if config['kind'] in ('haar', 'haar_hf_cnn'):
            assert not any(after.get(k,0) for k in ['Div','Abs','Sign'])
            assert after.get('ConvTranspose')==4
        opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
        opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
        checks=[]
        for label,x in inputs.items():
            actual=session.run(None,{'noisy_raw':x.numpy()})[0]
            diff=np.abs(actual-expected[label])
            passed=actual.shape==shape and bool(np.isfinite(actual).all() and np.allclose(actual,expected[label],rtol=1e-3,atol=1e-4))
            checks.append(dict(input=label,max_abs=float(diff.max()),mean_abs=float(diff.mean()),passed=passed))
            if not passed:
                raise RuntimeError(f'{config["name"]}: {checks[-1]}')
        report=dict(config=str(config_path),settings=config,checkpoint_sha256=hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest(),
                    metadata=metadata,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    operators_before=before,operators_after=after,checks=checks,shape=list(shape),
                    torch_version=torch.__version__,onnx_version=onnx.__version__,ort_version=ort.__version__,
                    onnxsim_version=simplifier_version,reference_kind=reference_kind,
                    deployed_parameter_count=sum(p.numel() for p in model.parameters()),
                    hf_parameter_count=sum(p.numel() for p in model.wavelet.processors.parameters()) if config['kind']=='haar_hf_cnn' else 0)
        if config['kind']=='haar_hf_cnn':
            torch.save(dict(seed=config['seed'],variant=config['hf_cnn_variant'],
                            state_dict=model.wavelet.processors.state_dict()),
                       path.with_name(path.stem+'_hf_untrained.pth'))
        path.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(config['name'], 'nodes',sum(before.values()),'->',sum(after.values()),
              'max_abs',max(c['max_abs'] for c in checks),'ops',after,flush=True)
        del session,model,graph,expected,inputs
        gc.collect()


if __name__ == '__main__':
    main()
