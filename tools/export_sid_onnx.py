"""Export fixed NCHW packed RAW checkpoints and verify with ONNX Runtime CPU."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnx
import onnxruntime as ort
import torch
from utils.model_factory import build_denoiser_from_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoints',nargs='+')
    parser.add_argument('--height',type=int,default=360)
    parser.add_argument('--width',type=int,default=640)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    for filename in args.checkpoints:
        path=Path(filename)
        outdir=path.parent.parent/'onnx';outdir.mkdir(exist_ok=True)
        model,meta=build_denoiser_from_checkpoint(str(path),'cpu')
        output=outdir/f"{meta['model']}_{path.stem}_deploy_1x4x{args.height}x{args.width}.onnx"
        x=torch.rand(1,4,args.height,args.width)
        with torch.no_grad():
            torch.onnx.export(model,x,str(output),opset_version=17,
                              input_names=['noisy_raw'],output_names=['denoised_raw'],
                              do_constant_folding=True,dynamic_axes=None)
        graph=onnx.load(str(output));onnx.checker.check_model(graph,full_check=True)
        shape=[d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
        assert shape==list(x.shape),shape
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(output),sess_options=options,providers=['CPUExecutionProvider'])
        checks=[]
        for label,test in [('uniform',x),('signed_raw',torch.randn_like(x)*0.1),('zeros',torch.zeros_like(x))]:
            with torch.no_grad():expected=model(test).numpy()
            actual=session.run(None,{'noisy_raw':test.numpy()})[0]
            assert actual.shape==tuple(test.shape)
            assert np.isfinite(actual).all()
            diff=np.abs(actual-expected)
            passed=bool(np.allclose(actual,expected,rtol=1e-3,atol=1e-4))
            checks.append(dict(input=label,max_abs=float(diff.max()),mean_abs=float(diff.mean()),passed=passed))
            if not passed:raise RuntimeError(f'Parity failed: {output}: {checks[-1]}')
        report=dict(checkpoint=str(path),onnx=str(output),input_shape=shape,layout='NCHW',dtype='float32',
                    opset=17,model=meta,checks=checks,torch_version=torch.__version__,
                    onnx_version=onnx.__version__,onnxruntime_version=ort.__version__,
                    bytes=output.stat().st_size,onnx_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        output.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(report,ensure_ascii=False),flush=True)
        del session,model,graph;gc.collect()

if __name__=='__main__':main()
