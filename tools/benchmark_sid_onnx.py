"""Fixed-shape ORT CPU/CUDA latency benchmark; CUDA I/O binding + synchronized timing."""
import argparse,json,time,gc,subprocess,os
from pathlib import Path
import numpy as np
import torch  # Load CUDA/cuDNN libraries before ORT CUDA provider.
import onnxruntime as ort

def measure(fn,warmup,runs,sync=lambda:None):
    for _ in range(warmup):fn()
    sync(); times=[]
    for _ in range(runs):
        sync();start=time.perf_counter();fn();sync();times.append((time.perf_counter()-start)*1000)
    return dict(mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),min_ms=float(min(times)),fps=1000/float(np.mean(times)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='ref-doc/onnx_benchmark_results.json');a=p.parse_args()
    names=['sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb','sid_sony_mrlfn_paper_s2d_k4_n4_d32','sid_sony_nafnet_new','sid_sony_paper_fair']
    report=dict(ort=ort.__version__,providers=ort.get_available_providers(),shape=[1,4,360,640],cpu_threads=4,cpu_warmup=5,cpu_runs=20,gpu_warmup=10,gpu_runs=50,gpu=0,tf32=False,results=[])
    report['gpu_before']=subprocess.check_output(['nvidia-smi'],text=True)
    x=np.random.default_rng(2026).random((1,4,360,640),dtype=np.float32)
    for name, prefix in zip(names, ['learning_dwt_repncb', 'mrlfn', 'nafnet', 'unet']):
        path=Path('experiments')/name/'onnx'/f'{prefix}_latest_deploy_1x4x360x640.onnx'
        result=dict(name=name,path=str(path));opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
        opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])
        result['cpu']=measure(lambda:s.run(None,{'noisy_raw':x}),5,20)
        reference=s.run(None,{'noisy_raw':x})[0];del s;gc.collect()
        s=ort.InferenceSession(str(path),opts,providers=[('CUDAExecutionProvider',{'device_id':0,'use_tf32':0,'cudnn_conv_algo_search':'HEURISTIC'}),'CPUExecutionProvider'])
        if 'CUDAExecutionProvider' not in s.get_providers():raise RuntimeError('CUDA unavailable; refusing CPU fallback benchmark')
        result['gpu_provider_options']=s.get_provider_options()
        device_input=ort.OrtValue.ortvalue_from_numpy(x,'cuda',0)
        device_output=ort.OrtValue.ortvalue_from_shape_and_type(x.shape,np.float32,'cuda',0)
        binding=s.io_binding();binding.bind_ortvalue_input('noisy_raw',device_input);binding.bind_ortvalue_output('denoised_raw',device_output)
        sync=lambda:torch.cuda.synchronize(0)
        result['gpu_device']=measure(lambda:s.run_with_iobinding(binding),10,50,sync)
        actual=device_output.numpy();result['gpu_cpu_max_abs']=float(np.abs(actual-reference).max())
        assert np.allclose(actual,reference,atol=1e-4,rtol=1e-3)
        result['gpu_host_io']=measure(lambda:s.run(None,{'noisy_raw':x}),10,50,sync)
        report['results'].append(result)
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        del s,binding,device_input,device_output;gc.collect()
    report['gpu_after']=subprocess.check_output(['nvidia-smi'],text=True)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
