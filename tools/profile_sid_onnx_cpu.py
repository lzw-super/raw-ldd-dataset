"""Interleaved CPU retest and optimized ORT per-node profiling."""
import json,time,collections
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT=Path('ref-doc/onnx_cpu_profile');ROOT.mkdir(exist_ok=True)
paths={
'haar':'experiments/sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb/onnx/learning_dwt_repncb_latest_deploy_1x4x360x640.onnx',
'mrlfn':'experiments/sid_sony_mrlfn_paper_s2d_k4_n4_d32/onnx/mrlfn_latest_deploy_1x4x360x640.onnx'}
x=np.random.default_rng(2026).random((1,4,360,640),dtype=np.float32)
report={'ort':ort.__version__,'shape':list(x.shape),'warmup':10,'rounds':3,'runs_per_round':20,'timing':{},'profiles':{}}
for threads in [1,4]:
 sessions={}
 for name,path in paths.items():
  o=ort.SessionOptions();o.intra_op_num_threads=threads;o.inter_op_num_threads=1
  o.add_session_config_entry('session.intra_op.allow_spinning','0')
  sessions[name]=ort.InferenceSession(path,o,providers=['CPUExecutionProvider'])
  for _ in range(10):sessions[name].run(None,{'noisy_raw':x})
 times={name:[] for name in paths}
 for rnd in range(3):
  for name in (list(paths) if rnd%2==0 else list(reversed(paths))):
   block=[]
   for _ in range(20):
    start=time.perf_counter();sessions[name].run(None,{'noisy_raw':x});block.append((time.perf_counter()-start)*1000)
   times[name].append(block)
 report['timing'][str(threads)]={name:dict(mean_ms=float(np.mean(v)),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),round_means_ms=np.mean(v,axis=1).tolist()) for name,v in times.items()}
 print(threads,report['timing'][str(threads)],flush=True)
 del sessions
for name,path in paths.items():
 o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1;o.enable_profiling=True
 o.add_session_config_entry('session.intra_op.allow_spinning','0')
 o.profile_file_prefix=str(ROOT/name);o.optimized_model_filepath=str(ROOT/(name+'_optimized.onnx'))
 s=ort.InferenceSession(path,o,providers=['CPUExecutionProvider'])
 for _ in range(15):s.run(None,{'noisy_raw':x})
 events=json.loads(Path(s.end_profiling()).read_text())
 runs=[e for e in events if e.get('name')=='model_run'];cutoff=runs[5]['ts']
 nodes=collections.defaultdict(float);ops=collections.defaultdict(float);groups=collections.defaultdict(float)
 for e in events:
  if e.get('cat')!='Node' or not e['name'].endswith('_kernel_time') or e['ts']<cutoff:continue
  key=e['name'];ms=e['dur']/1000/10;nodes[key]+=ms;ops[e['args'].get('op_name','?')]+=ms
  if '/wavelet/predictor/' in key:g='threshold_cnn'
  elif '/wavelet/ll_restorer/' in key:g='ll_restorer'
  elif '/refiner/' in key:g='refiner'
  elif '/wavelet/' in key:g='wavelet_transform_and_atlas_ops'
  else:g='runtime_reorders_or_other'
  groups[g]+=ms
 report['profiles'][name]={'groups_ms':dict(groups),'ops_ms':dict(sorted(ops.items(),key=lambda kv:-kv[1])),'nodes_ms':dict(sorted(nodes.items(),key=lambda kv:-kv[1])),'profiled_runs':10}
 del s
# Attribute original wavelet node names to atlas phases for this exported graph.
import onnx
original=onnx.load(paths['haar'])
indices={n.name:i for i,n in enumerate(original.graph.node)}
end_dwt=indices['/wavelet/Concat_8']
start_predictor=min(i for n,i in indices.items() if '/wavelet/predictor/' in n)
end_predictor=max(i for n,i in indices.items() if '/wavelet/predictor/' in n)
start_ll=min(i for n,i in indices.items() if '/wavelet/ll_restorer/' in n)
end_ll=max(i for n,i in indices.items() if '/wavelet/ll_restorer/' in n)
end_replace=indices['/wavelet/ScatterND_22']
detail=collections.defaultdict(float)
for key,ms in report['profiles']['haar']['nodes_ms'].items():
    name=key.removesuffix('_kernel_time')
    if '/wavelet/predictor/' in name:g='threshold_cnn'
    elif '/wavelet/ll_restorer/' in name:g='ll_restorer'
    elif '/refiner/s2d' in name or '/refiner/d2s' in name:g='refiner_s2d_d2s'
    elif '/refiner/' in name:g='refiner_convs_and_other'
    elif name in indices:
        i=indices[name]
        if i<=end_dwt:g='haar_dwt_and_atlas'
        elif i<start_predictor:g='maps_and_normalization'
        elif i<start_ll:g='threshold_and_shrink'
        elif i<=end_replace:g='ll_atlas_replacement'
        else:g='haar_iwt_and_crop'
    else:g='runtime_reorders_and_unmapped'
    detail[g]+=ms
report['profiles']['haar']['detailed_groups_ms']=dict(detail)
mrlfn_detail=collections.defaultdict(float)
for name,ms in report['profiles']['mrlfn']['nodes_ms'].items():
    if '/space_to_depth/' in name:g='s2d'
    elif '/depth_to_space/' in name:g='d2s'
    elif '/shallow_conv/' in name:g='shallow_conv'
    elif '/shallow_fusion/' in name:g='shallow_fusion'
    elif '/deep_fusion/' in name:g='deep_fusion'
    elif '/output_conv/' in name:g='output_conv'
    elif '/blocks/blocks.' in name:g='block_'+name.split('/blocks/blocks.')[1].split('/')[0]
    elif name.startswith('/Concat'):g='concat'
    else:g='runtime_reorders_and_other'
    mrlfn_detail[g]+=ms
report['profiles']['mrlfn']['detailed_groups_ms']=dict(mrlfn_detail)
(ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print('Done',flush=True)
