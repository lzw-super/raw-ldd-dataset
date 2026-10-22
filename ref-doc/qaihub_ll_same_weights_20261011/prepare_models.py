import sys
sys.path.insert(0,'.')
from pathlib import Path
import torch,json,hashlib,yaml,numpy as np,copy
from utils.model_factory import build_denoiser_from_checkpoint
src=Path('experiments/npu_4k/inputs/haar_hf_repncb_prelu_residual_snapshot.pth');cp=torch.load(src,map_location='cpu');root=Path('experiments/npu_ll_same_weights_20261011');audit={'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'epoch':cp['epoch'],'variants':{}}
full=cp['model_deploy'];prefix='wavelet.ll_restorer.';removed=[k for k in full if k.startswith(prefix)];assert removed
for variant in ['full','bypass']:
 args=copy.deepcopy(cp['args']);args['dwt_ll_mode']='bypass' if variant=='bypass' else 'll_only';sd={k:v.clone() for k,v in full.items() if variant=='full' or not k.startswith(prefix)}
 path=root/'inputs'/f'{variant}.pth';torch.save({'args':args,'model_deploy':sd,'epoch':cp['epoch'],'inference_graph':'deploy'},path)
 model,meta=build_denoiser_from_checkpoint(str(path),'cpu');assert set(model.state_dict())==set(sd)
 assert all(torch.equal(v,full[k]) for k,v in model.state_dict().items())
 config={'name':f'haar_same_weights_{variant}_2k','kind':'learning_dwt_repncb','checkpoint':str(path),'height':720,'width':1280,'simplify':True}
 Path(f'configs/deploy/haar_same_weights_{variant}_2k.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 audit['variants'][variant]={'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'all_retained_tensors_equal_source':True,'retained_tensor_count':len(sd),'removed_keys':removed if variant=='bypass' else [],'deployed_parameters':sum(p.numel() for p in model.parameters())}
# Fixed random calibration is for controlled speed-only measurement, not accuracy.
rng=np.random.default_rng(20261011);arrays=[rng.random((1,4,720,1280),dtype=np.float32) for _ in range(3)]
np.savez(root/'inputs/calibration.npz',**{f'sample_{i}':x for i,x in enumerate(arrays)})
audit['calibration']={'path':str(root/'inputs/calibration.npz'),'seed':20261011,'samples':3,'distribution':'uniform [0,1), float32; speed-only','sha256':hashlib.sha256((root/'inputs/calibration.npz').read_bytes()).hexdigest(),'tensor_sha256':[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]}
Path('ref-doc/qaihub_ll_same_weights_20261011/provenance.json').write_text(json.dumps(audit,indent=2)+'\n');print(audit['variants'])
