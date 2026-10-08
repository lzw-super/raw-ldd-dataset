"""Resumable Qualcomm Workbench W8A8 speed-only benchmark. No token in artifacts.
Install qai-hub separately; authenticate with QAI_HUB_API_TOKEN or --token-file.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import qai_hub as hub

ROOT=Path('ref-doc/qaihub_w8a8_20261007')
EXPERIMENTS={
 'haar_ll_no_norm':'sid_sony_learning_dwt_haar_l3_d32_atlas_ll3_cnn_concat1x1_repncb_w32_ll_repncb_static_hf_depth5_soft_ll_no_norm',
 'mrlfn':'sid_sony_mrlfn_paper_s2d_k4_n4_d32',
 'nafnet':'sid_sony_nafnet_new',
 'unet':'sid_sony_paper_new',
 'splitternet':'sid_sony_splitternet',
}
COMPILE='--target_runtime qnn_dlc --quantize_full_type int8 --quantize_io --compute_unit npu --qairt_version 2.50'
PROFILE='--compute_unit npu --qairt_version 2.50 --max_profiler_iterations 100 --max_profiler_time 600'


def main():
 global ROOT
 p=argparse.ArgumentParser();p.add_argument('--token-file',type=Path);p.add_argument('--rounds',type=int,default=3);p.add_argument('--watch',action='store_true')
 p.add_argument('--output-dir',type=Path,default=ROOT,help='Use a new directory for an independent benchmark; existing jobs are resumed.')
 a=p.parse_args()
 if a.rounds<1:p.error('--rounds must be positive')
 ROOT=a.output_dir
 token=a.token_file.read_text().strip() if a.token_file else os.environ['QAI_HUB_API_TOKEN']
 client=hub.Client(hub.ClientConfig(api_token=token));device=hub.Device('Samsung Galaxy S24')
 ROOT.mkdir(parents=True,exist_ok=True);manifest=ROOT/'jobs.json'
 state=json.loads(manifest.read_text()) if manifest.exists() else dict(device=device.name,shape=[1,4,360,640],sdk=str(hub.__version__),compile_options=COMPILE,profile_options=PROFILE,calibration='Platform-generated random calibration; performance only',models={})
 def save():
  temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(manifest)
 for label,exp in EXPERIMENTS.items():
  record=state['models'].setdefault(label,{})
  folder=ROOT/label;folder.mkdir(exist_ok=True)
  if 'compile_job' not in record:
   candidates=list((Path('experiments')/exp/'onnx').glob('*best*1x4x360x640_qai_w8a8_source.onnx'))
   if len(candidates)!=1:raise RuntimeError(f'{label}: expected one source ONNX')
   source=candidates[0];record.update(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint=str(Path('experiments')/exp/'checkpoints/best.pth'))
   job=client.submit_compile_job(str(source),device=device,name=f'SID_W8A8_{label}_360x640',options=COMPILE)
   record['compile_job']=job.job_id;save();print(label,'submitted compile',job.job_id,flush=True)
  job=client.get_job(record['compile_job']);status=job.get_status();record['compile_status']=str(status);record['compile_code']=status.code;save()
  print(label,'compile',status,flush=True)
  if status.failure and not record.get('failure_logs'):
   job.download_job_logs(str(folder/'compile_logs'));record['failure_logs']=True;save()
  if not status.success:continue
  if not (folder/'compile_logs').exists():job.download_job_logs(str(folder/'compile_logs'))
  target=job.get_target_model();record['target_model']=target.model_id
  if not (folder/'model.dlc').exists():target.download(str(folder/'model.dlc'))
  profiles=record.setdefault('profile_jobs',[])
  for round_index in range(a.rounds):
   if len(profiles)<=round_index:
    prof=client.submit_inference_job(target,device=device,inputs=None,profile=True,name=f'SID_W8A8_{label}_round{round_index+1}',options=PROFILE)
    profiles.append({'id':prof.job_id});save();print(label,'submitted profile',prof.job_id,flush=True)
   entry=profiles[round_index];prof=client.get_job(entry['id']);status=prof.get_status();entry['status']=str(status);entry['code']=status.code
   if status.success and not entry.get('downloaded'):
    prof.download_profile(str(folder/f'profile_{round_index+1}.json'));entry['downloaded']=True
   if status.success and not (folder/f'profile_logs_{round_index+1}').exists():
    prof.download_job_logs(str(folder/f'profile_logs_{round_index+1}'))
   if status.failure and not entry.get('failure_logs'):
    prof.download_job_logs(str(folder/f'profile_logs_{round_index+1}'));entry['failure_logs']=True
   save();print(label,'profile',round_index+1,status,flush=True)
 save()

if __name__=='__main__':
 import sys
 while True:
  main()
  if '--watch' not in sys.argv:break
  current=json.loads((ROOT/'jobs.json').read_text())
  finished=all(r.get('compile_code')=='FAILED' or (r.get('profile_jobs') and all(p.get('code') in ('SUCCESS','FAILED') for p in r['profile_jobs'])) for r in current['models'].values())
  if len(current['models'])==len(EXPERIMENTS) and finished:break
  time.sleep(30)
