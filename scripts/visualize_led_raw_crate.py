"""Single-frame LED model inference on headerless RAW-crate, with fixed ISP."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_led_mine import configure_cuda,load_options


def render(packed, wb):
    import cv2
    h,w,_=packed.shape
    p=np.clip(packed*np.asarray([wb[0],1,wb[2],1],np.float32),0,1)
    mosaic=np.empty((h*2,w*2),np.float32)
    for c,(y,x) in enumerate([(0,0),(0,1),(1,1),(1,0)]):mosaic[y::2,x::2]=p[...,c]
    bgr=cv2.demosaicing((mosaic*65535).astype(np.uint16),cv2.COLOR_BayerRGGB2BGR).astype(np.float32)/65535
    return (255*np.clip(np.where(bgr<=.0031308,12.92*bgr,1.055*bgr**(1/2.4)-.055),0,1)).astype(np.uint8)


def panel(a,b,label,width=768,crop=False,model_name='Rep-NCB'):
    import cv2
    if crop:
        h,w=a.shape[:2];a=a[h//2-256:h//2+256,w//2-256:w//2+256];b=b[h//2-256:h//2+256,w//2-256:w//2+256]
    tiles=[]
    for im,title in [(a,'Noisy input'),(b,model_name+' latest')]:
        h=round(im.shape[0]*width/im.shape[1]);im=cv2.resize(im,(width,h),interpolation=cv2.INTER_NEAREST if crop else cv2.INTER_AREA)
        im=cv2.copyMakeBorder(im,36,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(im,title+' | '+label,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        tiles.append(im)
    return np.concatenate(tiles,axis=1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=ROOT.parent/'RViDeformer/mydata_test/RAW-crate/RAW')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'experiments/ledmine_sid_repncb/models/net_g_latest.pth')
    p.add_argument('--config',type=Path,default=ROOT/'configs/ledmine_sid_repncb.yaml')
    p.add_argument('--output',type=Path)
    p.add_argument('--ffmpeg',default='/opt/conda/envs/RViDeformer/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
    p.add_argument('--reuse-predictions',action='store_true')
    p.add_argument('--input-gain',type=float,default=64)
    p.add_argument('--cuda-visible-devices',default='0')
    a=p.parse_args();configure_cuda(a.config,a.cuda_visible_devices)
    import cv2,torch,subprocess
    from models.led_mine_adapter import CurrentRepoDenoiser
    led=ROOT.parent/'LED-mine';sys.path.insert(0,str(led))
    opt=load_options(a.config.resolve(),led)
    cfg=dict(opt['network_g']);cfg.pop('type')
    model_name={'repncb':'Rep-NCB','nafnet':'NAFNet','mrlfn':'MRLFN','splitternet':'SplitterNet'}[cfg['architecture']]
    a.output=a.output or a.checkpoint.resolve().parent.parent/f'RAW-crate_latest_gain{a.input_gain:g}'
    model=CurrentRepoDenoiser(**cfg)
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['params'],strict=True)
    model=model.cuda().eval();torch.set_num_threads(4)
    files=sorted(a.input.glob('*.raw'))
    assert files and a.input_gain>0
    if a.output.exists() and any(a.output.iterdir()) and not a.reuse_predictions:raise FileExistsError(a.output)
    for sub in ['noisy','denoised','comparison','detail']:(a.output/sub).mkdir(parents=True,exist_ok=True)
    result=np.lib.format.open_memmap(a.output/'denoised_packed.npy',mode='r+' if a.reuse_predictions else 'w+',dtype='float32',shape=(len(files),1536,2048,4))
    wb=None;stats=[];writers=[]
    try:
        for i,f in enumerate(files):
            raw=np.fromfile(f,dtype='<u2');assert raw.size==4096*3072
            raw=raw.reshape(3072,4096)
            packed=np.stack([np.maximum(raw[y::2,x::2].astype(np.float32)-64,0)/959 for y,x in [(0,0),(0,1),(1,1),(1,0)]],axis=-1)
            if wb is None:
                means=packed.mean(axis=(0,1));g=(means[1]+means[3])/2
                wb=[float(np.clip(g/max(means[0],1e-12),.25,4)),1.,float(np.clip(g/max(means[2],1e-12),.25,4))]
            inp=np.clip(packed*a.input_gain,0,1)
            with torch.no_grad():
                prediction=result[i].copy() if a.reuse_predictions else model(torch.from_numpy(inp.transpose(2,0,1).copy())[None].cuda())[0].permute(1,2,0).cpu().numpy()
            assert np.isfinite(prediction).all()
            result[i]=prediction
            noisy,denoised=render(inp,wb),render(prediction,wb)
            full=panel(noisy,denoised,f'{i:04d}',model_name=model_name)
            detail=panel(noisy,denoised,f'{i:04d}',512,True,model_name)
            for folder,im in [('noisy',noisy),('denoised',denoised),('comparison',full),('detail',detail)]:
                assert cv2.imwrite(str(a.output/folder/f'frame_{i:04d}.png'),im)
            stats.append({'file':str(f.resolve()),'raw_min':int(raw.min()),'raw_max':int(raw.max()),'input_saturated_fraction':float(np.mean(packed*a.input_gain>=1)),'prediction_min':float(prediction.min()),'prediction_max':float(prediction.max())})
            print(f'{i+1}/{len(files)} {f.name}',flush=True)
    finally:
        for w in writers:w.close()
        result.flush()
    for folder,name in [('comparison','comparison_5fps.mp4'),('detail','detail_5fps.mp4')]:
        subprocess.run([a.ffmpeg,'-v','error','-framerate','5','-i',str(a.output/folder/'frame_%04d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(a.output/name)],check=True)
    report={'model':model_name,'config':str(a.config.resolve()),'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'frames':len(files),'raw_shape':[3072,4096],'packed_shape':[1536,2048,4],'packed_order':['R','G1','B','G2'],'assumptions':{'bayer':'RGGB','black':64,'white':1023,'dtype':'little-endian uint16','input_gain':a.input_gain},'white_balance_rgb':wb,'isp':'fixed first-frame gray-world WB, bilinear demosaic, identity CCM, sRGB transfer','output':'float32 amplified-domain model output, no divide-back; negative values retained in npy','mode':'single-frame full-resolution FP32, no tiling, no temporal feedback','fps':'5 for presentation only; capture fps unknown','clean_reference':False,'stats':stats,'status':'complete'}
    (a.output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    body=f'<meta charset="utf-8"><h1>RAW-crate · {model_name} latest</h1><p>{len(files)} 帧；左为输入，右为去噪。完整分辨率单帧推理；5fps 仅为预览播放速度。RGGB、黑64、白1023为假设，入网增益{a.input_gain:g}，固定白平衡及单位CCM；无干净GT，不计算PSNR/SSIM。</p>'
    for name in ['comparison_5fps.mp4','detail_5fps.mp4']:body+=f'<video src="{name}" controls loop style="max-width:100%"></video>'
    for i in [0,len(files)//2,len(files)-1]:
        body+=f'<h2>Frame {i:04d}</h2><img src="comparison/frame_{i:04d}.png" style="max-width:100%"><img src="detail/frame_{i:04d}.png" style="max-width:100%">'
    (a.output/'report.html').write_text(body)

if __name__=='__main__':main()
