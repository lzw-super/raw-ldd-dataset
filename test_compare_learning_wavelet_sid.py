"""Paired comparison: learned thresholds, initialization, and two BayesShrink boundaries."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.real_dataset import SIDEvalDataset
from learning_wt.learning_dwt import LearningDWT, learning_dwt_kwargs
from models.wavelet_denoiser import WaveletDenoiser
from utils.model_factory import build_denoiser_from_checkpoint
from utils.utils import ELDIlluminanceCorrect, PMN_metric, rggb_to_srgb


def measure(pred, gt):
    corrected = ELDIlluminanceCorrect().correct(pred, gt).clamp(0,1)
    return {'raw': {k:float(v) for k,v in PMN_metric(pred.clamp(0,1),gt).items()},
            'corrected': {k:float(v) for k,v in PMN_metric(corrected,gt).items()},
            'channel_bias': (pred.clamp(0,1)-gt).mean((0,2,3)).tolist()}


def comparison_image(path, images, data):
    width, top, bottom = 480, 355, 300
    canvas=Image.new('RGB',(len(images)*width,top+bottom),'white')
    draw=ImageDraw.Draw(canvas)
    for i,(label,tensor) in enumerate(images.items()):
        rgb=rggb_to_srgb(tensor[0].clamp(0,1).permute(1,2,0).numpy(),
                         wb=data['wb'].numpy(),ccm=data['ccm'].numpy(),gamma=3,format='rggb',uint8=True)
        image=Image.fromarray(rgb); image.thumbnail((width,top-30))
        canvas.paste(image,(i*width,30));draw.text((i*width+5,8),label,fill='black')
        h,w=rgb.shape[:2]; y,x=h//2-128,w//2-128
        canvas.paste(Image.fromarray(rgb[y:y+256,x:x+256]),(i*width,top+30))
        draw.text((i*width+5,top+8),'Center crop (no GT correction)',fill='black')
    canvas.save(path)


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    if (output/'per_image.jsonl').exists():
        raise ValueError('Choose a fresh output directory')
    checkpoint=torch.load(args.checkpoint,map_location='cpu')
    options=checkpoint['args']
    model,meta=build_denoiser_from_checkpoint(args.checkpoint,args.device)
    if meta['model'] != 'learning_dwt':
        raise ValueError('Expected a learning_dwt checkpoint')
    kwargs=learning_dwt_kwargs(options)
    if kwargs['wavelet'] != 'sym4' or kwargs['levels'] != 3:
        raise ValueError('This comparison fixes sym4 and three LL-recursive levels')
    torch.manual_seed(options.get('seed',2026))
    initial=LearningDWT(**kwargs).to(args.device).eval()
    baselines={mode:WaveletDenoiser(3,'sym4',mode) for mode in ('symmetric','periodization')}
    trained_steps = checkpoint.get('global_step')
    del checkpoint
    rows=[]; summaries=[]
    for ratio in args.ratios:
        dataset=SIDEvalDataset(eval_ratio=ratio,max_items=args.max_samples,clip_low=False,clip_high=True)
        for index in range(len(dataset)):
            data=dataset[index]; noisy,gt=data['lr'],data['hr']
            images={'input':noisy}; times={}
            for mode,denoiser in baselines.items():
                start=time.perf_counter()
                images['bayes_'+mode]=torch.from_numpy(denoiser(noisy[0].numpy())).unsqueeze(0)
                times['bayes_'+mode]=time.perf_counter()-start
            gpu_input=noisy.to(args.device)
            for name,net in (('initial',initial),('learned',model)):
                if str(args.device).startswith('cuda'):torch.cuda.synchronize()
                start=time.perf_counter();pred=net(gpu_input)
                if str(args.device).startswith('cuda'):torch.cuda.synchronize()
                times[name]=time.perf_counter()-start
                images[name]=pred.cpu()
            row=dict(ratio=ratio,index=index,name=data['name'],
                     short=dataset.data_info[index]['short'][0],long=dataset.data_info[index]['long'],
                     methods={name:measure(pred,gt) for name,pred in images.items()},seconds=times)
            rows.append(row)
            with (output/'per_image.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
            if index<args.visualize_samples:
                comparison_image(output/f'x{ratio}_{index:02d}.png',
                                 dict([(name,images[name]) for name in ('input','bayes_symmetric','learned')]+[('GT',gt)]),data)
            print(f'x{ratio} {index+1}/{len(dataset)} '+ ' '.join(f'{k}={v["raw"]["psnr"]:.3f}' for k,v in row['methods'].items()),flush=True)
        selected=[r for r in rows if r['ratio']==ratio]
        summary={'ratio':ratio,'count':len(selected),'methods':{}}
        for method in selected[0]['methods']:
            summary['methods'][method]={mode:{metric:float(np.mean([r['methods'][method][mode][metric] for r in selected]))
                                               for metric in ('psnr','ssim')} for mode in ('raw','corrected')}
            summary['methods'][method]['channel_bias']=np.mean([r['methods'][method]['channel_bias'] for r in selected],axis=0).tolist()
        summaries.append(summary)
        result=dict(config=vars(args),model=meta,training_global_step=trained_steps,summaries=summaries,
                    notes='Matched local SIDEvalDataset index (mixed prefixes 1/2); no test tuning. Uncorrected and GT-corrected RAW metrics. CPU Bayes vs GPU CNN timings are not hardware-matched.')
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        del dataset


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--ratios',type=int,nargs='+',choices=[100,250,300],default=[100,250,300])
    parser.add_argument('--max-samples',type=int,default=None)
    parser.add_argument('--visualize-samples',type=int,default=3)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.max_samples is not None and args.max_samples<1:parser.error('--max-samples must be positive')
    main(args)
