"""Evaluate classical wavelet BayesShrink on exactly the SIDEvalDataset pairs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.real_dataset import SIDEvalDataset
from models.wavelet_denoiser import WaveletDenoiser
from utils.utils import ELDIlluminanceCorrect, PMN_metric
from utils.utils import rggb_to_srgb


def metrics(pred, gt, correct=False):
    if correct:
        pred = ELDIlluminanceCorrect().correct(pred, gt)
    pred = pred.clamp(0, 1)
    return {k: float(v) for k, v in PMN_metric(pred, gt.clamp(0, 1)).items()}, pred


def save_comparison(path, tensors, data):
    rgbs = [rggb_to_srgb(t[0].clamp(0, 1).permute(1, 2, 0).numpy(),
                        wb=data['wb'].numpy(), ccm=data['ccm'].numpy(),
                        gamma=3, format='rggb', uint8=True) for t in tensors]
    w, h = 640, 440
    canvas = Image.new('RGB', (3*w, h+360), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (rgb, label) in enumerate(zip(rgbs, ('Input', 'Wavelet (no GT correction)', 'GT'))):
        image = Image.fromarray(rgb)
        image.thumbnail((w, h-30))
        canvas.paste(image, (i*w, 30))
        draw.text((i*w+10, 8), label, fill='black')
        height, width = rgb.shape[:2]
        y, x = height//2-128, width//2-128
        crop = Image.fromarray(rgb[y:y+256, x:x+256]).resize((320,320))
        canvas.paste(crop, (i*w, h+30))
        draw.text((i*w+10, h+8), 'Center crop, 256x256 RGB pixels', fill='black')
    canvas.save(path)


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    denoiser = WaveletDenoiser(args.levels, args.wavelet)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    for ratio in args.ratios:
        dataset = SIDEvalDataset(clip_low=False, clip_high=True, eval_ratio=ratio, max_items=args.max_samples)
        for idx in range(len(dataset)):
            data = dataset[idx]
            gt, noisy = data['hr'], data['lr']
            start = time.perf_counter()
            pred = torch.from_numpy(denoiser(noisy[0].numpy())).unsqueeze(0)
            seconds = time.perf_counter()-start
            raw_input, _ = metrics(noisy, gt)
            input_corrected, _ = metrics(noisy, gt, True)
            raw_pred, _ = metrics(pred, gt)
            corrected, _ = metrics(pred, gt, True)
            row = dict(ratio=ratio, index=idx, name=data['name'],
                       short=dataset.data_info[idx]['short'][0], long=dataset.data_info[idx]['long'],
                       actual_ratio=float(dataset.data_info[idx]['ratio'][0]),
                       input=raw_input, input_corrected=input_corrected,
                       wavelet=raw_pred, wavelet_corrected=corrected, seconds=seconds)
            rows.append(row)
            with (output/'per_image.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            if idx < args.visualize_samples:
                save_comparison(output/f'x{ratio}_{idx:02d}.png', (noisy, pred, gt), data)
            print(f'x{ratio} {idx+1}/{len(dataset)} input={raw_input["psnr"]:.3f} '
                  f'wavelet={raw_pred["psnr"]:.3f} corrected={corrected["psnr"]:.3f}', flush=True)
        selected = [r for r in rows if r['ratio'] == ratio]
        summary = dict(ratio=ratio, count=len(selected))
        for key in ('input', 'input_corrected', 'wavelet', 'wavelet_corrected'):
            summary[key] = {metric: float(np.mean([r[key][metric] for r in selected])) for metric in ('psnr','ssim')}
        summary['mean_seconds'] = float(np.mean([r['seconds'] for r in selected]))
        summaries.append(summary)
        (output/'results.json').write_text(json.dumps(dict(config=vars(args), summaries=summaries), indent=2)+'\n')
        del dataset
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--wavelet', default='sym4', help='Orthogonal PyWavelets basis, e.g. haar/db2/sym4/coif1')
    parser.add_argument('--ratios', type=int, nargs='+', choices=(100,250,300), default=[100,250,300])
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--visualize-samples', type=int, default=3, help='First N images per ratio')
    parser.add_argument('--output-dir', default='test_res/wavelet_sym4_l3')
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('--max-samples must be positive')
    if (Path(args.output_dir)/'per_image.jsonl').exists():
        parser.error('Output already contains results; choose a fresh --output-dir')
    main(args)
