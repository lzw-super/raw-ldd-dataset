"""Evaluate an LED checkpoint and save exact metrics beside its experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import yaml
from run_led_mine import ROOT, configure_cuda, load_options, prepare_options, check_paths, plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--led-root', type=Path, default=ROOT.parent / 'LED-mine')
    parser.add_argument('--cuda-visible-devices')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--split', choices=['led-valid', 'sid-test'], default='sid-test')
    parser.add_argument('--sid-test-list', type=Path, default=ROOT.parent.parent / 'Datasets/SID-sub/Sony_test_list.txt')
    args = parser.parse_args()
    config, checkpoint, led_root = args.config.resolve(), args.checkpoint.resolve(), args.led_root.resolve()
    selected = configure_cuda(config, args.cuda_visible_devices)
    sys.path.insert(0, str(led_root))
    import torch
    from led.utils.registry import ARCH_REGISTRY
    from models.led_mine_adapter import CurrentRepoDenoiser
    from led.utils.options import parse_options
    from led.utils import get_root_logger
    from led.data import build_dataset, build_dataloader
    from led.models import build_model
    ARCH_REGISTRY.register()(CurrentRepoDenoiser)
    opt = prepare_options(load_options(config, led_root), 'test', checkpoint)
    check_paths(opt)
    output = args.output_dir or checkpoint.parent.parent / ('evaluation_latest_sid_test' if args.split == 'sid-test' else 'evaluation_latest')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_test_list = None
    if args.split == 'sid-test':
        source_test_list = args.sid_test_list.resolve()
        groups = {100: [], 250: [], 300: []}
        for line in source_test_list.read_text().splitlines():
            if not line.strip():
                continue
            short, long = line.split()[:2]
            exposure = lambda p: float(re.search(r'_(\d+(?:\.\d+)?)s', Path(p).name)[1])
            ratio = round(min(exposure(long) / exposure(short), 300))
            if ratio not in groups:
                raise ValueError(f'Unexpected ratio: {ratio}')
            groups[ratio].append(f'./short/{Path(short).name} ./long/{Path(long).name}')
        for ratio, rows in groups.items():
            if not rows:
                raise ValueError(f'Empty test ratio {ratio}')
            pairs = output / f'sid_test_{ratio}.txt'
            pairs.write_text('\n'.join(rows)+'\n')
            ds = opt['datasets'][f'val_{ratio}']
            ds['data_pair_list'] = str(pairs)
            ds['name'] = f'SIDSonyOfficialTest{ratio}'
            for row in rows:
                for relative in row.split():
                    path = Path(ds['dataroot']) / (relative + '.npz')
                    if not path.is_file():
                        raise FileNotFoundError(path)
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory() as temporary:
        resolved = Path(temporary) / 'eval.yaml'
        resolved.write_text(yaml.safe_dump(plain(opt), sort_keys=False))
        sys.argv = [sys.argv[0], '-opt', str(resolved)]
        opt, _ = parse_options(str(ROOT), is_train=False)
    opt['val']['pbar'] = False
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    logger = get_root_logger(logger_name='led', log_file=str(output / 'test.log'))
    model = build_model(opt)
    report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'config': str(config), 'cuda_visible_devices': selected,
              'split': args.split, 'source_test_list': str(source_test_list) if source_test_list else None,
              'protocol': plain(opt['val']), 'results': {}, 'status': 'running'}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    for key, dataset_opt in sorted(opt['datasets'].items()):
        dataset = build_dataset(dataset_opt)
        loader = build_dataloader(dataset, dataset_opt, num_gpu=opt['num_gpu'], dist=False,
                                  sampler=None, seed=opt['manual_seed'])
        logger.info('Evaluating %s: %d pairs', key, len(dataset))
        model.validation(loader, current_iter='latest', tb_logger=None, save_img=False)
        report['results'][key] = {'count': len(dataset), 'pair_list': dataset_opt['data_pair_list'],
                                 **{k: float(v) for k,v in model.metric_results.items()}}
        (output / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    values = list(report['results'].values())
    report['mean_three_ratios'] = {k: sum(v[k] for v in values)/len(values) for k in ('psnr','ssim')}
    report['mean_all_pairs'] = {k: sum(v[k]*v['count'] for v in values)/sum(v['count'] for v in values) for k in ('psnr','ssim')}
    report['status'] = 'complete'
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    lines = ['# Latest 权重 LED SID 指标评估', '', f'权重：`{checkpoint}`', '',
             ('SID Sony_test_list.txt 完整测试划分（598 对，其中 93 对被 LED 用于训练验证）。' if args.split == 'sid-test' else 'LED valid 子集（93 对）。') + ' RAW 域，亮度校正开启，裁边 2 像素；完整图像 FP32 推理。', '',
             '| 档位 | 数量 | PSNR (dB) | SSIM |', '|---|---:|---:|---:|']
    for key,v in report['results'].items():
        lines.append(f"| {key} | {v['count']} | {v['psnr']:.6f} | {v['ssim']:.6f} |")
    for key,label in [('mean_three_ratios','三档等权平均'),('mean_all_pairs','全部图像等权平均')]:
        v=report[key];lines.append(f"| {label} | — | {v['psnr']:.6f} | {v['ssim']:.6f} |")
    (output / '指标汇总.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
