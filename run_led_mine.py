"""Run sibling LED's synthetic-noise protocol with architectures from this repo."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import yaml

ROOT = Path(__file__).resolve().parent


def configure_cuda(config, override=None):
    """Select visible GPUs before importing LED/PyTorch (top-level config only)."""
    with config.open(encoding='utf-8') as handle:
        settings = yaml.safe_load(handle)
    configured = settings.get('cuda_visible_devices', '0')
    selected = override if override is not None else os.environ.get(
        'CUDA_VISIBLE_DEVICES', configured)
    if selected is None or isinstance(selected, (bool, list, dict)):
        raise ValueError('cuda_visible_devices must be a GPU ID string, e.g. "0" or "1"')
    selected = str(selected).strip()
    os.environ['CUDA_VISIBLE_DEVICES'] = selected
    print(f'CUDA_VISIBLE_DEVICES={selected}', flush=True)
    return selected


def load_options(config, led_root):
    from led.utils.options import yaml_load, _merge_from_base
    previous = Path.cwd()
    try:
        os.chdir(led_root)
        opt = _merge_from_base(yaml_load(str(config)), str(config))
    finally:
        os.chdir(previous)
    # Remove inherited base references: the archived YAML is fully resolved.
    opt.pop('base', None)
    for dataset in opt['datasets'].values():
        for key in ('dataroot', 'data_pair_list'):
            if dataset.get(key):
                dataset[key] = str((led_root / dataset[key]).resolve())
    for key in ('predefined_noise_g', 'CRF'):
        if opt['path'].get(key):
            opt['path'][key] = str((led_root / opt['path'][key]).resolve())
    return opt


def prepare_options(opt, mode, checkpoint=None):
    opt = deepcopy(opt)
    if mode == 'test':
        if checkpoint is None:
            raise ValueError('--mode test requires --checkpoint (an LED params checkpoint)')
        opt['datasets'].pop('train', None)
        opt.pop('noise_g', None)
        opt.pop('train', None)
        opt['path'].update(pretrain_network_g=str(Path(checkpoint).resolve()),
                           param_key_g='params', resume_state=None)
        opt['val']['save_img'] = False
    return opt


def check_paths(opt):
    paths = []
    for dataset in opt['datasets'].values():
        paths.extend(dataset[k] for k in ('dataroot', 'data_pair_list') if dataset.get(k))
    paths.extend(opt['path'][k] for k in ('CRF', 'predefined_noise_g', 'pretrain_network_g')
                 if opt['path'].get(k))
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--led-root', type=Path, default=ROOT.parent / 'LED-mine')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--cuda-visible-devices', help='Override GPU visibility, e.g. 0 or 1')
    args, extra = parser.parse_known_args()
    if args.mode == 'train' and args.checkpoint:
        parser.error('--checkpoint is for test; configure path.pretrain_network_g for warm starts')
    led_root = args.led_root.resolve()
    config = args.config.resolve()
    selected_cuda = configure_cuda(config, args.cuda_visible_devices)
    sys.path.insert(0, str(led_root))
    from led.utils.registry import ARCH_REGISTRY
    from models.led_mine_adapter import CurrentRepoDenoiser
    ARCH_REGISTRY.register()(CurrentRepoDenoiser)
    opt = prepare_options(load_options(config, led_root), args.mode, args.checkpoint)
    opt['cuda_visible_devices'] = selected_cuda
    check_paths(opt)
    if args.check_only:
        from led.archs import build_network
        net = build_network(opt['network_g'])
        print(f"{opt['name']}: paths OK; parameters={sum(p.numel() for p in net.parameters()):,}")
        print(yaml.safe_dump(plain(opt), sort_keys=False))
        return
    # LED parses argv itself; keep its resume/distributed/override flags available.
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix='ledmine_config_') as temporary:
        resolved = Path(temporary) / config.name
        resolved.write_text(yaml.safe_dump(plain(opt), sort_keys=False), encoding='utf-8')
        sys.argv = [sys.argv[0], '-opt', str(resolved), *extra]
        if args.mode == 'train':
            from led.train import train_pipeline
            train_pipeline(str(ROOT))
        else:
            from led.test import test_pipeline
            test_pipeline(str(ROOT))


if __name__ == '__main__':
    main()
