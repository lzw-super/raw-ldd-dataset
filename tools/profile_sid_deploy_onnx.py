"""Count parameters and arithmetic of previously exported SID deployment graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.haar_soft_export import HaarSoftExport
from tools.profile_hf_cnn_onnx import count
from utils.model_factory import build_denoiser_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-reports', type=Path, nargs='+', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    reports = {}
    for path in args.export_reports:
        incoming = json.loads(path.read_text())
        if reports.keys() & incoming.keys():
            raise ValueError('Duplicate model labels in export reports')
        reports.update(incoming)
    results = {}
    for name in args.models:
        report = reports[name]
        config = report['settings']
        source = Path(report['source'])
        checkpoint = Path(config['checkpoint'])
        for path, expected in [(source, report['source_sha256']),
                               (checkpoint, report['checkpoint_sha256'])]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Artifact changed: {path}')
        model, metadata = build_denoiser_from_checkpoint(checkpoint, 'cpu')
        frozen_thresholds = 0
        if config['kind'] == 'haar':
            model.wavelet = HaarSoftExport(model.wavelet, config['hf_layout'], config['hf_formula'])
            frozen_thresholds = model.wavelet.thresholds.numel()
        elif config['kind'] not in ('learning_dwt_repncb', 'mrlfn', 'splitternet', 'unet', 'nafnet'):
            raise ValueError(f'Unsupported export wrapper: {config["kind"]}')
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != report['deployed_parameter_count']:
            raise ValueError(f'{name}: deployment parameter count changed')
        row = count(source)
        row.update(shape=report['shape'], config=report['config'],
                   checkpoint=str(checkpoint), checkpoint_sha256=report['checkpoint_sha256'],
                   onnx_sha256=report['source_sha256'], metadata=metadata,
                   deployed_nn_parameters=parameters, frozen_learned_thresholds=frozen_thresholds,
                   deployed_learned_coefficients=parameters + frozen_thresholds,
                   convolution_flops_2_per_mac=2 * row['conv_macs'])
        results[name] = row
        print(name, parameters, '+', frozen_thresholds, 'coefficients;',
              row['conv_macs'] / 1e9, 'GMAC;', row['arithmetic_ops'] / 1e9, 'GOp')
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        convention='Deployment parameters exclude fixed transform buffers; learned frozen thresholds '
                   'are listed separately. Dense Conv/ConvTranspose includes fixed kernels and zero '
                   'entries; 1 MAC=2 ops; bias=1/output; PReLU/LeakyReLU=2/output; '
                   'Sigmoid=4/output; Pow/Sqrt=1/output; mean=input elements; '
                   'max reduction=input-output elements; data movement=0 arithmetic ops. '
                   'Algorithmic estimate, not hardware instruction count or DLC storage size.',
        results=results), indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
