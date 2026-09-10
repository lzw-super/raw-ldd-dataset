from argparse import Namespace

import numpy as np
import pytest
import torch

from datasets.sid_real_validation import SIDRealValidationDataset
from train_sid_sony import real_validate, save_checkpoint, load_checkpoint, prune_checkpoints
from utils.utils import ELDIlluminanceCorrect, PMN_metric


def test_real_validation_matches_test_script_and_averages_image_psnr():
    torch.manual_seed(7)
    batches = []
    expected = []
    for noise in (0.01, 0.2):
        hr = torch.rand(1, 4, 12, 12)
        lr = hr + noise * torch.randn_like(hr)
        batches.append({'hr': hr.unsqueeze(1), 'lr': lr.unsqueeze(1)})
        pred = ELDIlluminanceCorrect().correct(lr, hr).clamp(0, 1)
        expected.append(PMN_metric(pred, hr)['psnr'])
    model = torch.nn.Identity().train()
    result = real_validate(model, batches, Namespace(), torch.device('cpu'))
    assert result['real_psnr'] == pytest.approx(np.mean(expected))
    assert result['validation_pairs'] == 2
    assert model.training


def test_checkpoint_best_metadata_and_pruning(tmp_path):
    model = torch.nn.Conv2d(4, 4, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    for filename in ('best.pth', 'latest.pth', 'epoch_0003.pth'):
        save_checkpoint(tmp_path / filename, model, optimizer, scheduler, scaler,
                        3, 12, Namespace(), 35.7, 2)
    state = torch.load(tmp_path / 'latest.pth', map_location='cpu')
    assert state['best_psnr'] == 35.7
    assert state['best_epoch'] == 2
    assert load_checkpoint(tmp_path / 'best.pth', model, optimizer, scheduler, scaler) == (3, 12)
    prune_checkpoints(tmp_path, 0)
    assert (tmp_path / 'best.pth').exists()
    assert (tmp_path / 'latest.pth').exists()
    assert not (tmp_path / 'epoch_0003.pth').exists()


def test_real_pair_preprocessing_keeps_negative_input():
    dataset = SIDRealValidationDataset.__new__(SIDRealValidationDataset)
    dataset.wl, dataset.bl = 16383, 512
    dataset.clip_low, dataset.clip_high = -float('inf'), 1
    dataset.data_info = [dict(name='20001', ratio=[100], ISO=200, wb=np.ones(4), ccm=np.eye(3))]
    dataset.load_raw_pair = lambda _: (np.full((8, 8), 1000, np.float32), np.full((8, 8), 510, np.float32), 0)
    dataset.get_darkshading = lambda _: 1
    pair = dataset[0]
    assert pair['lr'].shape == (1, 4, 4, 4)
    assert pair['lr'].mean().item() == pytest.approx(-300 / (16383 - 512))
    assert pair['hr'].mean().item() == pytest.approx(488 / (16383 - 512))
