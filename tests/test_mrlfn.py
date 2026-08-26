from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from losses.mrlfn_loss import RawReconstructionChromaticLoss
from models.mrlfn_arch import MRLFN
from utils.model_factory import build_denoiser_from_checkpoint


def test_n4_d16_deployment_equivalence_and_size() -> None:
    torch.manual_seed(7)
    training_model = MRLFN(feature_channels=16, num_blocks=4).eval()
    deployed_model = training_model.deploy().eval()
    sample = torch.randn(1, 4, 31, 37)
    with torch.no_grad():
        expected = training_model(sample)
        actual = deployed_model(sample)
    assert sum(parameter.numel() for parameter in training_model.parameters()) == 38_580
    assert sum(parameter.numel() for parameter in deployed_model.parameters()) == 32_244
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_composite_loss_uses_current_repository_channel_layout() -> None:
    pred = torch.zeros(1, 4, 1, 1)
    target = torch.zeros_like(pred)
    pred[:, 3] = 1.0  # B in this repository's [R,G1,G2,B] packing.
    criterion = RawReconstructionChromaticLoss(channel_order=(0, 1, 3, 2))
    total, raw, chromatic = criterion.per_sample_components(pred, target)
    torch.testing.assert_close(raw, torch.tensor([0.25]))
    torch.testing.assert_close(chromatic, torch.tensor([1.0]))
    torch.testing.assert_close(total, torch.tensor([0.55]))


def test_factory_prefers_deploy_weights(tmp_path: Path) -> None:
    training_model = MRLFN(feature_channels=16, num_blocks=4).eval()
    deployed_model = training_model.deploy().eval()
    checkpoint = tmp_path / "mrlfn.pth"
    torch.save(
        {
            "model": training_model.state_dict(),
            "model_deploy": deployed_model.state_dict(),
            "args": {"model": "mrlfn", "feature_channels": 16, "num_blocks": 4, "model_bias": True},
        },
        checkpoint,
    )
    loaded, meta = build_denoiser_from_checkpoint(str(checkpoint), device="cpu")
    assert getattr(loaded, "deploy_mode", False)
    assert meta["graph_state"] == "deploy"
    assert meta["weight_source"] == "model_deploy"
