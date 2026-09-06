from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from losses.mrlfn_loss import RawReconstructionChromaticLoss
from models.mrlfn_arch import MRLFN, PackedBayerDepthToSpace, PackedBayerSpaceToDepth
from train_sid_sony import build_scheduler
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


def test_paper_s2d_uses_single_plane_bayer_channel_order_and_is_invertible() -> None:
    # The four inputs encode a 4x4 mosaic split into row-major Bayer planes.
    packed = torch.tensor(
        [
            [
                [[0.0, 2.0], [8.0, 10.0]],
                [[1.0, 3.0], [9.0, 11.0]],
                [[4.0, 6.0], [12.0, 14.0]],
                [[5.0, 7.0], [13.0, 15.0]],
            ]
        ]
    )
    reduced = PackedBayerSpaceToDepth(4)(packed)
    assert reduced.shape == (1, 16, 1, 1)
    torch.testing.assert_close(reduced.flatten(), torch.arange(16, dtype=packed.dtype))
    torch.testing.assert_close(PackedBayerDepthToSpace(4)(reduced), packed)


def test_paper_aligned_model_has_figure6_shapes_and_deploy_equivalence() -> None:
    torch.manual_seed(11)
    training_model = MRLFN(
        feature_channels=32,
        num_blocks=4,
        space_to_depth_factor=4,
    ).eval()
    assert training_model.shallow_conv.in_channels == 16
    assert training_model.shallow_fusion.in_channels == 16
    assert training_model.output_conv.out_channels == 16
    assert sum(parameter.numel() for parameter in training_model.deploy().parameters()) == 130_672

    observed_stem_shapes: list[tuple[int, ...]] = []
    handle = training_model.shallow_conv.register_forward_pre_hook(
        lambda _module, inputs: observed_stem_shapes.append(tuple(inputs[0].shape))
    )
    sample = torch.randn(1, 4, 18, 22)
    with torch.no_grad():
        expected = training_model(sample)
    handle.remove()
    with torch.no_grad():
        actual = training_model.deploy().eval()(sample)
    assert observed_stem_shapes == [(1, 16, 9, 11)]
    assert expected.shape == sample.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_paper_s2d_rejects_nondivisible_mosaic_shape() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        PackedBayerSpaceToDepth(4)(torch.randn(1, 4, 17, 18))


def test_factory_restores_paper_s2d_checkpoint(tmp_path: Path) -> None:
    training_model = MRLFN(feature_channels=32, num_blocks=4, space_to_depth_factor=4).eval()
    checkpoint = tmp_path / "mrlfn_paper_s2d.pth"
    torch.save(
        {
            "model": training_model.state_dict(),
            "model_deploy": training_model.deploy().state_dict(),
            "args": {
                "model": "mrlfn",
                "feature_channels": 32,
                "num_blocks": 4,
                "model_bias": True,
                "space_to_depth_factor": 4,
            },
        },
        checkpoint,
    )
    loaded, meta = build_denoiser_from_checkpoint(str(checkpoint), device="cpu")
    assert loaded.space_to_depth_factor == 4
    assert meta["space_to_depth_factor"] == 4
    assert loaded(torch.randn(1, 4, 16, 20)).shape == (1, 4, 16, 20)


def test_factory_infers_s2d_factor_from_bare_paper_checkpoint(tmp_path: Path) -> None:
    deployed_model = MRLFN(
        feature_channels=32,
        num_blocks=4,
        space_to_depth_factor=4,
    ).deploy()
    checkpoint = tmp_path / "bare_mrlfn_paper_s2d.pth"
    torch.save(deployed_model.state_dict(), checkpoint)
    loaded, meta = build_denoiser_from_checkpoint(
        str(checkpoint),
        device="cpu",
        model="mrlfn",
        feature_channels=32,
        num_blocks=4,
    )
    assert loaded.space_to_depth_factor == 4
    assert meta["space_to_depth_factor"] == 4


def test_paper_cosine_scheduler_has_no_warmup_and_reaches_zero() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = Adam([parameter], lr=1e-4)
    scheduler = build_scheduler(optimizer, total_steps=4, warmup_steps=0, min_learning_rate=0.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    learning_rates = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert learning_rates == pytest.approx([8.5355339e-5, 5e-5, 1.4644661e-5, 0.0])
