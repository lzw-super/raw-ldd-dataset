"""Deterministic held-out synthetic evaluation for the phone DNG pipeline."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

import numpy as np
import torch

from datasets.phone_synthetic_train import PhoneSyntheticTrainDataset
from noise.dng_noise_synthesis import synthesize_phone_noise


def _new_stats() -> dict[str, float | int]:
    return {"absolute_error": 0.0, "squared_error": 0.0, "pixels": 0, "samples": 0}


def _summarize(stats: dict[str, float | int]) -> dict[str, float | int]:
    pixels = max(1, int(stats["pixels"]))
    mse = float(stats["squared_error"]) / pixels
    return {
        "samples": int(stats["samples"]),
        "pixels": int(stats["pixels"]),
        "l1": float(stats["absolute_error"]) / pixels,
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
    }


def _accumulate(stats: dict[str, float | int], prediction: torch.Tensor, target: torch.Tensor) -> None:
    difference = prediction - target
    stats["absolute_error"] = float(stats["absolute_error"]) + float(difference.abs().sum().cpu())
    stats["squared_error"] = float(stats["squared_error"]) + float(difference.square().sum().cpu())
    stats["pixels"] = int(stats["pixels"]) + int(difference.numel())
    stats["samples"] = int(stats["samples"]) + 1


@torch.inference_mode()
def evaluate_phone_synthetic(
    model: torch.nn.Module,
    dataset: PhoneSyntheticTrainDataset,
    *,
    device: torch.device,
    ratios: Iterable[float] = (100.0, 250.0, 300.0),
    max_samples: int = 0,
    seed: int = 10_001,
    synthesis: str = "hybrid",
) -> dict[str, Any]:
    """Evaluate a fixed set of synthetic patches without perturbing training RNG.

    The input dataset must be built from held-out long-exposure clean DNGs and
    held-out dark residuals.  The returned PSNR is only a synthetic result:
    model(noisy synthesized from a pseudo-clean 10 s frame) versus that frame.
    It is deliberately not named or formatted as real noisy-clean PSNR.
    """
    selected_ratios = tuple(float(value) for value in ratios)
    if not selected_ratios or any(value <= 0 for value in selected_ratios):
        raise ValueError("ratios must contain one or more positive values")
    sample_count = len(dataset) if int(max_samples) <= 0 else min(int(max_samples), len(dataset))
    if sample_count <= 0:
        raise ValueError("Synthetic validation dataset is empty")

    original_training_state = model.training
    original_ratios = dataset.ratios
    original_rng = dataset._rng
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    result_by_ratio: dict[str, dict[str, Any]] = {}
    global_stats = _new_stats()
    model.eval()
    try:
        # Torch's Poisson noise is part of the benchmark.  Forking preserves
        # the random state that the training loop will use afterwards.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            for ratio_index, ratio in enumerate(selected_ratios):
                dataset.ratios = (ratio,)
                dataset._rng = np.random.default_rng(int(seed) + 1_000_003 * ratio_index)
                ratio_stats = _new_stats()
                iso_stats: dict[int, dict[str, float | int]] = defaultdict(_new_stats)
                for index in range(sample_count):
                    item = dataset[index]
                    clean = item["clean"].unsqueeze(0).to(device, non_blocking=True)
                    dark = item["dark_residual_dn"].unsqueeze(0).to(device, non_blocking=True)
                    s4 = item["noise_profile_s4"].unsqueeze(0).to(device, non_blocking=True)
                    o4 = item["noise_profile_o4"].unsqueeze(0).to(device, non_blocking=True)
                    dark_dynamic_range = item["dark_dynamic_range"].view(1).to(device, non_blocking=True)
                    ratio_tensor = item["ratio"].view(1).to(device, non_blocking=True)
                    noisy, target = synthesize_phone_noise(
                        clean,
                        dark,
                        s4,
                        dark_dynamic_range,
                        ratio_tensor,
                        mode=synthesis,
                        noise_profile_o4=o4,
                    )
                    prediction = torch.clamp(model(noisy), 0.0, 1.0)
                    iso = int(item["iso_exif"])
                    _accumulate(ratio_stats, prediction, target)
                    _accumulate(iso_stats[iso], prediction, target)
                    _accumulate(global_stats, prediction, target)
                result_by_ratio[f"{ratio:g}"] = {
                    **_summarize(ratio_stats),
                    "by_iso": {str(iso): _summarize(stats) for iso, stats in sorted(iso_stats.items())},
                }
    finally:
        dataset.ratios = original_ratios
        dataset._rng = original_rng
        model.train(original_training_state)

    global_summary = _summarize(global_stats)
    return {
        "synthetic_heldout_psnr": global_summary["psnr"],
        "synthetic_heldout_l1": global_summary["l1"],
        "synthetic_heldout_samples": global_summary["samples"],
        "synthetic_heldout_pixels": global_summary["pixels"],
        "synthetic_heldout_seed": int(seed),
        "synthetic_heldout_psnr_by_ratio": {
            ratio: float(metrics["psnr"]) for ratio, metrics in result_by_ratio.items()
        },
        "synthetic_heldout_l1_by_ratio": {
            ratio: float(metrics["l1"]) for ratio, metrics in result_by_ratio.items()
        },
        "synthetic_heldout_ratios": result_by_ratio,
    }
