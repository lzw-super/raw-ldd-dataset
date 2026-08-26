"""Helpers that make reparameterized evaluation/checkpointing explicit."""

from __future__ import annotations

from typing import Any

import torch


def has_deploy_graph(model: torch.nn.Module) -> bool:
    return callable(getattr(model, "deploy", None))


@torch.no_grad()
def prepare_model_for_inference(model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    """Return an eval model; fuse a copied graph when the model supports it."""
    inference_model = model.deploy() if has_deploy_graph(model) else model
    inference_model.eval()
    return inference_model, "deploy" if has_deploy_graph(model) else "native"


@torch.no_grad()
def deploy_state_dict(model: torch.nn.Module) -> dict[str, Any] | None:
    """Return fused weights for a reparameterizable model, otherwise ``None``."""
    if not has_deploy_graph(model):
        return None
    deployed = model.deploy()
    return deployed.state_dict()
