#!/usr/bin/env python3
"""Export a compact MRLFN checkpoint containing only fused deployment weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_factory import build_denoiser_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Training or already-deployed MRLFN checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = torch.load(args.checkpoint, map_location="cpu")
    source_args = source.get("args", {}) if isinstance(source, dict) else {}
    model, meta = build_denoiser_from_checkpoint(args.checkpoint, device="cpu", model="mrlfn")
    if meta["graph_state"] != "deploy":
        raise RuntimeError("MRLFN export did not produce a deployment graph")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_args = dict(source_args)
    export_args.update(
        {
            "model": "mrlfn",
            "feature_channels": meta["feature_channels"],
            "num_blocks": meta["num_blocks"],
            "model_bias": meta["model_bias"],
            "space_to_depth_factor": meta["space_to_depth_factor"],
        }
    )
    payload = {
        "model_deploy": model.state_dict(),
        "args": export_args,
        "checkpoint_format": 2,
        "inference_graph": "deploy",
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "conversion": meta["weight_source"],
    }
    torch.save(payload, output)
    print(f"Saved fused MRLFN deployment checkpoint: {output}")


if __name__ == "__main__":
    main()
