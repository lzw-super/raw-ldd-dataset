#!/usr/bin/env python3
"""Build auditable SID full-RAW train/validation manifests."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.sid_synthetic_train import build_sid_patch_manifest, build_sid_raw_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", default="/home/shared_files/dataset/SID/Sony_train_long_patches")
    parser.add_argument("--pair-list", default="/home/shared_files/dataset/SID/Sony_train_list.txt")
    parser.add_argument("--sid-long-dir", default="/home/shared_files/dataset/SID/Sony/long")
    parser.add_argument("--output", default="infos/SID_train_clean_raw.json")
    parser.add_argument("--validation-output", default="infos/SID_validation_clean_raw.json")
    parser.add_argument("--clean-source", choices=["raw", "packed"], default="raw")
    args = parser.parse_args()
    if args.clean_source == "raw":
        manifest = build_sid_raw_manifest(args.sid_long_dir, args.output, scene_prefixes=("0",))
        validation = build_sid_raw_manifest(args.sid_long_dir, args.validation_output, scene_prefixes=("2",))
        print(
            f"Wrote {args.output}: {manifest['summary']['records']} full RAW train scenes; "
            f"{args.validation_output}: {validation['summary']['records']} held-out scenes"
        )
        print(f"Training ISO values: {manifest['summary']['iso_values']}")
        return
    manifest = build_sid_patch_manifest(args.patch_dir, args.pair_list, args.output, args.sid_long_dir)
    summary = manifest["summary"]
    print(f"Wrote {args.output}: {summary['records']} patches from {summary['scenes']} train scenes")
    print(f"ISO values: {summary['iso_values']}")
    print(f"Ignored non-train patches: {summary['ignored_nontrain_patches']}")


if __name__ == "__main__":
    main()
