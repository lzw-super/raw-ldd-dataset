#!/usr/bin/env python3
"""Build the auditable, leakage-safe SID packed-patch training manifest."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.sid_synthetic_train import build_sid_patch_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", default="/home/shared_files/dataset/SID/Sony_train_long_patches")
    parser.add_argument("--pair-list", default="/home/shared_files/dataset/SID/Sony_train_list.txt")
    parser.add_argument("--sid-long-dir", default="/home/shared_files/dataset/SID/Sony/long")
    parser.add_argument("--output", default="infos/SID_train_clean_patches.json")
    args = parser.parse_args()
    manifest = build_sid_patch_manifest(args.patch_dir, args.pair_list, args.output, args.sid_long_dir)
    summary = manifest["summary"]
    print(f"Wrote {args.output}: {summary['records']} patches from {summary['scenes']} train scenes")
    print(f"ISO values: {summary['iso_values']}")
    print(f"Ignored non-train patches: {summary['ignored_nontrain_patches']}")


if __name__ == "__main__":
    main()
