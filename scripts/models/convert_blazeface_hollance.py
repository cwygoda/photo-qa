#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "packaging",
#     "safetensors",
#     "torch",
# ]
# ///
"""
Convert hollance/BlazeFace-PyTorch weights to safetensors format.

This converts the pretrained BlazeFace weights from:
https://github.com/hollance/BlazeFace-PyTorch

The hollance implementation has BatchNorm folded into conv biases,
so this produces a model WITHOUT separate BatchNorm layers.

The Rust implementation must be updated to match (no BatchNorm).
"""

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_weights(hollance_weights: dict) -> dict:
    """
    Convert hollance weight naming to our Rust naming convention.

    hollance structure:
        backbone1.0.weight/bias -> conv0
        backbone1.{2-12}.convs.0 -> backbone1.{0-10}.depthwise
        backbone1.{2-12}.convs.1 -> backbone1.{0-10}.pointwise
        backbone2.{0-4}.convs.0 -> backbone2.{0-4}.depthwise
        backbone2.{0-4}.convs.1 -> backbone2.{0-4}.pointwise
        classifier_8 -> classifier_16 (operates on 16x16 feature map)
        classifier_16 -> classifier_8 (operates on 8x8 feature map)
        regressor_8 -> regressor_16
        regressor_16 -> regressor_8
    """
    converted = {}

    for key, value in hollance_weights.items():
        new_key = key

        # Initial conv: backbone1.0.{weight,bias} -> conv0.{weight,bias}
        if key in ("backbone1.0.weight", "backbone1.0.bias"):
            new_key = key.replace("backbone1.0.", "conv0.")

        # BlazeBlocks in backbone1: indices 2-12 -> 0-10
        elif key.startswith("backbone1."):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                old_idx = int(parts[1])
                if old_idx >= 2:
                    new_idx = old_idx - 2
                    # convs.0 -> depthwise, convs.1 -> pointwise
                    if parts[2] == "convs":
                        conv_type = "depthwise" if parts[3] == "0" else "pointwise"
                        param_type = parts[4]  # weight or bias
                        new_key = f"backbone1.{new_idx}.{conv_type}.{param_type}"

        # BlazeBlocks in backbone2
        elif key.startswith("backbone2."):
            parts = key.split(".")
            if len(parts) >= 4 and parts[2] == "convs":
                idx = parts[1]
                conv_type = "depthwise" if parts[3] == "0" else "pointwise"
                param_type = parts[4]
                new_key = f"backbone2.{idx}.{conv_type}.{param_type}"

        # Detection heads: swap 8<->16 naming (hollance naming is confusing)
        # hollance classifier_8 (88->2) operates on 16x16 features
        # hollance classifier_16 (96->6) operates on 8x8 features
        elif key.startswith("classifier_8"):
            new_key = key.replace("classifier_8", "classifier_16")
        elif key.startswith("classifier_16"):
            new_key = key.replace("classifier_16", "classifier_8")
        elif key.startswith("regressor_8"):
            new_key = key.replace("regressor_8", "regressor_16")
        elif key.startswith("regressor_16"):
            new_key = key.replace("regressor_16", "regressor_8")

        converted[new_key] = value.float()

    return converted


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Convert hollance BlazeFace weights to safetensors"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("hollance_blazeface.pth"),
        help="Input hollance .pth file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("weights/blazeface.safetensors"),
        help="Output safetensors file"
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Download from: https://github.com/hollance/BlazeFace-PyTorch/raw/master/blazeface.pth")
        return 1

    print(f"Loading weights from {args.input}...")
    hollance_weights = torch.load(args.input, map_location="cpu", weights_only=True)

    print(f"Converting {len(hollance_weights)} weight tensors...")
    converted = convert_weights(hollance_weights)

    # Verify conversion
    print(f"\nConverted keys ({len(converted)}):")
    for key in sorted(converted.keys())[:20]:
        print(f"  {key}: {list(converted[key].shape)}")
    if len(converted) > 20:
        print(f"  ... and {len(converted) - 20} more")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {args.output}...")
    save_file(converted, args.output)

    sha256 = compute_sha256(args.output)
    file_size = args.output.stat().st_size

    print(f"\nConversion complete!")
    print(f"  File: {args.output}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"  SHA256: {sha256}")
    print(f"\nNOTE: Rust BlazeFace implementation must be updated to remove BatchNorm!")

    return 0


if __name__ == "__main__":
    exit(main())
