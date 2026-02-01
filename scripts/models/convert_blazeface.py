#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "safetensors",
#     "torch",
# ]
# ///
"""
Convert BlazeFace PyTorch model to safetensors format.

NOTE: The original BlazeFace implementation (hollance/BlazeFace-PyTorch) uses
a simpler architecture without BatchNorm. Our Rust implementation uses BatchNorm
for better accuracy. This script initializes a model matching our Rust architecture.

For production use, this model should be trained on face detection datasets
such as WIDER FACE or similar.

Architecture matches: crates/photo-qa-core/src/inference/blazeface.rs
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file


class BlazeBlock(nn.Module):
    """BlazeFace building block with depthwise separable convolution + BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride == 2:
            padding = 0
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        else:
            padding = (kernel_size - 1) // 2
            self.max_pool = None

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            x_padded = nn.functional.pad(x, (0, 2, 0, 2), mode="constant", value=0)
            h = self.depthwise(x_padded)
            residual = self.max_pool(x)
        else:
            h = self.depthwise(x)
            residual = x

        h = self.bn_dw(h)
        h = self.relu(h)
        h = self.pointwise(h)
        h = self.bn_pw(h)

        if self.channel_pad > 0:
            residual = nn.functional.pad(residual, (0, 0, 0, 0, 0, self.channel_pad))

        return self.relu(h + residual)


class BlazeFace(nn.Module):
    """BlazeFace face detection model with BatchNorm (matching Rust implementation)."""

    def __init__(self):
        super().__init__()

        # Initial convolution: 5x5, stride 2, produces 64x64 feature map
        self.conv0 = nn.Conv2d(3, 24, 5, stride=2, padding=0)
        self.bn0 = nn.BatchNorm2d(24)

        # Backbone 1: produces 16x16 feature map with 88 channels
        self.backbone1 = nn.ModuleList([
            BlazeBlock(24, 24, 3, 1),   # 0
            BlazeBlock(24, 28, 3, 1),   # 1
            BlazeBlock(28, 32, 3, 2),   # 2: -> 32x32
            BlazeBlock(32, 36, 3, 1),   # 3
            BlazeBlock(36, 42, 3, 1),   # 4
            BlazeBlock(42, 48, 3, 2),   # 5: -> 16x16
            BlazeBlock(48, 56, 3, 1),   # 6
            BlazeBlock(56, 64, 3, 1),   # 7
            BlazeBlock(64, 72, 3, 1),   # 8
            BlazeBlock(72, 80, 3, 1),   # 9
            BlazeBlock(80, 88, 3, 1),   # 10
        ])

        # Backbone 2: produces 8x8 feature map with 96 channels
        self.backbone2 = nn.ModuleList([
            BlazeBlock(88, 96, 3, 2),   # 0: -> 8x8
            BlazeBlock(96, 96, 3, 1),   # 1
            BlazeBlock(96, 96, 3, 1),   # 2
            BlazeBlock(96, 96, 3, 1),   # 3
            BlazeBlock(96, 96, 3, 1),   # 4
        ])

        # Detection heads for 16x16 (512 anchors = 16*16*2)
        self.classifier_16 = nn.Conv2d(88, 2, 1, bias=False)
        self.regressor_16 = nn.Conv2d(88, 32, 1, bias=False)

        # Detection heads for 8x8 (384 anchors = 8*8*6)
        self.classifier_8 = nn.Conv2d(96, 6, 1, bias=False)
        self.regressor_8 = nn.Conv2d(96, 96, 1, bias=False)

    def forward(self, x):
        # Asymmetric padding for initial conv
        x = nn.functional.pad(x, (1, 2, 1, 2), mode="constant", value=0)
        x = nn.functional.relu(self.bn0(self.conv0(x)))

        for block in self.backbone1:
            x = block(x)

        h = x
        for block in self.backbone2:
            h = block(h)

        c1 = self.classifier_16(x)
        c2 = self.classifier_8(h)
        r1 = self.regressor_16(x)
        r2 = self.regressor_8(h)

        return c1, c2, r1, r2


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate BlazeFace safetensors model")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("blazeface.safetensors"),
        help="Output safetensors file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight initialization"
    )
    args = parser.parse_args()

    print("Generating BlazeFace model (BatchNorm architecture)...")
    print()
    print("NOTE: This generates randomly initialized weights matching the")
    print("Rust Candle architecture. For production, train on face detection")
    print("datasets like WIDER FACE.")
    print()

    torch.manual_seed(args.seed)
    model = BlazeFace()
    model.eval()

    # Collect weights matching Rust naming
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.data.float()

    for name, buffer in model.named_buffers():
        # Skip num_batches_tracked
        if "num_batches_tracked" not in name:
            weights[name] = buffer.float()

    print(f"Saving to {args.output}...")
    save_file(weights, args.output)

    sha256 = compute_sha256(args.output)
    file_size = args.output.stat().st_size

    print(f"\nGeneration complete!")
    print(f"  File: {args.output}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  SHA256: {sha256}")
    print(f"\nWeight keys: {len(weights)}")
    for key in sorted(weights.keys())[:15]:
        print(f"  {key}: {list(weights[key].shape)}")
    if len(weights) > 15:
        print(f"  ... and {len(weights) - 15} more")


if __name__ == "__main__":
    main()
