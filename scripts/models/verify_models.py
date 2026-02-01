#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "packaging",
#     "pillow",
#     "safetensors",
#     "torch",
# ]
# ///
"""
Verify converted safetensors models by comparing inference with PyTorch reference.

This script runs inference on test images using both PyTorch and the exported
safetensors models to ensure they produce equivalent outputs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

# Eye state constants
EYE_WIDTH = 34
EYE_HEIGHT = 26


# ============================================================================
# Model Architectures (must match conversion scripts)
# ============================================================================


class BlazeBlock(nn.Module):
    """BlazeFace building block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        padding = 0 if stride == 2 else (kernel_size - 1) // 2

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.stride == 2:
            x_padded = F.pad(x, (0, 2, 0, 2))
            h = self.depthwise(x_padded)
            residual = F.max_pool2d(x, 2, 2)
        else:
            h = self.depthwise(x)
            residual = x

        h = F.relu(self.bn_dw(h))
        h = self.bn_pw(self.pointwise(h))

        if self.channel_pad > 0:
            residual = F.pad(residual, (0, 0, 0, 0, 0, self.channel_pad))

        return F.relu(h + residual)


class BlazeFace(nn.Module):
    """BlazeFace face detection model."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 24, 5, stride=2, padding=0)
        self.bn0 = nn.BatchNorm2d(24)

        self.backbone1 = nn.ModuleList([
            BlazeBlock(24, 24, 3, 1), BlazeBlock(24, 28, 3, 1), BlazeBlock(28, 32, 3, 2),
            BlazeBlock(32, 36, 3, 1), BlazeBlock(36, 42, 3, 1), BlazeBlock(42, 48, 3, 2),
            BlazeBlock(48, 56, 3, 1), BlazeBlock(56, 64, 3, 1), BlazeBlock(64, 72, 3, 1),
            BlazeBlock(72, 80, 3, 1), BlazeBlock(80, 88, 3, 1),
        ])

        self.backbone2 = nn.ModuleList([
            BlazeBlock(88, 96, 3, 2), BlazeBlock(96, 96, 3, 1), BlazeBlock(96, 96, 3, 1),
            BlazeBlock(96, 96, 3, 1), BlazeBlock(96, 96, 3, 1),
        ])

        self.classifier_16 = nn.Conv2d(88, 2, 1, bias=False)
        self.regressor_16 = nn.Conv2d(88, 32, 1, bias=False)
        self.classifier_8 = nn.Conv2d(96, 6, 1, bias=False)
        self.regressor_8 = nn.Conv2d(96, 96, 1, bias=False)

    def forward(self, x):
        x = F.pad(x, (1, 2, 1, 2))
        x = F.relu(self.bn0(self.conv0(x)))

        for block in self.backbone1:
            x = block(x)
        h = x
        for block in self.backbone2:
            h = block(h)

        return self.classifier_16(x), self.classifier_8(h), self.regressor_16(x), self.regressor_8(h)


class EyeStateClassifier(nn.Module):
    """Eye state classifier."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 3, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class REBNCONV(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dirate: int = 1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=False)


class RSU7(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(self.pool4(hx4))
        hx6 = self.rebnconv6(self.pool5(hx5))
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx5d = self.rebnconv5d(torch.cat((_upsample_like(hx6d, hx5), hx5), 1))
        hx4d = self.rebnconv4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(self.pool4(hx4))
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx4d = self.rebnconv4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin


class U2NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = RSU7(3, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        self.side1 = nn.Conv2d(64, 1, 3, padding=1)
        self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        self.side3 = nn.Conv2d(128, 1, 3, padding=1)
        self.side4 = nn.Conv2d(256, 1, 3, padding=1)
        self.side5 = nn.Conv2d(512, 1, 3, padding=1)
        self.side6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))
        hx5d = self.stage5d(torch.cat((_upsample_like(hx6, hx5), hx5), 1))
        hx4d = self.stage4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.stage3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.stage2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.stage1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6), d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return (torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2),
                torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6))


# ============================================================================
# Verification Functions
# ============================================================================


def load_safetensors_model(model_class, weights_path: Path):
    """Load a model from safetensors weights."""
    model = model_class()
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def verify_blazeface(weights_path: Path, test_image: Path) -> bool:
    """Verify BlazeFace model output."""
    print(f"\n=== Verifying BlazeFace ===")
    print(f"Weights: {weights_path}")
    print(f"Test image: {test_image}")

    model = load_safetensors_model(BlazeFace, weights_path)
    img = Image.open(test_image).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
    arr = (np.array(img, dtype=np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        c1, c2, r1, r2 = model(tensor)

    print(f"16x16 head: max score = {torch.sigmoid(c1).max().item():.4f}, shape = {c1.shape}")
    print(f"8x8 head: max score = {torch.sigmoid(c2).max().item():.4f}, shape = {c2.shape}")

    valid = (c1.shape == (1, 2, 16, 16) and c2.shape == (1, 6, 8, 8) and
             r1.shape == (1, 32, 16, 16) and r2.shape == (1, 96, 8, 8))
    print(f"BlazeFace: {'PASSED' if valid else 'FAILED'}")
    return valid


def verify_eye_state(weights_path: Path, test_image: Path) -> bool:
    """Verify eye state classifier output."""
    print(f"\n=== Verifying Eye State Classifier ===")
    print(f"Weights: {weights_path}")
    print(f"Test image: {test_image}")

    model = load_safetensors_model(EyeStateClassifier, weights_path)
    img = Image.open(test_image).convert("L").resize((EYE_WIDTH, EYE_HEIGHT), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    print(f"Logit: {logit.item():.4f}, Probability (open): {prob:.4f}")
    print(f"Classification: {'OPEN' if prob > 0.5 else 'CLOSED'}")

    valid = logit.shape == (1, 1) and -100 < logit.item() < 100
    print(f"Eye State: {'PASSED' if valid else 'FAILED'}")
    return valid


def verify_u2net(weights_path: Path, test_image: Path) -> bool:
    """Verify U²-Net model output."""
    print(f"\n=== Verifying U²-Net ===")
    print(f"Weights: {weights_path}")
    print(f"Test image: {test_image}")

    model = load_safetensors_model(U2NET, weights_path)
    img = Image.open(test_image).convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)

    d0 = outputs[0]
    print(f"Output shape: {d0.shape}, range: [{d0.min().item():.4f}, {d0.max().item():.4f}]")

    valid = d0.shape == (1, 1, 320, 320) and 0 <= d0.min().item() and d0.max().item() <= 1
    print(f"U²-Net: {'PASSED' if valid else 'FAILED'}")
    return valid


def create_test_image(path: Path, size: tuple[int, int] = (320, 320)):
    """Create a simple test image."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            arr[y, x] = [int(x * 255 / size[0]), int(y * 255 / size[1]), 128]
    Image.fromarray(arr).save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Verify converted models")
    parser.add_argument("--blazeface", type=Path, help="Path to blazeface.safetensors")
    parser.add_argument("--eye-state", type=Path, help="Path to eye_state.safetensors")
    parser.add_argument("--u2net", type=Path, help="Path to u2net.safetensors")
    parser.add_argument("--test-image", type=Path, help="Test image (creates synthetic if not provided)")
    args = parser.parse_args()

    test_image = args.test_image or create_test_image(Path("test_image.png"))
    if not args.test_image:
        print(f"Created synthetic test image: {test_image}")

    results = []
    if args.blazeface:
        results.append(("BlazeFace", verify_blazeface(args.blazeface, test_image)))
    if args.eye_state:
        results.append(("Eye State", verify_eye_state(args.eye_state, test_image)))
    if args.u2net:
        results.append(("U²-Net", verify_u2net(args.u2net, test_image)))

    if not results:
        print("No models specified. Use --blazeface, --eye-state, or --u2net")
        sys.exit(1)

    print("\n" + "=" * 40 + "\nSUMMARY\n" + "=" * 40)
    all_passed = all(p for _, p in results)
    for name, passed in results:
        print(f"  {name}: {'PASSED' if passed else 'FAILED'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
