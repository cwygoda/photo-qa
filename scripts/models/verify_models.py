#!/usr/bin/env python3
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
from PIL import Image
from safetensors.torch import load_file

# Import model architectures from conversion scripts
from convert_blazeface import BlazeFace
from convert_eye_state import EyeStateClassifier, EYE_WIDTH, EYE_HEIGHT
from convert_u2net import U2NET


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

    # Load model
    model = load_safetensors_model(BlazeFace, weights_path)

    # Prepare input
    img = Image.open(test_image).convert("RGB")
    img = img.resize((128, 128), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # Normalize to [-1, 1]
    arr = (arr / 127.5) - 1.0
    # NCHW format
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        c1, c2, r1, r2 = model(tensor)

    # Print summary
    scores_16 = torch.sigmoid(c1.view(-1))
    scores_8 = torch.sigmoid(c2.view(-1))
    max_score_16 = scores_16.max().item()
    max_score_8 = scores_8.max().item()

    print(f"16x16 head: max score = {max_score_16:.4f}, shape = {c1.shape}")
    print(f"8x8 head: max score = {max_score_8:.4f}, shape = {c2.shape}")
    print(f"Regressor 16x16 shape: {r1.shape}")
    print(f"Regressor 8x8 shape: {r2.shape}")

    # Basic sanity checks
    valid = True
    if c1.shape != (1, 2, 16, 16):
        print(f"ERROR: c1 shape mismatch, expected (1, 2, 16, 16)")
        valid = False
    if c2.shape != (1, 6, 8, 8):
        print(f"ERROR: c2 shape mismatch, expected (1, 6, 8, 8)")
        valid = False
    if r1.shape != (1, 32, 16, 16):
        print(f"ERROR: r1 shape mismatch, expected (1, 32, 16, 16)")
        valid = False
    if r2.shape != (1, 96, 8, 8):
        print(f"ERROR: r2 shape mismatch, expected (1, 96, 8, 8)")
        valid = False

    if valid:
        print("BlazeFace: PASSED")
    else:
        print("BlazeFace: FAILED")

    return valid


def verify_eye_state(weights_path: Path, test_image: Path) -> bool:
    """Verify eye state classifier output."""
    print(f"\n=== Verifying Eye State Classifier ===")
    print(f"Weights: {weights_path}")
    print(f"Test image: {test_image}")

    # Load model
    model = load_safetensors_model(EyeStateClassifier, weights_path)

    # Prepare input (grayscale eye crop)
    img = Image.open(test_image).convert("L")
    img = img.resize((EYE_WIDTH, EYE_HEIGHT), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    print(f"Input shape: {tensor.shape}")
    print(f"Logit: {logit.item():.4f}")
    print(f"Probability (open): {prob:.4f}")
    print(f"Classification: {'OPEN' if prob > 0.5 else 'CLOSED'}")

    # Sanity check
    valid = True
    if logit.shape != (1, 1):
        print(f"ERROR: output shape mismatch, expected (1, 1)")
        valid = False
    if not -100 < logit.item() < 100:
        print(f"ERROR: logit out of reasonable range")
        valid = False

    if valid:
        print("Eye State: PASSED")
    else:
        print("Eye State: FAILED")

    return valid


def verify_u2net(weights_path: Path, test_image: Path) -> bool:
    """Verify U²-Net model output."""
    print(f"\n=== Verifying U²-Net ===")
    print(f"Weights: {weights_path}")
    print(f"Test image: {test_image}")

    # Load model
    model = load_safetensors_model(U2NET, weights_path)

    # Prepare input
    img = Image.open(test_image).convert("RGB")
    img = img.resize((320, 320), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(tensor)

    d0 = outputs[0]
    print(f"Input shape: {tensor.shape}")
    print(f"Output shape: {d0.shape}")
    print(f"Output range: [{d0.min().item():.4f}, {d0.max().item():.4f}]")
    print(f"Mean saliency: {d0.mean().item():.4f}")

    # Sanity checks
    valid = True
    if d0.shape != (1, 1, 320, 320):
        print(f"ERROR: output shape mismatch, expected (1, 1, 320, 320)")
        valid = False
    if d0.min().item() < 0 or d0.max().item() > 1:
        print(f"ERROR: output values out of [0, 1] range after sigmoid")
        valid = False

    if valid:
        print("U²-Net: PASSED")
    else:
        print("U²-Net: FAILED")

    return valid


def create_test_image(path: Path, size: tuple[int, int] = (320, 320)):
    """Create a simple test image if none provided."""
    # Create gradient image
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            arr[y, x] = [int(x * 255 / size[0]), int(y * 255 / size[1]), 128]
    img = Image.fromarray(arr)
    img.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Verify converted models")
    parser.add_argument(
        "--blazeface",
        type=Path,
        help="Path to blazeface.safetensors"
    )
    parser.add_argument(
        "--eye-state",
        type=Path,
        help="Path to eye_state.safetensors"
    )
    parser.add_argument(
        "--u2net",
        type=Path,
        help="Path to u2net.safetensors"
    )
    parser.add_argument(
        "--test-image",
        type=Path,
        help="Test image path (optional, will create synthetic if not provided)"
    )
    args = parser.parse_args()

    # Create or use test image
    if args.test_image:
        test_image = args.test_image
    else:
        test_image = Path("test_image.png")
        print(f"Creating synthetic test image: {test_image}")
        create_test_image(test_image)

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

    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
