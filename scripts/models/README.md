# Model Conversion Scripts

Scripts to generate and convert ML models to safetensors format for photo-qa.

All scripts use [PEP 723](https://peps.python.org/pep-0723/) inline metadata,
so dependencies are installed automatically by `uv run`.

## Quick Start (Development)

Generate models with random initialization for testing:

```bash
./convert_blazeface.py -o blazeface.safetensors
./convert_eye_state.py --random-init -o eye_state.safetensors
./convert_u2net.py --random-init -o u2net.safetensors

# Copy to model cache
mkdir -p ~/Library/Application\ Support/photo-qa/models  # macOS
cp *.safetensors ~/Library/Application\ Support/photo-qa/models/
```

## Models

### BlazeFace (Face Detection)

Generates BlazeFace model matching the Rust Candle implementation.

```bash
# Generate with random weights (for development/testing)
python convert_blazeface.py -o blazeface.safetensors
```

**Note:** The Rust implementation uses BatchNorm, which differs from the
original BlazeFace. For production, train on WIDER FACE or similar dataset.

### Eye State Classifier

Trains a CNN on CEW (Closed Eyes in the Wild) dataset.

```bash
# Random initialization (for testing)
python convert_eye_state.py --random-init -o eye_state.safetensors

# Train on CEW dataset
# Download: http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html
python convert_eye_state.py --data-dir path/to/cew -o eye_state.safetensors
```

CEW Dataset structure:
```
cew/
  closed_eyes/*.jpg
  open_eyes/*.jpg
```

### U²-Net (Saliency Detection)

Generates U²-Net model for salient object detection.

```bash
# Random initialization (for testing)
python convert_u2net.py --random-init -o u2net.safetensors

# Convert pretrained weights (download manually from Google Drive first)
# https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view
python convert_u2net.py --weights u2net_pretrained.pth -o u2net.safetensors
```

Source: [U-2-Net](https://github.com/xuebinqin/U-2-Net)

## Verification

Verify converted models produce expected outputs:

```bash
# Verify all models
python verify_models.py \
  --blazeface blazeface.safetensors \
  --eye-state eye_state.safetensors \
  --u2net u2net.safetensors \
  --test-image test.jpg

# Verify single model
python verify_models.py --blazeface blazeface.safetensors
```

## Output Files

After conversion, you'll have:

| Model | File | Size |
|-------|------|------|
| BlazeFace | `blazeface.safetensors` | ~1.5 MB |
| Eye State | `eye_state.safetensors` | ~2.5 MB |
| U²-Net | `u2net.safetensors` | ~176 MB |

## Updating models.rs

After generating the safetensors files, update the SHA256 checksums in
`crates/photo-qa-adapters/src/models.rs`:

```bash
sha256sum *.safetensors
```

## GitHub Release

1. Create safetensors files using the scripts above
2. Compute SHA256 checksums
3. Update `models.rs` with real checksums
4. Create release tag `models-v1`
5. Upload safetensors files to release

```bash
# Create release
gh release create models-v1 \
  blazeface.safetensors \
  eye_state.safetensors \
  u2net.safetensors \
  --title "ML Models v1" \
  --notes "Initial model release for photo-qa"
```
