#!/bin/bash
# Create GitHub release with model files
#
# Prerequisites:
# 1. Generate model files (see README.md)
# 2. Ensure gh CLI is installed and authenticated
#
# Usage: ./release.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WEIGHTS_DIR="weights"
MODELS=(
    "blazeface.safetensors"
    "eye_state.safetensors"
    "u2net.safetensors"
)

TAG="models-v1"
TITLE="ML Models v1"

# Check all models exist
echo "Checking model files..."
MODEL_PATHS=()
for model in "${MODELS[@]}"; do
    model_path="$WEIGHTS_DIR/$model"
    if [[ ! -f "$model_path" ]]; then
        echo "Error: Missing $model_path"
        echo "Run the conversion scripts first (see README.md)"
        exit 1
    fi
    echo "  ✓ $model ($(du -h "$model_path" | cut -f1))"
    MODEL_PATHS+=("$model_path")
done

# Print checksums
echo ""
echo "SHA256 checksums:"
for model_path in "${MODEL_PATHS[@]}"; do
    shasum -a 256 "$model_path"
done

echo ""
echo "Creating release $TAG..."
echo ""

# Check if release exists
if gh release view "$TAG" &>/dev/null; then
    echo "Release $TAG already exists."
    read -p "Delete and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gh release delete "$TAG" --yes
        git tag -d "$TAG" 2>/dev/null || true
        git push origin ":refs/tags/$TAG" 2>/dev/null || true
    else
        echo "Aborting."
        exit 1
    fi
fi

# Create release
gh release create "$TAG" \
    "${MODEL_PATHS[@]}" \
    --title "$TITLE" \
    --notes "$(cat <<'EOF'
## Photo-QA ML Models

Pre-trained models for photo quality analysis.

### Models

| Model | Description | Size | Status |
|-------|-------------|------|--------|
| blazeface.safetensors | Face detection with keypoints | ~0.4 MB | ✅ MediaPipe pretrained |
| eye_state.safetensors | Open/closed eye classification | ~2 MB | ✅ Trained on CEW (96% acc) |
| u2net.safetensors | Saliency detection | ~176 MB | ✅ Official pretrained |

### Usage

Models are downloaded automatically on first use:

```bash
photo-qa check image.jpg
```

Or manually:

```bash
photo-qa models fetch
```

### Checksums

See the release assets for SHA256 checksums.
EOF
)"

echo ""
echo "Release created: https://github.com/cwygoda/photo-qa/releases/tag/$TAG"
echo ""
echo "Next steps:"
echo "1. Update crates/photo-qa-adapters/src/models.rs with real checksums"
echo "2. Run 'cargo run -- models fetch' to test download"
