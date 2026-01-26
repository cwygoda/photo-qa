#!/usr/bin/env bash
# Fetch test datasets for photo-qa
#
# Usage:
#   ./scripts/fetch-test-data.sh         # Download all datasets
#   ./scripts/fetch-test-data.sh blur    # Download blur dataset only
#   ./scripts/fetch-test-data.sh eyes    # Download closed eyes dataset only
#   ./scripts/fetch-test-data.sh exposure # Download exposure dataset only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FIXTURES_DIR="$PROJECT_ROOT/tests/fixtures"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Create fixtures directory
mkdir -p "$FIXTURES_DIR"

# Download and extract a dataset
download_dataset() {
    local name="$1"
    local url="$2"
    local dest_dir="$FIXTURES_DIR/$name"

    if [[ -d "$dest_dir" ]]; then
        info "Dataset '$name' already exists, skipping"
        return 0
    fi

    info "Downloading $name dataset..."
    mkdir -p "$dest_dir"

    local tmp_file
    tmp_file=$(mktemp)

    if curl -fsSL "$url" -o "$tmp_file"; then
        # Detect file type and extract
        local file_type
        file_type=$(file -b --mime-type "$tmp_file")

        case "$file_type" in
            application/zip)
                unzip -q "$tmp_file" -d "$dest_dir"
                ;;
            application/gzip|application/x-gzip)
                tar -xzf "$tmp_file" -C "$dest_dir"
                ;;
            application/x-tar)
                tar -xf "$tmp_file" -C "$dest_dir"
                ;;
            *)
                # Assume it's a single file
                mv "$tmp_file" "$dest_dir/"
                return 0
                ;;
        esac

        rm -f "$tmp_file"
        info "Downloaded and extracted $name"
    else
        rm -f "$tmp_file"
        error "Failed to download $name from $url"
    fi
}

# Dataset configurations
# Note: Some datasets require manual download due to licensing
# These URLs are placeholders - update with actual sources

fetch_blur_datasets() {
    info "Fetching blur detection datasets..."

    # CERTH blur dataset (from Kaggle)
    # Note: Kaggle datasets may require authentication
    warn "CERTH/Kaggle blur dataset requires manual download from:"
    warn "  https://www.kaggle.com/datasets/kwentar/blur-dataset"
    warn "  Extract to: $FIXTURES_DIR/blur/"

    # Create placeholder directory
    mkdir -p "$FIXTURES_DIR/blur"
    echo "Download blur dataset from Kaggle and extract here" > "$FIXTURES_DIR/blur/README.txt"
}

fetch_eyes_datasets() {
    info "Fetching closed eyes detection datasets..."

    # CEW (Closed Eyes in the Wild) dataset
    # Note: Academic dataset, may require registration
    warn "CEW dataset requires manual download from:"
    warn "  https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html"
    warn "  Extract to: $FIXTURES_DIR/eyes/"

    mkdir -p "$FIXTURES_DIR/eyes"
    echo "Download CEW dataset and extract here" > "$FIXTURES_DIR/eyes/README.txt"
}

fetch_exposure_datasets() {
    info "Fetching exposure datasets..."

    # VV-Dataset (publicly available)
    warn "VV-Dataset requires manual download from:"
    warn "  https://sites.google.com/site/vonikakis/datasets"
    warn "  Extract to: $FIXTURES_DIR/exposure/"

    mkdir -p "$FIXTURES_DIR/exposure"
    echo "Download VV-Dataset and extract here" > "$FIXTURES_DIR/exposure/README.txt"
}

# Main
main() {
    local dataset="${1:-all}"

    info "Photo QA Test Data Fetcher"
    info "Fixtures directory: $FIXTURES_DIR"
    echo ""

    case "$dataset" in
        blur)
            fetch_blur_datasets
            ;;
        eyes)
            fetch_eyes_datasets
            ;;
        exposure)
            fetch_exposure_datasets
            ;;
        all)
            fetch_blur_datasets
            fetch_eyes_datasets
            fetch_exposure_datasets
            ;;
        *)
            error "Unknown dataset: $dataset. Use 'blur', 'eyes', 'exposure', or 'all'"
            ;;
    esac

    echo ""
    info "Done! Check $FIXTURES_DIR for downloaded datasets."
    warn "Some datasets require manual download due to licensing restrictions."
}

main "$@"
