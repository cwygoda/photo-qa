#!/usr/bin/env bash
# Fetch test datasets for photo-qa
#
# Usage:
#   ./scripts/fetch-test-data.sh              # Download all datasets
#   ./scripts/fetch-test-data.sh blur         # Download blur dataset only
#   ./scripts/fetch-test-data.sh eyes         # Download closed eyes dataset only
#   ./scripts/fetch-test-data.sh exposure     # Download exposure dataset only
#   ./scripts/fetch-test-data.sh status       # Show download status of all datasets
#   ./scripts/fetch-test-data.sh verify <name> # Verify dataset integrity
#   ./scripts/fetch-test-data.sh interactive  # Guided download

set -euo pipefail

# Require Bash 4+ for associative arrays
if ((BASH_VERSINFO[0] < 4)); then
    echo "Error: This script requires Bash 4.0 or later." >&2
    echo "Current version: $BASH_VERSION" >&2
    echo "On macOS, install via: brew install bash" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FIXTURES_DIR="$PROJECT_ROOT/tests/fixtures"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

status_icon() {
    if [[ "$1" == "ok" ]]; then
        echo -e "${GREEN}✓${NC}"
    elif [[ "$1" == "partial" ]]; then
        echo -e "${YELLOW}~${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
}

# Create fixtures directory
mkdir -p "$FIXTURES_DIR"

# Dataset metadata
declare -A DATASET_URLS
declare -A DATASET_SIZES
declare -A DATASET_CHECKSUMS

DATASET_URLS["blur"]="https://www.kaggle.com/datasets/kwentar/blur-dataset"
DATASET_SIZES["blur"]="~50MB"
DATASET_CHECKSUMS["blur"]=""

DATASET_URLS["eyes"]="https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html"
DATASET_SIZES["eyes"]="~100MB"
DATASET_CHECKSUMS["eyes"]=""

DATASET_URLS["exposure"]="https://sites.google.com/site/vonikakis/datasets"
DATASET_SIZES["exposure"]="~200MB"
DATASET_CHECKSUMS["exposure"]=""

# Check if a dataset is downloaded
check_dataset_status() {
    local name="$1"
    local dir="$FIXTURES_DIR/$name"

    if [[ ! -d "$dir" ]]; then
        echo "missing"
        return
    fi

    # Check if directory has content (not just README)
    local file_count
    file_count=$(find "$dir" -type f ! -name "README.txt" ! -name "*.md" 2>/dev/null | wc -l | tr -d ' ')

    if [[ "$file_count" -eq 0 ]]; then
        echo "empty"
    elif [[ "$file_count" -lt 10 ]]; then
        echo "partial"
    else
        echo "ok"
    fi
}

# Count images in a dataset
count_images() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo "0"
        return
    fi
    find "$dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) 2>/dev/null | wc -l | tr -d ' '
}

# Show status of all datasets
show_status() {
    echo ""
    echo -e "${BLUE}=== Dataset Status ===${NC}"
    echo ""
    printf "%-12s %-10s %-12s %-10s %s\n" "Dataset" "Status" "Images" "Size" "Location"
    echo "---------------------------------------------------------------------"

    for dataset in blur eyes exposure; do
        local dir="$FIXTURES_DIR/$dataset"
        local status
        status=$(check_dataset_status "$dataset")
        local icon
        icon=$(status_icon "$status")
        local images
        images=$(count_images "$dir")
        local size="${DATASET_SIZES[$dataset]:-unknown}"

        printf "%-12s %s %-8s %-12s %-10s %s\n" "$dataset" "$icon" "$status" "$images" "$size" "$dir"
    done

    echo ""
    echo "Legend: ${GREEN}✓${NC} ok  ${YELLOW}~${NC} partial  ${RED}✗${NC} missing/empty"
    echo ""
}

# Verify dataset integrity
verify_dataset() {
    local name="$1"
    local dir="$FIXTURES_DIR/$name"

    echo ""
    echo -e "${BLUE}=== Verifying Dataset: $name ===${NC}"
    echo ""

    if [[ ! -d "$dir" ]]; then
        echo -e "${RED}Directory not found: $dir${NC}"
        return 1
    fi

    local status
    status=$(check_dataset_status "$name")
    local images
    images=$(count_images "$dir")

    echo "Directory: $dir"
    echo "Status:    $status"
    echo "Images:    $images"
    echo ""

    # List subdirectories
    echo "Structure:"
    if command -v tree &> /dev/null; then
        tree -L 2 -d "$dir" 2>/dev/null || find "$dir" -maxdepth 2 -type d
    else
        find "$dir" -maxdepth 2 -type d
    fi
    echo ""

    # Check for ground truth files
    echo "Ground truth files:"
    local gt_files
    gt_files=$(find "$dir" -type f \( -iname "*ground*truth*" -o -iname "*labels*" -o -iname "*annotations*" -o -iname "*.csv" \) 2>/dev/null || true)
    if [[ -n "$gt_files" ]]; then
        echo "$gt_files"
    else
        echo "  None found (may need to create)"
    fi
    echo ""

    # Check checksum if available
    local checksum="${DATASET_CHECKSUMS[$name]:-}"
    if [[ -n "$checksum" ]]; then
        echo "Verifying checksum..."
        # TODO: Implement checksum verification
        echo "  Checksum verification not yet implemented"
    fi

    return 0
}

# Generate ground truth template
generate_ground_truth_template() {
    local name="$1"
    local dir="$FIXTURES_DIR/$name"
    local gt_file="$dir/ground_truth.csv"

    if [[ ! -d "$dir" ]]; then
        error "Dataset directory not found: $dir"
    fi

    if [[ -f "$gt_file" ]]; then
        warn "Ground truth file already exists: $gt_file"
        return 0
    fi

    info "Generating ground truth template for $name..."

    # Create CSV header based on dataset type
    case "$name" in
        blur)
            echo "filename,is_blurry,blur_type,notes" > "$gt_file"
            ;;
        exposure)
            echo "filename,is_underexposed,is_overexposed,exposure_level,notes" > "$gt_file"
            ;;
        eyes)
            echo "filename,has_closed_eyes,face_count,closed_count,notes" > "$gt_file"
            ;;
        *)
            echo "filename,label,notes" > "$gt_file"
            ;;
    esac

    # Add placeholder entries for each image
    local count=0
    while IFS= read -r -d '' img; do
        local basename
        basename=$(basename "$img")
        case "$name" in
            blur)
                echo "$basename,,," >> "$gt_file"
                ;;
            exposure)
                echo "$basename,,,," >> "$gt_file"
                ;;
            eyes)
                echo "$basename,,,," >> "$gt_file"
                ;;
            *)
                echo "$basename,," >> "$gt_file"
                ;;
        esac
        ((count++))
    done < <(find "$dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -print0 2>/dev/null)

    info "Generated template with $count entries: $gt_file"
    info "Please fill in the labels manually"
}

# Download and extract a dataset
download_dataset() {
    local name="$1"
    local url="$2"
    local dest_dir="$FIXTURES_DIR/$name"

    if [[ -d "$dest_dir" ]]; then
        local status
        status=$(check_dataset_status "$name")
        if [[ "$status" == "ok" ]]; then
            info "Dataset '$name' already exists, skipping"
            return 0
        fi
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
fetch_blur_datasets() {
    info "Fetching blur detection datasets..."

    # CERTH blur dataset (from Kaggle)
    warn "CERTH/Kaggle blur dataset requires manual download from:"
    warn "  ${DATASET_URLS[blur]}"
    warn "  Extract to: $FIXTURES_DIR/blur/"
    echo ""
    warn "Steps:"
    warn "  1. Log in to Kaggle"
    warn "  2. Download the dataset"
    warn "  3. Extract to $FIXTURES_DIR/blur/"
    warn "  4. Run: $0 verify blur"

    mkdir -p "$FIXTURES_DIR/blur"
    cat > "$FIXTURES_DIR/blur/README.txt" << 'EOF'
Blur Detection Dataset

Download from:
  https://www.kaggle.com/datasets/kwentar/blur-dataset

Expected structure:
  blur/
  ├── sharp/
  │   └── *.jpg
  └── defocused_blurred/
      └── *.jpg

After downloading, run: ./scripts/fetch-test-data.sh verify blur
EOF
}

fetch_eyes_datasets() {
    info "Fetching closed eyes detection datasets..."

    warn "CEW dataset requires manual download from:"
    warn "  ${DATASET_URLS[eyes]}"
    warn "  Extract to: $FIXTURES_DIR/eyes/"

    mkdir -p "$FIXTURES_DIR/eyes"
    cat > "$FIXTURES_DIR/eyes/README.txt" << 'EOF'
Closed Eyes in the Wild (CEW) Dataset

Download from:
  https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html

Expected structure:
  eyes/
  ├── open_eyes/
  │   └── *.jpg
  └── closed_eyes/
      └── *.jpg

After downloading, run: ./scripts/fetch-test-data.sh verify eyes
EOF
}

fetch_exposure_datasets() {
    info "Fetching exposure datasets..."

    warn "VV-Dataset requires manual download from:"
    warn "  ${DATASET_URLS[exposure]}"
    warn "  Extract to: $FIXTURES_DIR/exposure/"

    mkdir -p "$FIXTURES_DIR/exposure"
    cat > "$FIXTURES_DIR/exposure/README.txt" << 'EOF'
VV Exposure Dataset

Download from:
  https://sites.google.com/site/vonikakis/datasets

Expected structure:
  exposure/
  ├── underexposed/
  │   └── *.jpg
  ├── overexposed/
  │   └── *.jpg
  └── normal/
      └── *.jpg

After downloading, run: ./scripts/fetch-test-data.sh verify exposure
EOF
}

# Interactive mode
interactive_mode() {
    echo ""
    echo -e "${BLUE}=== Interactive Dataset Setup ===${NC}"
    echo ""

    show_status

    echo "What would you like to do?"
    echo ""
    echo "  1) Download all datasets (shows instructions)"
    echo "  2) Download specific dataset"
    echo "  3) Verify a dataset"
    echo "  4) Generate ground truth template"
    echo "  5) Exit"
    echo ""

    read -r -p "Enter choice [1-5]: " choice

    case "$choice" in
        1)
            fetch_blur_datasets
            echo ""
            fetch_eyes_datasets
            echo ""
            fetch_exposure_datasets
            ;;
        2)
            echo ""
            echo "Available datasets: blur, eyes, exposure"
            read -r -p "Enter dataset name: " dataset
            case "$dataset" in
                blur) fetch_blur_datasets ;;
                eyes) fetch_eyes_datasets ;;
                exposure) fetch_exposure_datasets ;;
                *) error "Unknown dataset: $dataset" ;;
            esac
            ;;
        3)
            echo ""
            echo "Available datasets: blur, eyes, exposure"
            read -r -p "Enter dataset name to verify: " dataset
            verify_dataset "$dataset"
            ;;
        4)
            echo ""
            echo "Available datasets: blur, eyes, exposure"
            read -r -p "Enter dataset name: " dataset
            generate_ground_truth_template "$dataset"
            ;;
        5)
            exit 0
            ;;
        *)
            error "Invalid choice"
            ;;
    esac
}

# Main
main() {
    local command="${1:-all}"
    local arg="${2:-}"

    info "Photo QA Test Data Manager"
    info "Fixtures directory: $FIXTURES_DIR"

    case "$command" in
        status)
            show_status
            ;;
        verify)
            if [[ -z "$arg" ]]; then
                error "Usage: $0 verify <dataset>"
            fi
            verify_dataset "$arg"
            ;;
        interactive)
            interactive_mode
            ;;
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
            echo ""
            fetch_eyes_datasets
            echo ""
            fetch_exposure_datasets
            ;;
        ground-truth)
            if [[ -z "$arg" ]]; then
                error "Usage: $0 ground-truth <dataset>"
            fi
            generate_ground_truth_template "$arg"
            ;;
        *)
            error "Unknown command: $command. Use 'blur', 'eyes', 'exposure', 'all', 'status', 'verify <name>', 'interactive', or 'ground-truth <name>'"
            ;;
    esac

    echo ""
    info "Done!"
    if [[ "$command" != "status" && "$command" != "verify" ]]; then
        warn "Some datasets require manual download due to licensing restrictions."
        info "Run '$0 status' to check download status."
    fi
}

main "$@"
