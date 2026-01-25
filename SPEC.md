# Photo QA - Technical Specification

A Rust-based CLI tool for automated photo quality assessment.

## Overview

Photo QA analyzes images to detect common quality issues:
- **Blur detection** - motion blur vs defocus blur classification
- **Exposure analysis** - underexposed and overexposed image detection
- **Closed eyes detection** - per-face analysis in portraits/group shots

CLI-first with planned TUI support via ratatui.

## Architecture

### Hexagonal Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Adapters                             │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────────────┐│
│  │   CLI   │  │   TUI   │  │Filesystem│  │  Model Loader   ││
│  │ (clap)  │  │(ratatui)│  │ Adapter  │  │  (auto-download)││
│  └────┬────┘  └────┬────┘  └────┬─────┘  └───────┬─────────┘│
└───────┼────────────┼────────────┼────────────────┼──────────┘
        │            │            │                │
┌───────┴────────────┴────────────┴────────────────┴──────────┐
│                         Ports                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ ImageSource  │  │ ResultOutput │  │ProgressSink  │       │
│  │    trait     │  │    trait     │  │    trait     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
        │                    │                 │
┌───────┴────────────────────┴─────────────────┴─────────────┐
│                      Domain Core                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   QaModule trait                    │   │
│  │  ┌─────────┐  ┌──────────┐  ┌────────────────────┐  │   │
│  │  │  Blur   │  │ Exposure │  │    ClosedEyes      │  │   │
│  │  │ Module  │  │  Module  │  │      Module        │  │   │
│  │  └─────────┘  └──────────┘  └────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Inference Engine (candle)              │   │
│  │  ┌───────────┐  ┌───────────┐  ┌─────────────────┐  │   │
│  │  │ RetinaFace│  │ Saliency  │  │ 68-pt Landmarks │  │   │
│  │  │  (faces)  │  │(U²/IS-Net)│  │   (eye EAR)     │  │   │
│  │  └───────────┘  └───────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### Workspace Structure

```
photo-qa/
├── Cargo.toml              # Workspace manifest
├── crates/
│   ├── photo-qa-core/      # Domain logic, QA modules, inference
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── domain/     # Core types, QaModule trait
│   │   │   ├── modules/    # blur, exposure, eyes implementations
│   │   │   ├── inference/  # candle wrappers, model loading
│   │   │   └── ports/      # trait definitions
│   │   └── Cargo.toml
│   ├── photo-qa-cli/       # CLI binary
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── commands/   # check, models, etc.
│   │   │   └── output/     # JSON, progress formatting
│   │   └── Cargo.toml
│   └── photo-qa-adapters/  # Filesystem, stdout, model download
│       ├── src/
│       │   ├── lib.rs
│       │   ├── fs.rs
│       │   ├── stdout.rs
│       │   └── models.rs
│       └── Cargo.toml
├── models/                 # .gitignored, auto-downloaded
├── tests/                  # Integration tests
│   └── fixtures/           # .gitignored, script-downloaded
├── scripts/
│   └── fetch-test-data.sh  # Downloads test datasets
├── justfile
├── lefthook.yml
└── mise.toml
```

## QA Modules

### Blur Detection

**Approach**: Hybrid subject detection + sharpness analysis

1. **Subject Detection**
   - Primary: Edge-density heuristic (Laplacian variance in sliding windows)
   - Fallback: Saliency model (U²-Net or IS-Net) if edge detection is ambiguous
   - Evaluate sharpness in detected subject region, not whole image

2. **Blur Type Classification**
   - Heuristic-based (FFT directional analysis + Laplacian patterns)
   - Motion blur: directional FFT energy distribution
   - Defocus blur: uniform Laplacian variance drop
   - Extensible for ML classifier later

3. **Output**
   - `blur_score`: 0.0 (sharp) to 1.0 (blurry)
   - `blur_type`: `motion` | `defocus` | `mixed` | `sharp`
   - Subject region bounding box

**Models** (test both, pick winner):
- U²-Net (~44MB safetensors)
- IS-Net (~43MB safetensors)

### Exposure Analysis

**Approach**: Histogram-based adaptive analysis with percentile bounds

1. **Analysis**
   - Convert to luminance channel
   - Compute histogram (256 bins)
   - Calculate: 5th percentile, 95th percentile, mean, std deviation
   - Flag underexposed if significant shadow clipping (5th percentile region)
   - Flag overexposed if significant highlight clipping (95th percentile region)

2. **Output**
   - `under_score`: 0.0 (ok) to 1.0 (severely underexposed)
   - `over_score`: 0.0 (ok) to 1.0 (severely overexposed)
   - Histogram summary statistics

**No ML models required** - pure algorithmic analysis.

### Closed Eyes Detection

**Approach**: Face detection → landmark extraction → per-face EAR calculation

1. **Face Detection**
   - RetinaFace model (safetensors, ~30MB estimated)
   - Returns face bounding boxes + confidence

2. **Landmark Extraction**
   - 68-point landmark model (dlib-compatible, safetensors)
   - Points 37-42: left eye, 43-48: right eye

3. **Eye Aspect Ratio (EAR)**
   - `EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)`
   - Per-face calibration: estimate open-eye EAR from face geometry
   - Threshold relative to estimated baseline

4. **Output**
   - Per-face results array
   - Each face: `{bbox, left_ear, right_ear, eyes_closed: bool, confidence}`
   - Image not flagged overall; consumer decides policy

**Models**:
- RetinaFace (safetensors)
- 68-point facial landmarks (safetensors)

## CLI Interface

### Commands

```bash
# Default: run all checks
photo-qa [OPTIONS] <PATHS>...

# Subcommands
photo-qa check [OPTIONS] <PATHS>...    # Explicit check command
photo-qa models fetch                   # Download required models
photo-qa models list                    # Show model status
photo-qa models path                    # Print model directory
```

### Options

```
Input:
  <PATHS>...                  Files or directories to analyze
  --recursive, -r             Recurse into subdirectories

Checks (default: all enabled):
  --blur                      Enable blur detection
  --exposure                  Enable exposure analysis
  --eyes                      Enable closed-eyes detection
  --no-blur                   Disable blur detection
  --no-exposure               Disable exposure analysis
  --no-eyes                   Disable closed-eyes detection

Thresholds (override defaults):
  --blur-threshold <0.0-1.0>  Blur score threshold (default: 0.5)
  --under-threshold <0.0-1.0> Underexposure threshold (default: 0.3)
  --over-threshold <0.0-1.0>  Overexposure threshold (default: 0.3)
  --ear-threshold <0.0-1.0>   Eye aspect ratio threshold (default: 0.2)

Output:
  --format <json|jsonl>       Output format (default: jsonl)
  --exif                      Include EXIF metadata in output
  --progress                  Show progress bar (default: auto-detect TTY)
  --quiet, -q                 Suppress progress output

RAW files:
  --raw-full-decode           Full RAW decode (default: use embedded preview)

Verbosity:
  -v, --verbose               Increase verbosity (-v, -vv, -vvv)

Config:
  --config <PATH>             Path to config file
```

### Exit Codes

- `0`: Success, no issues found
- `1`: Success, issues found in one or more images
- `2`: Runtime error (invalid input, model load failure, etc.)

## Output Format

JSON Lines (one object per line per image):

```json
{
  "path": "/path/to/image.jpg",
  "timestamp": "2024-01-15T10:30:00Z",
  "dimensions": {"width": 4000, "height": 3000},
  "issues": [
    {
      "type": "blur",
      "score": 0.72,
      "details": {
        "blur_type": "motion",
        "subject_bbox": [100, 100, 800, 600]
      }
    },
    {
      "type": "exposure",
      "score": 0.45,
      "details": {
        "direction": "under",
        "under_score": 0.45,
        "over_score": 0.02
      }
    },
    {
      "type": "closed_eyes",
      "score": 0.85,
      "details": {
        "faces": [
          {
            "bbox": [200, 150, 300, 350],
            "left_ear": 0.15,
            "right_ear": 0.18,
            "eyes_closed": true,
            "confidence": 0.95
          }
        ]
      }
    }
  ],
  "exif": {
    "camera": "Canon EOS R5",
    "lens": "RF 24-70mm F2.8",
    "iso": 3200,
    "shutter_speed": "1/60",
    "aperture": "f/2.8"
  }
}
```

## Configuration

Layered configuration: XDG config < project file < CLI flags

### Config Locations

1. `~/.config/photo-qa/config.toml` (XDG)
2. `.photo-qa.toml` in cwd or parent directories
3. CLI flags override all

### Config File Format

```toml
[general]
recursive = true
format = "jsonl"
include_exif = false

[blur]
enabled = true
threshold = 0.5
# subject_detection = "hybrid"  # future

[exposure]
enabled = true
under_threshold = 0.3
over_threshold = 0.3

[eyes]
enabled = true
ear_threshold = 0.2

[models]
# path = "~/.local/share/photo-qa/models"  # default
# download_url = "https://github.com/..."  # override source

[output]
progress = "auto"  # auto | always | never
```

## Image Format Support

### Raster Formats
Via `image` crate:
- JPEG
- PNG
- TIFF
- WebP
- BMP
- GIF (first frame)

### RAW Formats
Via `rawloader` crate:
- Canon CR2, CR3
- Nikon NEF
- Sony ARW
- Fuji RAF
- Adobe DNG
- Others supported by rawloader

**RAW Strategy**: Full decode to RGB by default. Memory-intensive but accurate for QA.

## Inference Engine

### Runtime: Candle

- Pure Rust ML framework by Hugging Face
- Native safetensors support
- Metal (macOS), CUDA, CPU backends
- Auto-detect GPU; fall back to CPU silently

### Model Format

All models stored as safetensors (Candle's preferred format).

### Model Management

**Auto-download on first run:**
1. Check `~/.local/share/photo-qa/models/` (XDG_DATA_HOME)
2. If missing, download from GitHub Releases
3. Verify SHA256 checksums
4. Cache indefinitely

**Manual management:**
```bash
photo-qa models fetch          # Force (re)download
photo-qa models list           # Show installed models
photo-qa models path           # Print model directory
```

### Model Inventory

| Model | Purpose | Size (est.) | Source |
|-------|---------|-------------|--------|
| RetinaFace | Face detection | ~30MB | Convert from PyTorch |
| 68-point landmarks | Eye landmarks | ~5MB | Convert from dlib |
| U²-Net | Saliency (blur) | ~44MB | Convert from PyTorch |
| IS-Net | Saliency (blur) | ~43MB | Convert from PyTorch |

## Concurrency & Progress

### Single-threaded (v1)
- Sequential image processing
- Simpler debugging, predictable behavior
- Parallelism deferred to future version

### Progress Reporting

- Progress bar via `indicatif` crate
- Auto-detect TTY; disable for pipes
- Two-level events for future TUI:
  - Progress events (image started, % complete)
  - Final result events (per-image results)

### Channel Architecture (for TUI)

- Bounded channels (capacity ~100)
- Backpressure if consumer lags
- Domain emits events; CLI/TUI subscribes

## Error Handling

**Policy: Skip + Warn**

- Log warning per unreadable/corrupt file
- Continue processing remaining files
- Exit 0 if any images processed successfully
- Exit 2 only on fatal errors (no valid inputs, model load failure)

```
WARN: Skipping /path/to/corrupt.jpg: Invalid JPEG header
WARN: Skipping /path/to/unknown.xyz: Unsupported format
```

## Testing Strategy

### Test Levels

1. **Unit tests** - Mock inference outputs; test domain logic
2. **Integration tests** - Synthetic generated images; test full pipeline
3. **E2E tests** - Real datasets (not in repo); test accuracy

### Test Datasets

Script downloads from public sources:

| Dataset | Purpose | Source |
|---------|---------|--------|
| CERTH Blur | Blur detection | Academic dataset |
| RealBlur | Blur with ground truth | POSTECH |
| CEW (Closed Eyes in Wild) | Closed eyes | NUAA |
| VV-Dataset | Exposure (challenging) | Vonikakis |
| Afifi Multi-Exposure | Exposure levels | MIT-Adobe derived |

```bash
./scripts/fetch-test-data.sh    # Downloads all
./scripts/fetch-test-data.sh blur  # Downloads blur only
```

Test fixtures stored in `tests/fixtures/` (.gitignored).

## Development Setup

### Prerequisites

```bash
# Install mise (tool version manager)
curl https://mise.run | sh

# Install tools
mise install

# Install lefthook
lefthook install
```

### mise.toml

```toml
[tools]
rust = "1.75"
```

### Justfile

```just
default:
    @just --list

# Development
build:
    cargo build

check:
    cargo check --all-targets

test:
    cargo test

test-integration:
    cargo test --test '*'

# Quality
fmt:
    cargo fmt

fmt-check:
    cargo fmt --check

lint:
    cargo clippy --all-targets -- -D warnings

# Full validation
ci: fmt-check lint test

# Models
fetch-models:
    cargo run -- models fetch

# Test data
fetch-test-data:
    ./scripts/fetch-test-data.sh

# Release
release-dry-run:
    cargo build --release
```

### Lefthook

```yaml
pre-commit:
  parallel: true
  commands:
    fmt:
      run: cargo fmt --check
    clippy:
      run: cargo clippy --all-targets -- -D warnings
```

## CI/CD

### GitHub Actions

**On push/PR:**
- Format check (`cargo fmt --check`)
- Lint (`cargo clippy`)
- Test (`cargo test`)
- Build (Linux + macOS)

**On push to main:**
- semantic-release determines version
- Build release binaries (Linux x86_64, macOS x86_64, macOS ARM64)
- Create GitHub Release with binaries
- Attach model files to release

### semantic-release

Conventional commits trigger releases:
- `feat:` → minor version
- `fix:` → patch version
- `feat!:` or `BREAKING CHANGE:` → major version

## Future Extensions

### Planned Modules

- **Duplicate detection** - pHash-based near-duplicate finding
- **Noise detection** - ISO noise / grain analysis
- **Composition** - Rule of thirds, horizon level
- **Color cast** - White balance issues

### Architecture Provisions

- `QaModule` trait designed for extensibility
- Static compilation now; WASM plugin system future
- Channel-based events ready for TUI integration

## Logging

### Crate: `tracing` + `tracing-subscriber`

```rust
// Verbosity levels
-v    → INFO
-vv   → DEBUG
-vvv  → TRACE
```

Environment override: `RUST_LOG=photo_qa=debug`

## License

MIT

## References

### Datasets
- [CERTH Blur Dataset](https://github.com/priyabagaria/Image-Blur-Detection)
- [RealBlur Dataset](https://cg.postech.ac.kr/research/realblur/)
- [CEW - Closed Eyes in Wild](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html)
- [VV-Dataset](https://sites.google.com/site/vonikakis/datasets)
- [Multi-Scale Exposure Correction](https://arxiv.org/abs/2003.11596)

### Models & Techniques
- [RetinaFace](https://arxiv.org/abs/1905.00641)
- [U²-Net](https://arxiv.org/abs/2005.09007)
- [IS-Net](https://arxiv.org/abs/2203.03041)
- [Eye Aspect Ratio (EAR)](https://www.sciencedirect.com/science/article/pii/S0923596517301303)
- [Candle ML Framework](https://github.com/huggingface/candle)
