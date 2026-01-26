# Photo QA - Task Breakdown

## Phase 1: Project Foundation

### 1.1 Workspace Setup
- [x] Create Cargo workspace with three crates
- [x] Configure `photo-qa-core`, `photo-qa-cli`, `photo-qa-adapters`
- [x] Set up strict clippy lints (`pedantic`, `nursery`, `unwrap_used`, `expect_used`)
- [x] Deny `unsafe_code`
- [x] Configure MSRV 1.82

### 1.2 Development Tooling
- [x] Create `mise.toml` with Rust toolchain
- [x] Create `justfile` with build/test/lint targets
- [x] Set up `lefthook.yml` for pre-commit hooks
- [x] Add `just bootstrap` target

### 1.3 CI/CD Pipeline
- [x] GitHub Actions: fmt/lint/test on push/PR
- [x] Multi-platform build (Linux x86_64, macOS x86_64, macOS ARM64)
- [x] semantic-release configuration
- [x] Release binary publishing to GitHub Releases

---

## Phase 2: Domain Core (`photo-qa-core`)

### 2.1 Core Types
- [x] Define `Issue` type (type, score, details)
- [x] Define `AnalysisResult` type (path, timestamp, dimensions, issues, exif)
- [x] Define `ImageDimensions` type
- [x] Define `BoundingBox` type

### 2.2 Port Traits
- [x] `ImageSource` trait - load images from various sources
- [x] `ResultOutput` trait - emit analysis results
- [x] `ProgressSink` trait - progress events for CLI/TUI

### 2.3 QaModule Trait
- [ ] Define `QaModule` trait with `analyze(&Image) -> Vec<Issue>`
- [ ] Define module registration/discovery pattern
- [ ] Define module configuration interface

### 2.4 Image Abstraction
- [ ] Create internal `Image` type wrapping decoded data
- [ ] Support RGB/RGBA pixel access
- [ ] Metadata extraction interface (dimensions, format)

---

## Phase 3: Image Loading (`photo-qa-adapters`)

### 3.1 Raster Format Support
- [ ] Add `image` crate dependency
- [ ] Implement JPEG loading
- [ ] Implement PNG loading
- [ ] Implement TIFF loading
- [ ] Implement WebP loading
- [ ] Implement BMP loading
- [ ] Implement GIF loading (first frame)

### 3.2 RAW Format Support
- [ ] Add `rawloader` crate dependency
- [ ] Implement CR2/CR3 (Canon) loading
- [ ] Implement NEF (Nikon) loading
- [ ] Implement ARW (Sony) loading
- [ ] Implement RAF (Fuji) loading
- [ ] Implement DNG loading
- [ ] Add `--raw-full-decode` flag support

### 3.3 Filesystem Adapter
- [ ] Implement `ImageSource` for filesystem
- [ ] Directory traversal (recursive option)
- [ ] File type filtering by extension
- [ ] Error handling: skip unreadable files with warning

---

## Phase 4: Inference Engine (`photo-qa-core/inference`)

### 4.1 Candle Integration
- [ ] Add `candle-core`, `candle-nn` dependencies
- [ ] Configure Metal backend (macOS)
- [ ] Configure CUDA backend (optional feature)
- [ ] CPU fallback with auto-detection

### 4.2 Model Loading
- [ ] Safetensors loading utility
- [ ] Model path resolution (XDG_DATA_HOME)
- [ ] SHA256 checksum verification
- [ ] Lazy model loading (on first use)

### 4.3 Model Management Adapter
- [ ] Model inventory definition (name, size, checksum, URL)
- [ ] Download from GitHub Releases
- [ ] Progress reporting during download
- [ ] `models fetch` implementation
- [ ] `models list` implementation
- [ ] `models path` implementation

---

## Phase 5: QA Modules

### 5.1 Exposure Module (simplest, no ML)
- [ ] Convert image to luminance channel
- [ ] Compute 256-bin histogram
- [ ] Calculate percentiles (5th, 95th)
- [ ] Calculate mean, std deviation
- [ ] Compute `under_score` (shadow clipping)
- [ ] Compute `over_score` (highlight clipping)
- [ ] Configurable thresholds
- [ ] Unit tests with synthetic images

### 5.2 Blur Module
- [ ] **Sharpness Analysis**
  - [ ] Laplacian variance calculation
  - [ ] Sliding window implementation
  - [ ] Per-region sharpness scoring
- [ ] **Subject Detection (Heuristic)**
  - [ ] Edge-density heuristic (primary)
  - [ ] Define "subject region" extraction
- [ ] **Subject Detection (ML Fallback)**
  - [ ] U²-Net model integration
  - [ ] IS-Net model integration
  - [ ] Benchmark both, select winner
  - [ ] Saliency map → bounding box conversion
- [ ] **Blur Type Classification**
  - [ ] FFT directional analysis
  - [ ] Motion blur detection (directional energy)
  - [ ] Defocus blur detection (uniform variance drop)
  - [ ] Classification: `motion | defocus | mixed | sharp`
- [ ] Output: `blur_score`, `blur_type`, `subject_bbox`
- [ ] Unit tests with blur dataset samples

### 5.3 Closed Eyes Module
- [ ] **Face Detection**
  - [ ] RetinaFace model integration
  - [ ] Face bounding box extraction
  - [ ] Confidence score filtering
- [ ] **Landmark Extraction**
  - [ ] 68-point landmark model integration
  - [ ] Eye point extraction (37-42, 43-48)
- [ ] **EAR Calculation**
  - [ ] Implement EAR formula
  - [ ] Per-face baseline estimation
  - [ ] Relative threshold comparison
- [ ] Output: per-face `{bbox, left_ear, right_ear, eyes_closed, confidence}`
- [ ] Unit tests with CEW dataset samples

---

## Phase 6: CLI (`photo-qa-cli`)

### 6.1 Command Structure
- [ ] Root command (default: run all checks)
- [ ] `check` subcommand (explicit)
- [ ] `models` subcommand group
  - [ ] `models fetch`
  - [ ] `models list`
  - [ ] `models path`

### 6.2 Input Options
- [ ] `<PATHS>...` positional arguments
- [ ] `--recursive, -r` flag
- [ ] Path validation and expansion

### 6.3 Check Toggles
- [ ] `--blur` / `--no-blur`
- [ ] `--exposure` / `--no-exposure`
- [ ] `--eyes` / `--no-eyes`
- [ ] Default: all enabled

### 6.4 Threshold Options
- [ ] `--blur-threshold <0.0-1.0>`
- [ ] `--under-threshold <0.0-1.0>`
- [ ] `--over-threshold <0.0-1.0>`
- [ ] `--ear-threshold <0.0-1.0>`
- [ ] Validation and default values

### 6.5 Output Options
- [ ] `--format <json|jsonl>` (default: jsonl)
- [ ] `--exif` flag for EXIF metadata
- [ ] `--progress` / `--quiet` flags
- [ ] TTY auto-detection for progress

### 6.6 Output Formatting
- [ ] JSON serialization of `AnalysisResult`
- [ ] JSONL streaming output
- [ ] Pretty JSON option
- [ ] Progress bar via `indicatif`

### 6.7 Exit Codes
- [ ] `0`: success, no issues
- [ ] `1`: success, issues found
- [ ] `2`: runtime error

### 6.8 Logging
- [ ] `tracing` + `tracing-subscriber` setup
- [ ] `-v` / `-vv` / `-vvv` verbosity levels
- [ ] `RUST_LOG` environment override

---

## Phase 7: Configuration

### 7.1 Config File Support
- [ ] TOML parsing via `toml` crate
- [ ] XDG config path (`~/.config/photo-qa/config.toml`)
- [ ] Project-local `.photo-qa.toml`
- [ ] Parent directory traversal for config

### 7.2 Config Layering
- [ ] Load XDG config (lowest priority)
- [ ] Override with project config
- [ ] Override with CLI flags (highest priority)

### 7.3 Config Schema
- [ ] `[general]` section
- [ ] `[blur]` section
- [ ] `[exposure]` section
- [ ] `[eyes]` section
- [ ] `[models]` section
- [ ] `[output]` section

---

## Phase 8: Testing Infrastructure

### 8.1 Unit Tests
- [ ] Mock inference outputs
- [ ] Test domain logic in isolation
- [ ] Test threshold calculations
- [ ] Test configuration merging

### 8.2 Integration Tests
- [ ] Synthetic test image generation
- [ ] Full pipeline tests
- [ ] CLI argument parsing tests
- [ ] Output format verification

### 8.3 Test Data Management
- [ ] `scripts/fetch-test-data.sh`
- [ ] CERTH Blur dataset download
- [ ] RealBlur dataset download
- [ ] CEW dataset download
- [ ] VV-Dataset download
- [ ] Afifi Multi-Exposure download
- [ ] `.gitignore` test fixtures

### 8.4 E2E Accuracy Tests
- [ ] Blur detection accuracy benchmarks
- [ ] Exposure detection accuracy benchmarks
- [ ] Closed eyes detection accuracy benchmarks
- [ ] Threshold tuning based on results

---

## Phase 9: Model Preparation

### 9.1 Model Conversion
- [ ] Convert RetinaFace PyTorch → safetensors
- [ ] Convert 68-point landmarks dlib → safetensors
- [ ] Convert U²-Net PyTorch → safetensors
- [ ] Convert IS-Net PyTorch → safetensors

### 9.2 Model Validation
- [ ] Verify inference output matches original
- [ ] Benchmark inference speed
- [ ] Document model sizes and checksums

### 9.3 Model Distribution
- [ ] Upload models to GitHub Releases
- [ ] Create model manifest with checksums
- [ ] Test auto-download flow

---

## Phase 10: Polish & Release

### 10.1 Error Handling Review
- [ ] Audit all error paths
- [ ] Ensure skip+warn behavior
- [ ] Graceful model download failures
- [ ] Helpful error messages

### 10.2 Performance
- [ ] Profile with large image sets
- [ ] Memory usage optimization
- [ ] Consider lazy image decoding

### 10.3 Documentation
- [ ] README with usage examples
- [ ] `--help` text review
- [ ] CHANGELOG setup

### 10.4 Release Prep
- [ ] Version 0.1.0 checklist
- [ ] Binary smoke tests (all platforms)
- [ ] Model bundle verification
