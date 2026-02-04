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
- [x] Define `QaModule` trait with `analyze(&Image) -> Vec<Issue>`
- [x] Define module registration/discovery pattern
- [x] Define module configuration interface

### 2.4 Image Abstraction
- [x] Create internal `Image` type wrapping decoded data
- [x] Support RGB/RGBA pixel access
- [x] Metadata extraction interface (dimensions, format)

---

## Phase 3: Image Loading (`photo-qa-adapters`)

### 3.1 Raster Format Support
- [x] Add `image` crate dependency
- [x] Implement JPEG loading
- [x] Implement PNG loading
- [x] Implement TIFF loading
- [x] Implement WebP loading
- [x] Implement BMP loading
- [x] Implement GIF loading (first frame)

### 3.2 RAW Format Support
- [x] Add `rawloader` crate dependency
- [x] Implement CR2/CR3 (Canon) loading
- [x] Implement NEF (Nikon) loading
- [x] Implement ARW (Sony) loading
- [x] Implement RAF (Fuji) loading
- [x] Implement DNG loading
- [ ] Add `--raw-full-decode` flag support

### 3.3 Filesystem Adapter
- [x] Implement `ImageSource` for filesystem
- [x] Directory traversal (recursive option)
- [x] File type filtering by extension
- [x] Error handling: skip unreadable files with warning

---

## Phase 4: Inference Engine (`photo-qa-core/inference`)

### 4.1 Candle Integration
- [x] Add `candle-core`, `candle-nn` dependencies
- [x] Configure Metal backend (macOS)
- [x] Configure CUDA backend (optional feature)
- [x] CPU fallback with auto-detection

### 4.2 Model Loading
- [x] Safetensors loading utility
- [x] Model path resolution (XDG_DATA_HOME)
- [x] SHA256 checksum verification
- [x] Lazy model loading (on first use)

### 4.3 Model Management Adapter
- [x] Model inventory definition (name, size, checksum, URL)
- [x] Download from GitHub Releases
- [x] Progress reporting during download
- [x] `models fetch` implementation
- [x] `models list` implementation
- [x] `models path` implementation

---

## Phase 5: QA Modules

### 5.1 Exposure Module (simplest, no ML)
- [x] Convert image to luminance channel
- [x] Compute 256-bin histogram
- [x] Calculate percentiles (5th, 95th)
- [x] Calculate mean, std deviation
- [x] Compute `under_score` (shadow clipping)
- [x] Compute `over_score` (highlight clipping)
- [x] Configurable thresholds
- [x] Unit tests with synthetic images

### 5.2 Blur Module
- [x] **Sharpness Analysis**
  - [x] Laplacian variance calculation
  - [x] Sliding window implementation
  - [x] Per-region sharpness scoring
- [x] **Subject Detection (Heuristic)**
  - [x] Edge-density heuristic (primary)
  - [x] Define "subject region" extraction
- [ ] **Subject Detection (ML Fallback)** (deferred to Phase 9)
  - [ ] U²-Net model integration
  - [ ] IS-Net model integration
  - [ ] Benchmark both, select winner
  - [ ] Saliency map → bounding box conversion
- [x] **Blur Type Classification**
  - [x] FFT directional analysis
  - [x] Motion blur detection (directional energy)
  - [x] Defocus blur detection (uniform variance drop)
  - [x] Classification: `motion | defocus | mixed | sharp`
- [x] Output: `blur_score`, `blur_type`, `subject_bbox`
- [x] Unit tests with synthetic images

### 5.3 Closed Eyes Module
- [x] **Face Detection**
  - [x] BlazeFace model integration (instead of RetinaFace - simpler architecture)
  - [x] Face bounding box extraction
  - [x] Confidence score filtering
- [x] **Eye State Classification**
  - [x] Eye state CNN classifier (instead of 68-point landmarks)
  - [x] Crops eye regions from BlazeFace keypoints
  - [x] Binary open/closed classification
- [x] **EAR Approximation**
  - [x] Map classifier probability to EAR-equivalent score
  - [x] Per-face threshold comparison
- [x] Output: per-face `{bbox, left_ear, right_ear, eyes_closed, confidence}`
- [x] Unit tests (basic config/module tests; full CEW tests pending model files)

---

## Phase 6: CLI (`photo-qa-cli`)

### 6.1 Command Structure
- [x] Root command (default: run all checks)
- [x] `check` subcommand (explicit)
- [x] `models` subcommand group
  - [x] `models fetch`
  - [x] `models list`
  - [x] `models path`

### 6.2 Input Options
- [x] `<PATHS>...` positional arguments
- [x] `--recursive, -r` flag
- [x] Path validation and expansion

### 6.3 Check Toggles
- [x] `--blur` / `--no-blur`
- [x] `--exposure` / `--no-exposure`
- [x] `--eyes` / `--no-eyes`
- [x] Default: all enabled

### 6.4 Threshold Options
- [x] `--blur-threshold <0.0-1.0>`
- [x] `--under-threshold <0.0-1.0>`
- [x] `--over-threshold <0.0-1.0>`
- [x] `--ear-threshold <0.0-1.0>`
- [x] Validation and default values

### 6.5 Output Options
- [x] `--format <json|jsonl>` (default: jsonl)
- [x] `--exif` flag for EXIF metadata
- [x] `--progress` / `--quiet` flags
- [x] TTY auto-detection for progress

### 6.6 Output Formatting
- [x] JSON serialization of `AnalysisResult`
- [x] JSONL streaming output
- [x] Pretty JSON option (`--pretty` flag)
- [x] Progress bar via `indicatif`

### 6.7 Exit Codes
- [x] `0`: success, no issues
- [x] `1`: success, issues found
- [x] `2`: runtime error

### 6.8 Logging
- [x] `tracing` + `tracing-subscriber` setup
- [x] `-v` / `-vv` / `-vvv` verbosity levels
- [x] `RUST_LOG` environment override

---

## Phase 7: Configuration

### 7.1 Config File Support
- [x] TOML parsing via `toml` crate
- [x] XDG config path (`~/.config/photo-qa/config.toml`)
- [x] Project-local `.photo-qa.toml`
- [x] Parent directory traversal for config

### 7.2 Config Layering
- [x] Load XDG config (lowest priority)
- [x] Override with project config
- [x] Override with CLI flags (highest priority)

### 7.3 Config Schema
- [x] `[general]` section
- [x] `[blur]` section
- [x] `[exposure]` section
- [x] `[eyes]` section
- [x] `[models]` section
- [x] `[output]` section

---

## Phase 8: Testing Infrastructure

### 8.1 Unit Tests
- [x] Mock inference outputs (`photo-qa-test-support` crate)
- [x] Test domain logic in isolation (blur/exposure threshold tests)
- [x] Test threshold calculations (boundary tests in blur.rs/exposure.rs)
- [x] Test configuration merging (config.rs merge priority tests)

### 8.2 Integration Tests
- [x] Synthetic test image generation (`SyntheticImageBuilder`)
- [x] Full pipeline tests (`tests/pipeline.rs`)
- [x] CLI argument parsing tests (`tests/cli_args.rs`)
- [x] Output format verification (`tests/output_format.rs`)

### 8.3 Test Data Management
- [x] `scripts/fetch-test-data.sh` (status, verify, interactive commands)
- [x] CERTH Blur dataset download instructions
- [x] CEW dataset download instructions
- [x] VV-Dataset download instructions
- [x] Ground truth template generation

### 8.4 E2E Accuracy Tests
- [x] Accuracy benchmark framework (`tests/accuracy.rs`)
- [x] Blur detection accuracy benchmarks (ignored test)
- [x] Exposure detection accuracy benchmarks (ignored test)
- [x] Closed eyes detection accuracy benchmarks (ignored test)
- [x] Threshold sweep utilities

---

## Phase 9: Model Preparation

### 9.1 Model Conversion Scripts
- [x] Create `scripts/models/requirements.txt` (PyTorch, safetensors deps)
- [x] Create `scripts/models/convert_blazeface.py` (MediaPipe → safetensors)
- [x] Create `scripts/models/convert_eye_state.py` (train + export to safetensors)
- [x] Create `scripts/models/convert_u2net.py` (official weights → safetensors)

### 9.2 Model Validation
- [x] Create `scripts/models/verify_models.py` (unified verification script)
- [ ] Benchmark inference speed (PyTorch vs Candle)

### 9.3 Model Distribution
- [x] Generate safetensors files for all models (random init for dev)
- [x] Create `scripts/models/release.sh` (GitHub release workflow)
- [x] Verify models load in Rust via `cargo run -- check`
- [x] Train/obtain production-ready model weights
  - [x] U²-Net: Official pretrained from Google Drive
  - [x] Eye State: Trained on CEW dataset (96% val accuracy)
  - [x] BlazeFace: Pretrained from hollance/BlazeFace-PyTorch (MediaPipe weights)
- [x] Update `models.rs` with production checksums
- [x] Create GitHub Release `models-v1` with trained models
- [x] Test auto-download flow end-to-end

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
