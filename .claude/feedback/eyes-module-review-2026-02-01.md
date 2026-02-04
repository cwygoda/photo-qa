# Code Review: Closed Eyes Detection Module (BlazeFace + Eye State Classifier)

**Reviewed:** 2026-02-01
**Files:** `blazeface.rs`, `eye_state.rs`, `eyes.rs`, `models.rs`, `inference/mod.rs`, `TASKS.md`
**Scope:** New ML-based closed eyes detection pipeline using BlazeFace face detection and eye state binary classifier

## Summary

Solid implementation of the eyes detection pipeline. The architecture change from RetinaFace/68-landmarks to BlazeFace/binary-classifier is pragmatic - simpler, fewer models, and well-suited for the binary open/closed use case. Code is clean, passes clippy with pedantic settings, and follows existing patterns. A few edge cases and potential improvements below.

## Critical Issues

### 1. Duplicate sigmoid function
**Location:** `blazeface.rs:478` and `eye_state.rs:242`
**Problem:** Same `sigmoid` function defined twice - DRY violation, minor but easy to miss if one gets updated.
**Fix:**
```rust
// Add to inference/mod.rs or create inference/utils.rs
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
```

### 2. Model hash placeholders will pass validation
**Location:** `models.rs:36-37, 43-44`
**Problem:** SHA256 hashes are all zeros - model downloads will fail integrity check or (worse) pass with corrupted files if hash validation isn't enforced.
```rust
sha256: "0000000000000000000000000000000000000000000000000000000000000000", // TODO: Update with real hash
```
**Fix:** Generate real hashes once model files are released, or add explicit `// SKIP_VALIDATION` flag that the downloader recognizes.

## Improvements

### 3. Eye crop bounds could go negative
**Location:** `eye_state.rs:197-198`
**Current:**
```rust
let x = eye_center[0] - eye_w / 2.0;
let y = eye_center[1] - eye_h / 2.0;
```
**Suggested:**
```rust
let x = (eye_center[0] - eye_w / 2.0).max(0.0);
let y = (eye_center[1] - eye_h / 2.0).max(0.0);
```
**Why:** If eye keypoint is near image edge, negative coords passed to `preprocess()` could cause issues. The pixel conversion in `preprocess` handles this partially via `min(image.width()-1)`, but cleaner to clamp early.

### 4. NMS removes first element inefficiently
**Location:** `blazeface.rs:466`
**Current:**
```rust
let det = detections.remove(0);
```
**Suggested:**
```rust
// Use VecDeque or swap_remove pattern
let det = detections.swap_remove(0);
detections.sort_by(...); // re-sort after swap
// Or better: use indices instead of mutation
```
**Why:** `remove(0)` is O(n) per iteration, making NMS O(n^2). Typically fine for <20 faces but could matter at scale. Consider `VecDeque::pop_front()` or index-based approach.

### 5. Lazy model loading error message is stringly-typed
**Location:** `eyes.rs:77-80`
**Current:**
```rust
let result = self.models
    .get_or_init(|| self.load_models().map_err(|e| e.to_string()));
result.as_ref().map_err(|e| anyhow::anyhow!("{e}"))
```
**Suggested:**
```rust
// Consider Arc<anyhow::Error> or custom error type to preserve context
// Or accept that error backtrace is lost on repeated access
```
**Why:** Converting to String loses the error chain/backtrace. If `get_models()` is called multiple times after a failure, subsequent calls get a degraded error. Acceptable tradeoff for `OnceLock` simplicity, but worth documenting.

### 6. BlazeFace backbone comment mismatch
**Location:** `blazeface.rs:215`
```rust
(42, 48, 3, 2), // stride 2: but we only have 16x16 now
```
**Why:** Comment says "but we only have 16x16 now" but this is the 6th layer - spatial dims depend on prior strides. Either clarify the expected feature map size at each stage or remove the misleading suffix.

### 7. Consider batch processing
**Location:** `eyes.rs:117-139`
**Current:** Processes eyes one at a time with separate forward passes.
**Suggested:** Batch both eyes together for single inference call when possible.
**Why:** Single batched inference is more efficient, especially on GPU. Minor for now but worth considering if performance matters.

## Minor/Style

- `blazeface.rs:460`: `unwrap_or(std::cmp::Ordering::Equal)` for NaN handling is correct but could add comment explaining the NaN case
- `eyes.rs:5-7`: Inner `#![allow(...)]` attributes in module - unusual but valid; consider moving to outer attribute on module or per-function
- `eye_state.rs:193-194`: Magic numbers `0.25` and `0.15` for eye region size could be named constants
- `eyes.rs:136-137`: `unwrap_or_else` with warning but assumes open - might want configurable fallback behavior
- `blazeface.rs:191`: `#[allow(clippy::needless_pass_by_value)]` - VarBuilder is typically consumed, so this is fine

## Positive Notes

- Clean separation: BlazeFace handles detection, EyeStateClassifier handles classification - good single responsibility
- Proper use of `OnceLock` for lazy model loading without external crate (`once_cell` used elsewhere but std's `OnceLock` works)
- Comprehensive doc comments with architecture explanations
- EAR approximation from classifier probability is clever - maintains API compatibility with landmark-based approaches
- Test coverage is reasonable; `test_analyze_without_models` validates graceful degradation
- Consistent use of normalized coordinates throughout the pipeline

## Action Items

- [ ] Extract shared `sigmoid` function (critical - DRY)
- [ ] Replace placeholder SHA256 hashes before release (critical - security)
- [ ] Clamp eye crop coordinates to non-negative (improvement - edge case)
- [ ] Consider `VecDeque` for NMS or document O(n^2) as acceptable (minor - perf)
- [ ] Add named constants for eye region size ratios (style - readability)
