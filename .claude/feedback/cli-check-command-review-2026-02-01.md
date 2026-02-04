# Code Review: CLI Check Command Implementation

**Reviewed:** 2026-02-01
**Files:** `crates/photo-qa-cli/src/commands/check.rs`, `crates/photo-qa-cli/src/main.rs`, `crates/photo-qa-cli/src/commands/mod.rs`, related files
**Scope:** Full implementation of the `check` command with exit codes, EXIF extraction, output formatting

## Summary

Solid implementation completing Phase 6 of the CLI. Clean architecture following hexagonal patterns. A few issues around error handling, unused imports, and mixed output concerns need attention before shipping.

## Critical Issues

### 1. Unused `ResultOutput` import
**Location:** `crates/photo-qa-cli/src/commands/check.rs:6`
**Problem:** `ResultOutput` is imported but never used as a trait bound - only `JsonOutput` methods are called directly.
**Fix:**
```rust
use photo_qa_core::{
    AnalysisResult, BlurConfig, BlurModule, ExposureConfig, ExposureModule, EyesConfig, EyesModule,
    ImageDimensions, ImageSource, ProgressEvent, QaModule,
};
```

### 2. Mixed stdout usage for JSON format
**Location:** `crates/photo-qa-cli/src/commands/check.rs:228-232`
**Problem:** JSON array format uses raw `println!` while JSONL uses the `JsonOutput` adapter. This bypasses the abstraction and could cause issues if output destination changes.
**Fix:**
```rust
// For JSON format, write directly to the same writer
if matches!(args.format, OutputFormat::Json) {
    // Consider adding a write_raw method to JsonOutput or use the adapter consistently
    output.write_all_json(&all_results, args.pretty)?;
}
```

## Improvements

### 1. `#[allow(dead_code)]` annotations suggest unused fields
**Location:** `crates/photo-qa-cli/src/commands/check.rs:86-93`
**Current:**
```rust
pub struct CheckResult {
    #[allow(dead_code)]
    pub processed: usize,
    #[allow(dead_code)]
    pub skipped: usize,
    #[allow(dead_code)]
    pub with_issues: usize,
    pub exit_code: ExitCode,
}
```
**Suggested:**
```rust
/// Result of running the check command.
/// Fields are public for testing and future CLI summary output.
pub struct CheckResult {
    pub processed: usize,
    pub skipped: usize,
    pub with_issues: usize,
    pub exit_code: ExitCode,
}
```
**Why:** Either remove the `pub` or use the fields (e.g., summary output). `#[allow(dead_code)]` on pub fields is a code smell.

### 2. Error path in EXIF extraction gives no path context
**Location:** `crates/photo-qa-cli/src/commands/check.rs:254`
**Current:**
```rust
let path = format!("image_{index}");
progress.on_event(ProgressEvent::Skipped {
    path,
    reason: e.to_string(),
});
```
**Suggested:**
Consider logging the actual path where the error occurred if available from the iterator, or improve the error message to include more context.

### 3. Module loading could be lazy
**Location:** `crates/photo-qa-cli/src/commands/check.rs:130-149`
**Why:** The `EyesModule` does expensive model loading. If there are no images to process, this work is wasted. Consider lazy initialization or checking image count first.

### 4. `iso_timestamp()` uses `unwrap_or_else` on infallible operation
**Location:** `crates/photo-qa-cli/src/commands/check.rs:277-280`
**Current:**
```rust
fn iso_timestamp() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| String::from("1970-01-01T00:00:00Z"))
}
```
**Why:** RFC3339 formatting of `OffsetDateTime` is infallible. The fallback masks any actual issues. Consider using `.expect("RFC3339 format is infallible")` if clippy allows, or document why the fallback exists.

### 5. `json.rs` still has `#[allow(dead_code)]` annotations
**Location:** `crates/photo-qa-cli/src/output/json.rs:9-16`
**Why:** These are now used; remove the dead_code allows.

## Minor/Style

- `crates/photo-qa-cli/src/commands/check.rs:5`: Consider grouping std imports together (`std::collections::HashMap`, `std::io::IsTerminal`, `std::path::PathBuf`)
- `crates/photo-qa-cli/src/main.rs:10`: Alias `ExitCode as AppExitCode` works but is slightly verbose; consider renaming the custom enum to `AppExitCode` at definition
- `crates/photo-qa-cli/src/commands/check.rs:259`: `image.path.clone()` called twice in close proximity; could clone once

## Positive Notes

- Clean separation: `build_modules()` and `process_images()` functions are well-factored
- Exit code handling is correct and follows CLI conventions (0=success, 1=issues found, 2=error)
- Progress bar TTY detection logic is correct
- EXIF extraction handles errors gracefully with `Option` return
- The `ProgressEvent` pattern is a nice abstraction for progress reporting
- Good use of `--no-*` flags for disabling modules (cleaner than `--blur/--no-blur` pairs)

## Action Items

- [ ] Remove unused `ResultOutput` import from check.rs
- [ ] Fix mixed stdout usage for JSON format (use adapter consistently)
- [ ] Remove `#[allow(dead_code)]` from JsonOutput in json.rs
- [ ] Consider removing `#[allow(dead_code)]` from CheckResult fields or making them private
- [ ] Add summary output to stderr showing processed/skipped/issues counts
