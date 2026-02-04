# Code Review: Check Command Implementation

**Reviewed:** 2026-02-01
**Files:** `crates/photo-qa-cli/src/commands/check.rs`, `crates/photo-qa-cli/src/main.rs`, `crates/photo-qa-cli/src/output/json.rs`, `crates/photo-qa-cli/src/output/progress.rs`, `crates/photo-qa-adapters/src/lib.rs`
**Scope:** CLI check command wiring, exit codes, output formatting

## Summary

Solid implementation of the check command with proper hexagonal architecture wiring. Exit codes handled cleanly via enum with explicit values. QA module initialization is well-structured with graceful ML model fallback. One bug in EXIF extraction, otherwise clean.

## Critical Issues

### 1. EXIF Reader Borrows File Incorrectly

**Location:** `crates/photo-qa-cli/src/commands/check.rs:296`
**Problem:** `BufReader::new(&file)` borrows `File` instead of taking ownership. EXIF library may need ownership for seeking.

**Current:**
```rust
let file = File::open(path).ok()?;
let mut reader = BufReader::new(&file);
let exif = exif::Reader::new().read_from_container(&mut reader).ok()?;
```

**Fix:**
```rust
let file = File::open(path).ok()?;
let mut reader = BufReader::new(file);  // Take ownership
let exif = exif::Reader::new().read_from_container(&mut reader).ok()?;
```

**Why:** `read_from_container` may require `Seek` which works better with owned `File`. Current code compiles but may fail at runtime on some formats.

## Improvements

### 1. JSON Array Output Uses Adapter Nicely

**Location:** `crates/photo-qa-cli/src/output/json.rs:30-42`
**Observation:** Good - `write_array()` method added to `JsonOutput` for JSON format. This keeps output consistent through the adapter pattern. The `#[allow(clippy::significant_drop_tightening)]` is correctly applied.

### 2. Consider `info!` Instead of `warn!` for Model Fetch Hint

**Location:** `crates/photo-qa-cli/src/commands/check.rs:133-143`
**Current:**
```rust
warn!(
    "Eyes module disabled: {} not found. Run `photo-qa models fetch`.",
    bf.display()
);
```

**Suggested:** Use `info!` for actionable user guidance, reserve `warn!` for unexpected states. Model not existing yet is expected on first run.

### 3. Progress Bar Position Fix is Correct

**Location:** `crates/photo-qa-cli/src/output/progress.rs:56-64`
**Current:**
```rust
ProgressEvent::Started { path, total, .. } => {
    // Don't set_position here - let Completed/Skipped inc(1) handle it
    bar.set_message(path);
}
```

**Observation:** The diff shows removal of `set_position(index)` in favor of letting `inc(1)` handle progression. This is the right fix - prevents position jumps. Comment documents intent well.

## Minor/Style

- `check.rs:90`: `CheckResult` fields marked `#[allow(dead_code)]` - fine for now, struct exposed for programmatic use
- `main.rs:47`: Error handling refactored nicely - returns `std::process::ExitCode` instead of `Result<()>`
- `output/mod.rs`: Cleaned up `#[allow(unused_imports)]` - exports now used
- `adapters/lib.rs`: Good - re-exports `model_path` and `models_dir` for CLI use

## Positive Notes

- Clean separation: `build_modules()` and `process_images()` are well-factored helpers
- Exit code enum with explicit values (`Success = 0`, `IssuesFound = 1`, `Error = 2`) is clear
- Graceful degradation when ML models missing - warns but continues with other modules
- JSONL vs JSON array output handled cleanly via format flag
- Progress bar TTY auto-detection (`std::io::stderr().is_terminal()`) is correct
- `time` crate used properly for RFC3339 timestamps

## Action Items

- [ ] Fix EXIF reader to take ownership of File
- [ ] Consider changing model-not-found from `warn!` to `info!`
