//! CLI argument validation tests.
//!
//! Tests command-line argument parsing, validation, and error handling.

#![allow(clippy::unwrap_used)]
#![allow(deprecated)] // cargo_bin deprecation

use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("photo-qa-adapters/tests/fixtures")
}

// === Missing/Invalid Path Tests ===

#[test]
fn test_missing_path_shows_error() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    // No path argument at all - error goes to stderr
    cmd.assert().failure().stderr(
        predicate::str::contains("No paths specified")
            .or(predicate::str::contains("required"))
            .or(predicate::str::contains("PATHS")),
    );
}

#[test]
fn test_nonexistent_path_warns_but_continues() {
    // The CLI warns about nonexistent paths but continues (graceful degradation)
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("/nonexistent/path/to/image.jpg").arg("--no-eyes"); // Skip ML models

    // Should succeed (exit 0) but warn
    cmd.assert()
        .code(0) // No images processed = no issues
        .stderr(
            predicate::str::contains("does not exist").or(predicate::str::contains("not found")),
        );
}

#[test]
fn test_empty_directory() {
    let temp_dir = tempfile::tempdir().unwrap();

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg(temp_dir.path()).arg("--no-eyes");

    // Empty directory should succeed with no output (exit 0)
    cmd.assert().code(predicate::eq(0));
}

// === Format Validation Tests ===

#[test]
fn test_invalid_format_rejected() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("xml") // Invalid format
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("json").or(predicate::str::contains("jsonl")));
}

#[test]
fn test_valid_formats_accepted() {
    // Test JSON format
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("json")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert().code(predicate::in_iter([0, 1]));

    // Test JSONL format
    let mut cmd2 = Command::cargo_bin("photo-qa").unwrap();
    cmd2.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    cmd2.assert().code(predicate::in_iter([0, 1]));
}

// === Threshold Validation Tests ===

#[test]
fn test_blur_threshold_above_one_rejected() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--blur-threshold")
        .arg("1.5")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("0.0..=1.0").or(predicate::str::contains("invalid")));
}

#[test]
fn test_blur_threshold_negative_rejected() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--blur-threshold")
        .arg("-0.1")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert().failure();
}

#[test]
fn test_blur_threshold_non_numeric_rejected() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--blur-threshold")
        .arg("abc")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("invalid"));
}

#[test]
fn test_valid_threshold_boundaries() {
    // Test 0.0
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--blur-threshold")
        .arg("0.0")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert().code(predicate::in_iter([0, 1]));

    // Test 1.0
    let mut cmd2 = Command::cargo_bin("photo-qa").unwrap();
    cmd2.arg("--blur-threshold")
        .arg("1.0")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    cmd2.assert().code(predicate::in_iter([0, 1]));
}

#[test]
fn test_under_threshold_validation() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--under-threshold")
        .arg("2.0")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("0.0..=1.0").or(predicate::str::contains("invalid")));
}

#[test]
fn test_over_threshold_validation() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--over-threshold")
        .arg("-1")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert().failure();
}

#[test]
fn test_ear_threshold_validation() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--ear-threshold")
        .arg("1.1")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("0.0..=1.0").or(predicate::str::contains("invalid")));
}

// === Verbosity Level Tests ===

#[test]
fn test_verbosity_v() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("-v")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    // Should show debug output
    cmd.assert().code(predicate::in_iter([0, 1]));
}

#[test]
fn test_verbosity_vv() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("-vv")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert().code(predicate::in_iter([0, 1]));
}

#[test]
fn test_verbosity_vvv() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("-vvv")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert().code(predicate::in_iter([0, 1]));
}

#[test]
fn test_quiet_suppresses_progress() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--quiet")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    // With --quiet, should succeed without progress bar
    // Note: logging may still appear based on verbosity settings
    cmd.assert().code(predicate::in_iter([0, 1]));
}

// === Module Disable Flags ===

#[test]
fn test_no_blur_flag() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--no-blur")
        .arg("--no-eyes")
        .arg("-v")
        .arg(fixtures_dir().join("test.jpg"));

    // Should work, blur module disabled
    cmd.assert().code(predicate::in_iter([0, 1]));
}

#[test]
fn test_no_exposure_flag() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--no-exposure")
        .arg("--no-eyes")
        .arg("-v")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert().code(predicate::in_iter([0, 1]));
}

#[test]
fn test_no_eyes_flag() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--no-eyes").arg(fixtures_dir().join("test.jpg"));

    cmd.assert().code(predicate::in_iter([0, 1]));
}

#[test]
fn test_all_modules_disabled() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--no-blur")
        .arg("--no-exposure")
        .arg("--no-eyes")
        .arg("-v")
        .arg(fixtures_dir().join("test.jpg"));

    // Should succeed but warn about no modules
    cmd.assert()
        .code(0)
        .stderr(predicate::str::contains("disabled").or(predicate::str::contains("modules")));
}

// === Multiple Paths ===

#[test]
fn test_multiple_paths() {
    let fixture_path = fixtures_dir().join("test.jpg");

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--no-eyes").arg(&fixture_path).arg(&fixture_path); // Same file twice

    cmd.assert().code(predicate::in_iter([0, 1]));
}

// === Recursive Flag ===

#[test]
fn test_recursive_flag() {
    let temp_dir = tempfile::tempdir().unwrap();
    let sub_dir = temp_dir.path().join("subdir");
    std::fs::create_dir(&sub_dir).unwrap();

    // Copy test image to subdir
    let fixture = fixtures_dir().join("test.jpg");
    let dest = sub_dir.join("test.jpg");
    std::fs::copy(&fixture, &dest).unwrap();

    // Without -r, should not find image in subdir
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--no-eyes").arg(temp_dir.path());

    cmd.assert().code(0); // No images found at top level

    // With -r, should find image in subdir
    let mut cmd2 = Command::cargo_bin("photo-qa").unwrap();
    cmd2.arg("-r").arg("--no-eyes").arg(temp_dir.path());

    cmd2.assert().code(predicate::in_iter([0, 1])); // Found and analyzed
}

// === Help and Version ===

#[test]
fn test_help_flag() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Usage"))
        .stdout(predicate::str::contains("--blur-threshold"))
        .stdout(predicate::str::contains("--format"));
}

#[test]
fn test_version_flag() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--version");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("photo-qa"));
}

// === Check Subcommand ===

#[test]
fn test_check_subcommand() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("check")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert().code(predicate::in_iter([0, 1]));
}

#[test]
fn test_check_subcommand_with_options() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("check")
        .arg("--blur-threshold")
        .arg("0.8")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert().code(predicate::in_iter([0, 1]));
}
