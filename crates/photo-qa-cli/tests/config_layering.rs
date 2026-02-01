//! Integration tests for configuration layering.
//!
//! Tests the full priority chain: hardcoded defaults < XDG config < project config < CLI args

#![allow(clippy::unwrap_used)] // Test code uses unwrap for brevity
#![allow(deprecated)] // cargo_bin deprecation warning

use std::fs;
use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;

/// Get path to test fixtures
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("photo-qa-adapters/tests/fixtures")
}

#[test]
fn test_cli_threshold_validation_rejects_invalid() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--blur-threshold")
        .arg("2.0")
        .arg(fixtures_dir().join("test.jpg"));

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("2 is not in 0.0..=1.0"));
}

#[test]
fn test_cli_threshold_validation_accepts_valid() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--blur-threshold")
        .arg("0.9")
        .arg("--no-eyes") // Skip ML models
        .arg(fixtures_dir().join("test.jpg"));

    // Should not fail on threshold validation (may fail on other things)
    cmd.assert().code(predicate::in_iter([0, 1])); // 0 = no issues, 1 = issues found
}

#[test]
fn test_project_config_applies_format() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join(".photo-qa.toml");

    // Create project config with JSON format
    fs::write(
        &config_path,
        r"
[output]
format = 'json'
",
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.current_dir(temp_dir.path())
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    // Output should be JSON array format per config
    cmd.assert()
        .code(1) // Issues found
        .stdout(predicate::str::starts_with("[")); // JSON array format
}

#[test]
fn test_cli_overrides_project_config() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join(".photo-qa.toml");

    // Create project config with JSON format
    fs::write(
        &config_path,
        r"
[output]
format = 'json'
",
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.current_dir(temp_dir.path())
        .arg("--format")
        .arg("jsonl") // CLI overrides config
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    // CLI --format jsonl should override config format = "json"
    cmd.assert()
        .code(1)
        .stdout(predicate::str::starts_with("{")); // JSONL format (single object per line)
}

#[test]
fn test_config_disables_module() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join(".photo-qa.toml");

    // Disable blur and exposure via config
    fs::write(
        &config_path,
        r"
[blur]
enabled = false

[exposure]
enabled = false
",
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.current_dir(temp_dir.path())
        .arg("--no-eyes")
        .arg("-v") // Verbose to see module status
        .arg(fixtures_dir().join("test.jpg"));

    // With blur and exposure disabled, and eyes skipped, no modules = exit 0
    cmd.assert()
        .code(0)
        .stderr(predicate::str::contains("All QA modules disabled"));
}
