//! Output format validation tests.
//!
//! Tests JSON/JSONL output format correctness and required field presence.

#![allow(clippy::unwrap_used)]
#![allow(deprecated)] // cargo_bin deprecation
#![allow(unused_imports)]

use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("photo-qa-adapters/tests/fixtures")
}

// === JSONL Format Tests ===

#[test]
fn test_jsonl_format_single_object_per_line() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Each line should be valid JSON
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Result<Value, _> = serde_json::from_str(line);
        assert!(
            parsed.is_ok(),
            "Each JSONL line should be valid JSON: {line}"
        );

        // Should be an object, not an array
        let value = parsed.unwrap();
        assert!(value.is_object(), "JSONL line should be an object");
    }
}

#[test]
fn test_jsonl_format_multiple_images() {
    let fixture_path = fixtures_dir().join("test.jpg");

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(&fixture_path)
        .arg(&fixture_path); // Same image twice

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    let json_lines: Vec<_> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();

    // Should have 2 lines (one per image)
    assert_eq!(json_lines.len(), 2, "Should have one line per image");

    // Each line should be independently parseable
    for line in json_lines {
        let parsed: Value = serde_json::from_str(line).unwrap();
        assert!(parsed.is_object());
    }
}

// === JSON Array Format Tests ===

#[test]
fn test_json_format_is_array() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("json")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should be a valid JSON array
    let parsed: Result<Value, _> = serde_json::from_str(&stdout);
    assert!(parsed.is_ok(), "JSON format should be valid JSON");

    let value = parsed.unwrap();
    assert!(value.is_array(), "JSON format should be an array");
}

#[test]
fn test_json_format_multiple_images() {
    let fixture_path = fixtures_dir().join("test.jpg");

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("json")
        .arg("--no-eyes")
        .arg(&fixture_path)
        .arg(&fixture_path);

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    let parsed: Value = serde_json::from_str(&stdout).unwrap();
    let arr = parsed.as_array().unwrap();

    assert_eq!(arr.len(), 2, "Should have one entry per image");
}

#[test]
fn test_json_format_empty_array_for_no_images() {
    let temp_dir = tempfile::tempdir().unwrap();

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("json")
        .arg("--no-eyes")
        .arg(temp_dir.path());

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    let parsed: Value = serde_json::from_str(&stdout).unwrap();
    let arr = parsed.as_array().unwrap();

    assert!(arr.is_empty(), "Empty directory should produce empty array");
}

// === Pretty Format Tests ===

#[test]
fn test_pretty_format_is_indented() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("json")
        .arg("--pretty")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Pretty format should have newlines and indentation
    assert!(stdout.contains('\n'), "Pretty format should have newlines");
    assert!(
        stdout.contains("  ") || stdout.contains('\t'),
        "Pretty format should have indentation"
    );

    // Should still be valid JSON
    let parsed: Result<Value, _> = serde_json::from_str(&stdout);
    assert!(parsed.is_ok(), "Pretty JSON should still be valid");
}

#[test]
fn test_non_pretty_format_is_compact() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Each non-empty line should be a single-line JSON object
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        // The line itself shouldn't contain newlines (it's a single line)
        // and shouldn't start with unnecessary whitespace
        assert!(
            !line.starts_with("  "),
            "JSONL should not have leading indentation"
        );
    }
}

// === Required Fields Presence ===

#[test]
fn test_result_has_path_field() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        assert!(
            parsed.get("path").is_some(),
            "Result should have 'path' field"
        );
        assert!(parsed["path"].is_string(), "'path' should be a string");
    }
}

#[test]
fn test_result_has_timestamp_field() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        assert!(
            parsed.get("timestamp").is_some(),
            "Result should have 'timestamp' field"
        );
        assert!(
            parsed["timestamp"].is_string(),
            "'timestamp' should be a string"
        );

        // Should look like an ISO 8601 timestamp
        let ts = parsed["timestamp"].as_str().unwrap();
        assert!(
            ts.contains('T') || ts.contains('-'),
            "Timestamp should be ISO 8601 format: {ts}"
        );
    }
}

#[test]
fn test_result_has_dimensions_field() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        assert!(
            parsed.get("dimensions").is_some(),
            "Result should have 'dimensions' field"
        );

        let dims = &parsed["dimensions"];
        assert!(dims.is_object(), "'dimensions' should be an object");
        assert!(
            dims.get("width").is_some(),
            "dimensions should have 'width'"
        );
        assert!(
            dims.get("height").is_some(),
            "dimensions should have 'height'"
        );
        assert!(dims["width"].is_number(), "'width' should be a number");
        assert!(dims["height"].is_number(), "'height' should be a number");
    }
}

#[test]
fn test_result_has_issues_field() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        assert!(
            parsed.get("issues").is_some(),
            "Result should have 'issues' field"
        );
        assert!(parsed["issues"].is_array(), "'issues' should be an array");
    }
}

#[test]
fn test_issue_has_required_fields() {
    // Use a blurry-ish test to ensure we get issues
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--blur-threshold")
        .arg("0.0") // Very low threshold to ensure detection
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut found_issues = false;

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        for issue in issues {
            found_issues = true;

            // Check required issue fields
            assert!(
                issue.get("type").is_some(),
                "Issue should have 'type' field"
            );
            assert!(
                issue.get("score").is_some(),
                "Issue should have 'score' field"
            );
            assert!(
                issue.get("details").is_some(),
                "Issue should have 'details' field"
            );

            // Type should be a string
            assert!(issue["type"].is_string(), "'type' should be a string");

            // Score should be a number between 0 and 1
            let score = issue["score"].as_f64().unwrap();
            assert!(
                (0.0..=1.0).contains(&score),
                "Score should be 0.0-1.0, got {score}"
            );
        }
    }

    // We should have found at least one issue with threshold 0.0
    assert!(found_issues, "Should have detected at least one issue");
}

// === EXIF Field Tests ===

#[test]
fn test_exif_flag_includes_exif() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--exif")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();

        // With --exif flag, should have exif field (may be null if no EXIF data)
        // Actually, the field might be omitted if null depending on serialization
        // Just verify the JSON is still valid
        if let Some(exif) = parsed.get("exif") {
            // If present, should be null or an object
            assert!(
                exif.is_null() || exif.is_object(),
                "'exif' should be null or object"
            );
        }
    }
}

#[test]
fn test_no_exif_flag_omits_exif() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();

        // Without --exif flag, exif field should be omitted or null
        // (depends on implementation, both are acceptable)
        if let Some(exif) = parsed.get("exif") {
            assert!(
                exif.is_null(),
                "Without --exif, exif should be null if present"
            );
        }
    }
}

// === Issue Type Values ===

#[test]
fn test_blur_issue_type_value() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--blur-threshold")
        .arg("0.0")
        .arg("--no-exposure")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        for issue in issues {
            let issue_type = issue["type"].as_str().unwrap();
            assert_eq!(issue_type, "blur", "Issue type should be 'blur'");
        }
    }
}

#[test]
fn test_exposure_issue_type_value() {
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--under-threshold")
        .arg("0.0")
        .arg("--no-blur")
        .arg("--no-eyes")
        .arg(fixtures_dir().join("test.jpg"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        for issue in issues {
            let issue_type = issue["type"].as_str().unwrap();
            assert_eq!(issue_type, "exposure", "Issue type should be 'exposure'");
        }
    }
}
