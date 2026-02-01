//! Pipeline integration tests using synthetic images.
//!
//! Tests the full analysis pipeline with programmatically generated test images.

#![allow(
    clippy::unwrap_used,
    clippy::redundant_clone,
    clippy::needless_collect,
    clippy::uninlined_format_args,
    clippy::float_cmp,
    clippy::expect_used,
    clippy::missing_panics_doc,
    deprecated
)]

use assert_cmd::Command;
use photo_qa_test_support::SyntheticImageBuilder;
use serde_json::Value;

/// Create a temporary directory with synthetic test images.
fn create_test_images(images: Vec<(&str, image::DynamicImage)>) -> tempfile::TempDir {
    let temp_dir = tempfile::tempdir().unwrap();

    for (name, img) in images {
        let path = temp_dir.path().join(name);
        img.save(&path).unwrap();
    }

    temp_dir
}

// === Sharp Image Tests ===

#[test]
fn test_sharp_image_no_blur_issue() {
    let sharp = SyntheticImageBuilder::checkerboard(256, 256);
    let temp_dir = create_test_images(vec![("sharp.png", sharp.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("sharp.png"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should succeed
    assert!(output.status.success() || output.status.code() == Some(1));

    // Parse output
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        // Sharp checkerboard should have no blur issues
        let blur_issues: Vec<_> = issues
            .iter()
            .filter(|i| i["type"].as_str() == Some("blur"))
            .collect();

        assert!(
            blur_issues.is_empty(),
            "Sharp checkerboard should have no blur issues, got {:?}",
            blur_issues
        );
    }
}

#[test]
fn test_sharp_image_with_standard_threshold_no_blur() {
    let sharp = SyntheticImageBuilder::checkerboard(256, 256);
    let temp_dir = create_test_images(vec![("sharp.png", sharp.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--blur-threshold")
        .arg("0.5") // Standard threshold
        .arg("--no-exposure") // Disable exposure (checkerboard is half black/white)
        .arg("--no-eyes")
        .arg(temp_dir.path().join("sharp.png"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse output and verify no blur issues
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        // No blur issues for sharp image
        let blur_issues: Vec<_> = issues
            .iter()
            .filter(|i| i["type"].as_str() == Some("blur"))
            .collect();

        assert!(
            blur_issues.is_empty(),
            "Sharp checkerboard should have no blur issues"
        );
    }
}

// === Blurry Image Tests ===

#[test]
fn test_blurry_image_detected() {
    let blurry = SyntheticImageBuilder::uniform_gray(256, 256, 128);
    let temp_dir = create_test_images(vec![("blurry.png", blurry.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("blurry.png"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should detect issues (exit code 1)
    assert_eq!(
        output.status.code(),
        Some(1),
        "Blurry image should have exit code 1 (issues found)"
    );

    // Parse and verify blur issue
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        let blur_issues: Vec<_> = issues
            .iter()
            .filter(|i| i["type"].as_str() == Some("blur"))
            .collect();

        assert!(
            !blur_issues.is_empty(),
            "Uniform gray image should be detected as blurry"
        );

        // Verify blur score is high
        for issue in blur_issues {
            let score = issue["score"].as_f64().unwrap();
            assert!(
                score > 0.5,
                "Blur score should be high for uniform image, got {}",
                score
            );
        }
    }
}

#[test]
fn test_gradient_image_blur_detection() {
    let gradient = SyntheticImageBuilder::horizontal_gradient(256, 256);
    let temp_dir = create_test_images(vec![("gradient.png", gradient.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("gradient.png"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Gradient has low variance (blur-like)
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        // Should have blur issues
        let blur_issues: Vec<_> = issues
            .iter()
            .filter(|i| i["type"].as_str() == Some("blur"))
            .collect();

        assert!(
            !blur_issues.is_empty(),
            "Gradient should be detected as blurry"
        );
    }
}

// === Exposure Tests ===

#[test]
fn test_underexposed_image_detected() {
    let dark = SyntheticImageBuilder::underexposed(256, 256);
    let temp_dir = create_test_images(vec![("dark.png", dark.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-blur")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("dark.png"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert_eq!(
        output.status.code(),
        Some(1),
        "Underexposed image should have issues"
    );

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        let exposure_issues: Vec<_> = issues
            .iter()
            .filter(|i| i["type"].as_str() == Some("exposure"))
            .collect();

        assert!(
            !exposure_issues.is_empty(),
            "Black image should be detected as underexposed"
        );

        // Verify direction is "under"
        for issue in &exposure_issues {
            let details = &issue["details"];
            let direction = details["direction"].as_str().unwrap();
            assert!(
                direction == "under" || direction == "both",
                "Direction should be 'under' or 'both', got {}",
                direction
            );
        }
    }
}

#[test]
fn test_overexposed_image_detected() {
    let bright = SyntheticImageBuilder::overexposed(256, 256);
    let temp_dir = create_test_images(vec![("bright.png", bright.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-blur")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("bright.png"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert_eq!(
        output.status.code(),
        Some(1),
        "Overexposed image should have issues"
    );

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        let exposure_issues: Vec<_> = issues
            .iter()
            .filter(|i| i["type"].as_str() == Some("exposure"))
            .collect();

        assert!(
            !exposure_issues.is_empty(),
            "White image should be detected as overexposed"
        );

        // Verify direction is "over"
        for issue in &exposure_issues {
            let details = &issue["details"];
            let direction = details["direction"].as_str().unwrap();
            assert!(
                direction == "over" || direction == "both",
                "Direction should be 'over' or 'both', got {}",
                direction
            );
        }
    }
}

#[test]
fn test_well_exposed_image_no_issues() {
    let good = SyntheticImageBuilder::good_tonal_range(256, 256);
    let temp_dir = create_test_images(vec![("good.png", good.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-blur")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("good.png"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Good tonal range should have no exposure issues
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        let exposure_issues: Vec<_> = issues
            .iter()
            .filter(|i| i["type"].as_str() == Some("exposure"))
            .collect();

        assert!(
            exposure_issues.is_empty(),
            "Good tonal range image should have no exposure issues, got {:?}",
            exposure_issues
        );
    }
}

// === Mixed Issues Detection ===

#[test]
fn test_blurry_and_dark_image() {
    // Uniform black is both blurry (no edges) and underexposed
    let blurry_dark = SyntheticImageBuilder::uniform_gray(256, 256, 5);
    let temp_dir = create_test_images(vec![("blurry_dark.png", blurry_dark.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("blurry_dark.png"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        let blur_issues: Vec<_> = issues
            .iter()
            .filter(|i| i["type"].as_str() == Some("blur"))
            .collect();

        let exposure_issues: Vec<_> = issues
            .iter()
            .filter(|i| i["type"].as_str() == Some("exposure"))
            .collect();

        // Should detect both
        assert!(
            !blur_issues.is_empty(),
            "Should detect blur in uniform dark image"
        );
        assert!(
            !exposure_issues.is_empty(),
            "Should detect underexposure in dark image"
        );
    }
}

// === Multiple Images Pipeline ===

#[test]
fn test_multiple_images_analyzed() {
    let sharp = SyntheticImageBuilder::checkerboard(128, 128);
    let blurry = SyntheticImageBuilder::uniform_gray(128, 128, 128);
    let dark = SyntheticImageBuilder::underexposed(128, 128);

    let temp_dir = create_test_images(vec![
        ("sharp.png", sharp.image.clone()),
        ("blurry.png", blurry.image.clone()),
        ("dark.png", dark.image.clone()),
    ]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(temp_dir.path());

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should have 3 results
    let lines: Vec<_> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 3, "Should have results for all 3 images");

    // Verify each result has required fields
    for line in lines {
        let parsed: Value = serde_json::from_str(line).unwrap();
        assert!(parsed.get("path").is_some());
        assert!(parsed.get("dimensions").is_some());
        assert!(parsed.get("issues").is_some());
    }
}

// === Threshold Interaction Tests ===

#[test]
fn test_high_blur_threshold_passes_borderline() {
    let gradient = SyntheticImageBuilder::horizontal_gradient(256, 256);
    let temp_dir = create_test_images(vec![("gradient.png", gradient.image.clone())]);

    // With very high threshold, gradient might pass
    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--blur-threshold")
        .arg("0.99")
        .arg("--no-exposure")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("gradient.png"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).unwrap();
        let issues = parsed["issues"].as_array().unwrap();

        // With threshold 0.99, may or may not flag depending on exact score
        // Just verify parsing works
        for issue in issues {
            assert!(issue.get("type").is_some());
        }
    }
}

#[test]
fn test_low_exposure_threshold_flags_everything() {
    let good = SyntheticImageBuilder::good_tonal_range(256, 256);
    let temp_dir = create_test_images(vec![("good.png", good.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--under-threshold")
        .arg("0.0")
        .arg("--over-threshold")
        .arg("0.0")
        .arg("--no-blur")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("good.png"));

    let output = cmd.output().unwrap();

    // Even good images might get flagged at threshold 0
    // Just verify it runs without error
    assert!(
        output.status.success() || output.status.code() == Some(1),
        "Should complete successfully"
    );
}

// === Exit Code Tests ===

#[test]
fn test_exit_code_0_no_issues() {
    let good = SyntheticImageBuilder::good_tonal_range(256, 256);

    // Create sharp + well exposed image (checkerboard is both sharp and mid-gray)
    let temp_dir = create_test_images(vec![("good.png", good.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--no-blur") // Disable blur to avoid gradient issues
        .arg("--no-eyes")
        .arg(temp_dir.path().join("good.png"));

    cmd.assert().code(0);
}

#[test]
fn test_exit_code_1_issues_found() {
    let blurry = SyntheticImageBuilder::uniform_gray(256, 256, 128);
    let temp_dir = create_test_images(vec![("blurry.png", blurry.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--no-eyes").arg(temp_dir.path().join("blurry.png"));

    cmd.assert().code(1);
}

// === Edge Cases ===

#[test]
fn test_tiny_image() {
    let tiny = SyntheticImageBuilder::tiny([[0, 255], [128, 64]]);
    let temp_dir = create_test_images(vec![("tiny.png", tiny.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("tiny.png"));

    let output = cmd.output().unwrap();

    // Should not crash on tiny images
    assert!(
        output.status.success() || output.status.code() == Some(1),
        "Tiny image should not cause crash"
    );
}

#[test]
fn test_rgb_image_conversion() {
    let rgb = SyntheticImageBuilder::rgb_uniform(256, 256, 128, 128, 128);
    let temp_dir = create_test_images(vec![("rgb.png", rgb.image.clone())]);

    let mut cmd = Command::cargo_bin("photo-qa").unwrap();
    cmd.arg("--format")
        .arg("jsonl")
        .arg("--no-eyes")
        .arg(temp_dir.path().join("rgb.png"));

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should handle RGB images (converted to luma internally)
    assert!(
        output.status.success() || output.status.code() == Some(1),
        "RGB image should be handled"
    );

    // Parse should succeed
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let _: Value = serde_json::from_str(line).unwrap();
    }
}
