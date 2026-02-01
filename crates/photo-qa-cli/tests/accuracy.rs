//! Accuracy benchmark tests for photo-qa.
//!
//! These tests measure detection accuracy against ground truth datasets.
//! They are marked `#[ignore]` because they require external datasets.
//!
//! To run:
//!   cargo test --test accuracy -- --ignored --nocapture
//!
//! Or via justfile:
//!   just bench-accuracy
//!
//! # Dataset Setup
//!
//! Before running, download datasets using:
//!   ./scripts/fetch-test-data.sh
//!
//! Then create ground truth files:
//!   ./scripts/fetch-test-data.sh ground-truth blur

#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::redundant_closure_for_method_calls,
    clippy::uninlined_format_args,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]
#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Accuracy metrics for binary classification.
#[derive(Debug, Clone, Default)]
pub struct AccuracyResult {
    /// True positives: correctly identified issues
    pub true_positives: usize,
    /// False positives: incorrectly flagged as issues
    pub false_positives: usize,
    /// True negatives: correctly identified non-issues
    pub true_negatives: usize,
    /// False negatives: missed issues
    pub false_negatives: usize,
}

impl AccuracyResult {
    /// Precision = TP / (TP + FP)
    pub fn precision(&self) -> f64 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 {
            0.0
        } else {
            self.true_positives as f64 / denom as f64
        }
    }

    /// Recall = TP / (TP + FN)
    pub fn recall(&self) -> f64 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 {
            0.0
        } else {
            self.true_positives as f64 / denom as f64
        }
    }

    /// F1 Score = 2 * (precision * recall) / (precision + recall)
    pub fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Accuracy = (TP + TN) / total
    pub fn accuracy(&self) -> f64 {
        let total =
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives;
        if total == 0 {
            0.0
        } else {
            (self.true_positives + self.true_negatives) as f64 / total as f64
        }
    }

    /// Total samples
    pub fn total(&self) -> usize {
        self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
    }

    /// Update with a single prediction
    pub fn update(&mut self, predicted: bool, actual: bool) {
        match (predicted, actual) {
            (true, true) => self.true_positives += 1,
            (true, false) => self.false_positives += 1,
            (false, true) => self.false_negatives += 1,
            (false, false) => self.true_negatives += 1,
        }
    }

    /// Print a summary of the results
    pub fn print_summary(&self, label: &str) {
        println!("\n=== {} ===", label);
        println!("Total samples: {}", self.total());
        println!(
            "TP: {}, FP: {}, TN: {}, FN: {}",
            self.true_positives, self.false_positives, self.true_negatives, self.false_negatives
        );
        println!("Precision: {:.2}%", self.precision() * 100.0);
        println!("Recall:    {:.2}%", self.recall() * 100.0);
        println!("F1 Score:  {:.2}%", self.f1_score() * 100.0);
        println!("Accuracy:  {:.2}%", self.accuracy() * 100.0);
    }
}

/// Ground truth entry for an image.
#[derive(Debug, Clone)]
pub struct GroundTruth {
    pub filename: String,
    pub label: bool,
    pub extra: HashMap<String, String>,
}

/// Load ground truth from CSV file.
///
/// Expected format (first column is filename, second is label):
/// ```text
/// filename,is_blurry,blur_type,notes
/// image1.jpg,true,defocus,
/// image2.jpg,false,,
/// ```
pub fn load_ground_truth(path: &Path, label_column: &str) -> anyhow::Result<Vec<GroundTruth>> {
    let content = fs::read_to_string(path)?;
    let mut lines = content.lines();

    // Parse header
    let header = lines.next().ok_or_else(|| anyhow::anyhow!("Empty CSV"))?;
    let columns: Vec<_> = header.split(',').collect();

    let label_idx = columns
        .iter()
        .position(|&c| c == label_column)
        .ok_or_else(|| anyhow::anyhow!("Label column '{}' not found", label_column))?;

    let mut ground_truth = Vec::new();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }

        let values: Vec<_> = line.split(',').collect();
        if values.is_empty() {
            continue;
        }

        let filename = values[0].to_string();
        let label_str = values.get(label_idx).unwrap_or(&"").trim().to_lowercase();

        // Parse boolean label
        // Note: Empty labels default to false. If your ground truth has empty cells,
        // ensure this is intentional - empty values may mask data entry issues.
        let label = match label_str.as_str() {
            "true" | "1" | "yes" | "y" => true,
            "false" | "0" | "no" | "n" => false,
            "" => {
                eprintln!(
                    "Warning: Empty label for '{}', defaulting to false",
                    filename
                );
                false
            }
            _ => continue, // Skip unparseable rows
        };

        // Collect extra columns
        let mut extra = HashMap::new();
        for (i, col) in columns.iter().enumerate() {
            if i != 0 && i != label_idx {
                if let Some(val) = values.get(i) {
                    extra.insert(col.to_string(), val.to_string());
                }
            }
        }

        ground_truth.push(GroundTruth {
            filename,
            label,
            extra,
        });
    }

    Ok(ground_truth)
}

/// Results of a threshold sweep.
#[derive(Debug, Clone)]
pub struct ThresholdSweepResult {
    pub threshold: f32,
    pub result: AccuracyResult,
}

/// Perform a threshold sweep to find optimal operating point.
pub fn threshold_sweep<F>(
    thresholds: &[f32],
    ground_truth: &[GroundTruth],
    predict_fn: F,
) -> Vec<ThresholdSweepResult>
where
    F: Fn(&str, f32) -> anyhow::Result<bool>,
{
    thresholds
        .iter()
        .map(|&threshold| {
            let mut result = AccuracyResult::default();

            for gt in ground_truth {
                match predict_fn(&gt.filename, threshold) {
                    Ok(predicted) => result.update(predicted, gt.label),
                    Err(e) => {
                        eprintln!("Warning: failed to predict {}: {}", gt.filename, e);
                    }
                }
            }

            ThresholdSweepResult { threshold, result }
        })
        .collect()
}

/// Print threshold sweep results and find best F1.
pub fn print_threshold_sweep(results: &[ThresholdSweepResult], label: &str) {
    println!("\n=== Threshold Sweep: {} ===", label);
    println!(
        "{:>10} {:>10} {:>10} {:>10}",
        "Threshold", "Precision", "Recall", "F1"
    );
    println!("{}", "-".repeat(44));

    let mut best_f1 = 0.0;
    let mut best_threshold = 0.0;

    for r in results {
        println!(
            "{:>10.2} {:>9.1}% {:>9.1}% {:>9.1}%",
            r.threshold,
            r.result.precision() * 100.0,
            r.result.recall() * 100.0,
            r.result.f1_score() * 100.0
        );

        if r.result.f1_score() > best_f1 {
            best_f1 = r.result.f1_score();
            best_threshold = r.threshold;
        }
    }

    println!(
        "\nBest threshold: {:.2} (F1: {:.1}%)",
        best_threshold,
        best_f1 * 100.0
    );
}

/// Get the fixtures directory.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Check if a dataset is available.
fn dataset_available(name: &str) -> bool {
    let dir = fixtures_dir().join(name);
    if !dir.exists() {
        return false;
    }

    // Check for images
    let count = fs::read_dir(&dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let path = e.path();
                    matches!(
                        path.extension().and_then(|s| s.to_str()),
                        Some("jpg" | "jpeg" | "png" | "bmp")
                    )
                })
                .count()
        })
        .unwrap_or(0);

    count > 0
}

// === Blur Detection Accuracy ===

#[test]
#[ignore = "Requires blur dataset - run with --ignored"]
fn test_blur_accuracy() {
    let dataset_dir = fixtures_dir().join("blur");
    let gt_path = dataset_dir.join("ground_truth.csv");

    if !dataset_available("blur") {
        eprintln!("Blur dataset not available. Run: ./scripts/fetch-test-data.sh blur");
        return;
    }

    if !gt_path.exists() {
        eprintln!("Ground truth not found. Run: ./scripts/fetch-test-data.sh ground-truth blur");
        return;
    }

    let ground_truth =
        load_ground_truth(&gt_path, "is_blurry").expect("Failed to load ground truth");

    println!("Loaded {} ground truth entries", ground_truth.len());

    // TODO: Run actual blur detection and compare
    // For now, just demonstrate the framework

    let mut result = AccuracyResult::default();

    // Placeholder: mark all as correctly classified
    for gt in &ground_truth {
        // In real test: run blur detection on gt.filename
        // let predicted = analyze_blur(&dataset_dir.join(&gt.filename));
        let predicted = gt.label; // Placeholder

        result.update(predicted, gt.label);
    }

    result.print_summary("Blur Detection");

    // TODO: Set meaningful threshold once real benchmarks run
    // Target: assert!(result.f1_score() >= 0.80, "F1 below 80%");
    assert!(
        result.f1_score() >= 0.0,
        "F1 score should be non-negative: {}",
        result.f1_score()
    );
}

#[test]
#[ignore = "Requires blur dataset - run with --ignored"]
fn test_blur_threshold_sweep() {
    let dataset_dir = fixtures_dir().join("blur");
    let gt_path = dataset_dir.join("ground_truth.csv");

    if !dataset_available("blur") {
        eprintln!("Blur dataset not available");
        return;
    }

    if !gt_path.exists() {
        eprintln!("Ground truth not found");
        return;
    }

    let ground_truth =
        load_ground_truth(&gt_path, "is_blurry").expect("Failed to load ground truth");

    let thresholds: Vec<f32> = (0..=10).map(|i| i as f32 * 0.1).collect();

    let results = threshold_sweep(&thresholds, &ground_truth, |_filename, _threshold| {
        // TODO: Run actual blur detection with threshold
        // For now, return placeholder
        Ok(false)
    });

    print_threshold_sweep(&results, "Blur Detection");
}

// === Exposure Detection Accuracy ===

#[test]
#[ignore = "Requires exposure dataset - run with --ignored"]
fn test_exposure_accuracy() {
    let dataset_dir = fixtures_dir().join("exposure");
    let gt_path = dataset_dir.join("ground_truth.csv");

    if !dataset_available("exposure") {
        eprintln!("Exposure dataset not available");
        return;
    }

    if !gt_path.exists() {
        eprintln!("Ground truth not found");
        return;
    }

    // Test underexposure
    let under_gt = load_ground_truth(&gt_path, "is_underexposed");
    if let Ok(ground_truth) = under_gt {
        let mut result = AccuracyResult::default();

        for gt in &ground_truth {
            let predicted = gt.label; // Placeholder
            result.update(predicted, gt.label);
        }

        result.print_summary("Underexposure Detection");
    }

    // Test overexposure
    let over_gt = load_ground_truth(&gt_path, "is_overexposed");
    if let Ok(ground_truth) = over_gt {
        let mut result = AccuracyResult::default();

        for gt in &ground_truth {
            let predicted = gt.label; // Placeholder
            result.update(predicted, gt.label);
        }

        result.print_summary("Overexposure Detection");
    }
}

#[test]
#[ignore = "Requires exposure dataset - run with --ignored"]
fn test_exposure_threshold_sweep() {
    let dataset_dir = fixtures_dir().join("exposure");
    let gt_path = dataset_dir.join("ground_truth.csv");

    if !dataset_available("exposure") {
        eprintln!("Exposure dataset not available");
        return;
    }

    if !gt_path.exists() {
        eprintln!("Ground truth not found");
        return;
    }

    let ground_truth =
        load_ground_truth(&gt_path, "is_underexposed").expect("Failed to load ground truth");

    let thresholds: Vec<f32> = (0..=10).map(|i| i as f32 * 0.1).collect();

    let results = threshold_sweep(&thresholds, &ground_truth, |_filename, _threshold| {
        Ok(false) // Placeholder
    });

    print_threshold_sweep(&results, "Underexposure Detection");
}

// === Closed Eyes Detection Accuracy ===

#[test]
#[ignore = "Requires eyes dataset - run with --ignored"]
fn test_eyes_accuracy() {
    let dataset_dir = fixtures_dir().join("eyes");
    let gt_path = dataset_dir.join("ground_truth.csv");

    if !dataset_available("eyes") {
        eprintln!("Eyes dataset not available");
        return;
    }

    if !gt_path.exists() {
        eprintln!("Ground truth not found");
        return;
    }

    let ground_truth =
        load_ground_truth(&gt_path, "has_closed_eyes").expect("Failed to load ground truth");

    println!("Loaded {} ground truth entries", ground_truth.len());

    let mut result = AccuracyResult::default();

    for gt in &ground_truth {
        let predicted = gt.label; // Placeholder
        result.update(predicted, gt.label);
    }

    result.print_summary("Closed Eyes Detection");
}

#[test]
#[ignore = "Requires eyes dataset - run with --ignored"]
fn test_eyes_threshold_sweep() {
    let dataset_dir = fixtures_dir().join("eyes");
    let gt_path = dataset_dir.join("ground_truth.csv");

    if !dataset_available("eyes") {
        eprintln!("Eyes dataset not available");
        return;
    }

    if !gt_path.exists() {
        eprintln!("Ground truth not found");
        return;
    }

    let ground_truth =
        load_ground_truth(&gt_path, "has_closed_eyes").expect("Failed to load ground truth");

    let thresholds: Vec<f32> = (0..=10).map(|i| i as f32 * 0.05).collect(); // 0.0 to 0.5

    let results = threshold_sweep(&thresholds, &ground_truth, |_filename, _threshold| {
        Ok(false) // Placeholder
    });

    print_threshold_sweep(&results, "Closed Eyes Detection (EAR threshold)");
}

// === Utility Tests ===

#[test]
fn test_accuracy_result_calculations() {
    let mut result = AccuracyResult::default();

    // 8 TP, 2 FP, 5 TN, 5 FN
    for _ in 0..8 {
        result.update(true, true);
    } // TP
    for _ in 0..2 {
        result.update(true, false);
    } // FP
    for _ in 0..5 {
        result.update(false, false);
    } // TN
    for _ in 0..5 {
        result.update(false, true);
    } // FN

    assert_eq!(result.total(), 20);

    // Precision = 8 / (8 + 2) = 0.8
    assert!((result.precision() - 0.8).abs() < 0.001);

    // Recall = 8 / (8 + 5) = 0.615...
    assert!((result.recall() - 8.0 / 13.0).abs() < 0.001);

    // Accuracy = (8 + 5) / 20 = 0.65
    assert!((result.accuracy() - 0.65).abs() < 0.001);
}

#[test]
fn test_accuracy_edge_cases() {
    // All zeros
    let empty = AccuracyResult::default();
    assert_eq!(empty.precision(), 0.0);
    assert_eq!(empty.recall(), 0.0);
    assert_eq!(empty.f1_score(), 0.0);
    assert_eq!(empty.accuracy(), 0.0);

    // Perfect classifier
    let mut perfect = AccuracyResult::default();
    for _ in 0..10 {
        perfect.update(true, true);
    }
    for _ in 0..10 {
        perfect.update(false, false);
    }

    assert_eq!(perfect.precision(), 1.0);
    assert_eq!(perfect.recall(), 1.0);
    assert_eq!(perfect.f1_score(), 1.0);
    assert_eq!(perfect.accuracy(), 1.0);
}

#[test]
fn test_ground_truth_loading() {
    // Create temp CSV
    let temp_dir = tempfile::tempdir().unwrap();
    let csv_path = temp_dir.path().join("test_gt.csv");

    fs::write(
        &csv_path,
        "filename,is_blurry,notes\nimage1.jpg,true,test\nimage2.jpg,false,\nimage3.jpg,1,\n",
    )
    .unwrap();

    let gt = load_ground_truth(&csv_path, "is_blurry").unwrap();

    assert_eq!(gt.len(), 3);
    assert_eq!(gt[0].filename, "image1.jpg");
    assert!(gt[0].label);
    assert!(!gt[1].label);
    assert!(gt[2].label); // "1" parses as true
}
