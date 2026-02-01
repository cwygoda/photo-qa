//! Blur detection module.
//!
//! Detects motion blur and defocus blur using hybrid approach:
//! - Laplacian variance for sharpness measurement
//! - Sliding window for per-region analysis
//! - Edge-density heuristic for subject detection
//! - FFT directional analysis for blur type classification
//!
//! # Threshold Interaction
//!
//! The blur detection pipeline uses several thresholds that interact:
//!
//! - **`BlurConfig::laplacian_sharp_threshold`** (default: 100.0): Laplacian variance
//!   above this indicates a sharp image. Used to compute `blur_score` (linear mapping
//!   from variance to 0.0-1.0).
//!
//! - **`BlurConfig::edge_density_threshold`** (default: 0.15): Minimum edge density
//!   ratio for a region to be considered part of the subject. Higher values require
//!   stronger edges.
//!
//! - **`BlurConfig::threshold`** (default: 0.5): Final blur score threshold. Images
//!   with `blur_score >= threshold` are flagged as having blur issues.
//!
//! - **`MOTION_THRESHOLD`** / **`DEFOCUS_THRESHOLD`**: Internal constants for blur
//!   type classification. Directional energy ratio above `MOTION_THRESHOLD` (1.5)
//!   indicates motion blur; below `DEFOCUS_THRESHOLD` (1.2) indicates defocus.

use std::f64::consts::PI;

use crate::domain::{
    BlurDetails, BlurType, BoundingBox, ImageInfo, Issue, IssueDetails, IssueType, QaModule,
};

/// Configuration for blur detection.
#[derive(Debug, Clone)]
pub struct BlurConfig {
    /// Threshold for blur score (0.0-1.0). Images above this are flagged.
    pub threshold: f32,
    /// Minimum Laplacian variance for sharp images.
    /// Below this, image is considered blurry.
    pub laplacian_sharp_threshold: f64,
    /// Window size for sliding window analysis.
    pub window_size: u32,
    /// Window stride (overlap control).
    pub window_stride: u32,
    /// Minimum edge density ratio to consider region as subject.
    pub edge_density_threshold: f64,
}

impl Default for BlurConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            laplacian_sharp_threshold: 100.0,
            window_size: 64,
            window_stride: 32,
            edge_density_threshold: 0.15,
        }
    }
}

/// Internal blur analysis results.
///
/// Provides detailed analysis data beyond just the issue report,
/// useful for debugging, logging, and future ML integration.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields exposed for consumers and future saliency model integration
pub struct BlurAnalysis {
    /// Overall sharpness score (Laplacian variance).
    /// Higher values indicate sharper images.
    pub global_variance: f64,
    /// Subject region sharpness (if detected).
    /// Used when subject detection identifies a region of interest.
    pub subject_variance: Option<f64>,
    /// Detected subject bounding box.
    pub subject_bbox: Option<BoundingBox>,
    /// Blur score 0.0 (sharp) to 1.0 (severely blurred).
    pub blur_score: f32,
    /// Classified blur type.
    pub blur_type: BlurType,
    /// Motion blur angle (degrees, 0-180) if motion blur detected.
    /// Indicates the direction of camera/subject movement.
    pub motion_angle: Option<f64>,
}

impl BlurAnalysis {
    /// Analyze image for blur.
    #[must_use]
    pub fn analyze(image: &image::GrayImage, config: &BlurConfig) -> Self {
        let (width, height) = image.dimensions();

        // Step 1: Global Laplacian variance
        let global_variance = compute_laplacian_variance(image);

        // Step 2: Sliding window analysis for regional sharpness
        let region_scores = compute_region_sharpness(image, config);

        // Step 3: Subject detection via edge density heuristic
        let (subject_bbox, subject_variance) =
            detect_subject_region(image, &region_scores, config, width, height);

        // Step 4: Blur type classification via FFT
        let (blur_type, motion_angle) = classify_blur_type(image, global_variance, config);

        // Step 5: Compute final blur score
        let reference_variance = subject_variance.unwrap_or(global_variance);
        let blur_score = compute_blur_score(reference_variance, config);

        Self {
            global_variance,
            subject_variance,
            subject_bbox,
            blur_score,
            blur_type,
            motion_angle,
        }
    }
}

/// Compute Laplacian variance as a sharpness measure.
/// Higher values indicate sharper images.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn compute_laplacian_variance(image: &image::GrayImage) -> f64 {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 {
        return 0.0;
    }

    // Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    let mut sum = 0i64;
    let mut sum_sq = 0i64;
    let mut count = 0u64;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = i32::from(image.get_pixel(x, y).0[0]);
            let top = i32::from(image.get_pixel(x, y - 1).0[0]);
            let bottom = i32::from(image.get_pixel(x, y + 1).0[0]);
            let left = i32::from(image.get_pixel(x - 1, y).0[0]);
            let right = i32::from(image.get_pixel(x + 1, y).0[0]);

            // Laplacian response
            let lap = top + bottom + left + right - 4 * center;

            sum += i64::from(lap);
            sum_sq += i64::from(lap) * i64::from(lap);
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    // Variance = E[X²] - E[X]²
    let mean = sum as f64 / count as f64;
    let mean_sq = sum_sq as f64 / count as f64;
    mean.mul_add(-mean, mean_sq)
}

/// Region sharpness score with coordinates.
#[derive(Debug, Clone, Copy)]
struct RegionScore {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    /// Regional Laplacian variance (reserved for future weighted scoring).
    #[allow(dead_code)]
    variance: f64,
    edge_density: f64,
}

/// Compute sharpness scores for sliding windows across the image.
#[allow(clippy::cast_possible_truncation)]
fn compute_region_sharpness(image: &image::GrayImage, config: &BlurConfig) -> Vec<RegionScore> {
    let (img_width, img_height) = image.dimensions();
    let window = config.window_size;
    let stride = config.window_stride;

    if img_width < window || img_height < window {
        return vec![];
    }

    let mut scores = Vec::new();

    let mut y = 0u32;
    while y + window <= img_height {
        let mut x = 0u32;
        while x + window <= img_width {
            // Extract window
            let sub_image = image::imageops::crop_imm(image, x, y, window, window).to_image();
            let variance = compute_laplacian_variance(&sub_image);
            let edge_density = compute_edge_density(&sub_image);

            scores.push(RegionScore {
                x,
                y,
                width: window,
                height: window,
                variance,
                edge_density,
            });

            x += stride;
        }
        y += stride;
    }

    scores
}

/// Sobel magnitude threshold for edge detection.
/// Squared to avoid sqrt in hot loop (50.0² = 2500.0).
const EDGE_MAGNITUDE_SQ_THRESHOLD: i32 = 2500;

/// Compute edge density using Sobel gradients.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]
fn compute_edge_density(image: &image::GrayImage) -> f64 {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 {
        return 0.0;
    }

    let mut edge_count = 0u64;
    let mut total = 0u64;

    // Sobel edge detection
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let p = |dx: i32, dy: i32| -> i32 {
                let px = (x as i32 + dx) as u32;
                let py = (y as i32 + dy) as u32;
                i32::from(image.get_pixel(px, py).0[0])
            };

            // Sobel X: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            let gx = -p(-1, -1) + p(1, -1) - 2 * p(-1, 0) + 2 * p(1, 0) - p(-1, 1) + p(1, 1);

            // Sobel Y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            let gy = -p(-1, -1) - 2 * p(0, -1) - p(1, -1) + p(-1, 1) + 2 * p(0, 1) + p(1, 1);

            // Use squared magnitude to avoid sqrt in hot loop
            let magnitude_sq = gx * gx + gy * gy;

            if magnitude_sq > EDGE_MAGNITUDE_SQ_THRESHOLD {
                edge_count += 1;
            }
            total += 1;
        }
    }

    if total == 0 {
        return 0.0;
    }

    edge_count as f64 / total as f64
}

/// Detect subject region using edge density heuristic.
/// Returns the bounding box and sharpness of the detected subject region.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn detect_subject_region(
    image: &image::GrayImage,
    region_scores: &[RegionScore],
    config: &BlurConfig,
    img_width: u32,
    img_height: u32,
) -> (Option<BoundingBox>, Option<f64>) {
    if region_scores.is_empty() {
        return (None, None);
    }

    // Find regions with high edge density (likely subject)
    let subject_regions: Vec<&RegionScore> = region_scores
        .iter()
        .filter(|r| r.edge_density >= config.edge_density_threshold)
        .collect();

    if subject_regions.is_empty() {
        // No clear subject detected, use center-weighted approach
        let center_x = img_width / 2;
        let center_y = img_height / 2;
        let center_size = (img_width.min(img_height) / 3).max(64);

        let start_x = center_x.saturating_sub(center_size / 2);
        let start_y = center_y.saturating_sub(center_size / 2);
        let actual_width = center_size.min(img_width - start_x);
        let actual_height = center_size.min(img_height - start_y);

        if actual_width < 3 || actual_height < 3 {
            return (None, None);
        }

        let center_region =
            image::imageops::crop_imm(image, start_x, start_y, actual_width, actual_height)
                .to_image();
        let center_variance = compute_laplacian_variance(&center_region);

        return (
            Some(BoundingBox::new(
                start_x,
                start_y,
                actual_width,
                actual_height,
            )),
            Some(center_variance),
        );
    }

    // Compute bounding box of subject regions
    let min_x = subject_regions.iter().map(|r| r.x).min().unwrap_or(0);
    let min_y = subject_regions.iter().map(|r| r.y).min().unwrap_or(0);
    let max_x = subject_regions
        .iter()
        .map(|r| r.x + r.width)
        .max()
        .unwrap_or(0);
    let max_y = subject_regions
        .iter()
        .map(|r| r.y + r.height)
        .max()
        .unwrap_or(0);

    let bbox_width = max_x.saturating_sub(min_x);
    let bbox_height = max_y.saturating_sub(min_y);

    if bbox_width < 3 || bbox_height < 3 {
        return (None, None);
    }

    // Compute variance in subject region
    let subject_image =
        image::imageops::crop_imm(image, min_x, min_y, bbox_width, bbox_height).to_image();
    let subject_variance = compute_laplacian_variance(&subject_image);

    (
        Some(BoundingBox::new(min_x, min_y, bbox_width, bbox_height)),
        Some(subject_variance),
    )
}

/// Threshold for motion blur detection (directional energy ratio).
const MOTION_THRESHOLD: f64 = 1.5;
/// Threshold below which blur is classified as defocus.
const DEFOCUS_THRESHOLD: f64 = 1.2;

/// Classify blur type using FFT directional analysis.
/// Motion blur shows directional energy peaks; defocus blur is more uniform.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn classify_blur_type(
    image: &image::GrayImage,
    global_variance: f64,
    config: &BlurConfig,
) -> (BlurType, Option<f64>) {
    // If image is sharp, no blur classification needed
    if global_variance >= config.laplacian_sharp_threshold {
        return (BlurType::Sharp, None);
    }

    let (width, height) = image.dimensions();

    // Use smaller dimension for FFT (must be power of 2 for efficiency)
    let fft_size = 64u32.min(width.min(height).next_power_of_two() / 2);
    if fft_size < 16 {
        // Image too small for reliable FFT analysis
        return (BlurType::Defocus, None);
    }

    // Extract center region for FFT
    let start_x = (width - fft_size) / 2;
    let start_y = (height - fft_size) / 2;
    let center_region =
        image::imageops::crop_imm(image, start_x, start_y, fft_size, fft_size).to_image();

    // Compute angular energy distribution via FFT
    let (directional_ratio, dominant_angle) = compute_fft_directional_analysis(&center_region);

    // Motion blur: high directional ratio (energy concentrated in one direction)
    // Defocus blur: low directional ratio (energy spread uniformly)
    if directional_ratio > MOTION_THRESHOLD {
        (BlurType::Motion, Some(dominant_angle))
    } else if directional_ratio < DEFOCUS_THRESHOLD {
        (BlurType::Defocus, None)
    } else {
        (BlurType::Mixed, None)
    }
}

/// Number of angular bins for directional energy analysis.
const NUM_ANGLES: usize = 36;

/// Compute FFT directional analysis for blur type classification.
///
/// Returns `(directional_ratio, dominant_angle)`.
/// Directional ratio > 1.0 indicates directional blur (motion).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
fn compute_fft_directional_analysis(image: &image::GrayImage) -> (f64, f64) {
    let (width, height) = image.dimensions();

    if width < 8 {
        return (1.0, 0.0);
    }

    // Simple DFT-based analysis (using radial energy binning)
    // For each angle, sum the energy at that direction
    let mut angle_energy = [0.0f64; NUM_ANGLES];
    let center = f64::from(width) / 2.0;

    // Compute gradient-based directional energy (approximates FFT spectrum analysis)
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = i32::from(image.get_pixel(x + 1, y).0[0])
                - i32::from(image.get_pixel(x - 1, y).0[0]);
            let gy = i32::from(image.get_pixel(x, y + 1).0[0])
                - i32::from(image.get_pixel(x, y - 1).0[0]);

            let magnitude_sq = gx * gx + gy * gy;
            if magnitude_sq < 1 {
                continue;
            }
            let magnitude = f64::from(magnitude_sq).sqrt();

            // Gradient direction perpendicular to edge
            let angle = f64::from(gy).atan2(f64::from(gx));
            let normalized_angle = (angle + PI) / (2.0 * PI);
            let bin = ((normalized_angle * NUM_ANGLES as f64) as usize).min(NUM_ANGLES - 1);

            // Weight by distance from center (frequency weighting)
            let dx = f64::from(x) - center;
            let dy = f64::from(y) - center;
            let dist = dx.hypot(dy);
            let freq_weight = (dist / center).min(1.0);

            angle_energy[bin] += magnitude * freq_weight;
        }
    }

    // Find dominant angle and compute directional ratio
    let total_energy: f64 = angle_energy.iter().sum();
    if total_energy < 1.0 {
        return (1.0, 0.0);
    }

    let max_energy = angle_energy.iter().copied().fold(0.0f64, f64::max);
    let max_bin = angle_energy
        .iter()
        .position(|&e| (e - max_energy).abs() < 0.001)
        .unwrap_or(0);
    let dominant_angle = (max_bin as f64 / NUM_ANGLES as f64) * 180.0;

    // Include neighboring bins for robust peak detection
    let prev_bin = (max_bin + NUM_ANGLES - 1) % NUM_ANGLES;
    let next_bin = (max_bin + 1) % NUM_ANGLES;
    let peak_energy = angle_energy[prev_bin] + angle_energy[max_bin] + angle_energy[next_bin];

    // Directional ratio: peak energy vs average energy
    let avg_energy = total_energy / NUM_ANGLES as f64;
    let directional_ratio = if avg_energy > 0.0 {
        (peak_energy / 3.0) / avg_energy
    } else {
        1.0
    };

    (directional_ratio, dominant_angle)
}

/// Compute blur score from Laplacian variance.
/// Maps variance to 0.0 (sharp) to 1.0 (severely blurred).
#[allow(clippy::cast_possible_truncation)]
fn compute_blur_score(variance: f64, config: &BlurConfig) -> f32 {
    let threshold = config.laplacian_sharp_threshold;

    if variance >= threshold {
        // Sharp image
        0.0
    } else if variance <= 0.0 {
        // Completely blurred
        1.0
    } else {
        // Linear interpolation between threshold and 0
        let score = 1.0 - (variance / threshold);
        (score as f32).clamp(0.0, 1.0)
    }
}

/// Blur detection QA module.
pub struct BlurModule {
    config: BlurConfig,
}

impl BlurModule {
    /// Creates a new blur detection module with the given configuration.
    #[must_use]
    pub const fn new(config: BlurConfig) -> Self {
        Self { config }
    }

    /// Returns the module configuration.
    #[must_use]
    pub const fn config(&self) -> &BlurConfig {
        &self.config
    }
}

impl Default for BlurModule {
    fn default() -> Self {
        Self::new(BlurConfig::default())
    }
}

impl QaModule for BlurModule {
    fn name(&self) -> &'static str {
        "blur"
    }

    fn analyze(&self, image: &ImageInfo) -> anyhow::Result<Vec<Issue>> {
        let luma = image.to_luma8();
        let analysis = BlurAnalysis::analyze(&luma, &self.config);

        // Only report if blur exceeds threshold
        if analysis.blur_score < self.config.threshold {
            return Ok(vec![]);
        }

        Ok(vec![Issue {
            issue_type: IssueType::Blur,
            score: analysis.blur_score,
            details: IssueDetails::Blur(BlurDetails {
                blur_type: analysis.blur_type,
                subject_bbox: analysis.subject_bbox,
            }),
        }])
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::suboptimal_flops,
    clippy::uninlined_format_args
)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BlurConfig::default();
        assert!((config.threshold - 0.5).abs() < f32::EPSILON);
        assert!((config.laplacian_sharp_threshold - 100.0).abs() < f64::EPSILON);
        assert_eq!(config.window_size, 64);
        assert_eq!(config.window_stride, 32);
    }

    #[test]
    fn test_module_name() {
        let module = BlurModule::default();
        assert_eq!(module.name(), "blur");
    }

    #[test]
    fn test_sharp_synthetic_image() {
        // Create high-contrast checkerboard pattern (very sharp edges)
        let img = image::GrayImage::from_fn(100, 100, |x, y| {
            if (x / 5 + y / 5) % 2 == 0 {
                image::Luma([255u8])
            } else {
                image::Luma([0u8])
            }
        });

        let variance = compute_laplacian_variance(&img);
        // Checkerboard should have high variance due to sharp edges
        assert!(
            variance > 1000.0,
            "checkerboard variance should be high, got {variance}"
        );
    }

    #[test]
    fn test_blurry_synthetic_image() {
        // Uniform gray - no edges, very low variance
        let img = image::GrayImage::from_fn(100, 100, |_, _| image::Luma([128u8]));

        let variance = compute_laplacian_variance(&img);
        assert!(
            variance < 1.0,
            "uniform image should have near-zero variance, got {variance}"
        );
    }

    #[test]
    fn test_gradient_image() {
        // Smooth gradient - low variance (simulates defocus blur)
        let img = image::GrayImage::from_fn(100, 100, |x, _| {
            let val = ((x * 255) / 100) as u8;
            image::Luma([val])
        });

        let variance = compute_laplacian_variance(&img);
        // Gradient has low but non-zero variance
        assert!(
            variance < 10.0,
            "gradient should have low variance, got {variance}"
        );
    }

    #[test]
    fn test_edge_density_uniform() {
        let img = image::GrayImage::from_fn(50, 50, |_, _| image::Luma([128u8]));
        let density = compute_edge_density(&img);
        assert!(
            density < 0.01,
            "uniform image should have near-zero edge density, got {density}"
        );
    }

    #[test]
    fn test_edge_density_edges() {
        // Vertical bars with sharp edges
        let img = image::GrayImage::from_fn(50, 50, |x, _| {
            if x % 10 < 5 {
                image::Luma([0u8])
            } else {
                image::Luma([255u8])
            }
        });
        let density = compute_edge_density(&img);
        assert!(
            density > 0.1,
            "bar pattern should have high edge density, got {density}"
        );
    }

    #[test]
    fn test_blur_analysis_sharp_image() {
        let img = image::GrayImage::from_fn(128, 128, |x, y| {
            if (x / 8 + y / 8) % 2 == 0 {
                image::Luma([255u8])
            } else {
                image::Luma([0u8])
            }
        });

        let config = BlurConfig::default();
        let analysis = BlurAnalysis::analyze(&img, &config);

        assert!(
            analysis.blur_score < 0.1,
            "sharp image should have low blur score, got {}",
            analysis.blur_score
        );
        assert_eq!(analysis.blur_type, BlurType::Sharp);
        // Verify variance fields
        assert!(
            analysis.global_variance > 100.0,
            "checkerboard should have high variance, got {}",
            analysis.global_variance
        );
        // Subject variance should be detected for uniform sharp image
        if let Some(sv) = analysis.subject_variance {
            assert!(sv > 0.0, "subject variance should be positive");
        }
    }

    #[test]
    fn test_blur_analysis_blurry_image() {
        let img = image::GrayImage::from_fn(128, 128, |_, _| image::Luma([128u8]));

        let config = BlurConfig::default();
        let analysis = BlurAnalysis::analyze(&img, &config);

        assert!(
            analysis.blur_score > 0.9,
            "uniform image should have high blur score, got {}",
            analysis.blur_score
        );
        // Verify analysis fields are populated
        assert!(
            analysis.global_variance < 1.0,
            "uniform image should have near-zero variance"
        );
        assert!(
            analysis.motion_angle.is_none(),
            "no motion angle for defocus"
        );
    }

    #[test]
    fn test_blur_module_no_issue_sharp() {
        let img = image::GrayImage::from_fn(128, 128, |x, y| {
            if (x / 8 + y / 8) % 2 == 0 {
                image::Luma([255u8])
            } else {
                image::Luma([0u8])
            }
        });
        let info = ImageInfo::new("sharp.jpg", image::DynamicImage::ImageLuma8(img));

        let module = BlurModule::default();
        let issues = module.analyze(&info).expect("analysis should succeed");

        assert!(issues.is_empty(), "sharp image should have no issues");
    }

    #[test]
    fn test_blur_module_detects_blur() {
        let img = image::GrayImage::from_fn(128, 128, |_, _| image::Luma([128u8]));
        let info = ImageInfo::new("blurry.jpg", image::DynamicImage::ImageLuma8(img));

        let module = BlurModule::default();
        let issues = module.analyze(&info).expect("analysis should succeed");

        assert_eq!(issues.len(), 1, "should detect blur issue");
        assert_eq!(issues[0].issue_type, IssueType::Blur);
        assert!(
            issues[0].score > 0.5,
            "blur score should be high, got {}",
            issues[0].score
        );
    }

    #[test]
    fn test_blur_type_classification() {
        // Uniform = likely defocus (no directional energy)
        let uniform = image::GrayImage::from_fn(64, 64, |_, _| image::Luma([128u8]));
        let config = BlurConfig {
            laplacian_sharp_threshold: 1000.0, // Force blur classification
            ..Default::default()
        };
        let variance = compute_laplacian_variance(&uniform);
        let (blur_type, _) = classify_blur_type(&uniform, variance, &config);
        assert_eq!(blur_type, BlurType::Defocus);
    }

    #[test]
    fn test_motion_blur_detection() {
        // Horizontal stripes (simulates motion blur in one direction)
        let motion = image::GrayImage::from_fn(64, 64, |_, y| {
            // Horizontal bands with blur transition
            let val = ((y as f64 / 8.0).sin() * 127.0 + 128.0) as u8;
            image::Luma([val])
        });

        let config = BlurConfig {
            laplacian_sharp_threshold: 10000.0, // Force blur classification
            ..Default::default()
        };
        let variance = compute_laplacian_variance(&motion);
        let (blur_type, angle) = classify_blur_type(&motion, variance, &config);

        // Should detect directional blur (Motion or Mixed)
        // The exact angle depends on the gradient direction interpretation
        assert!(
            matches!(blur_type, BlurType::Motion | BlurType::Mixed),
            "horizontal stripes should be detected as motion/mixed blur, got {:?}",
            blur_type
        );

        // If motion is detected, angle should exist
        if blur_type == BlurType::Motion {
            assert!(angle.is_some(), "motion blur should report an angle");
        }
    }

    #[test]
    fn test_subject_detection() {
        // Image with sharp center, blurry edges
        let img = image::GrayImage::from_fn(128, 128, |x, y| {
            let cx = 64i32;
            let cy = 64i32;
            let dx = (x as i32 - cx).abs();
            let dy = (y as i32 - cy).abs();
            let dist = dx.max(dy);

            if dist < 32 {
                // Sharp checkerboard in center
                if (x / 4 + y / 4) % 2 == 0 {
                    image::Luma([255u8])
                } else {
                    image::Luma([0u8])
                }
            } else {
                // Blurry edges
                image::Luma([128u8])
            }
        });

        let config = BlurConfig::default();
        let region_scores = compute_region_sharpness(&img, &config);
        let (bbox, _variance) = detect_subject_region(&img, &region_scores, &config, 128, 128);

        assert!(bbox.is_some(), "should detect subject region");
        let bbox = bbox.expect("bbox");

        // Subject should be roughly in center
        let (cx, cy) = bbox.center();
        assert!(
            (cx as i32 - 64).abs() < 40,
            "subject center x should be near image center, got {cx}"
        );
        assert!(
            (cy as i32 - 64).abs() < 40,
            "subject center y should be near image center, got {cy}"
        );
    }

    #[test]
    fn test_custom_threshold() {
        // Image with moderate sharpness (not completely uniform, not fully sharp)
        // Gradient produces low but non-zero variance
        let img = image::GrayImage::from_fn(128, 128, |x, _| {
            let val = ((x * 255) / 128) as u8;
            image::Luma([val])
        });
        let info = ImageInfo::new("gradient.jpg", image::DynamicImage::ImageLuma8(img));

        // Low threshold - should flag the gradient as blurry
        let low_threshold_config = BlurConfig {
            threshold: 0.1,
            ..Default::default()
        };
        let module_low = BlurModule::new(low_threshold_config);
        let issues_low = module_low.analyze(&info).expect("analysis should succeed");
        assert!(
            !issues_low.is_empty(),
            "low threshold should flag gradient as blurry"
        );

        // Very high threshold - should not flag the gradient
        let high_threshold_config = BlurConfig {
            threshold: 1.1, // Above max possible score
            ..Default::default()
        };
        let module_high = BlurModule::new(high_threshold_config);
        let issues_high = module_high.analyze(&info).expect("analysis should succeed");
        assert!(
            issues_high.is_empty(),
            "threshold above 1.0 should not flag anything"
        );
    }

    #[test]
    fn test_small_image() {
        // Very small image should not crash
        let img = image::GrayImage::from_fn(10, 10, |x, y| image::Luma([(x + y) as u8]));
        let info = ImageInfo::new("small.jpg", image::DynamicImage::ImageLuma8(img));

        let module = BlurModule::default();
        let result = module.analyze(&info);

        assert!(result.is_ok(), "small image should not cause error");
    }

    #[test]
    fn test_blur_score_range() {
        let config = BlurConfig::default();

        // Score should be 0 for high variance
        let score_sharp = compute_blur_score(200.0, &config);
        assert!(
            (score_sharp - 0.0).abs() < f32::EPSILON,
            "high variance should give 0 score"
        );

        // Score should be 1 for zero variance
        let score_blurry = compute_blur_score(0.0, &config);
        assert!(
            (score_blurry - 1.0).abs() < f32::EPSILON,
            "zero variance should give 1.0 score"
        );

        // Score should be 0.5 for half threshold
        let score_mid = compute_blur_score(50.0, &config);
        assert!(
            (score_mid - 0.5).abs() < 0.01,
            "half threshold should give ~0.5 score, got {score_mid}"
        );
    }

    #[test]
    fn test_blur_module_rgb_input() {
        // Verify RGB images are properly converted to luma for analysis
        let rgb = image::RgbImage::from_fn(128, 128, |_, _| image::Rgb([128u8, 128, 128]));
        let info = ImageInfo::new("rgb.jpg", image::DynamicImage::ImageRgb8(rgb));

        let module = BlurModule::default();
        let issues = module.analyze(&info).expect("analysis should succeed");

        // Uniform RGB should be detected as blurry (same as uniform luma)
        assert!(
            !issues.is_empty(),
            "uniform RGB should be detected as blurry"
        );
        assert_eq!(issues[0].issue_type, IssueType::Blur);
    }

    #[test]
    fn test_blur_details_serialization() {
        let details = BlurDetails {
            blur_type: BlurType::Motion,
            subject_bbox: Some(BoundingBox::new(10, 20, 100, 80)),
        };

        let json = serde_json::to_string(&details).expect("serialize");
        assert!(json.contains("motion"));
        assert!(json.contains("subject_bbox"));
    }
}
