//! Exposure analysis module.
//!
//! Analyzes images for under/overexposure using histogram-based
//! adaptive analysis with percentile bounds.

use crate::domain::{
    ExposureDetails, ExposureDirection, ImageInfo, Issue, IssueDetails, IssueType, QaModule,
};

/// Configuration for exposure analysis.
#[derive(Debug, Clone)]
pub struct ExposureConfig {
    /// Threshold for underexposure score (0.0-1.0).
    pub under_threshold: f32,
    /// Threshold for overexposure score (0.0-1.0).
    pub over_threshold: f32,
    /// Shadow clipping boundary (0-255). Pixels below this are considered clipped shadows.
    pub shadow_clip_level: u8,
    /// Highlight clipping boundary (0-255). Pixels above this are considered clipped highlights.
    pub highlight_clip_level: u8,
}

impl Default for ExposureConfig {
    fn default() -> Self {
        Self {
            under_threshold: 0.3,
            over_threshold: 0.3,
            shadow_clip_level: 8,
            highlight_clip_level: 247,
        }
    }
}

/// 256-bin histogram of luminance values.
#[derive(Debug, Clone)]
pub struct Histogram {
    bins: [u64; 256],
    total: u64,
}

impl Histogram {
    /// Compute histogram from grayscale image.
    #[must_use]
    pub fn from_luma(image: &image::GrayImage) -> Self {
        let mut bins = [0u64; 256];
        for pixel in image.pixels() {
            bins[usize::from(pixel.0[0])] += 1;
        }
        let total = bins.iter().sum();
        Self { bins, total }
    }

    /// Returns the total pixel count.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.total
    }

    /// Calculate percentile value (0.0-1.0 → luminance 0-255).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn percentile(&self, p: f64) -> u8 {
        if self.total == 0 {
            return 0;
        }
        // Safe: total * p is always positive and within u64 range for reasonable totals
        let target = (self.total as f64 * p).round() as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                // Safe: i is always 0-255 (bin index)
                return i as u8;
            }
        }
        255
    }

    /// Calculate mean luminance.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let sum: u64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &count)| (i as u64) * count)
            .sum();
        // Precision loss acceptable for statistical purposes
        sum as f64 / self.total as f64
    }

    /// Calculate standard deviation of luminance.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let diff = (i as f64) - mean;
                diff * diff * (count as f64)
            })
            .sum::<f64>()
            / (self.total as f64);
        variance.sqrt()
    }

    /// Count pixels below or equal to a threshold.
    #[must_use]
    pub fn count_below(&self, threshold: u8) -> u64 {
        self.bins[..=usize::from(threshold)].iter().sum()
    }

    /// Count pixels above or equal to a threshold.
    #[must_use]
    pub fn count_above(&self, threshold: u8) -> u64 {
        self.bins[usize::from(threshold)..].iter().sum()
    }

    /// Fraction of pixels below or equal to threshold.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn fraction_below(&self, threshold: u8) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count_below(threshold) as f64 / self.total as f64
    }

    /// Fraction of pixels above or equal to threshold.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn fraction_above(&self, threshold: u8) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count_above(threshold) as f64 / self.total as f64
    }
}

/// Exposure analysis results (internal).
#[derive(Debug, Clone)]
pub struct ExposureAnalysis {
    /// Histogram of luminance values.
    pub histogram: Histogram,
    /// 5th percentile luminance.
    pub p5: u8,
    /// 95th percentile luminance.
    pub p95: u8,
    /// Mean luminance.
    pub mean: f64,
    /// Standard deviation of luminance.
    pub std_dev: f64,
    /// Underexposure score (0.0-1.0).
    pub under_score: f32,
    /// Overexposure score (0.0-1.0).
    pub over_score: f32,
}

impl ExposureAnalysis {
    /// Analyze exposure from grayscale image.
    #[must_use]
    pub fn analyze(image: &image::GrayImage, config: &ExposureConfig) -> Self {
        let histogram = Histogram::from_luma(image);

        let p5 = histogram.percentile(0.05);
        let p95 = histogram.percentile(0.95);
        let mean = histogram.mean();
        let std_dev = histogram.std_dev();

        // Shadow clipping: fraction of pixels in deep shadows
        let shadow_fraction = histogram.fraction_below(config.shadow_clip_level);

        // Highlight clipping: fraction of pixels in blown highlights
        let highlight_fraction = histogram.fraction_above(config.highlight_clip_level);

        // Combine with tonal range analysis for robust scoring
        // Low p5 + high shadow fraction → underexposed
        // High p95 + high highlight fraction → overexposed
        let under_score = compute_under_score(p5, shadow_fraction, mean);
        let over_score = compute_over_score(p95, highlight_fraction, mean);

        Self {
            histogram,
            p5,
            p95,
            mean,
            std_dev,
            under_score,
            over_score,
        }
    }
}

/// Compute underexposure score based on shadows and tonal range.
#[allow(clippy::cast_possible_truncation)]
fn compute_under_score(p5: u8, shadow_fraction: f64, mean: f64) -> f32 {
    // Factors:
    // 1. How dark is the 5th percentile? (0 = severe clipping)
    // 2. What fraction of image is in deep shadows?
    // 3. How low is the mean? (dark overall)

    // p5 contribution: 0-16 is increasingly bad
    let p5_score = if p5 < 16 {
        (16.0 - f64::from(p5)) / 16.0
    } else {
        0.0
    };

    // Shadow fraction contribution: >5% is problematic, >15% is severe
    let shadow_score = (shadow_fraction / 0.15).min(1.0);

    // Mean contribution: mean < 50 indicates dark image
    let mean_score = if mean < 50.0 {
        (50.0 - mean) / 50.0
    } else {
        0.0
    };

    // Weighted combination
    let combined = p5_score * 0.4 + shadow_score * 0.4 + mean_score * 0.2;
    (combined as f32).clamp(0.0, 1.0)
}

/// Compute overexposure score based on highlights and tonal range.
#[allow(clippy::cast_possible_truncation)]
fn compute_over_score(p95: u8, highlight_fraction: f64, mean: f64) -> f32 {
    // Factors:
    // 1. How bright is the 95th percentile? (255 = severe clipping)
    // 2. What fraction of image is in blown highlights?
    // 3. How high is the mean? (bright overall)

    // p95 contribution: 240-255 is increasingly bad
    let p95_score = if p95 > 240 {
        f64::from(p95 - 240) / 15.0
    } else {
        0.0
    };

    // Highlight fraction contribution: >5% is problematic, >15% is severe
    let highlight_score = (highlight_fraction / 0.15).min(1.0);

    // Mean contribution: mean > 200 indicates bright image
    let mean_score = if mean > 200.0 {
        (mean - 200.0) / 55.0
    } else {
        0.0
    };

    // Weighted combination
    let combined = p95_score * 0.4 + highlight_score * 0.4 + mean_score * 0.2;
    (combined as f32).clamp(0.0, 1.0)
}

/// Exposure analysis QA module.
pub struct ExposureModule {
    config: ExposureConfig,
}

impl ExposureModule {
    /// Creates a new exposure analysis module with the given configuration.
    #[must_use]
    pub const fn new(config: ExposureConfig) -> Self {
        Self { config }
    }

    /// Returns the module configuration.
    #[must_use]
    pub const fn config(&self) -> &ExposureConfig {
        &self.config
    }
}

impl Default for ExposureModule {
    fn default() -> Self {
        Self::new(ExposureConfig::default())
    }
}

impl QaModule for ExposureModule {
    fn name(&self) -> &'static str {
        "exposure"
    }

    fn analyze(&self, image: &ImageInfo) -> anyhow::Result<Vec<Issue>> {
        let luma = image.to_luma8();
        let analysis = ExposureAnalysis::analyze(&luma, &self.config);

        let under_exceeded = analysis.under_score >= self.config.under_threshold;
        let over_exceeded = analysis.over_score >= self.config.over_threshold;

        if !under_exceeded && !over_exceeded {
            return Ok(vec![]);
        }

        let direction = match (under_exceeded, over_exceeded) {
            (true, true) => ExposureDirection::Both,
            (true, false) => ExposureDirection::Under,
            (false, true) => ExposureDirection::Over,
            (false, false) => unreachable!(),
        };

        // Use the higher score for severity
        let score = analysis.under_score.max(analysis.over_score);

        Ok(vec![Issue {
            issue_type: IssueType::Exposure,
            score,
            details: IssueDetails::Exposure(ExposureDetails {
                direction,
                under_score: analysis.under_score,
                over_score: analysis.over_score,
            }),
        }])
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ExposureConfig::default();
        assert!((config.under_threshold - 0.3).abs() < f32::EPSILON);
        assert!((config.over_threshold - 0.3).abs() < f32::EPSILON);
        assert_eq!(config.shadow_clip_level, 8);
        assert_eq!(config.highlight_clip_level, 247);
    }

    #[test]
    fn test_module_name() {
        let module = ExposureModule::default();
        assert_eq!(module.name(), "exposure");
    }

    #[test]
    fn test_histogram_from_uniform() {
        // Create 256x1 image with all values 0-255
        let mut img = image::GrayImage::new(256, 1);
        for (x, _, pixel) in img.enumerate_pixels_mut() {
            pixel.0[0] = x as u8;
        }

        let hist = Histogram::from_luma(&img);
        assert_eq!(hist.total(), 256);

        // Each bin should have exactly 1
        for count in hist.bins {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_histogram_percentiles() {
        // Create uniform distribution 0-255, 100 samples each
        let mut img = image::GrayImage::new(256, 100);
        for (x, _, pixel) in img.enumerate_pixels_mut() {
            pixel.0[0] = x as u8;
        }

        let hist = Histogram::from_luma(&img);

        // p50 should be ~127-128
        let p50 = hist.percentile(0.5);
        assert!(p50 > 120 && p50 < 136, "p50 should be ~128, got {p50}");

        // p5 should be ~12-13
        let p5 = hist.percentile(0.05);
        assert!(p5 < 20, "p5 should be ~13, got {p5}");

        // p95 should be ~242-243
        let p95 = hist.percentile(0.95);
        assert!(p95 > 235, "p95 should be ~243, got {p95}");
    }

    #[test]
    fn test_histogram_mean_uniform() {
        // All pixels = 128
        let img = image::GrayImage::from_fn(100, 100, |_, _| image::Luma([128u8]));
        let hist = Histogram::from_luma(&img);

        let mean = hist.mean();
        assert!(
            (mean - 128.0).abs() < 0.001,
            "mean should be 128, got {mean}"
        );
    }

    #[test]
    fn test_histogram_std_dev_uniform() {
        // All same value → std_dev = 0
        let img = image::GrayImage::from_fn(100, 100, |_, _| image::Luma([100u8]));
        let hist = Histogram::from_luma(&img);

        let std_dev = hist.std_dev();
        assert!(std_dev.abs() < 0.001, "std_dev should be 0, got {std_dev}");
    }

    #[test]
    fn test_very_dark_image() {
        // All pixels = 0 (completely black)
        let img = image::GrayImage::from_fn(100, 100, |_, _| image::Luma([0u8]));
        let info = ImageInfo::new("dark.jpg", image::DynamicImage::ImageLuma8(img));

        let module = ExposureModule::default();
        let issues = module.analyze(&info).expect("analysis should succeed");

        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, IssueType::Exposure);

        let IssueDetails::Exposure(details) = &issues[0].details else {
            panic!("expected ExposureDetails");
        };
        assert!(
            matches!(details.direction, ExposureDirection::Under),
            "expected Under, got {:?}",
            details.direction
        );
        assert!(
            details.under_score > 0.5,
            "under_score should be high, got {}",
            details.under_score
        );
    }

    #[test]
    fn test_very_bright_image() {
        // All pixels = 255 (completely white)
        let img = image::GrayImage::from_fn(100, 100, |_, _| image::Luma([255u8]));
        let info = ImageInfo::new("bright.jpg", image::DynamicImage::ImageLuma8(img));

        let module = ExposureModule::default();
        let issues = module.analyze(&info).expect("analysis should succeed");

        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, IssueType::Exposure);

        let IssueDetails::Exposure(details) = &issues[0].details else {
            panic!("expected ExposureDetails");
        };
        assert!(
            matches!(details.direction, ExposureDirection::Over),
            "expected Over, got {:?}",
            details.direction
        );
        assert!(
            details.over_score > 0.5,
            "over_score should be high, got {}",
            details.over_score
        );
    }

    #[test]
    fn test_well_exposed_image() {
        // Middle gray - should be well exposed
        let img = image::GrayImage::from_fn(100, 100, |_, _| image::Luma([128u8]));
        let info = ImageInfo::new("normal.jpg", image::DynamicImage::ImageLuma8(img));

        let module = ExposureModule::default();
        let issues = module.analyze(&info).expect("analysis should succeed");

        assert!(
            issues.is_empty(),
            "well-exposed image should have no issues"
        );
    }

    #[test]
    fn test_high_contrast_image() {
        // Half black, half white - both shadows and highlights clipped
        let img = image::GrayImage::from_fn(100, 100, |x, _| {
            if x < 50 {
                image::Luma([0u8])
            } else {
                image::Luma([255u8])
            }
        });
        let info = ImageInfo::new("contrast.jpg", image::DynamicImage::ImageLuma8(img));

        let module = ExposureModule::default();
        let issues = module.analyze(&info).expect("analysis should succeed");

        assert_eq!(issues.len(), 1);
        let IssueDetails::Exposure(details) = &issues[0].details else {
            panic!("expected ExposureDetails");
        };
        assert!(
            matches!(details.direction, ExposureDirection::Both),
            "expected Both, got {:?}",
            details.direction
        );
    }

    #[test]
    fn test_gradient_image() {
        // Smooth gradient from 50 to 200 - well exposed
        let img = image::GrayImage::from_fn(100, 100, |x, _| {
            // x is 0-99, maps to 50-199
            let val = 50 + ((x * 150) / 100) as u8;
            image::Luma([val])
        });
        let info = ImageInfo::new("gradient.jpg", image::DynamicImage::ImageLuma8(img));

        let module = ExposureModule::default();
        let issues = module.analyze(&info).expect("analysis should succeed");

        assert!(
            issues.is_empty(),
            "smooth gradient should be well exposed, got {issues:?}"
        );
    }

    #[test]
    fn test_custom_thresholds() {
        // Dark image with high threshold should not trigger
        let img = image::GrayImage::from_fn(100, 100, |_, _| image::Luma([30u8]));
        let info = ImageInfo::new("dark.jpg", image::DynamicImage::ImageLuma8(img));

        let config = ExposureConfig {
            under_threshold: 0.9, // Very high threshold
            over_threshold: 0.9,
            ..Default::default()
        };
        let module = ExposureModule::new(config);
        let issues = module.analyze(&info).expect("analysis should succeed");

        assert!(
            issues.is_empty(),
            "high threshold should not trigger on slightly dark image"
        );
    }

    #[test]
    fn test_exposure_analysis_fields() {
        // Verify ExposureAnalysis exposes expected statistics
        let img = image::GrayImage::from_fn(100, 100, |x, _| {
            // Gradient 50-149 (x is 0-99)
            let val = 50 + x as u8;
            image::Luma([val])
        });
        let config = ExposureConfig::default();
        let analysis = ExposureAnalysis::analyze(&img, &config);

        // Check all fields are accessible and reasonable
        assert_eq!(analysis.histogram.total(), 10000);
        assert!(analysis.p5 >= 50 && analysis.p5 <= 60, "p5={}", analysis.p5);
        assert!(
            analysis.p95 >= 140 && analysis.p95 <= 150,
            "p95={}",
            analysis.p95
        );
        assert!(
            analysis.mean > 90.0 && analysis.mean < 110.0,
            "mean={}",
            analysis.mean
        );
        assert!(analysis.std_dev > 20.0, "std_dev={}", analysis.std_dev);
        assert!(
            analysis.under_score < 0.1,
            "under_score={}",
            analysis.under_score
        );
        assert!(
            analysis.over_score < 0.1,
            "over_score={}",
            analysis.over_score
        );
    }
}
