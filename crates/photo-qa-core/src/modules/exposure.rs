//! Exposure analysis module.
//!
//! Analyzes images for under/overexposure using histogram-based
//! adaptive analysis with percentile bounds.

use crate::domain::ImageInfo;
use crate::domain::Issue;
use crate::domain::QaModule;

/// Configuration for exposure analysis.
#[derive(Debug, Clone)]
pub struct ExposureConfig {
    /// Threshold for underexposure score (0.0-1.0).
    pub under_threshold: f32,
    /// Threshold for overexposure score (0.0-1.0).
    pub over_threshold: f32,
}

impl Default for ExposureConfig {
    fn default() -> Self {
        Self {
            under_threshold: 0.3,
            over_threshold: 0.3,
        }
    }
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

    fn analyze(&self, _image: &ImageInfo) -> anyhow::Result<Vec<Issue>> {
        // TODO: Implement exposure analysis
        // 1. Convert to luminance channel
        // 2. Compute histogram (256 bins)
        // 3. Calculate 5th and 95th percentiles
        // 4. Score under/overexposure based on clipping
        let _ = &self.config; // Will be used when implemented
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ExposureConfig::default();
        assert!((config.under_threshold - 0.3).abs() < f32::EPSILON);
        assert!((config.over_threshold - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_module_name() {
        let module = ExposureModule::default();
        assert_eq!(module.name(), "exposure");
    }
}
