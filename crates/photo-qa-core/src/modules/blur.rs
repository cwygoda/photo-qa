//! Blur detection module.
//!
//! Detects motion blur and defocus blur using hybrid approach:
//! - Edge-density heuristic for subject detection
//! - Saliency model fallback for ambiguous cases
//! - FFT analysis for blur type classification

use crate::domain::ImageInfo;
use crate::domain::Issue;
use crate::domain::QaModule;

/// Configuration for blur detection.
#[derive(Debug, Clone)]
pub struct BlurConfig {
    /// Threshold for blur score (0.0-1.0). Images above this are flagged.
    pub threshold: f32,
}

impl Default for BlurConfig {
    fn default() -> Self {
        Self { threshold: 0.5 }
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
        // TODO: Implement blur detection
        // 1. Convert to grayscale
        // 2. Detect subject region (edge density or saliency model)
        // 3. Calculate Laplacian variance in subject region
        // 4. Classify blur type via FFT analysis

        let _ = (image, &self.config);

        // Placeholder: return no issues
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BlurConfig::default();
        assert!((config.threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_module_name() {
        let module = BlurModule::default();
        assert_eq!(module.name(), "blur");
    }
}
