//! Closed eyes detection module.
//!
//! Detects closed eyes in faces using:
//! - `RetinaFace` for face detection
//! - 68-point landmarks for eye region extraction
//! - Eye Aspect Ratio (EAR) with per-face calibration

use crate::domain::ImageInfo;
use crate::domain::Issue;
use crate::domain::QaModule;

/// Configuration for closed eyes detection.
#[derive(Debug, Clone)]
pub struct EyesConfig {
    /// Eye aspect ratio threshold for closed eyes detection.
    pub ear_threshold: f32,
}

impl Default for EyesConfig {
    fn default() -> Self {
        Self { ear_threshold: 0.2 }
    }
}

/// Closed eyes detection QA module.
pub struct EyesModule {
    config: EyesConfig,
}

impl EyesModule {
    /// Creates a new closed eyes detection module with the given configuration.
    #[must_use]
    pub const fn new(config: EyesConfig) -> Self {
        Self { config }
    }
}

impl Default for EyesModule {
    fn default() -> Self {
        Self::new(EyesConfig::default())
    }
}

impl QaModule for EyesModule {
    fn name(&self) -> &'static str {
        "eyes"
    }

    fn analyze(&self, image: &ImageInfo) -> anyhow::Result<Vec<Issue>> {
        // TODO: Implement closed eyes detection
        // 1. Run RetinaFace to detect faces
        // 2. For each face, extract 68-point landmarks
        // 3. Calculate EAR for each eye
        // 4. Calibrate threshold based on face geometry
        // 5. Report per-face results

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
        let config = EyesConfig::default();
        assert!((config.ear_threshold - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_module_name() {
        let module = EyesModule::default();
        assert_eq!(module.name(), "eyes");
    }
}
