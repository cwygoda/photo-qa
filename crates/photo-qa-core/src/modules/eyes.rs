//! Closed eyes detection module.
//!
//! Detects closed eyes in faces using:
//! - `BlazeFace` for face detection and eye keypoint localization
//! - Eye state classifier for open/closed binary classification
//! - EAR (Eye Aspect Ratio) approximation from classifier output

// Allow common ML/image code patterns
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use tracing::{debug, warn};

use crate::domain::{
    BoundingBox, EyesDetails, FaceResult, ImageInfo, Issue, IssueDetails, IssueType, QaModule,
};
use crate::inference::{get_device, load_safetensors, BlazeFace, EyeStateClassifier};

/// Configuration for closed eyes detection.
#[derive(Debug, Clone)]
pub struct EyesConfig {
    /// Eye aspect ratio threshold for closed eyes detection.
    /// Eyes with EAR below this threshold are considered closed.
    pub ear_threshold: f32,

    /// Minimum face detection confidence.
    pub min_face_confidence: f32,

    /// Path to `BlazeFace` model weights.
    pub blazeface_model_path: Option<std::path::PathBuf>,

    /// Path to eye state classifier model weights.
    pub eye_state_model_path: Option<std::path::PathBuf>,
}

impl Default for EyesConfig {
    fn default() -> Self {
        Self {
            ear_threshold: 0.2,
            min_face_confidence: 0.75,
            blazeface_model_path: None,
            eye_state_model_path: None,
        }
    }
}

impl EyesConfig {
    /// Sets the path to the `BlazeFace` model.
    #[must_use]
    pub fn with_blazeface_path(mut self, path: impl AsRef<Path>) -> Self {
        self.blazeface_model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the path to the eye state model.
    #[must_use]
    pub fn with_eye_state_path(mut self, path: impl AsRef<Path>) -> Self {
        self.eye_state_model_path = Some(path.as_ref().to_path_buf());
        self
    }
}

/// Lazily initialized models for eyes detection.
struct EyesModels {
    blazeface: BlazeFace,
    eye_state: EyeStateClassifier,
}

/// Closed eyes detection QA module.
pub struct EyesModule {
    config: EyesConfig,
    models: OnceLock<Result<EyesModels, String>>,
}

impl EyesModule {
    /// Creates a new closed eyes detection module with the given configuration.
    #[must_use]
    pub const fn new(config: EyesConfig) -> Self {
        Self {
            config,
            models: OnceLock::new(),
        }
    }

    /// Loads or returns the cached models.
    fn get_models(&self) -> Result<&EyesModels> {
        let result = self
            .models
            .get_or_init(|| self.load_models().map_err(|e| e.to_string()));

        result.as_ref().map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Loads the `BlazeFace` and eye state models.
    fn load_models(&self) -> Result<EyesModels> {
        let device = get_device();

        // Load BlazeFace
        let blazeface_path = self
            .config
            .blazeface_model_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("BlazeFace model path not configured"))?;

        debug!("Loading BlazeFace from {}", blazeface_path.display());
        let vb = load_safetensors(blazeface_path, &device)
            .context("Failed to load BlazeFace weights")?;
        let blazeface = BlazeFace::new(vb).context("Failed to create BlazeFace model")?;

        // Load eye state classifier
        let eye_state_path = self
            .config
            .eye_state_model_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Eye state model path not configured"))?;

        debug!(
            "Loading eye state classifier from {}",
            eye_state_path.display()
        );
        let vb = load_safetensors(eye_state_path, &device)
            .context("Failed to load eye state weights")?;
        let eye_state =
            EyeStateClassifier::new(vb).context("Failed to create eye state classifier")?;

        Ok(EyesModels {
            blazeface,
            eye_state,
        })
    }

    /// Analyzes an image for closed eyes.
    fn analyze_image(&self, image: &ImageInfo) -> Result<Vec<FaceResult>> {
        let models = self.get_models()?;

        // Detect faces
        let detections = models
            .blazeface
            .detect(&image.image)
            .context("Face detection failed")?;

        debug!("Found {} faces", detections.len());

        let mut results = Vec::new();

        for det in detections {
            if det.score < self.config.min_face_confidence {
                debug!("Skipping low-confidence face: {:.2}", det.score);
                continue;
            }

            // Get eye keypoints
            let right_eye = det.right_eye();
            let left_eye = det.left_eye();

            // Classify each eye
            let (_, right_ear) = models
                .eye_state
                .classify_eye(&image.image, right_eye, &det.bbox)
                .unwrap_or_else(|e| {
                    warn!("Failed to classify right eye: {}", e);
                    (1.0, 0.3) // Assume open on error
                });

            let (_, left_ear) = models
                .eye_state
                .classify_eye(&image.image, left_eye, &det.bbox)
                .unwrap_or_else(|e| {
                    warn!("Failed to classify left eye: {}", e);
                    (1.0, 0.3) // Assume open on error
                });

            // Average EAR for both eyes
            let avg_ear = (right_ear + left_ear) / 2.0;
            let eyes_closed = avg_ear < self.config.ear_threshold;

            debug!(
                "Face at {:?}: right_ear={:.3}, left_ear={:.3}, closed={}",
                det.bbox, right_ear, left_ear, eyes_closed
            );

            // Convert normalized bbox to pixel coordinates
            let img_w = image.width as f32;
            let img_h = image.height as f32;

            results.push(FaceResult {
                bbox: BoundingBox {
                    x: (det.bbox[0] * img_w) as u32,
                    y: (det.bbox[1] * img_h) as u32,
                    width: ((det.bbox[2] - det.bbox[0]) * img_w) as u32,
                    height: ((det.bbox[3] - det.bbox[1]) * img_h) as u32,
                },
                left_ear,
                right_ear,
                eyes_closed,
                confidence: det.score,
            });
        }

        Ok(results)
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

    fn analyze(&self, image: &ImageInfo) -> Result<Vec<Issue>> {
        // Check if models are configured
        if self.config.blazeface_model_path.is_none() || self.config.eye_state_model_path.is_none()
        {
            debug!("Eyes module skipped: models not configured");
            return Ok(vec![]);
        }

        let faces = self.analyze_image(image)?;

        if faces.is_empty() {
            debug!("No faces detected");
            return Ok(vec![]);
        }

        // Check if any face has closed eyes
        let closed_count = faces.iter().filter(|f| f.eyes_closed).count();

        if closed_count == 0 {
            debug!("No closed eyes detected in {} faces", faces.len());
            return Ok(vec![]);
        }

        // Calculate severity based on proportion of faces with closed eyes
        let severity = closed_count as f32 / faces.len() as f32;

        Ok(vec![Issue {
            issue_type: IssueType::ClosedEyes,
            score: severity,
            details: IssueDetails::Eyes(EyesDetails { faces }),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EyesConfig::default();
        assert!((config.ear_threshold - 0.2).abs() < f32::EPSILON);
        assert!((config.min_face_confidence - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_module_name() {
        let module = EyesModule::default();
        assert_eq!(module.name(), "eyes");
    }

    #[test]
    fn test_config_builder() {
        let config = EyesConfig::default()
            .with_blazeface_path("/path/to/blazeface.safetensors")
            .with_eye_state_path("/path/to/eye_state.safetensors");

        assert!(config.blazeface_model_path.is_some());
        assert!(config.eye_state_model_path.is_some());
    }

    #[test]
    fn test_analyze_without_models() {
        // Without model paths configured, should return empty
        let module = EyesModule::default();

        let image = image::DynamicImage::new_rgb8(100, 100);
        let info = ImageInfo::new("test.jpg", image);

        let result = module.analyze(&info);
        assert!(result.is_ok());
        assert!(result.is_ok_and(|v| v.is_empty()));
    }
}
