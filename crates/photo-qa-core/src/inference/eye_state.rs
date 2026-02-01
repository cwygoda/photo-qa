//! Eye state classifier for open/closed detection.
//!
//! A simple CNN binary classifier trained on the CEW (Closed Eyes in the Wild) dataset.
//! Takes cropped eye region images and predicts whether the eye is open or closed.

// Allow common ML code patterns
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]

use anyhow::{Context, Result};
use candle_core::{Device, Module, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, VarBuilder};

use super::sigmoid;

/// Input width for eye region crops.
pub const EYE_WIDTH: usize = 34;
/// Input height for eye region crops.
pub const EYE_HEIGHT: usize = 26;

/// Eye region width as fraction of face width.
const EYE_REGION_WIDTH_RATIO: f32 = 0.25;
/// Eye region height as fraction of face height.
const EYE_REGION_HEIGHT_RATIO: f32 = 0.15;

/// Eye state classifier model.
///
/// Architecture: 3 conv layers with max pooling, followed by 2 FC layers.
/// Input: 34x26 grayscale eye region
/// Output: probability of eye being open (0.0 = closed, 1.0 = open)
pub struct EyeStateClassifier {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    fc1: Linear,
    fc2: Linear,
    device: Device,
}

impl EyeStateClassifier {
    /// Creates a new eye state classifier from weights.
    ///
    /// # Errors
    ///
    /// Returns an error if model weights cannot be loaded or are invalid.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        // Conv layer 1: 1 -> 32 channels, 3x3 kernel
        let conv1 = conv2d(
            1,
            32,
            3,
            Conv2dConfig {
                padding: 1,
                ..Conv2dConfig::default()
            },
            vb.pp("conv1"),
        )?;

        // Conv layer 2: 32 -> 64 channels, 3x3 kernel
        let conv2 = conv2d(
            32,
            64,
            3,
            Conv2dConfig {
                padding: 1,
                ..Conv2dConfig::default()
            },
            vb.pp("conv2"),
        )?;

        // Conv layer 3: 64 -> 128 channels, 3x3 kernel
        let conv3 = conv2d(
            64,
            128,
            3,
            Conv2dConfig {
                padding: 1,
                ..Conv2dConfig::default()
            },
            vb.pp("conv3"),
        )?;

        // After 3 max pools of 2x2:
        // 34x26 -> 17x13 -> 8x6 -> 4x3
        // Flattened: 128 * 4 * 3 = 1536
        let fc1 = linear(1536, 256, vb.pp("fc1"))?;
        let fc2 = linear(256, 1, vb.pp("fc2"))?;

        Ok(Self {
            conv1,
            conv2,
            conv3,
            fc1,
            fc2,
            device,
        })
    }

    /// Preprocesses an eye region crop for classification.
    ///
    /// # Arguments
    /// * `image` - Eye region crop from the original image
    /// * `x` - Left coordinate of eye region (normalized 0-1)
    /// * `y` - Top coordinate of eye region (normalized 0-1)
    /// * `w` - Width of eye region (normalized 0-1)
    /// * `h` - Height of eye region (normalized 0-1)
    ///
    /// # Returns
    /// Tensor of shape `(1, 1, 26, 34)` normalized to `[0, 1]`
    ///
    /// # Errors
    ///
    /// Returns an error if tensor creation fails.
    pub fn preprocess(
        &self,
        image: &image::DynamicImage,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
    ) -> Result<Tensor> {
        let img_w = image.width() as f32;
        let img_h = image.height() as f32;

        // Convert normalized coords to pixel coords
        let px = (x * img_w) as u32;
        let py = (y * img_h) as u32;
        let pw = (w * img_w) as u32;
        let ph = (h * img_h) as u32;

        // Ensure bounds are valid
        let px = px.min(image.width().saturating_sub(1));
        let py = py.min(image.height().saturating_sub(1));
        let pw = pw.min(image.width().saturating_sub(px)).max(1);
        let ph = ph.min(image.height().saturating_sub(py)).max(1);

        // Crop and resize
        let cropped = image.crop_imm(px, py, pw, ph);
        let resized = cropped.resize_exact(
            EYE_WIDTH as u32,
            EYE_HEIGHT as u32,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to grayscale
        let gray = resized.to_luma8();

        // Convert to tensor normalized to [0, 1]
        let data: Vec<f32> = gray.pixels().map(|p| f32::from(p[0]) / 255.0).collect();

        Tensor::from_vec(data, (1, 1, EYE_HEIGHT, EYE_WIDTH), &self.device)
            .context("Failed to create eye tensor")
    }

    /// Classifies whether an eye is open or closed.
    ///
    /// # Arguments
    /// * `x` - Preprocessed eye tensor of shape `(1, 1, 26, 34)`
    ///
    /// # Returns
    /// Probability of eye being open (0.0 = closed, 1.0 = open)
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn classify(&self, x: &Tensor) -> Result<f32> {
        let x = self.forward(x)?;
        let logit = x.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;
        Ok(sigmoid(logit))
    }

    /// Classifies an eye region from the original image.
    ///
    /// # Arguments
    /// * `image` - Original image
    /// * `eye_center` - Eye center coordinates `[x, y]` in normalized `[0,1]` coords
    /// * `face_bbox` - Face bounding box `[x_min, y_min, x_max, y_max]` in normalized coords
    ///
    /// # Returns
    /// `(probability_open, equivalent_ear)` tuple
    ///
    /// # Errors
    ///
    /// Returns an error if preprocessing or inference fails.
    pub fn classify_eye(
        &self,
        image: &image::DynamicImage,
        eye_center: [f32; 2],
        face_bbox: &[f32; 4],
    ) -> Result<(f32, f32)> {
        // Estimate eye region size based on face size
        let face_w = face_bbox[2] - face_bbox[0];
        let face_h = face_bbox[3] - face_bbox[1];

        let eye_w = face_w * EYE_REGION_WIDTH_RATIO;
        let eye_h = face_h * EYE_REGION_HEIGHT_RATIO;

        // Calculate eye region bounds centered on eye point, clamped to valid range
        let x = (eye_center[0] - eye_w / 2.0).max(0.0);
        let y = (eye_center[1] - eye_h / 2.0).max(0.0);

        let input = self.preprocess(image, x, y, eye_w, eye_h)?;
        let prob_open = self.classify(&input)?;

        // Convert probability to equivalent EAR (Eye Aspect Ratio)
        // Open eye: EAR ~0.25-0.35, Closed eye: EAR ~0.05-0.15
        // Map probability: 0.0 -> 0.05, 1.0 -> 0.35
        let ear = 0.05 + prob_open * 0.30;

        Ok((prob_open, ear))
    }
}

impl Module for EyeStateClassifier {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Conv1 + ReLU + MaxPool
        let x = self.conv1.forward(x)?;
        let x = x.relu()?;
        let x = x.max_pool2d(2)?;

        // Conv2 + ReLU + MaxPool
        let x = self.conv2.forward(&x)?;
        let x = x.relu()?;
        let x = x.max_pool2d(2)?;

        // Conv3 + ReLU + MaxPool
        let x = self.conv3.forward(&x)?;
        let x = x.relu()?;
        let x = x.max_pool2d(2)?;

        // Flatten
        let x = x.flatten_from(1)?;

        // FC1 + ReLU
        let x = self.fc1.forward(&x)?;
        let x = x.relu()?;

        // FC2 (logit output)
        self.fc2.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye_dimensions() {
        // Verify the FC layer input size calculation
        // 34x26 -> 17x13 -> 8x6 -> 4x3
        assert_eq!(EYE_WIDTH / 2 / 2 / 2, 4);
        assert_eq!(EYE_HEIGHT / 2 / 2 / 2, 3);
    }
}
