//! `BlazeFace` face detection model.
//!
//! Implements the `BlazeFace` architecture for fast face detection with keypoints.
//! Based on the paper "`BlazeFace`: Sub-millisecond Neural Face Detection on Mobile GPUs"
//! and the `PyTorch` implementation at <https://github.com/hollance/BlazeFace-PyTorch>.

// Allow common ML code patterns
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};

use super::sigmoid;

/// Input image size for `BlazeFace`.
pub const INPUT_SIZE: usize = 128;

/// Number of anchor boxes (detections).
const NUM_ANCHORS: usize = 896;

/// Confidence threshold for face detection.
const SCORE_THRESHOLD: f32 = 0.75;

/// Non-maximum suppression IOU threshold.
const NMS_THRESHOLD: f32 = 0.3;

/// A detected face with bounding box and keypoints.
#[derive(Debug, Clone)]
pub struct FaceDetection {
    /// Face bounding box `[x_min, y_min, x_max, y_max]` in normalized `[0,1]` coordinates.
    pub bbox: [f32; 4],
    /// Detection confidence score.
    pub score: f32,
    /// Six keypoints: right eye, left eye, nose, mouth, right ear, left ear.
    /// Each keypoint is `[x, y]` in normalized `[0,1]` coordinates.
    pub keypoints: [[f32; 2]; 6],
}

impl FaceDetection {
    /// Returns the right eye keypoint `[x, y]`.
    #[must_use]
    pub const fn right_eye(&self) -> [f32; 2] {
        self.keypoints[0]
    }

    /// Returns the left eye keypoint `[x, y]`.
    #[must_use]
    pub const fn left_eye(&self) -> [f32; 2] {
        self.keypoints[1]
    }
}

/// `BlazeBlock` - the core building block of `BlazeFace`.
///
/// Uses depthwise separable convolution with optional stride.
/// Note: This implementation uses biased convolutions (BatchNorm folded in)
/// to match the pretrained hollance/BlazeFace-PyTorch weights.
struct BlazeBlock {
    depthwise: Conv2d,
    pointwise: Conv2d,
    channel_pad: usize,
    stride: usize,
}

impl BlazeBlock {
    #[allow(clippy::similar_names)]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let padding = if stride == 2 {
            0
        } else {
            (kernel_size - 1) / 2
        };

        // Depthwise convolution (with bias - BatchNorm folded in)
        let depthwise = conv2d(
            in_channels,
            in_channels,
            kernel_size,
            Conv2dConfig {
                stride,
                padding,
                groups: in_channels,
                dilation: 1,
            },
            vb.pp("depthwise"),
        )?;

        // Pointwise convolution (with bias - BatchNorm folded in)
        let pointwise = conv2d(
            in_channels,
            out_channels,
            1,
            Conv2dConfig::default(),
            vb.pp("pointwise"),
        )?;

        let channel_pad = out_channels.saturating_sub(in_channels);

        Ok(Self {
            depthwise,
            pointwise,
            channel_pad,
            stride,
        })
    }
}

impl Module for BlazeBlock {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Pad input for stride=2 (asymmetric padding)
        let x_padded = if self.stride == 2 {
            x.pad_with_zeros(2, 0, 2)?.pad_with_zeros(3, 0, 2)?
        } else {
            x.clone()
        };

        // Depthwise + ReLU (bias included, no BatchNorm)
        let h = self.depthwise.forward(&x_padded)?;
        let h = h.relu()?;

        // Pointwise (bias included, no BatchNorm)
        let h = self.pointwise.forward(&h)?;

        // Residual connection
        let residual = if self.stride == 2 {
            // Max pool for spatial downsampling
            let pool_size = 2;
            x.max_pool2d(pool_size)?
        } else {
            x.clone()
        };

        // Pad channels if needed
        let residual = if self.channel_pad > 0 {
            residual.pad_with_zeros(1, 0, self.channel_pad)?
        } else {
            residual
        };

        // Add residual and apply ReLU
        (h + residual)?.relu()
    }
}

/// `BlazeFace` face detection model.
///
/// Uses pretrained weights from hollance/BlazeFace-PyTorch with BatchNorm
/// folded into convolutional biases.
pub struct BlazeFace {
    // Initial convolution (with bias, no BatchNorm)
    conv0: Conv2d,

    // Backbone 1 (produces 16x16 feature map)
    backbone1: Vec<BlazeBlock>,

    // Backbone 2 (produces 8x8 feature map)
    backbone2: Vec<BlazeBlock>,

    // Detection heads for 16x16
    classifier_16: Conv2d,
    regressor_16: Conv2d,

    // Detection heads for 8x8
    classifier_8: Conv2d,
    regressor_8: Conv2d,

    // Anchor boxes
    anchors: Tensor,

    device: Device,
}

impl BlazeFace {
    /// Creates a new `BlazeFace` model from weights.
    ///
    /// # Errors
    ///
    /// Returns an error if model weights cannot be loaded or are invalid.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        // Initial 5x5 conv: 3 -> 24 channels, stride 2 (with bias, no BatchNorm)
        let conv0 = conv2d(
            3,
            24,
            5,
            Conv2dConfig {
                stride: 2,
                padding: 0,
                ..Conv2dConfig::default()
            },
            vb.pp("conv0"),
        )?;

        // Backbone 1: 128 -> 64 -> 32 -> 16 -> 8 (produces 88 channels at 8x8 after all strides)
        // Input: 64x64 after initial conv, two stride-2 blocks reduce to 16x16
        let backbone1_config = [
            (24, 24, 3, 1),
            (24, 28, 3, 1),
            (28, 32, 3, 2), // stride 2: 64x64 -> 32x32
            (32, 36, 3, 1),
            (36, 42, 3, 1),
            (42, 48, 3, 2), // stride 2: 32x32 -> 16x16
            (48, 56, 3, 1),
            (56, 64, 3, 1),
            (64, 72, 3, 1),
            (72, 80, 3, 1),
            (80, 88, 3, 1),
        ];

        let mut backbone1 = Vec::new();
        for (i, (in_c, out_c, k, s)) in backbone1_config.iter().enumerate() {
            let block = BlazeBlock::new(*in_c, *out_c, *k, *s, &vb.pp(format!("backbone1.{i}")))?;
            backbone1.push(block);
        }

        // Backbone 2: produces 96 channels at 8x8
        let backbone2_config = [
            (88, 96, 3, 2), // stride 2: 16x16 -> 8x8
            (96, 96, 3, 1),
            (96, 96, 3, 1),
            (96, 96, 3, 1),
            (96, 96, 3, 1),
        ];

        let mut backbone2 = Vec::new();
        for (i, (in_c, out_c, k, s)) in backbone2_config.iter().enumerate() {
            let block = BlazeBlock::new(*in_c, *out_c, *k, *s, &vb.pp(format!("backbone2.{i}")))?;
            backbone2.push(block);
        }

        // Detection heads for 16x16 (512 anchors, 2 per location)
        let classifier_16 =
            conv2d(88, 2, 1, Conv2dConfig::default(), vb.pp("classifier_16"))?;
        let regressor_16 =
            conv2d(88, 32, 1, Conv2dConfig::default(), vb.pp("regressor_16"))?;

        // Detection heads for 8x8 (384 anchors, 6 per location)
        let classifier_8 =
            conv2d(96, 6, 1, Conv2dConfig::default(), vb.pp("classifier_8"))?;
        let regressor_8 = conv2d(96, 96, 1, Conv2dConfig::default(), vb.pp("regressor_8"))?;

        // Generate anchor boxes
        let anchors = Self::generate_anchors(&device)?;

        Ok(Self {
            conv0,
            backbone1,
            backbone2,
            classifier_16,
            regressor_16,
            classifier_8,
            regressor_8,
            anchors,
            device,
        })
    }

    /// Generates anchor boxes for the two feature map scales.
    fn generate_anchors(device: &Device) -> Result<Tensor> {
        let mut anchors = Vec::with_capacity(NUM_ANCHORS * 4);

        // 16x16 feature map: 2 anchors per location = 512 anchors
        for y in 0..16_u8 {
            for x in 0..16_u8 {
                for _ in 0..2 {
                    let cx = (f32::from(x) + 0.5) / 16.0;
                    let cy = (f32::from(y) + 0.5) / 16.0;
                    anchors.extend_from_slice(&[cx, cy, 1.0, 1.0]);
                }
            }
        }

        // 8x8 feature map: 6 anchors per location = 384 anchors
        for y in 0..8_u8 {
            for x in 0..8_u8 {
                for _ in 0..6 {
                    let cx = (f32::from(x) + 0.5) / 8.0;
                    let cy = (f32::from(y) + 0.5) / 8.0;
                    anchors.extend_from_slice(&[cx, cy, 1.0, 1.0]);
                }
            }
        }

        Tensor::from_vec(anchors, (NUM_ANCHORS, 4), device)
            .context("Failed to create anchors tensor")
    }

    /// Preprocesses an image for `BlazeFace` input.
    ///
    /// # Arguments
    /// * `image` - RGB image as `DynamicImage`
    ///
    /// # Returns
    /// Tensor of shape (1, 3, 128, 128) normalized to `[-1, 1]`
    ///
    /// # Errors
    ///
    /// Returns an error if tensor creation fails.
    pub fn preprocess(&self, image: &image::DynamicImage) -> Result<Tensor> {
        // Resize to 128x128
        let resized = image.resize_exact(
            INPUT_SIZE as u32,
            INPUT_SIZE as u32,
            image::imageops::FilterType::Lanczos3,
        );
        let rgb = resized.to_rgb8();

        // Convert to tensor and normalize to [-1, 1]
        let data: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                [
                    (f32::from(p[0]) / 127.5) - 1.0,
                    (f32::from(p[1]) / 127.5) - 1.0,
                    (f32::from(p[2]) / 127.5) - 1.0,
                ]
            })
            .collect();

        // Reshape to (1, 3, 128, 128) - NCHW format
        let tensor = Tensor::from_vec(data, (1, INPUT_SIZE, INPUT_SIZE, 3), &self.device)?;
        tensor
            .permute((0, 3, 1, 2))?
            .to_dtype(DType::F32)
            .context("Failed to preprocess image")
    }

    /// Runs face detection on a preprocessed input tensor.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // Initial convolution with asymmetric padding (bias included, no BatchNorm)
        let x = x.pad_with_zeros(2, 1, 2)?.pad_with_zeros(3, 1, 2)?;
        let x = self.conv0.forward(&x)?;
        let x = x.relu()?;

        // Backbone 1
        let mut h = x;
        for block in &self.backbone1 {
            h = block.forward(&h)?;
        }
        let feature_16 = h.clone();

        // Backbone 2
        for block in &self.backbone2 {
            h = block.forward(&h)?;
        }
        let feature_8 = h;

        // Detection heads for 16x16
        let c1 = self.classifier_16.forward(&feature_16)?;
        let c1 = c1.permute((0, 2, 3, 1))?.reshape((1, 512, 1))?;

        let r1 = self.regressor_16.forward(&feature_16)?;
        let r1 = r1.permute((0, 2, 3, 1))?.reshape((1, 512, 16))?;

        // Detection heads for 8x8
        let c2 = self.classifier_8.forward(&feature_8)?;
        let c2 = c2.permute((0, 2, 3, 1))?.reshape((1, 384, 1))?;

        let r2 = self.regressor_8.forward(&feature_8)?;
        let r2 = r2.permute((0, 2, 3, 1))?.reshape((1, 384, 16))?;

        // Concatenate outputs
        let scores = Tensor::cat(&[c1, c2], 1)?;
        let boxes = Tensor::cat(&[r1, r2], 1)?;

        Ok((scores, boxes))
    }

    /// Detects faces in an image.
    ///
    /// # Arguments
    /// * `image` - Input image
    ///
    /// # Returns
    /// Vector of detected faces with bounding boxes and keypoints.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn detect(&self, image: &image::DynamicImage) -> Result<Vec<FaceDetection>> {
        let input = self.preprocess(image)?;
        let (scores, boxes) = self.forward(&input)?;

        // Decode detections
        self.decode_detections(&scores, &boxes)
    }

    /// Decodes raw network output into face detections.
    fn decode_detections(&self, scores: &Tensor, boxes: &Tensor) -> Result<Vec<FaceDetection>> {
        let scores = scores.squeeze(0)?.to_vec2::<f32>()?;
        let boxes = boxes.squeeze(0)?.to_vec2::<f32>()?;
        let anchors = self.anchors.to_vec2::<f32>()?;

        let mut detections = Vec::new();
        let input_size_f32 = INPUT_SIZE as f32;

        for i in 0..NUM_ANCHORS {
            let score = sigmoid(scores[i][0]);

            if score < SCORE_THRESHOLD {
                continue;
            }

            let anchor = &anchors[i];
            let box_data = &boxes[i];

            // Decode bounding box (center format -> corner format)
            let cx = anchor[0] + box_data[0] / input_size_f32;
            let cy = anchor[1] + box_data[1] / input_size_f32;
            let w = box_data[2] / input_size_f32;
            let h = box_data[3] / input_size_f32;

            let x_min = (cx - w / 2.0).clamp(0.0, 1.0);
            let y_min = (cy - h / 2.0).clamp(0.0, 1.0);
            let x_max = (cx + w / 2.0).clamp(0.0, 1.0);
            let y_max = (cy + h / 2.0).clamp(0.0, 1.0);

            // Decode 6 keypoints
            let mut keypoints = [[0.0f32; 2]; 6];
            for k in 0..6 {
                let kp_x = anchor[0] + box_data[4 + k * 2] / input_size_f32;
                let kp_y = anchor[1] + box_data[4 + k * 2 + 1] / input_size_f32;
                keypoints[k] = [kp_x.clamp(0.0, 1.0), kp_y.clamp(0.0, 1.0)];
            }

            detections.push(FaceDetection {
                bbox: [x_min, y_min, x_max, y_max],
                score,
                keypoints,
            });
        }

        // Apply non-maximum suppression
        let detections = Self::nms(detections);

        Ok(detections)
    }

    /// Non-maximum suppression to remove overlapping detections.
    ///
    /// Note: This is O(n²) due to `remove(0)` being O(n). Acceptable for typical
    /// face counts (<20). For high-throughput scenarios, consider `VecDeque`.
    fn nms(mut detections: Vec<FaceDetection>) -> Vec<FaceDetection> {
        // Sort by score descending (NaN scores treated as equal)
        detections.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = Vec::new();

        while !detections.is_empty() {
            let det = detections.remove(0);
            let bbox = det.bbox;
            keep.push(det);

            detections.retain(|other| iou(&bbox, &other.bbox) < NMS_THRESHOLD);
        }

        keep
    }
}

/// Intersection over Union for two bounding boxes.
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);

    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);

    let union = area_a + area_b - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_no_overlap() {
        let a = [0.0, 0.0, 0.5, 0.5];
        let b = [0.6, 0.6, 1.0, 1.0];
        assert!((iou(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_iou_full_overlap() {
        let a = [0.0, 0.0, 1.0, 1.0];
        let b = [0.0, 0.0, 1.0, 1.0];
        assert!((iou(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let a = [0.0, 0.0, 0.5, 0.5];
        let b = [0.25, 0.25, 0.75, 0.75];
        // Intersection: 0.25 * 0.25 = 0.0625
        // Union: 0.25 + 0.25 - 0.0625 = 0.4375
        let expected = 0.0625 / 0.4375;
        assert!((iou(&a, &b) - expected).abs() < 1e-6);
    }
}
