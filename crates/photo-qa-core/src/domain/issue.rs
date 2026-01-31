//! Issue types detected during photo QA analysis.

use serde::{Deserialize, Serialize};

use super::BoundingBox;

/// A quality issue detected in an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    /// Type of issue detected.
    #[serde(rename = "type")]
    pub issue_type: IssueType,
    /// Severity score from 0.0 (ok) to 1.0 (severe).
    pub score: f32,
    /// Issue-specific details.
    pub details: IssueDetails,
}

/// The type of quality issue.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueType {
    /// Image blur (motion or defocus).
    Blur,
    /// Exposure problems (under or over).
    Exposure,
    /// Closed eyes detected in faces.
    ClosedEyes,
}

/// Issue-specific details.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IssueDetails {
    /// Blur detection details.
    Blur(BlurDetails),
    /// Exposure analysis details.
    Exposure(ExposureDetails),
    /// Closed eyes detection details.
    Eyes(EyesDetails),
}

/// Details for blur detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlurDetails {
    /// Type of blur detected.
    pub blur_type: BlurType,
    /// Bounding box of the detected subject region.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject_bbox: Option<BoundingBox>,
}

/// Type of blur detected.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlurType {
    /// Motion blur from camera or subject movement.
    Motion,
    /// Out-of-focus blur.
    Defocus,
    /// Mixed blur types.
    Mixed,
    /// Image is sharp (no blur detected).
    Sharp,
}

/// Details for exposure analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureDetails {
    /// Direction of exposure issue.
    pub direction: ExposureDirection,
    /// Underexposure score (0.0 to 1.0).
    pub under_score: f32,
    /// Overexposure score (0.0 to 1.0).
    pub over_score: f32,
}

/// Direction of exposure issue.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExposureDirection {
    /// Image is underexposed.
    Under,
    /// Image is overexposed.
    Over,
    /// Both under and overexposed regions.
    Both,
}

/// Details for closed eyes detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyesDetails {
    /// Per-face detection results.
    pub faces: Vec<FaceResult>,
}

/// Detection result for a single face.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceResult {
    /// Face bounding box.
    pub bbox: BoundingBox,
    /// Left eye aspect ratio.
    pub left_ear: f32,
    /// Right eye aspect ratio.
    pub right_ear: f32,
    /// Whether eyes are detected as closed.
    pub eyes_closed: bool,
    /// Detection confidence (0.0 to 1.0).
    pub confidence: f32,
}
