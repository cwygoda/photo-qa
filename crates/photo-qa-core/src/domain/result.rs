//! Analysis result types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::Issue;

/// Complete analysis result for a single image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Path to the analyzed image.
    pub path: String,
    /// Timestamp of analysis (ISO 8601).
    pub timestamp: String,
    /// Image dimensions.
    pub dimensions: ImageDimensions,
    /// Detected quality issues.
    pub issues: Vec<Issue>,
    /// Optional EXIF metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exif: Option<HashMap<String, String>>,
}

/// Image dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ImageDimensions {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// Basic image information extracted during loading.
#[derive(Debug, Clone)]
pub struct ImageInfo {
    /// Path to the image file.
    pub path: String,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Decoded image data.
    pub image: image::DynamicImage,
}
