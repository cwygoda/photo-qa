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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageDimensions {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl ImageDimensions {
    /// Create new dimensions.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// A bounding box in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// X coordinate of top-left corner.
    pub x: u32,
    /// Y coordinate of top-left corner.
    pub y: u32,
    /// Width of the box.
    pub width: u32,
    /// Height of the box.
    pub height: u32,
}

impl BoundingBox {
    /// Create a new bounding box.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Calculate the area of the bounding box.
    #[must_use]
    pub const fn area(&self) -> u32 {
        self.width * self.height
    }

    /// Check if a point is inside the bounding box.
    #[must_use]
    pub const fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    /// Get the center point of the bounding box.
    #[must_use]
    pub const fn center(&self) -> (u32, u32) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }
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

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_image_dimensions_new() {
        let dims = ImageDimensions::new(1920, 1080);
        assert_eq!(dims.width, 1920);
        assert_eq!(dims.height, 1080);
    }

    #[test]
    fn test_image_dimensions_serialization() {
        let dims = ImageDimensions::new(1920, 1080);
        let json = serde_json::to_string(&dims).expect("serialize");
        assert_eq!(json, r#"{"width":1920,"height":1080}"#);

        let parsed: ImageDimensions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, dims);
    }

    #[test]
    fn test_bounding_box_new() {
        let bbox = BoundingBox::new(10, 20, 100, 50);
        assert_eq!(bbox.x, 10);
        assert_eq!(bbox.y, 20);
        assert_eq!(bbox.width, 100);
        assert_eq!(bbox.height, 50);
    }

    #[test]
    fn test_bounding_box_area() {
        let bbox = BoundingBox::new(0, 0, 100, 50);
        assert_eq!(bbox.area(), 5000);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox::new(10, 20, 100, 50);

        // Inside
        assert!(bbox.contains(10, 20)); // top-left corner
        assert!(bbox.contains(50, 40)); // middle
        assert!(bbox.contains(109, 69)); // bottom-right (exclusive boundary)

        // Outside
        assert!(!bbox.contains(9, 20)); // left of box
        assert!(!bbox.contains(10, 19)); // above box
        assert!(!bbox.contains(110, 40)); // right of box (at boundary)
        assert!(!bbox.contains(50, 70)); // below box (at boundary)
    }

    #[test]
    fn test_bounding_box_center() {
        let bbox = BoundingBox::new(10, 20, 100, 50);
        assert_eq!(bbox.center(), (60, 45));
    }

    #[test]
    fn test_bounding_box_serialization() {
        let bbox = BoundingBox::new(10, 20, 100, 50);
        let json = serde_json::to_string(&bbox).expect("serialize");
        assert_eq!(json, r#"{"x":10,"y":20,"width":100,"height":50}"#);

        let parsed: BoundingBox = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, bbox);
    }
}
