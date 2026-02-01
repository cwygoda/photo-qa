//! Synthetic image builders for testing.

use image::{DynamicImage, GrayImage, Luma, RgbImage};
use photo_qa_core::domain::ImageInfo;

/// Builder for creating synthetic test images.
///
/// Provides convenience methods for generating images with specific
/// characteristics (sharp, blurry, underexposed, etc.).
pub struct SyntheticImageBuilder;

impl SyntheticImageBuilder {
    // === Sharp/High-Contrast Images ===

    /// Creates a high-contrast checkerboard pattern (very sharp edges).
    ///
    /// Useful for testing blur detection - should be detected as sharp.
    #[must_use]
    pub fn checkerboard(width: u32, height: u32) -> ImageInfo {
        Self::checkerboard_with_cell_size(width, height, 8)
    }

    /// Creates a checkerboard with custom cell size.
    #[must_use]
    pub fn checkerboard_with_cell_size(width: u32, height: u32, cell_size: u32) -> ImageInfo {
        let img = GrayImage::from_fn(width, height, |x, y| {
            if (x / cell_size + y / cell_size) % 2 == 0 {
                Luma([255u8])
            } else {
                Luma([0u8])
            }
        });
        ImageInfo::new("synthetic://checkerboard", DynamicImage::ImageLuma8(img))
    }

    /// Creates vertical bars pattern (sharp edges).
    #[must_use]
    pub fn vertical_bars(width: u32, height: u32, bar_width: u32) -> ImageInfo {
        let img = GrayImage::from_fn(width, height, |x, _| {
            if (x / bar_width) % 2 == 0 {
                Luma([255u8])
            } else {
                Luma([0u8])
            }
        });
        ImageInfo::new("synthetic://vertical_bars", DynamicImage::ImageLuma8(img))
    }

    /// Creates horizontal bars pattern (sharp edges).
    #[must_use]
    pub fn horizontal_bars(width: u32, height: u32, bar_height: u32) -> ImageInfo {
        let img = GrayImage::from_fn(width, height, |_, y| {
            if (y / bar_height) % 2 == 0 {
                Luma([255u8])
            } else {
                Luma([0u8])
            }
        });
        ImageInfo::new("synthetic://horizontal_bars", DynamicImage::ImageLuma8(img))
    }

    // === Blurry Images ===

    /// Creates a uniform gray image (no edges, simulates severe blur).
    ///
    /// Useful for testing blur detection - should be detected as blurry.
    #[must_use]
    pub fn uniform_gray(width: u32, height: u32, value: u8) -> ImageInfo {
        let img = GrayImage::from_fn(width, height, |_, _| Luma([value]));
        ImageInfo::new("synthetic://uniform_gray", DynamicImage::ImageLuma8(img))
    }

    /// Creates a smooth horizontal gradient (low variance, simulates defocus).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn horizontal_gradient(width: u32, height: u32) -> ImageInfo {
        let img = GrayImage::from_fn(width, height, |x, _| {
            let val = ((u32::from(u8::MAX) * x) / width.max(1)) as u8;
            Luma([val])
        });
        ImageInfo::new(
            "synthetic://horizontal_gradient",
            DynamicImage::ImageLuma8(img),
        )
    }

    /// Creates a vertical gradient.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn vertical_gradient(width: u32, height: u32) -> ImageInfo {
        let img = GrayImage::from_fn(width, height, |_, y| {
            let val = ((u32::from(u8::MAX) * y) / height.max(1)) as u8;
            Luma([val])
        });
        ImageInfo::new(
            "synthetic://vertical_gradient",
            DynamicImage::ImageLuma8(img),
        )
    }

    // === Exposure Images ===

    /// Creates a completely black image (severely underexposed).
    #[must_use]
    pub fn underexposed(width: u32, height: u32) -> ImageInfo {
        Self::uniform_gray(width, height, 0)
    }

    /// Creates a very dark image with slight variation.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn dark_image(width: u32, height: u32, max_brightness: u8) -> ImageInfo {
        let img = GrayImage::from_fn(width, height, |x, y| {
            // Add slight variation to avoid being completely uniform
            let val = ((x + y) % u32::from(max_brightness.max(1))) as u8;
            Luma([val])
        });
        ImageInfo::new("synthetic://dark", DynamicImage::ImageLuma8(img))
    }

    /// Creates a completely white image (severely overexposed).
    #[must_use]
    pub fn overexposed(width: u32, height: u32) -> ImageInfo {
        Self::uniform_gray(width, height, 255)
    }

    /// Creates a very bright image with slight variation.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn bright_image(width: u32, height: u32, min_brightness: u8) -> ImageInfo {
        let img = GrayImage::from_fn(width, height, |x, y| {
            let range = 255 - min_brightness;
            let val = min_brightness + ((x + y) % u32::from(range.max(1))) as u8;
            Luma([val])
        });
        ImageInfo::new("synthetic://bright", DynamicImage::ImageLuma8(img))
    }

    /// Creates a well-exposed middle-gray image.
    #[must_use]
    pub fn well_exposed(width: u32, height: u32) -> ImageInfo {
        Self::uniform_gray(width, height, 128)
    }

    /// Creates an image with good tonal range (50-200).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn good_tonal_range(width: u32, height: u32) -> ImageInfo {
        let img = GrayImage::from_fn(width, height, |x, _| {
            let val = 50 + ((x * 150) / width.max(1)) as u8;
            Luma([val])
        });
        ImageInfo::new(
            "synthetic://good_tonal_range",
            DynamicImage::ImageLuma8(img),
        )
    }

    // === Special Test Images ===

    /// Creates a 1x1 pixel image (edge case).
    #[must_use]
    pub fn single_pixel(value: u8) -> ImageInfo {
        let img = GrayImage::from_fn(1, 1, |_, _| Luma([value]));
        ImageInfo::new("synthetic://1x1", DynamicImage::ImageLuma8(img))
    }

    /// Creates a tiny 2x2 image (minimal processable size).
    #[must_use]
    pub fn tiny(values: [[u8; 2]; 2]) -> ImageInfo {
        let mut img = GrayImage::new(2, 2);
        for (y, row) in values.iter().enumerate() {
            for (x, &val) in row.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                img.put_pixel(x as u32, y as u32, Luma([val]));
            }
        }
        ImageInfo::new("synthetic://2x2", DynamicImage::ImageLuma8(img))
    }

    /// Creates an RGB color image.
    #[must_use]
    pub fn rgb_uniform(width: u32, height: u32, r: u8, g: u8, b: u8) -> ImageInfo {
        let img = RgbImage::from_fn(width, height, |_, _| image::Rgb([r, g, b]));
        ImageInfo::new("synthetic://rgb_uniform", DynamicImage::ImageRgb8(img))
    }

    /// Creates an image with a sharp center and blurry edges.
    ///
    /// Useful for testing subject detection.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn sharp_center_blurry_edges(width: u32, height: u32) -> ImageInfo {
        let cx = (width / 2) as i32;
        let cy = (height / 2) as i32;
        let radius = (width.min(height) / 4) as i32;

        let img = GrayImage::from_fn(width, height, |x, y| {
            let dx = (x as i32 - cx).abs();
            let dy = (y as i32 - cy).abs();
            let dist = dx.max(dy);

            if dist < radius {
                // Sharp checkerboard in center
                if (x / 4 + y / 4) % 2 == 0 {
                    Luma([255u8])
                } else {
                    Luma([0u8])
                }
            } else {
                // Blurry uniform gray on edges
                Luma([128u8])
            }
        });
        ImageInfo::new("synthetic://sharp_center", DynamicImage::ImageLuma8(img))
    }
}

/// Convenience functions for common test images.
///
/// These return `ImageInfo` directly for quick use in tests.
impl SyntheticImageBuilder {
    /// Returns a standard sharp test image (128x128 checkerboard).
    #[must_use]
    pub fn sharp_image() -> ImageInfo {
        Self::checkerboard(128, 128)
    }

    /// Returns a standard blurry test image (128x128 uniform gray).
    #[must_use]
    pub fn blurry_image() -> ImageInfo {
        Self::uniform_gray(128, 128, 128)
    }

    /// Returns a standard underexposed test image (128x128 black).
    #[must_use]
    pub fn underexposed_image() -> ImageInfo {
        Self::underexposed(128, 128)
    }

    /// Returns a standard overexposed test image (128x128 white).
    #[must_use]
    pub fn overexposed_image() -> ImageInfo {
        Self::overexposed(128, 128)
    }

    /// Returns a standard well-exposed test image (128x128 mid-gray).
    #[must_use]
    pub fn well_exposed_image() -> ImageInfo {
        Self::well_exposed(128, 128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkerboard_dimensions() {
        let img = SyntheticImageBuilder::checkerboard(100, 80);
        assert_eq!(img.width, 100);
        assert_eq!(img.height, 80);
        assert_eq!(img.path, "synthetic://checkerboard");
    }

    #[test]
    fn test_checkerboard_pattern() {
        let img = SyntheticImageBuilder::checkerboard_with_cell_size(16, 16, 8);
        let luma = img.to_luma8();

        // Top-left cell (0,0) should be white (255)
        assert_eq!(luma.get_pixel(0, 0).0[0], 255);
        // Next cell should be black (0)
        assert_eq!(luma.get_pixel(8, 0).0[0], 0);
    }

    #[test]
    fn test_uniform_gray() {
        let img = SyntheticImageBuilder::uniform_gray(50, 50, 100);
        let luma = img.to_luma8();

        for pixel in luma.pixels() {
            assert_eq!(pixel.0[0], 100);
        }
    }

    #[test]
    fn test_gradient_range() {
        let img = SyntheticImageBuilder::horizontal_gradient(256, 10);
        let luma = img.to_luma8();

        // First column should be dark
        assert!(luma.get_pixel(0, 0).0[0] < 5);
        // Last column should be bright
        assert!(luma.get_pixel(255, 0).0[0] > 250);
    }

    #[test]
    fn test_exposure_images() {
        let under = SyntheticImageBuilder::underexposed(10, 10);
        let over = SyntheticImageBuilder::overexposed(10, 10);
        let well = SyntheticImageBuilder::well_exposed(10, 10);

        // Underexposed should be all black
        assert!(under.to_luma8().pixels().all(|p| p.0[0] == 0));
        // Overexposed should be all white
        assert!(over.to_luma8().pixels().all(|p| p.0[0] == 255));
        // Well exposed should be mid-gray
        assert!(well.to_luma8().pixels().all(|p| p.0[0] == 128));
    }

    #[test]
    fn test_single_pixel() {
        let img = SyntheticImageBuilder::single_pixel(42);
        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
        assert_eq!(img.to_luma8().get_pixel(0, 0).0[0], 42);
    }

    #[test]
    fn test_tiny_image() {
        let img = SyntheticImageBuilder::tiny([[0, 255], [128, 64]]);
        let luma = img.to_luma8();

        assert_eq!(luma.get_pixel(0, 0).0[0], 0);
        assert_eq!(luma.get_pixel(1, 0).0[0], 255);
        assert_eq!(luma.get_pixel(0, 1).0[0], 128);
        assert_eq!(luma.get_pixel(1, 1).0[0], 64);
    }

    #[test]
    fn test_rgb_image() {
        let img = SyntheticImageBuilder::rgb_uniform(10, 10, 255, 0, 128);
        let rgb = img.to_rgb8();
        let pixel = rgb.get_pixel(5, 5);

        assert_eq!(pixel.0[0], 255); // R
        assert_eq!(pixel.0[1], 0); // G
        assert_eq!(pixel.0[2], 128); // B
    }

    #[test]
    fn test_sharp_center_blurry_edges() {
        let img = SyntheticImageBuilder::sharp_center_blurry_edges(128, 128);
        let luma = img.to_luma8();

        // Center should have high variance (checkerboard)
        let center_val = luma.get_pixel(64, 64).0[0];
        let adjacent_val = luma.get_pixel(68, 64).0[0]; // 4 pixels away

        // These should differ (checkerboard pattern)
        assert_ne!(center_val, adjacent_val);

        // Edge should be uniform gray
        let edge_val = luma.get_pixel(0, 0).0[0];
        assert_eq!(edge_val, 128);
    }

    #[test]
    fn test_convenience_functions() {
        // Just verify they don't panic and have expected dimensions
        let sharp = SyntheticImageBuilder::sharp_image();
        let blurry = SyntheticImageBuilder::blurry_image();

        assert_eq!(sharp.width, 128);
        assert_eq!(blurry.width, 128);
    }
}
