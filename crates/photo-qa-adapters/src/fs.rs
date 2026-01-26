//! Filesystem adapter for loading images.

use anyhow::{Context, Result};
use image::GenericImageView;
use photo_qa_core::{ImageInfo, ImageSource};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// Supported image extensions.
const RASTER_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "tiff", "tif", "webp", "bmp", "gif"];
const RAW_EXTENSIONS: &[&str] = &["cr2", "cr3", "nef", "arw", "raf", "dng", "orf", "rw2"];

/// Filesystem image source adapter.
pub struct FsImageSource {
    paths: Vec<PathBuf>,
    recursive: bool,
}

impl FsImageSource {
    /// Creates a new filesystem image source.
    ///
    /// # Arguments
    ///
    /// * `paths` - Files or directories to scan
    /// * `recursive` - Whether to recurse into subdirectories
    #[must_use]
    pub const fn new(paths: Vec<PathBuf>, recursive: bool) -> Self {
        Self { paths, recursive }
    }

    /// Collects all image files from the configured paths.
    fn collect_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();

        for path in &self.paths {
            if path.is_file() {
                if is_supported_image(path) {
                    files.push(path.clone());
                } else {
                    warn!("Unsupported file type: {}", path.display());
                }
            } else if path.is_dir() {
                self.collect_from_dir(path, &mut files);
            } else {
                warn!("Path does not exist: {}", path.display());
            }
        }

        files
    }

    fn collect_from_dir(&self, dir: &Path, files: &mut Vec<PathBuf>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                warn!("Failed to read directory {}: {e}", dir.display());
                return;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && is_supported_image(&path) {
                files.push(path);
            } else if path.is_dir() && self.recursive {
                self.collect_from_dir(&path, files);
            }
        }
    }
}

impl ImageSource for FsImageSource {
    fn images(&self) -> Box<dyn Iterator<Item = Result<ImageInfo>> + Send + '_> {
        let files = self.collect_files();
        debug!("Found {} image files", files.len());

        Box::new(files.into_iter().map(|path| load_image(&path)))
    }

    fn count_hint(&self) -> Option<usize> {
        Some(self.collect_files().len())
    }
}

/// Checks if a path has a supported image extension.
fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .is_some_and(|e| {
            RASTER_EXTENSIONS.contains(&e.as_str()) || RAW_EXTENSIONS.contains(&e.as_str())
        })
}

/// Loads an image from the filesystem.
fn load_image(path: &Path) -> Result<ImageInfo> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();

    let image = if RAW_EXTENSIONS.contains(&ext.as_str()) {
        load_raw_image(path)?
    } else {
        image::open(path).with_context(|| format!("Failed to open image: {}", path.display()))?
    };

    let (width, height) = image.dimensions();

    Ok(ImageInfo {
        path: path.to_string_lossy().into_owned(),
        width,
        height,
        image,
    })
}

/// Loads a RAW image file.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
fn load_raw_image(path: &Path) -> Result<image::DynamicImage> {
    use rawloader::decode_file;

    let raw =
        decode_file(path).with_context(|| format!("Failed to decode RAW: {}", path.display()))?;

    let width = raw.width;
    let height = raw.height;

    // Extract raw pixel data
    let raw_data: Vec<u16> = match raw.data {
        rawloader::RawImageData::Integer(data) => data,
        rawloader::RawImageData::Float(data) => {
            // Convert float data to u16
            data.into_iter()
                .map(|f| (f.clamp(0.0, 1.0) * 65535.0) as u16)
                .collect()
        }
    };

    // Get black/white levels for normalization
    let black = raw.blacklevels.iter().copied().max().unwrap_or(0) as f32;
    let white = raw.whitelevels.iter().copied().min().unwrap_or(65535) as f32;
    let range = (white - black).max(1.0);

    // Simple bilinear demosaic
    let mut rgb = vec![0u8; width * height * 3];
    let cfa = &raw.cfa;

    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = demosaic_pixel(&raw_data, width, height, x, y, cfa, black, range);
            let idx = (y * width + x) * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    let img = image::RgbImage::from_raw(width as u32, height as u32, rgb)
        .context("Failed to create image from RAW data")?;

    Ok(image::DynamicImage::from(img))
}

/// Demosaic a single pixel using bilinear interpolation.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::too_many_arguments
)]
fn demosaic_pixel(
    data: &[u16],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    cfa: &rawloader::CFA,
    black: f32,
    range: f32,
) -> (u8, u8, u8) {
    let color = cfa.color_at(y, x);

    // Sample neighboring pixels for interpolation
    let get_pixel = |px: usize, py: usize| -> f32 {
        if px < width && py < height {
            let val = data[py * width + px] as f32;
            ((val - black) / range).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };

    let current = get_pixel(x, y);

    // Simple bilinear interpolation based on Bayer pattern position
    let (r, g, b) = match color {
        0 => {
            // Red pixel
            let g = average_neighbors_cross(data, width, height, x, y, black, range);
            let b = average_neighbors_diagonal(data, width, height, x, y, black, range);
            (current, g, b)
        }
        1 => {
            // Green pixel (on red row or blue row)
            let r = average_neighbors_horizontal(data, width, height, x, y, black, range);
            let b = average_neighbors_vertical(data, width, height, x, y, black, range);
            (r, current, b)
        }
        2 => {
            // Blue pixel
            let r = average_neighbors_diagonal(data, width, height, x, y, black, range);
            let g = average_neighbors_cross(data, width, height, x, y, black, range);
            (r, g, current)
        }
        _ => {
            // Green pixel variant (swap r/b interpolation)
            let r = average_neighbors_vertical(data, width, height, x, y, black, range);
            let b = average_neighbors_horizontal(data, width, height, x, y, black, range);
            (r, current, b)
        }
    };

    // Apply simple gamma correction and convert to 8-bit
    let gamma = |v: f32| -> u8 { (v.powf(1.0 / 2.2) * 255.0) as u8 };
    (gamma(r), gamma(g), gamma(b))
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
fn average_neighbors_cross(
    data: &[u16],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    black: f32,
    range: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;

    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx >= 0 && (nx as usize) < width && ny >= 0 && (ny as usize) < height {
            let val = data[ny as usize * width + nx as usize] as f32;
            sum += ((val - black) / range).clamp(0.0, 1.0);
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
fn average_neighbors_diagonal(
    data: &[u16],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    black: f32,
    range: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;

    for (dx, dy) in [(-1i32, -1i32), (1, -1), (-1, 1), (1, 1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx >= 0 && (nx as usize) < width && ny >= 0 && (ny as usize) < height {
            let val = data[ny as usize * width + nx as usize] as f32;
            sum += ((val - black) / range).clamp(0.0, 1.0);
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
fn average_neighbors_horizontal(
    data: &[u16],
    width: usize,
    _height: usize,
    x: usize,
    y: usize,
    black: f32,
    range: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;

    for dx in [-1i32, 1] {
        let nx = x as i32 + dx;
        if nx >= 0 && (nx as usize) < width {
            let val = data[y * width + nx as usize] as f32;
            sum += ((val - black) / range).clamp(0.0, 1.0);
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
fn average_neighbors_vertical(
    data: &[u16],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    black: f32,
    range: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;

    for dy in [-1i32, 1] {
        let ny = y as i32 + dy;
        if ny >= 0 && (ny as usize) < height {
            let val = data[ny as usize * width + x] as f32;
            sum += ((val - black) / range).clamp(0.0, 1.0);
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_supported_image() {
        assert!(is_supported_image(Path::new("test.jpg")));
        assert!(is_supported_image(Path::new("test.JPEG")));
        assert!(is_supported_image(Path::new("test.png")));
        assert!(is_supported_image(Path::new("test.cr2")));
        assert!(is_supported_image(Path::new("test.NEF")));
        assert!(!is_supported_image(Path::new("test.txt")));
        assert!(!is_supported_image(Path::new("test")));
    }
}
