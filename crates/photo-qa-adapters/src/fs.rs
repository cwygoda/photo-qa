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
fn load_raw_image(path: &Path) -> Result<image::DynamicImage> {
    use rawloader::decode_file;

    let raw =
        decode_file(path).with_context(|| format!("Failed to decode RAW: {}", path.display()))?;

    // Convert raw data to RGB image
    // This is a simplified implementation - full implementation would do proper demosaicing
    let width = raw.width;
    let height = raw.height;

    // For now, just create a placeholder - real implementation needs proper RAW processing
    #[allow(clippy::cast_possible_truncation)]
    let img = image::DynamicImage::new_rgb8(width as u32, height as u32);

    Ok(img)
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
