//! Image source port for loading images from various sources.

use crate::domain::ImageInfo;

/// Port for loading images from a source.
pub trait ImageSource: Send + Sync {
    /// Returns an iterator over images from this source.
    ///
    /// # Errors
    ///
    /// Individual items may be errors if an image fails to load.
    fn images(&self) -> Box<dyn Iterator<Item = anyhow::Result<ImageInfo>> + Send + '_>;

    /// Returns the total number of images, if known.
    fn count_hint(&self) -> Option<usize>;
}
