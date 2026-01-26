//! QA module trait for extensible quality checks.

use super::{ImageInfo, Issue};

/// Trait for implementing quality assessment modules.
///
/// Each module analyzes an image for a specific type of quality issue
/// and returns detected issues with severity scores.
pub trait QaModule: Send + Sync {
    /// Returns the name of this QA module.
    fn name(&self) -> &'static str;

    /// Analyzes an image and returns any detected issues.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to analyze
    ///
    /// # Returns
    ///
    /// A vector of detected issues, empty if no issues found.
    ///
    /// # Errors
    ///
    /// Returns an error if analysis fails.
    fn analyze(&self, image: &ImageInfo) -> anyhow::Result<Vec<Issue>>;
}
