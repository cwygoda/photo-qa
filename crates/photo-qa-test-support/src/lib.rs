//! Test support utilities for photo-qa.
//!
//! Provides mocks, synthetic image builders, and utilities for testing
//! the photo-qa analysis pipeline.
//!
//! # Example
//!
//! ```
//! use photo_qa_test_support::{MockImageSource, SyntheticImageBuilder};
//!
//! // Create synthetic test images
//! let sharp = SyntheticImageBuilder::checkerboard(128, 128);
//! let blurry = SyntheticImageBuilder::uniform_gray(128, 128, 128);
//!
//! // Create mock image source
//! let source = MockImageSource::new(vec![sharp, blurry]);
//! ```

mod builders;
mod mocks;

pub use builders::SyntheticImageBuilder;
pub use mocks::{MockImageSource, MockProgressSink, MockResultOutput};
