//! Port definitions for hexagonal architecture.
//!
//! These traits define the boundaries between the domain core and external adapters.

mod image_source;
mod progress;
mod result_output;

pub use image_source::ImageSource;
pub use progress::{ProgressEvent, ProgressSink};
pub use result_output::ResultOutput;
