//! QA module implementations.
//!
//! Each module implements the `QaModule` trait for a specific type of quality check.

mod blur;
mod exposure;
mod eyes;

pub use blur::BlurModule;
pub use exposure::ExposureModule;
pub use eyes::EyesModule;
