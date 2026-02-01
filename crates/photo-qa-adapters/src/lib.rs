//! Photo QA Adapters - External adapters for photo-qa.
//!
//! This crate provides adapters for:
//! - Filesystem image source
//! - Model downloading and caching
//! - Output formatting

pub mod fs;
pub mod models;

pub use fs::FsImageSource;
pub use models::{model_path, models_dir};
