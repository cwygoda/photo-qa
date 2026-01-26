//! Photo QA Core - Domain logic and QA modules
//!
//! This crate contains the core domain types, QA module trait, and implementations
//! for blur detection, exposure analysis, and closed-eyes detection.

pub mod domain;
pub mod inference;
pub mod modules;
pub mod ports;

pub use domain::{
    AnalysisResult, BlurDetails, BlurType, ExposureDetails, EyesDetails, FaceResult, ImageInfo,
    Issue, IssueType,
};
pub use ports::{ImageSource, ProgressEvent, ProgressSink, ResultOutput};
