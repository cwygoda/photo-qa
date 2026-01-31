//! Photo QA Core - Domain logic and QA modules
//!
//! This crate contains the core domain types, QA module trait, and implementations
//! for blur detection, exposure analysis, and closed-eyes detection.

pub mod domain;
pub mod inference;
pub mod modules;
pub mod ports;

pub use domain::{
    AnalysisResult, BlurDetails, BlurType, BoundingBox, ExposureDetails, ExposureDirection,
    EyesDetails, FaceResult, ImageDimensions, ImageInfo, Issue, IssueDetails, IssueType, QaModule,
};
pub use modules::{BlurConfig, BlurModule, ExposureConfig, ExposureModule, EyesConfig, EyesModule};
pub use ports::{ImageSource, ProgressEvent, ProgressSink, ResultOutput};
