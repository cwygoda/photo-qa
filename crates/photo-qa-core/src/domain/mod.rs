//! Core domain types for photo QA analysis.

mod issue;
mod qa_module;
mod result;

pub use issue::{
    BlurDetails, BlurType, ExposureDetails, ExposureDirection, EyesDetails, FaceResult, Issue,
    IssueDetails, IssueType,
};
pub use qa_module::QaModule;
pub use result::{AnalysisResult, BoundingBox, ImageDimensions, ImageInfo};
