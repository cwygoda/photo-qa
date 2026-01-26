//! Progress reporting port for UI integration.

use crate::domain::AnalysisResult;

/// Events emitted during analysis for progress tracking.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Analysis started for an image.
    Started {
        /// Path to the image.
        path: String,
        /// Index in the batch (0-based).
        index: usize,
        /// Total images in batch, if known.
        total: Option<usize>,
    },
    /// Analysis completed for an image.
    Completed {
        /// The analysis result.
        result: AnalysisResult,
    },
    /// An image was skipped due to an error.
    Skipped {
        /// Path to the image.
        path: String,
        /// Reason for skipping.
        reason: String,
    },
    /// All images have been processed.
    Finished {
        /// Total images processed successfully.
        processed: usize,
        /// Total images skipped.
        skipped: usize,
    },
}

/// Port for receiving progress events.
pub trait ProgressSink: Send + Sync {
    /// Called when a progress event occurs.
    fn on_event(&self, event: ProgressEvent);
}
