//! Result output port for writing analysis results.

use crate::domain::AnalysisResult;

/// Port for outputting analysis results.
pub trait ResultOutput: Send + Sync {
    /// Writes a single analysis result.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    fn write(&self, result: &AnalysisResult) -> anyhow::Result<()>;

    /// Flushes any buffered output.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails.
    fn flush(&self) -> anyhow::Result<()>;
}
