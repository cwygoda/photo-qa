//! JSON output adapter.

use anyhow::Result;
use photo_qa_core::{AnalysisResult, ResultOutput};
use std::io::{self, Write};
use std::sync::Mutex;

/// JSON Lines output adapter.
#[allow(dead_code)]
pub struct JsonOutput {
    writer: Mutex<Box<dyn Write + Send>>,
}

impl JsonOutput {
    /// Creates a new JSON output writing to stdout.
    #[allow(dead_code)]
    #[must_use]
    pub fn stdout() -> Self {
        Self {
            writer: Mutex::new(Box::new(io::stdout())),
        }
    }

    /// Creates a new JSON output writing to the given writer.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(writer: Box<dyn Write + Send>) -> Self {
        Self {
            writer: Mutex::new(writer),
        }
    }
}

impl ResultOutput for JsonOutput {
    #[allow(clippy::significant_drop_tightening)]
    fn write(&self, result: &AnalysisResult) -> Result<()> {
        let json = serde_json::to_string(result)?;
        let mut writer = self
            .writer
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))?;
        writeln!(writer, "{json}")?;
        Ok(())
    }

    #[allow(clippy::significant_drop_tightening)]
    fn flush(&self) -> Result<()> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))?;
        writer.flush()?;
        Ok(())
    }
}
