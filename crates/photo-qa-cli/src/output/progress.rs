//! Progress bar adapter using indicatif.

use indicatif::{ProgressBar as IndicatifBar, ProgressStyle};
use photo_qa_core::{ProgressEvent, ProgressSink};

/// Progress bar adapter for CLI output.
#[allow(dead_code)]
pub struct ProgressBar {
    bar: Option<IndicatifBar>,
    quiet: bool,
}

impl ProgressBar {
    /// Creates a new progress bar.
    ///
    /// # Arguments
    ///
    /// * `total` - Total number of items, if known
    /// * `quiet` - If true, suppress all output
    /// * `show_bar` - If true, show progress bar; otherwise show per-item status
    #[allow(dead_code)]
    #[must_use]
    pub fn new(total: Option<u64>, quiet: bool, show_bar: bool) -> Self {
        if quiet {
            return Self {
                bar: None,
                quiet: true,
            };
        }

        let bar = if show_bar {
            let bar = total.map_or_else(IndicatifBar::new_spinner, IndicatifBar::new);

            if let Ok(style) = ProgressStyle::default_bar().template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            ) {
                bar.set_style(style.progress_chars("#>-"));
            }

            Some(bar)
        } else {
            None
        };

        Self { bar, quiet }
    }
}

impl ProgressSink for ProgressBar {
    fn on_event(&self, event: ProgressEvent) {
        if self.quiet {
            return;
        }

        match event {
            ProgressEvent::Started { path, index, total } => {
                if let Some(bar) = &self.bar {
                    if let Some(t) = total {
                        bar.set_length(t as u64);
                    }
                    bar.set_position(index as u64);
                    bar.set_message(path);
                }
            }
            ProgressEvent::Completed { result } => {
                if let Some(bar) = &self.bar {
                    bar.inc(1);
                } else {
                    let issues = result.issues.len();
                    if issues > 0 {
                        eprintln!("{}: {} issue(s)", result.path, issues);
                    }
                }
            }
            ProgressEvent::Skipped { path, reason } => {
                if let Some(bar) = &self.bar {
                    bar.inc(1);
                }
                eprintln!("WARN: Skipping {path}: {reason}");
            }
            ProgressEvent::Finished { processed, skipped } => {
                if let Some(bar) = &self.bar {
                    bar.finish_with_message(format!(
                        "Done: {processed} processed, {skipped} skipped"
                    ));
                }
            }
        }
    }
}
