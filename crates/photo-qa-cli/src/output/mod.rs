//! Output formatting for CLI.

mod json;
mod progress;

// These are exposed for future use when the check command is implemented
#[allow(unused_imports)]
pub use json::JsonOutput;
#[allow(unused_imports)]
pub use progress::ProgressBar;
