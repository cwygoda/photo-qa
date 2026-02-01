//! CLI command definitions and handlers.

pub mod check;
pub mod models;

use clap::{Parser, Subcommand};

/// Exit codes for CLI commands.
#[derive(Debug, Clone, Copy)]
pub enum ExitCode {
    /// Success, no issues found.
    Success = 0,
    /// Success, but issues were found.
    IssuesFound = 1,
    /// Runtime error.
    Error = 2,
}

impl From<ExitCode> for std::process::ExitCode {
    fn from(code: ExitCode) -> Self {
        Self::from(code as u8)
    }
}

/// Photo QA - Automated photo quality assessment
#[derive(Parser)]
#[command(name = "photo-qa")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Shared check arguments (paths, thresholds, flags).
    #[command(flatten)]
    pub check: check::CheckArgs,

    /// Increase verbosity (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// Available subcommands
#[derive(Subcommand)]
pub enum Commands {
    /// Analyze images for quality issues
    Check(check::CheckArgs),
    /// Manage ML models
    Models(models::ModelsArgs),
}
