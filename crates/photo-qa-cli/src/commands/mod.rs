//! CLI command definitions and handlers.

pub mod check;
pub mod models;

use clap::{Parser, Subcommand};

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
