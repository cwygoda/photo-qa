//! CLI command definitions and handlers.

pub mod check;
pub mod models;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Photo QA - Automated photo quality assessment
#[derive(Parser)]
#[command(name = "photo-qa")]
#[command(author, version, about, long_about = None)]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// Files or directories to analyze
    #[arg(global = true)]
    pub paths: Vec<PathBuf>,

    /// Recurse into subdirectories
    #[arg(short, long, global = true)]
    pub recursive: bool,

    /// Disable blur detection
    #[arg(long, global = true)]
    pub no_blur: bool,

    /// Disable exposure analysis
    #[arg(long, global = true)]
    pub no_exposure: bool,

    /// Disable closed-eyes detection
    #[arg(long, global = true)]
    pub no_eyes: bool,

    /// Blur score threshold (0.0-1.0)
    #[arg(long, default_value = "0.5", global = true)]
    pub blur_threshold: f32,

    /// Underexposure threshold (0.0-1.0)
    #[arg(long, default_value = "0.3", global = true)]
    pub under_threshold: f32,

    /// Overexposure threshold (0.0-1.0)
    #[arg(long, default_value = "0.3", global = true)]
    pub over_threshold: f32,

    /// Eye aspect ratio threshold (0.0-1.0)
    #[arg(long, default_value = "0.2", global = true)]
    pub ear_threshold: f32,

    /// Include EXIF metadata in output
    #[arg(long, global = true)]
    pub exif: bool,

    /// Show progress bar
    #[arg(long, global = true)]
    pub progress: bool,

    /// Suppress progress output
    #[arg(short, long, global = true)]
    pub quiet: bool,

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
