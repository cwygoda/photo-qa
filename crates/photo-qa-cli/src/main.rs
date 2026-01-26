//! Photo QA CLI - Automated photo quality assessment tool.

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod commands;
mod output;

use commands::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = match cli.verbose {
        0 => EnvFilter::new("warn"),
        1 => EnvFilter::new("info"),
        2 => EnvFilter::new("debug"),
        _ => EnvFilter::new("trace"),
    };

    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .init();

    match cli.command {
        Some(Commands::Check(ref args)) => commands::check::run(args),
        Some(Commands::Models(ref args)) => commands::models::run(args),
        None => {
            // Default behavior: run check with paths from cli
            if cli.paths.is_empty() {
                anyhow::bail!("No paths specified. Use --help for usage information.");
            }
            let args = commands::check::CheckArgs {
                paths: cli.paths,
                recursive: cli.recursive,
                blur: !cli.no_blur,
                exposure: !cli.no_exposure,
                eyes: !cli.no_eyes,
                blur_threshold: cli.blur_threshold,
                under_threshold: cli.under_threshold,
                over_threshold: cli.over_threshold,
                ear_threshold: cli.ear_threshold,
                exif: cli.exif,
                progress: cli.progress,
                quiet: cli.quiet,
            };
            commands::check::run(&args)
        }
    }
}
