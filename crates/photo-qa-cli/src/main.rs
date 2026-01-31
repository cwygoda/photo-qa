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
            // Default behavior: run check with flattened args
            if cli.check.paths.is_empty() {
                anyhow::bail!("No paths specified. Use --help for usage information.");
            }
            commands::check::run(&cli.check)
        }
    }
}
