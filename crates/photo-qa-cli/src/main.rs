//! Photo QA CLI - Automated photo quality assessment tool.

use clap::Parser;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod commands;
mod output;

use commands::{Cli, Commands, ExitCode};

fn main() -> std::process::ExitCode {
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

    let exit_code = match cli.command {
        Some(Commands::Check(ref args)) => match commands::check::run(args) {
            Ok(result) => result.exit_code,
            Err(e) => {
                eprintln!("error: {e:#}");
                ExitCode::Error
            }
        },
        Some(Commands::Models(ref args)) => match commands::models::run(args) {
            Ok(()) => ExitCode::Success,
            Err(e) => {
                eprintln!("error: {e:#}");
                ExitCode::Error
            }
        },
        None => {
            // Default behavior: run check with flattened args
            if cli.check.paths.is_empty() {
                eprintln!("error: No paths specified. Use --help for usage information.");
                return ExitCode::Error.into();
            }
            match commands::check::run(&cli.check) {
                Ok(result) => result.exit_code,
                Err(e) => {
                    eprintln!("error: {e:#}");
                    ExitCode::Error
                }
            }
        }
    };

    exit_code.into()
}
