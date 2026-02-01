//! Photo QA CLI - Automated photo quality assessment tool.

use clap::Parser;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod commands;
mod config;
mod output;

use commands::{Cli, Commands, ExitCode};
use config::AppConfig;

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

    // Load configuration from files (XDG + project-local)
    let config = AppConfig::load();

    let exit_code = match cli.command {
        Some(Commands::Check(args)) => {
            let merged = commands::check::CheckArgs::with_config(*args, &config);
            match commands::check::run(&merged) {
                Ok(result) => result.exit_code,
                Err(e) => {
                    eprintln!("error: {e:#}");
                    ExitCode::Error
                }
            }
        }
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
            let merged = commands::check::CheckArgs::with_config(cli.check, &config);
            match commands::check::run(&merged) {
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
