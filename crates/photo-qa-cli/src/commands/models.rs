//! Models command - manage ML models.

use anyhow::Result;
use clap::{Args, Subcommand};
use tracing::info;

/// Arguments for the models command
#[derive(Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: ModelsCommand,
}

/// Models subcommands
#[derive(Subcommand)]
pub enum ModelsCommand {
    /// Download required models
    Fetch,
    /// List installed models
    List,
    /// Print model directory path
    Path,
}

/// Run the models command.
pub fn run(args: &ModelsArgs) -> Result<()> {
    match args.command {
        ModelsCommand::Fetch => fetch_models(),
        ModelsCommand::List => list_models(),
        ModelsCommand::Path => print_path(),
    }
}

#[allow(clippy::unnecessary_wraps)]
fn fetch_models() -> Result<()> {
    info!("Fetching models...");
    // TODO: Implement model download from GitHub releases
    println!("Model download not yet implemented");
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn list_models() -> Result<()> {
    info!("Listing models...");
    // TODO: List installed models with status
    println!("Model listing not yet implemented");
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn print_path() -> Result<()> {
    let path = photo_qa_adapters::models::models_dir();
    println!("{}", path.display());
    Ok(())
}
