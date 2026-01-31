//! Models command - manage ML models.

use anyhow::Result;
use clap::{Args, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use photo_qa_adapters::models::{
    ensure_models_with_progress, list_models as adapter_list_models, models_dir, ProgressCallback,
    MODELS,
};
use std::sync::{Arc, Mutex};

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

fn fetch_models() -> Result<()> {
    let pb = Arc::new(ProgressBar::new(0));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}")
            .map_err(|e| anyhow::anyhow!("Invalid progress template: {e}"))?
            .progress_chars("#>-"),
    );

    let current_model: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let pb_clone = Arc::clone(&pb);
    let model_clone = Arc::clone(&current_model);

    let progress: ProgressCallback =
        Box::new(move |name: &str, downloaded: u64, total: Option<u64>| {
            let is_new_model = {
                let mut current = model_clone
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if *current == name {
                    false
                } else {
                    *current = name.to_string();
                    true
                }
            };
            if is_new_model {
                if let Some(t) = total {
                    pb_clone.set_length(t);
                }
                pb_clone.set_message(name.to_string());
            }
            pb_clone.set_position(downloaded);
        });

    ensure_models_with_progress(Some(&progress))?;

    pb.finish_with_message("All models downloaded");
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn list_models() -> Result<()> {
    let models = adapter_list_models();
    let dir = models_dir();

    println!("Models directory: {}", dir.display());
    println!();

    for (name, installed) in &models {
        let status = if *installed { "✓" } else { "✗" };
        let info = MODELS.iter().find(|m| m.name == name);
        let filename = info.map_or("unknown", |m| m.filename);
        println!("  {status} {name} ({filename})");
    }

    println!();
    let installed_count = models.iter().filter(|(_, installed)| *installed).count();
    println!("{}/{} models installed", installed_count, models.len());

    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn print_path() -> Result<()> {
    let path = models_dir();
    println!("{}", path.display());
    Ok(())
}
