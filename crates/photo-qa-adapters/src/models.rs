//! Model downloading and caching adapter.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info};

/// Placeholder checksum indicating verification should be skipped.
const PLACEHOLDER_CHECKSUM: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Model metadata.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name/identifier.
    pub name: &'static str,
    /// Download URL (GitHub releases).
    pub url: &'static str,
    /// Expected SHA256 hash. Set to all zeros to skip verification during development.
    pub sha256: &'static str,
    /// Filename in models directory.
    pub filename: &'static str,
}

/// Known models.
pub const MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "retinaface",
        url: "https://github.com/cwygoda/photo-qa/releases/download/models-v1/retinaface.safetensors",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000", // TODO: Update with real hash
        filename: "retinaface.safetensors",
    },
    ModelInfo {
        name: "landmarks68",
        url: "https://github.com/cwygoda/photo-qa/releases/download/models-v1/landmarks68.safetensors",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000", // TODO: Update with real hash
        filename: "landmarks68.safetensors",
    },
    ModelInfo {
        name: "u2net",
        url: "https://github.com/cwygoda/photo-qa/releases/download/models-v1/u2net.safetensors",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000", // TODO: Update with real hash
        filename: "u2net.safetensors",
    },
];

/// Returns the models directory path.
///
/// Uses `XDG_DATA_HOME/photo-qa/models` or `~/.local/share/photo-qa/models`.
#[must_use]
pub fn models_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("photo-qa")
        .join("models")
}

/// Ensures all required models are downloaded.
///
/// # Errors
///
/// Returns an error if:
/// - The models directory cannot be created
/// - A model download fails
/// - A model's checksum doesn't match
pub fn ensure_models() -> Result<()> {
    let dir = models_dir();
    fs::create_dir_all(&dir).context("Failed to create models directory")?;

    for model in MODELS {
        let path = dir.join(model.filename);
        if path.exists() {
            debug!("Model {} already exists", model.name);
        } else {
            download_model(model, &path)?;
        }
    }

    Ok(())
}

/// Downloads a model from its URL.
fn download_model(model: &ModelInfo, path: &PathBuf) -> Result<()> {
    info!("Downloading model: {}", model.name);

    let response = reqwest::blocking::get(model.url)
        .with_context(|| format!("Failed to download {}", model.name))?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed with status: {}", response.status());
    }

    let bytes = response
        .bytes()
        .with_context(|| format!("Failed to read response for {}", model.name))?;

    // Verify checksum (skip if placeholder)
    if model.sha256 == PLACEHOLDER_CHECKSUM {
        debug!(
            "Skipping checksum verification for {} (placeholder checksum)",
            model.name
        );
    } else {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());

        if hash != model.sha256 {
            anyhow::bail!(
                "Checksum mismatch for {}: expected {}, got {}. \
                 Try deleting {} and re-running to download a fresh copy.",
                model.name,
                model.sha256,
                hash,
                path.display()
            );
        }
    }

    fs::write(path, &bytes).with_context(|| format!("Failed to write {}", model.name))?;

    info!("Downloaded {} ({} bytes)", model.name, bytes.len());
    Ok(())
}

/// Returns the path to a specific model file.
#[must_use]
pub fn model_path(name: &str) -> Option<PathBuf> {
    MODELS
        .iter()
        .find(|m| m.name == name)
        .map(|m| models_dir().join(m.filename))
}

/// Checks if all models are installed.
#[must_use]
pub fn all_models_installed() -> bool {
    let dir = models_dir();
    MODELS.iter().all(|m| dir.join(m.filename).exists())
}

/// Lists installed models with their status.
#[must_use]
pub fn list_models() -> Vec<(String, bool)> {
    let dir = models_dir();
    MODELS
        .iter()
        .map(|m| (m.name.to_string(), dir.join(m.filename).exists()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_dir() {
        let dir = models_dir();
        assert!(dir.ends_with("photo-qa/models"));
    }

    #[test]
    fn test_model_path() {
        let path = model_path("retinaface");
        assert!(path.is_some());
        let path = path.unwrap_or_else(|| panic!("should have path"));
        assert!(path.ends_with("retinaface.safetensors"));
    }

    #[test]
    fn test_model_path_unknown() {
        let path = model_path("unknown");
        assert!(path.is_none());
    }
}
