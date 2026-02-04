//! Model downloading and caching adapter.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use tracing::{debug, info};

/// Placeholder checksum indicating verification should be skipped.
const PLACEHOLDER_CHECKSUM: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Progress callback for download operations.
///
/// Called with `(model_name, downloaded_bytes, total_bytes)`.
/// `total_bytes` is `None` if the server doesn't provide `Content-Length`.
pub type ProgressCallback = Box<dyn Fn(&str, u64, Option<u64>) + Send>;

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
        name: "blazeface",
        url:
            "https://github.com/cwygoda/photo-qa/releases/download/models-v1/blazeface.safetensors",
        // Pretrained from hollance/BlazeFace-PyTorch (MediaPipe weights)
        sha256: "ef7fa16472e60e0503d5b6e4d8f06d08bbc2038e19109de35f93c9c3361e71ab",
        filename: "blazeface.safetensors",
    },
    ModelInfo {
        name: "eye_state",
        url:
            "https://github.com/cwygoda/photo-qa/releases/download/models-v1/eye_state.safetensors",
        sha256: "366028c136659dcd11a7c3f2b14bf6e7f27717c10b2a627744afd646ef620856",
        filename: "eye_state.safetensors",
    },
    ModelInfo {
        name: "u2net",
        url: "https://github.com/cwygoda/photo-qa/releases/download/models-v1/u2net.safetensors",
        sha256: "ef544aa6404ffd064c270bdf296075d448f30a3b0734961045b7a4c9ae27a244",
        filename: "u2net.safetensors",
    },
];

// Thread-local override for models directory.
std::thread_local! {
    static MODELS_DIR_OVERRIDE: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

/// Set a custom models directory override.
///
/// This affects all subsequent calls to `models_dir()` and `model_path()` in the current thread.
pub fn set_models_dir(path: Option<PathBuf>) {
    MODELS_DIR_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = path;
    });
}

/// Returns the models directory path.
///
/// Priority:
/// 1. Override set via `set_models_dir()` or `PHOTO_QA_MODELS_DIR` env var
/// 2. `XDG_DATA_HOME/photo-qa/models` or `~/.local/share/photo-qa/models`
#[must_use]
pub fn models_dir() -> PathBuf {
    // Check thread-local override first
    let override_path = MODELS_DIR_OVERRIDE.with(|cell| cell.borrow().clone());
    if let Some(path) = override_path {
        return path;
    }

    // Check environment variable
    if let Ok(env_path) = std::env::var("PHOTO_QA_MODELS_DIR") {
        return PathBuf::from(env_path);
    }

    // Default to XDG data dir
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
    ensure_models_with_progress(None)
}

/// Ensures all required models are downloaded, with progress reporting.
///
/// # Arguments
///
/// * `progress` - Optional callback for download progress updates
///
/// # Errors
///
/// Returns an error if:
/// - The models directory cannot be created
/// - A model download fails
/// - A model's checksum doesn't match
pub fn ensure_models_with_progress(progress: Option<&ProgressCallback>) -> Result<()> {
    let dir = models_dir();
    fs::create_dir_all(&dir).context("Failed to create models directory")?;

    for model in MODELS {
        let path = dir.join(model.filename);
        if path.exists() {
            debug!("Model {} already exists", model.name);
        } else {
            download_model(model, &path, progress)?;
        }
    }

    Ok(())
}

/// Downloads a model from its URL.
fn download_model(
    model: &ModelInfo,
    path: &PathBuf,
    progress: Option<&ProgressCallback>,
) -> Result<()> {
    info!("Downloading model: {}", model.name);

    let response = reqwest::blocking::get(model.url)
        .with_context(|| format!("Failed to download {}", model.name))?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed with status: {}", response.status());
    }

    let total_size = response.content_length();
    let mut downloaded: u64 = 0;
    let mut hasher = Sha256::new();

    // Create temporary file for download
    let temp_path = path.with_extension("tmp");
    let mut file =
        File::create(&temp_path).with_context(|| format!("Failed to create {}", model.name))?;

    // Stream download with progress
    let mut reader = response;
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = std::io::Read::read(&mut reader, &mut buffer)
            .with_context(|| format!("Failed to read response for {}", model.name))?;

        if bytes_read == 0 {
            break;
        }

        hasher.update(&buffer[..bytes_read]);
        file.write_all(&buffer[..bytes_read])
            .with_context(|| format!("Failed to write {}", model.name))?;

        downloaded += bytes_read as u64;

        if let Some(cb) = progress {
            cb(model.name, downloaded, total_size);
        }
    }

    file.flush()
        .with_context(|| format!("Failed to flush {}", model.name))?;
    drop(file);

    // Verify checksum (skip if placeholder)
    if model.sha256 == PLACEHOLDER_CHECKSUM {
        debug!(
            "Skipping checksum verification for {} (placeholder checksum)",
            model.name
        );
    } else {
        let hash = format!("{:x}", hasher.finalize());

        if hash != model.sha256 {
            // Remove temp file on checksum failure
            let _ = fs::remove_file(&temp_path);
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

    // Rename temp file to final path
    fs::rename(&temp_path, path).with_context(|| format!("Failed to rename {}", model.name))?;

    info!("Downloaded {} ({} bytes)", model.name, downloaded);
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
        let path = model_path("blazeface");
        assert!(path.is_some());
        let path = path.unwrap_or_else(|| panic!("should have path"));
        assert!(path.ends_with("blazeface.safetensors"));
    }

    #[test]
    fn test_model_path_unknown() {
        let path = model_path("unknown");
        assert!(path.is_none());
    }
}
