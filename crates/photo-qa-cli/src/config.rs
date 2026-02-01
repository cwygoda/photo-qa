//! Configuration file support for photo-qa.
//!
//! Supports TOML configuration from:
//! - XDG config: `~/.config/photo-qa/config.toml` (lowest priority)
//! - Project-local: `.photo-qa.toml` (searched up directory tree)
//! - CLI flags (highest priority, applied separately)

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::{debug, info};

/// Top-level configuration structure.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// General options.
    pub general: GeneralConfig,
    /// Blur detection settings.
    pub blur: BlurConfig,
    /// Exposure analysis settings.
    pub exposure: ExposureConfig,
    /// Closed-eyes detection settings.
    pub eyes: EyesConfig,
    /// Output formatting settings.
    pub output: OutputConfig,
}

/// General configuration options.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    /// Recurse into subdirectories by default.
    pub recursive: Option<bool>,
}

/// Blur detection configuration.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct BlurConfig {
    /// Enable/disable blur detection.
    pub enabled: Option<bool>,
    /// Blur score threshold (0.0-1.0).
    pub threshold: Option<f32>,
    /// Laplacian sharpness threshold.
    pub laplacian_sharp_threshold: Option<f64>,
    /// Sliding window size in pixels.
    pub window_size: Option<u32>,
    /// Window stride in pixels.
    pub window_stride: Option<u32>,
    /// Edge density threshold for subject detection.
    pub edge_density_threshold: Option<f64>,
}

/// Exposure analysis configuration.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct ExposureConfig {
    /// Enable/disable exposure analysis.
    pub enabled: Option<bool>,
    /// Underexposure threshold (0.0-1.0).
    pub under_threshold: Option<f32>,
    /// Overexposure threshold (0.0-1.0).
    pub over_threshold: Option<f32>,
    /// Shadow clip level (0-255).
    pub shadow_clip_level: Option<u8>,
    /// Highlight clip level (0-255).
    pub highlight_clip_level: Option<u8>,
}

/// Closed-eyes detection configuration.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct EyesConfig {
    /// Enable/disable closed-eyes detection.
    pub enabled: Option<bool>,
    /// Eye aspect ratio threshold.
    pub ear_threshold: Option<f32>,
    /// Minimum face detection confidence.
    pub min_face_confidence: Option<f32>,
}

/// Output formatting configuration.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Output format: "json" or "jsonl".
    pub format: Option<String>,
    /// Pretty-print JSON output.
    pub pretty: Option<bool>,
    /// Include EXIF metadata.
    pub exif: Option<bool>,
    /// Show progress bar.
    pub progress: Option<bool>,
}

impl AppConfig {
    /// Load configuration from XDG and project-local files.
    ///
    /// Priority (lowest to highest):
    /// 1. XDG config: `~/.config/photo-qa/config.toml`
    /// 2. Project-local: `.photo-qa.toml` (searched up from cwd)
    ///
    /// Missing files are silently ignored. Invalid values are logged as warnings.
    pub fn load() -> Self {
        let mut config = Self::default();

        // Load XDG config (lowest priority)
        if let Some(xdg_path) = xdg_config_path() {
            if xdg_path.exists() {
                info!("Loading XDG config: {}", xdg_path.display());
                if let Some(xdg_config) = load_file(&xdg_path) {
                    config = xdg_config;
                }
            } else {
                debug!("XDG config not found: {}", xdg_path.display());
            }
        }

        // Load project-local config (higher priority, merged)
        if let Some(project_path) = find_project_config() {
            info!("Loading project config: {}", project_path.display());
            if let Some(project_config) = load_file(&project_path) {
                config.merge(project_config);
            }
        }

        // Validate merged config
        if let Err(e) = config.validate() {
            tracing::warn!("Config validation: {e}");
        }

        config
    }

    /// Validate configuration values are within acceptable ranges.
    fn validate(&self) -> Result<(), String> {
        // Threshold validations (0.0-1.0 range)
        if let Some(t) = self.blur.threshold {
            if !(0.0..=1.0).contains(&t) {
                return Err(format!("blur.threshold must be 0.0-1.0, got {t}"));
            }
        }
        if let Some(t) = self.exposure.under_threshold {
            if !(0.0..=1.0).contains(&t) {
                return Err(format!("exposure.under_threshold must be 0.0-1.0, got {t}"));
            }
        }
        if let Some(t) = self.exposure.over_threshold {
            if !(0.0..=1.0).contains(&t) {
                return Err(format!("exposure.over_threshold must be 0.0-1.0, got {t}"));
            }
        }
        if let Some(t) = self.eyes.ear_threshold {
            if !(0.0..=1.0).contains(&t) {
                return Err(format!("eyes.ear_threshold must be 0.0-1.0, got {t}"));
            }
        }
        if let Some(t) = self.eyes.min_face_confidence {
            if !(0.0..=1.0).contains(&t) {
                return Err(format!("eyes.min_face_confidence must be 0.0-1.0, got {t}"));
            }
        }

        // Output format validation
        if let Some(ref f) = self.output.format {
            if f != "json" && f != "jsonl" {
                return Err(format!(
                    "output.format must be 'json' or 'jsonl', got '{f}'"
                ));
            }
        }

        Ok(())
    }

    /// Merge another config into this one.
    /// Values from `other` override values in `self` when present.
    fn merge(&mut self, other: Self) {
        // General
        self.general.recursive = other.general.recursive.or(self.general.recursive);

        // Blur
        self.blur.enabled = other.blur.enabled.or(self.blur.enabled);
        self.blur.threshold = other.blur.threshold.or(self.blur.threshold);
        self.blur.laplacian_sharp_threshold = other
            .blur
            .laplacian_sharp_threshold
            .or(self.blur.laplacian_sharp_threshold);
        self.blur.window_size = other.blur.window_size.or(self.blur.window_size);
        self.blur.window_stride = other.blur.window_stride.or(self.blur.window_stride);
        self.blur.edge_density_threshold = other
            .blur
            .edge_density_threshold
            .or(self.blur.edge_density_threshold);

        // Exposure
        self.exposure.enabled = other.exposure.enabled.or(self.exposure.enabled);
        self.exposure.under_threshold = other
            .exposure
            .under_threshold
            .or(self.exposure.under_threshold);
        self.exposure.over_threshold = other
            .exposure
            .over_threshold
            .or(self.exposure.over_threshold);
        self.exposure.shadow_clip_level = other
            .exposure
            .shadow_clip_level
            .or(self.exposure.shadow_clip_level);
        self.exposure.highlight_clip_level = other
            .exposure
            .highlight_clip_level
            .or(self.exposure.highlight_clip_level);

        // Eyes
        self.eyes.enabled = other.eyes.enabled.or(self.eyes.enabled);
        self.eyes.ear_threshold = other.eyes.ear_threshold.or(self.eyes.ear_threshold);
        self.eyes.min_face_confidence = other
            .eyes
            .min_face_confidence
            .or(self.eyes.min_face_confidence);

        // Output
        self.output.format = other.output.format.or_else(|| self.output.format.take());
        self.output.pretty = other.output.pretty.or(self.output.pretty);
        self.output.exif = other.output.exif.or(self.output.exif);
        self.output.progress = other.output.progress.or(self.output.progress);
    }
}

/// Get the XDG config file path.
fn xdg_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("photo-qa").join("config.toml"))
}

/// Find project-local config by searching up from current directory.
fn find_project_config() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    find_config_in_parents(&cwd)
}

/// Search for `.photo-qa.toml` in the given directory and its parents.
fn find_config_in_parents(start: &Path) -> Option<PathBuf> {
    let mut current = Some(start);

    while let Some(dir) = current {
        let config_path = dir.join(".photo-qa.toml");
        if config_path.exists() {
            return Some(config_path);
        }
        current = dir.parent();
    }

    None
}

/// Load and parse a TOML config file.
fn load_file(path: &Path) -> Option<AppConfig> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("Failed to read config file {}: {}", path.display(), e);
            return None;
        }
    };

    match toml::from_str(&content) {
        Ok(config) => Some(config),
        Err(e) => {
            tracing::warn!("Failed to parse config file {}: {}", path.display(), e);
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert!(config.blur.threshold.is_none());
        assert!(config.exposure.under_threshold.is_none());
        assert!(config.eyes.ear_threshold.is_none());
    }

    #[test]
    fn test_parse_minimal_config() {
        let toml = "";
        let config: AppConfig = toml::from_str(toml).expect("parse empty config");
        assert!(config.blur.enabled.is_none());
    }

    #[test]
    fn test_parse_blur_section() {
        let toml = r"
[blur]
enabled = true
threshold = 0.6
";
        let config: AppConfig = toml::from_str(toml).expect("parse blur config");
        assert_eq!(config.blur.enabled, Some(true));
        assert_eq!(config.blur.threshold, Some(0.6));
    }

    #[test]
    fn test_parse_full_config() {
        let toml = r"
[general]
recursive = true

[blur]
enabled = true
threshold = 0.6
laplacian_sharp_threshold = 120.0
window_size = 128
window_stride = 64
edge_density_threshold = 0.2

[exposure]
enabled = true
under_threshold = 0.25
over_threshold = 0.35
shadow_clip_level = 10
highlight_clip_level = 245

[eyes]
enabled = false
ear_threshold = 0.15
min_face_confidence = 0.8

[output]
format = 'json'
pretty = true
exif = true
progress = false
";
        let config: AppConfig = toml::from_str(toml).expect("parse full config");

        assert_eq!(config.general.recursive, Some(true));
        assert_eq!(config.blur.threshold, Some(0.6));
        assert_eq!(config.blur.window_size, Some(128));
        assert_eq!(config.exposure.under_threshold, Some(0.25));
        assert_eq!(config.eyes.enabled, Some(false));
        assert_eq!(config.eyes.ear_threshold, Some(0.15));
        assert_eq!(config.output.format, Some("json".to_string()));
        assert_eq!(config.output.pretty, Some(true));
    }

    #[test]
    fn test_merge_configs() {
        let mut base: AppConfig = toml::from_str(
            r"
[blur]
threshold = 0.5

[exposure]
under_threshold = 0.3
",
        )
        .expect("parse base");

        let override_config: AppConfig = toml::from_str(
            r"
[blur]
threshold = 0.7

[eyes]
ear_threshold = 0.15
",
        )
        .expect("parse override");

        base.merge(override_config);

        // Blur threshold overridden
        assert_eq!(base.blur.threshold, Some(0.7));
        // Exposure preserved from base
        assert_eq!(base.exposure.under_threshold, Some(0.3));
        // Eyes added from override
        assert_eq!(base.eyes.ear_threshold, Some(0.15));
    }
}
