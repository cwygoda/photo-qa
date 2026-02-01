//! Check command - analyze images for quality issues.

use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, ValueEnum};
use photo_qa_adapters::{model_path, set_models_dir, FsImageSource};
use photo_qa_core::{
    AnalysisResult, BlurConfig, BlurModule, ExposureConfig, ExposureModule, EyesConfig, EyesModule,
    ImageDimensions, ImageSource, ProgressEvent, QaModule, ResultOutput,
};
use tracing::{debug, info, warn};

use super::ExitCode;
use crate::config::AppConfig;
use crate::output::{JsonOutput, ProgressBar};

/// Output format for results.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum OutputFormat {
    /// JSON Lines (one JSON object per line)
    #[default]
    Jsonl,
    /// Single JSON array
    Json,
}

/// Hardcoded default values for thresholds.
mod defaults {
    pub const BLUR_THRESHOLD: f32 = 0.5;
    pub const UNDER_THRESHOLD: f32 = 0.3;
    pub const OVER_THRESHOLD: f32 = 0.3;
    pub const EAR_THRESHOLD: f32 = 0.2;
}

/// Parse and validate a threshold value (0.0-1.0).
fn parse_threshold(s: &str) -> Result<f32, String> {
    let value: f32 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(format!("{value} is not in 0.0..=1.0"))
    }
}

/// Shared arguments for image analysis.
#[derive(Args, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct CheckArgs {
    /// Files or directories to analyze
    pub paths: Vec<PathBuf>,

    /// Recurse into subdirectories
    #[arg(short, long)]
    pub recursive: bool,

    /// Disable blur detection
    #[arg(long)]
    pub no_blur: bool,

    /// Disable exposure analysis
    #[arg(long)]
    pub no_exposure: bool,

    /// Disable closed-eyes detection
    #[arg(long)]
    pub no_eyes: bool,

    /// Blur score threshold (0.0-1.0)
    #[arg(long, value_parser = parse_threshold)]
    pub blur_threshold: Option<f32>,

    /// Underexposure threshold (0.0-1.0)
    #[arg(long, value_parser = parse_threshold)]
    pub under_threshold: Option<f32>,

    /// Overexposure threshold (0.0-1.0)
    #[arg(long, value_parser = parse_threshold)]
    pub over_threshold: Option<f32>,

    /// Eye aspect ratio threshold (0.0-1.0)
    #[arg(long, value_parser = parse_threshold)]
    pub ear_threshold: Option<f32>,

    /// Include EXIF metadata in output
    #[arg(long)]
    pub exif: bool,

    /// Show progress bar
    #[arg(long)]
    pub progress: bool,

    /// Suppress progress output
    #[arg(short, long)]
    pub quiet: bool,

    /// Output format
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,

    /// Pretty-print JSON output (only affects --format json)
    #[arg(long)]
    pub pretty: bool,

    /// Custom models directory (overrides default and config)
    #[arg(long, value_name = "DIR")]
    pub models_dir: Option<PathBuf>,

    /// Merged config (populated by `with_config`, not from CLI).
    #[arg(skip)]
    config: Option<AppConfig>,
}

impl CheckArgs {
    /// Apply configuration file values, respecting CLI precedence.
    ///
    /// Layering priority (lowest to highest):
    /// 1. Hardcoded defaults (in accessor methods)
    /// 2. Config file values (XDG, then project-local)
    /// 3. CLI arguments (already set on self)
    ///
    /// For boolean flags: CLI `--no-*` always wins. Config can enable/disable
    /// only when CLI flag wasn't explicitly set.
    pub fn with_config(mut args: Self, config: &AppConfig) -> Self {
        // Recursive: config applies only if CLI --recursive not passed
        if !args.recursive {
            args.recursive = config.general.recursive.unwrap_or(false);
        }

        // Module enables: CLI --no-* takes precedence, then config, then default (enabled)
        // If CLI passed --no-blur, args.no_blur=true and we skip config
        // If CLI didn't pass it, apply config.blur.enabled (inverted to no_blur)
        if !args.no_blur {
            if let Some(enabled) = config.blur.enabled {
                args.no_blur = !enabled;
            }
        }
        if !args.no_exposure {
            if let Some(enabled) = config.exposure.enabled {
                args.no_exposure = !enabled;
            }
        }
        if !args.no_eyes {
            if let Some(enabled) = config.eyes.enabled {
                args.no_eyes = !enabled;
            }
        }

        // Thresholds: CLI > config (accessor provides hardcoded fallback)
        args.blur_threshold = args.blur_threshold.or(config.blur.threshold);
        args.under_threshold = args.under_threshold.or(config.exposure.under_threshold);
        args.over_threshold = args.over_threshold.or(config.exposure.over_threshold);
        args.ear_threshold = args.ear_threshold.or(config.eyes.ear_threshold);

        // Output format: CLI > config (accessor provides fallback)
        if args.format.is_none() {
            args.format = config
                .output
                .format
                .as_ref()
                .and_then(|s| match s.as_str() {
                    "json" => Some(OutputFormat::Json),
                    "jsonl" => Some(OutputFormat::Jsonl),
                    _ => None,
                });
        }

        // Boolean output options: CLI flag wins, then config
        if !args.pretty {
            args.pretty = config.output.pretty.unwrap_or(false);
        }
        if !args.exif {
            args.exif = config.output.exif.unwrap_or(false);
        }
        if !args.progress {
            args.progress = config.output.progress.unwrap_or(false);
        }

        // Models directory: CLI > config
        if args.models_dir.is_none() {
            args.models_dir.clone_from(&config.models.dir);
        }

        // Store config for build_modules to access advanced settings
        args.config = Some(config.clone());

        args
    }

    /// Get blur threshold with fallback to hardcoded default.
    fn blur_threshold(&self) -> f32 {
        self.blur_threshold.unwrap_or(defaults::BLUR_THRESHOLD)
    }

    /// Get under-exposure threshold with fallback to hardcoded default.
    fn under_threshold(&self) -> f32 {
        self.under_threshold.unwrap_or(defaults::UNDER_THRESHOLD)
    }

    /// Get over-exposure threshold with fallback to hardcoded default.
    fn over_threshold(&self) -> f32 {
        self.over_threshold.unwrap_or(defaults::OVER_THRESHOLD)
    }

    /// Get EAR threshold with fallback to hardcoded default.
    fn ear_threshold(&self) -> f32 {
        self.ear_threshold.unwrap_or(defaults::EAR_THRESHOLD)
    }

    /// Get output format with fallback to JSONL.
    fn format(&self) -> OutputFormat {
        self.format.unwrap_or(OutputFormat::Jsonl)
    }
}

/// Result of running the check command.
#[allow(dead_code)] // Fields exposed for programmatic use
pub struct CheckResult {
    /// Number of images processed.
    pub processed: usize,
    /// Number of images skipped.
    pub skipped: usize,
    /// Number of images with issues.
    pub with_issues: usize,
    /// Exit code.
    pub exit_code: ExitCode,
}

/// Run the check command.
///
/// Expects `args` to have been processed through `with_config()` first
/// to apply configuration file settings.
pub fn run(args: &CheckArgs) -> Result<CheckResult> {
    info!("Running check command on {} paths", args.paths.len());

    if args.paths.is_empty() {
        anyhow::bail!("No paths specified");
    }

    // Apply models directory override if specified
    if let Some(ref models_dir) = args.models_dir {
        debug!("Using custom models directory: {}", models_dir.display());
        set_models_dir(Some(models_dir.clone()));
    }

    // Initialize image source
    let source = FsImageSource::new(args.paths.clone(), args.recursive);
    let total = source.count_hint();

    // Determine if we should show progress
    let show_progress = !args.quiet && (args.progress || std::io::stderr().is_terminal());

    // Initialize progress bar
    let progress_bar = ProgressBar::new(total.map(|t| t as u64), args.quiet, show_progress);

    // Initialize output adapter
    let output = JsonOutput::stdout();

    // Build QA modules based on args (which includes merged config)
    let modules = build_modules(args);

    if modules.is_empty() {
        warn!("All QA modules disabled, nothing to check");
        return Ok(CheckResult {
            processed: 0,
            skipped: 0,
            with_issues: 0,
            exit_code: ExitCode::Success,
        });
    }

    // Process images
    process_images(&source, &modules, &output, &progress_bar, args)
}

/// Build QA modules based on merged args (CLI + config).
fn build_modules(args: &CheckArgs) -> Vec<Box<dyn QaModule>> {
    let mut modules: Vec<Box<dyn QaModule>> = Vec::new();
    let config = args.config.as_ref();

    // Exposure module (no ML, always available)
    if !args.no_exposure {
        let module_config = ExposureConfig {
            under_threshold: args.under_threshold(),
            over_threshold: args.over_threshold(),
            shadow_clip_level: config
                .and_then(|c| c.exposure.shadow_clip_level)
                .unwrap_or(8),
            highlight_clip_level: config
                .and_then(|c| c.exposure.highlight_clip_level)
                .unwrap_or(247),
        };
        modules.push(Box::new(ExposureModule::new(module_config)));
        debug!("Enabled exposure module");
    }

    // Blur module (no ML)
    if !args.no_blur {
        let module_config = BlurConfig {
            threshold: args.blur_threshold(),
            laplacian_sharp_threshold: config
                .and_then(|c| c.blur.laplacian_sharp_threshold)
                .unwrap_or(100.0),
            window_size: config.and_then(|c| c.blur.window_size).unwrap_or(64),
            window_stride: config.and_then(|c| c.blur.window_stride).unwrap_or(32),
            edge_density_threshold: config
                .and_then(|c| c.blur.edge_density_threshold)
                .unwrap_or(0.15),
        };
        modules.push(Box::new(BlurModule::new(module_config)));
        debug!("Enabled blur module");
    }

    // Eyes module (requires ML models)
    if !args.no_eyes {
        let blazeface_path = model_path("blazeface");
        let eye_state_path = model_path("eye_state");

        match (blazeface_path, eye_state_path) {
            (Some(bf), Some(es)) => {
                if !bf.exists() {
                    info!(
                        "Eyes module disabled: {} not found. Run `photo-qa models fetch`.",
                        bf.display()
                    );
                } else if !es.exists() {
                    info!(
                        "Eyes module disabled: {} not found. Run `photo-qa models fetch`.",
                        es.display()
                    );
                } else {
                    let module_config = EyesConfig {
                        ear_threshold: args.ear_threshold(),
                        min_face_confidence: config
                            .and_then(|c| c.eyes.min_face_confidence)
                            .unwrap_or(0.75),
                        blazeface_model_path: Some(bf),
                        eye_state_model_path: Some(es),
                    };
                    modules.push(Box::new(EyesModule::new(module_config)));
                    debug!("Enabled eyes module");
                }
            }
            (None, _) | (_, None) => {
                info!("Eyes module disabled: unknown model configuration.");
            }
        }
    }

    modules
}

/// Process images through QA modules.
fn process_images(
    source: &FsImageSource,
    modules: &[Box<dyn QaModule>],
    output: &JsonOutput,
    progress: &ProgressBar,
    args: &CheckArgs,
) -> Result<CheckResult> {
    use photo_qa_core::ProgressSink;

    let total = source.count_hint();
    let mut processed = 0usize;
    let mut skipped = 0usize;
    let mut with_issues = 0usize;
    let mut all_results: Vec<AnalysisResult> = Vec::new();

    for (index, image_result) in source.images().enumerate() {
        let image = match image_result {
            Ok(img) => img,
            Err(e) => {
                // Note: error message contains the path via anyhow context
                progress.on_event(ProgressEvent::Skipped {
                    path: format!("image {index}"),
                    reason: e.to_string(),
                });
                skipped += 1;
                continue;
            }
        };

        let path = image.path.clone();

        progress.on_event(ProgressEvent::Started {
            path: path.clone(),
            index,
            total,
        });

        // Run all modules
        let mut issues = Vec::new();
        for module in modules {
            match module.analyze(&image) {
                Ok(mut module_issues) => {
                    issues.append(&mut module_issues);
                }
                Err(e) => {
                    warn!("Module {} failed for {}: {}", module.name(), path, e);
                }
            }
        }

        // Extract EXIF if requested
        let exif = if args.exif { extract_exif(&path) } else { None };

        // Track issues before moving
        let has_issues = !issues.is_empty();

        // Build result
        let result = AnalysisResult {
            path,
            timestamp: iso_timestamp(),
            dimensions: ImageDimensions::new(image.width, image.height),
            issues,
            exif,
        };

        if has_issues {
            with_issues += 1;
        }

        progress.on_event(ProgressEvent::Completed {
            result: result.clone(),
        });

        // Output based on format
        match args.format() {
            OutputFormat::Jsonl => {
                output.write(&result)?;
            }
            OutputFormat::Json => {
                all_results.push(result);
            }
        }

        processed += 1;
    }

    // For JSON format, output all results as array via adapter
    if matches!(args.format(), OutputFormat::Json) {
        output.write_array(&all_results, args.pretty)?;
    }

    output.flush()?;

    progress.on_event(ProgressEvent::Finished { processed, skipped });

    // Determine exit code
    let exit_code = if with_issues > 0 {
        ExitCode::IssuesFound
    } else {
        ExitCode::Success
    };

    Ok(CheckResult {
        processed,
        skipped,
        with_issues,
        exit_code,
    })
}

/// Extract EXIF metadata from an image file.
fn extract_exif(path: &str) -> Option<HashMap<String, String>> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let exif = exif::Reader::new().read_from_container(&mut reader).ok()?;

    let mut map = HashMap::new();
    for field in exif.fields() {
        let tag_name = field.tag.to_string();
        let value = field.display_value().with_unit(&exif).to_string();
        map.insert(tag_name, value);
    }

    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

/// Generate ISO 8601 UTC timestamp (RFC 3339 format).
fn iso_timestamp() -> String {
    match time::OffsetDateTime::now_utc().format(&time::format_description::well_known::Rfc3339) {
        Ok(ts) => ts,
        Err(e) => {
            debug!("Timestamp format failed: {e}");
            String::from("1970-01-01T00:00:00Z")
        }
    }
}
