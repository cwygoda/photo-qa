//! ML inference engine using Candle.
//!
//! Provides model loading and inference for:
//! - `RetinaFace` (face detection)
//! - 68-point facial landmarks
//! - U²-Net / IS-Net (saliency detection)

mod device;
mod loader;

pub use device::get_device;
pub use loader::{load_safetensors, LazyModel};

// TODO: Add model implementations
// mod retinaface;
// mod landmarks;
// mod saliency;
