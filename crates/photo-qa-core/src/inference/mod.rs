//! ML inference engine using Candle.
//!
//! Provides model loading and inference for:
//! - `RetinaFace` (face detection)
//! - 68-point facial landmarks
//! - U²-Net / IS-Net (saliency detection)

mod device;

pub use device::get_device;

// TODO: Add model implementations
// mod retinaface;
// mod landmarks;
// mod saliency;
