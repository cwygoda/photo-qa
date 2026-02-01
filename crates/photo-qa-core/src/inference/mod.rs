//! ML inference engine using Candle.
//!
//! Provides model loading and inference for:
//! - `BlazeFace` (face detection with keypoints)
//! - Eye state classifier (open/closed detection)
//! - U²-Net / IS-Net (saliency detection, deferred)

mod blazeface;
mod device;
mod eye_state;
mod loader;
mod utils;

pub use blazeface::{BlazeFace, FaceDetection, INPUT_SIZE as BLAZEFACE_INPUT_SIZE};
pub use device::get_device;
pub use eye_state::{EyeStateClassifier, EYE_HEIGHT, EYE_WIDTH};
pub use loader::{load_safetensors, LazyModel};

pub(crate) use utils::sigmoid;
