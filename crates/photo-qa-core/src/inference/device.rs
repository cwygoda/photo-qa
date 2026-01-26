//! Device selection for inference.

use candle_core::Device;
use tracing::info;

/// Returns the best available device for inference.
///
/// Automatically detects and uses GPU (Metal on macOS, CUDA on Linux/Windows)
/// if available, falling back to CPU.
#[must_use]
pub fn get_device() -> Device {
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            info!("Using Metal device for inference");
            return device;
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            info!("Using CUDA device for inference");
            return device;
        }
    }

    info!("Using CPU for inference");
    Device::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_device_returns_valid_device() {
        // Function should not panic and always return a device
        let _device = get_device();
    }
}
