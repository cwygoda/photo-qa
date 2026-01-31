//! Model loading utilities for safetensors format.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use once_cell::sync::OnceCell;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use tracing::debug;

/// A lazily-loaded model that defers loading until first access.
pub struct LazyModel<T> {
    path: std::path::PathBuf,
    device: Device,
    builder: fn(VarBuilder) -> Result<T>,
    model: OnceCell<T>,
}

impl<T: Send + Sync> LazyModel<T> {
    /// Creates a new lazy model loader.
    ///
    /// The model will not be loaded until `get()` is called.
    #[must_use]
    pub fn new(path: impl AsRef<Path>, device: Device, builder: fn(VarBuilder) -> Result<T>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            device,
            builder,
            model: OnceCell::new(),
        }
    }

    /// Gets the model, loading it if necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model file cannot be read
    /// - The safetensors data is invalid
    /// - The model builder fails
    pub fn get(&self) -> Result<&T> {
        self.model.get_or_try_init(|| {
            debug!("Loading model from {}", self.path.display());
            let vb = load_safetensors(&self.path, &self.device)?;
            (self.builder)(vb)
        })
    }

    /// Returns true if the model has been loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.model.get().is_some()
    }
}

/// Loads a safetensors file and creates a `VarBuilder` for the model.
///
/// # Arguments
///
/// * `path` - Path to the safetensors file
/// * `device` - Device to load tensors onto
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - The safetensors data is invalid
pub fn load_safetensors(path: impl AsRef<Path>, device: &Device) -> Result<VarBuilder<'static>> {
    let path = path.as_ref();
    debug!("Loading safetensors from {}", path.display());

    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read model file: {}", path.display()))?;

    let tensors = SafeTensors::deserialize(&data)
        .with_context(|| format!("Failed to parse safetensors: {}", path.display()))?;

    // Convert to HashMap<String, Tensor> for VarBuilder
    let mut tensor_map: HashMap<String, Tensor> = HashMap::new();

    for name in tensors.names() {
        let tensor_view = tensors
            .tensor(name)
            .with_context(|| format!("Failed to get tensor '{name}'"))?;

        let dtype = safetensors_dtype_to_candle(tensor_view.dtype())?;
        let shape: Vec<usize> = tensor_view.shape().to_vec();

        let tensor = Tensor::from_raw_buffer(tensor_view.data(), dtype, &shape, device)
            .with_context(|| format!("Failed to create tensor '{name}'"))?;

        tensor_map.insert(name.clone(), tensor);
    }

    // VarBuilder::from_tensors takes ownership
    Ok(VarBuilder::from_tensors(tensor_map, DType::F32, device))
}

/// Converts safetensors dtype to candle dtype.
fn safetensors_dtype_to_candle(dtype: safetensors::Dtype) -> Result<DType> {
    use safetensors::Dtype as S;
    match dtype {
        S::F32 => Ok(DType::F32),
        S::F64 => Ok(DType::F64),
        S::F16 => Ok(DType::F16),
        S::BF16 => Ok(DType::BF16),
        S::I64 => Ok(DType::I64),
        S::U8 => Ok(DType::U8),
        S::U32 => Ok(DType::U32),
        other => anyhow::bail!("Unsupported dtype: {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[allow(clippy::expect_used)]
    fn create_test_safetensors() -> NamedTempFile {
        use safetensors::serialize;
        use safetensors::tensor::TensorView;

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data_bytes: &[u8] = bytemuck::cast_slice(&data);

        let tensor = TensorView::new(safetensors::Dtype::F32, vec![2, 2], data_bytes)
            .expect("valid tensor view");

        let tensors = HashMap::from([("test_tensor".to_string(), tensor)]);
        let serialized = serialize(&tensors, &None).expect("serialize");

        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&serialized).expect("write");
        file
    }

    #[test]
    fn test_load_safetensors() {
        let file = create_test_safetensors();
        let result = load_safetensors(file.path(), &Device::Cpu);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_safetensors_missing_file() {
        let result = load_safetensors("/nonexistent/path.safetensors", &Device::Cpu);
        assert!(result.is_err());
    }
}
