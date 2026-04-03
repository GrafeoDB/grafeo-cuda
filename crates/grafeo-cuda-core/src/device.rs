use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, result};

use crate::error::CudaError;

/// Properties of a CUDA device.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name (e.g., "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// Device ordinal (0-based).
    pub ordinal: usize,
    /// Total global memory in bytes.
    pub total_memory: usize,
}

/// Wrapper around a CUDA context and stream for GPU graph operations.
///
/// Create one per GPU. Multiple `GpuGraph` instances can share the same device.
#[derive(Clone)]
pub struct GpuDevice {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) properties: DeviceProperties,
}

impl GpuDevice {
    /// Initialize a GPU device by ordinal (0 = first GPU).
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal)?;

        let cu_device = ctx.cu_device();
        let name = result::device::get_name(cu_device)
            .unwrap_or_else(|_| format!("CUDA Device {ordinal}"));
        let total_memory = ctx.total_mem().unwrap_or(0);

        let stream = ctx.default_stream();

        let properties = DeviceProperties {
            name,
            ordinal,
            total_memory,
        };

        Ok(Self {
            ctx,
            stream,
            properties,
        })
    }

    /// Returns the device properties.
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Returns a reference to the underlying CUDA context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the default stream for this device.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}
