use thiserror::Error;

/// Errors from GPU operations.
#[derive(Error, Debug)]
pub enum CudaError {
    /// No CUDA-capable device was found on this system.
    #[error("no CUDA device found")]
    NoDevice,

    /// The CUDA driver returned an error.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    /// The graph is too large to fit in GPU memory.
    #[error(
        "graph too large for GPU memory: {required} bytes required, {available} bytes available"
    )]
    OutOfMemory {
        /// Bytes needed for the CSR arrays.
        required: usize,
        /// Bytes available on the device.
        available: usize,
    },

    /// A kernel launch or execution failed.
    #[error("kernel error: {0}")]
    Kernel(String),

    /// Failed to export the graph from Grafeo to CSR format.
    #[error("CSR export failed: {0}")]
    Export(String),
}
