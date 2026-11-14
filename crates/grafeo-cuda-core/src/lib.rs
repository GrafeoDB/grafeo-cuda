//! GPU acceleration core for Grafeo.
//!
//! Provides device management, CSR graph export from Grafeo stores,
//! and host-device memory transfer. Algorithm kernels live in `grafeo-cuda-algos`.

pub mod csr_export;
pub mod device;
pub mod error;
pub mod gpu_graph;

pub use csr_export::{
    CsrArrays, export_to_csr, from_edge_pairs, from_edge_pairs_oriented, parse_tsv_to_edges,
    validate_csr,
};
pub use device::{DeviceProperties, GpuDevice};
pub use error::CudaError;
pub use gpu_graph::GpuGraph;
