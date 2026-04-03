//! GPU-accelerated graph algorithms for Grafeo.
//!
//! Each module implements one algorithm, dispatching to a CUDA kernel loaded from
//! pre-compiled PTX. All algorithms operate on `GpuGraph` (CSR in GPU memory).

// Kernel launches require unsafe: the driver cannot verify argument correctness.
#![allow(unsafe_code)]

pub mod bfs;
pub mod harness;
pub mod k_truss;
mod module_cache;
pub mod pagerank;
pub mod triangle_counting;
