use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use grafeo_engine::GrafeoDB;

use crate::csr_export::{CsrArrays, export_to_csr};
use crate::device::GpuDevice;
use crate::error::CudaError;

/// A graph stored in GPU memory as CSR arrays.
///
/// Created from a Grafeo database via `GpuGraph::from_grafeo()`. The graph is
/// a read-only snapshot: mutations in the source database after transfer are
/// not reflected.
pub struct GpuGraph {
    /// CSR row pointers on GPU.
    pub offsets: CudaSlice<u32>,
    /// CSR column indices on GPU.
    pub targets: CudaSlice<u32>,
    /// Number of nodes.
    pub num_nodes: u32,
    /// Number of edge entries in the CSR (may be 2x logical edges if symmetric).
    pub num_edges: u32,
    /// Reference to the stream this graph was created on.
    pub stream: Arc<CudaStream>,
}

impl GpuGraph {
    /// Transfer a Grafeo graph to GPU memory.
    ///
    /// Exports the current graph state as symmetric CSR (undirected) and copies
    /// the arrays to the GPU. This is the primary entry point for GPU acceleration.
    pub fn from_grafeo(db: &GrafeoDB, gpu_device: &GpuDevice) -> Result<Self, CudaError> {
        let store = db.store();
        let csr = export_to_csr(&**store, true); // symmetric for undirected algorithms
        Self::from_csr_arrays(csr, gpu_device)
    }

    /// Transfer pre-built CSR arrays to GPU memory.
    ///
    /// Useful for benchmarking with raw data that bypasses the Grafeo export step.
    pub fn from_csr_arrays(csr: CsrArrays, gpu_device: &GpuDevice) -> Result<Self, CudaError> {
        let stream = Arc::clone(gpu_device.stream());

        let offsets = stream.clone_htod(&csr.offsets)?;
        let targets = stream.clone_htod(&csr.targets)?;

        Ok(Self {
            offsets,
            targets,
            num_nodes: csr.num_nodes,
            num_edges: csr.num_edges,
            stream,
        })
    }

    /// Transfer pre-built CSR slices to GPU memory.
    ///
    /// Takes borrowed slices instead of owned vecs, copying the data.
    pub fn from_csr_slices(
        offsets: &[u32],
        targets: &[u32],
        gpu_device: &GpuDevice,
    ) -> Result<Self, CudaError> {
        let stream = Arc::clone(gpu_device.stream());

        let num_nodes = if offsets.is_empty() {
            0
        } else {
            (offsets.len() - 1) as u32
        };
        let num_edges = targets.len() as u32;

        let offsets_gpu = stream.clone_htod(offsets)?;
        let targets_gpu = stream.clone_htod(targets)?;

        Ok(Self {
            offsets: offsets_gpu,
            targets: targets_gpu,
            num_nodes,
            num_edges,
            stream,
        })
    }
}
