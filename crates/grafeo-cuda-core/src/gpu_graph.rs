use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use grafeo_engine::GrafeoDB;

use crate::csr_export::{CsrArrays, export_to_csr, validate_csr};
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
    ///
    /// Reads through [`GrafeoDB::snapshot`], so databases backed by an external
    /// store (`GrafeoDB::with_store`, `GrafeoDB::with_read_store`) work too.
    /// CSR index `i` is the node at position `i` of the store's `node_ids(view)`,
    /// see [`export_to_csr`].
    pub fn from_grafeo(db: &GrafeoDB, gpu_device: &GpuDevice) -> Result<Self, CudaError> {
        Self::from_csr_arrays(snapshot_csr(db)?, gpu_device)
    }

    /// Transfer pre-built CSR arrays to GPU memory.
    ///
    /// Useful for benchmarking with raw data that bypasses the Grafeo export step.
    /// The arrays are checked with [`CsrArrays::validate`] first.
    pub fn from_csr_arrays(csr: CsrArrays, gpu_device: &GpuDevice) -> Result<Self, CudaError> {
        csr.validate()?;
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
    /// The slices are checked with [`validate_csr`] first.
    pub fn from_csr_slices(
        offsets: &[u32],
        targets: &[u32],
        gpu_device: &GpuDevice,
    ) -> Result<Self, CudaError> {
        validate_csr(offsets, targets)?;
        let stream = Arc::clone(gpu_device.stream());

        let num_nodes = (offsets.len() - 1) as u32;
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

/// Exports the active graph of `db` as symmetric CSR, pinned at the current
/// committed epoch via [`GrafeoDB::snapshot`].
fn snapshot_csr(db: &GrafeoDB) -> Result<CsrArrays, CudaError> {
    let snapshot = db
        .snapshot()
        .map_err(|e| CudaError::Export(e.to_string()))?;
    // symmetric for undirected algorithms
    export_to_csr(snapshot.store(), snapshot.view(), true)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_core::RowGroupStore;
    use grafeo_engine::database::BatchEdge;
    use grafeo_engine::{Config, GrafeoDB};

    use super::snapshot_csr;

    /// Regression: `from_grafeo` used `GrafeoDB::store()`, which panics for
    /// databases created via `with_store` over an external store.
    #[test]
    fn snapshot_reads_external_store() {
        let store = Arc::new(RowGroupStore::new());
        let db = GrafeoDB::with_store(store, Config::in_memory()).unwrap();
        let alix = db.create_node(&["Person"]).unwrap();
        let gus = db.create_node(&["Person"]).unwrap();
        let vincent = db.create_node(&["Person"]).unwrap();
        db.batch_create_edges(vec![
            BatchEdge::new(alix, gus, "KNOWS"),
            BatchEdge::new(gus, vincent, "KNOWS"),
        ])
        .unwrap();

        let csr = snapshot_csr(&db).unwrap();
        assert_eq!(csr.num_nodes, 3);
        assert_eq!(csr.offsets, vec![0, 1, 3, 4]);
        assert_eq!(csr.targets, vec![1, 0, 2, 1]);
        csr.validate().unwrap();
    }

    #[test]
    fn snapshot_reads_built_in_store() {
        let db = GrafeoDB::new_in_memory();
        let amsterdam = db.create_node(&["City"]).unwrap();
        let berlin = db.create_node(&["City"]).unwrap();
        db.batch_create_edges(vec![BatchEdge::new(amsterdam, berlin, "ROAD")])
            .unwrap();

        let csr = snapshot_csr(&db).unwrap();
        assert_eq!(csr.offsets, vec![0, 1, 2]);
        assert_eq!(csr.targets, vec![1, 0]);
    }
}
