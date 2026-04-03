//! Python wrappers for `GpuDevice` and `GpuGraph`.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use grafeo_cuda_algos::{bfs, k_truss, pagerank, triangle_counting};
use grafeo_cuda_core::device::GpuDevice;
use grafeo_cuda_core::gpu_graph::GpuGraph;

/// A CUDA GPU device.
#[pyclass(name = "GpuDevice")]
pub struct PyGpuDevice {
    pub(crate) inner: GpuDevice,
}

#[pymethods]
impl PyGpuDevice {
    /// Initialize a GPU device by ordinal (0 = first GPU).
    #[new]
    #[pyo3(signature = (ordinal=0))]
    fn new(ordinal: usize) -> PyResult<Self> {
        let inner = GpuDevice::new(ordinal)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Device name.
    #[getter]
    fn name(&self) -> String {
        self.inner.properties().name.clone()
    }

    /// Total GPU memory in bytes.
    #[getter]
    fn total_memory(&self) -> usize {
        self.inner.properties().total_memory
    }
}

/// A graph stored in GPU memory (CSR format).
#[pyclass(name = "GpuGraph")]
pub struct PyGpuGraph {
    inner: GpuGraph,
}

#[pymethods]
impl PyGpuGraph {
    /// Create a GpuGraph from raw CSR arrays.
    ///
    /// Args:
    ///     offsets (list[int]): CSR row pointer array.
    ///     targets (list[int]): CSR column index array.
    ///     device: A GpuDevice instance.
    #[staticmethod]
    fn from_csr(offsets: Vec<u32>, targets: Vec<u32>, device: &PyGpuDevice) -> PyResult<Self> {
        let inner = GpuGraph::from_csr_slices(&offsets, &targets, &device.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of nodes.
    #[getter]
    fn num_nodes(&self) -> u32 {
        self.inner.num_nodes
    }

    /// Number of edge entries in CSR.
    #[getter]
    fn num_edges(&self) -> u32 {
        self.inner.num_edges
    }

    /// Count all triangles in the graph on GPU.
    ///
    /// Returns:
    ///     dict with keys 'total_triangles' (int) and 'time_seconds' (float).
    fn triangle_count<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = triangle_counting::triangle_count(&self.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("total_triangles", result.total_triangles)?;
        dict.set_item("time_seconds", result.time_seconds)?;
        Ok(dict)
    }

    /// Run PageRank on GPU.
    ///
    /// Args:
    ///     damping (float): Damping factor (default 0.85).
    ///     max_iterations (int): Maximum iterations (default 100).
    ///     tolerance (float): Convergence tolerance (default 1e-6).
    ///
    /// Returns:
    ///     dict with keys 'ranks' (list[float]), 'iterations' (int),
    ///     'converged' (bool), 'time_seconds' (float).
    #[pyo3(signature = (damping=0.85, max_iterations=100, tolerance=1e-6))]
    fn pagerank<'py>(
        &self,
        py: Python<'py>,
        damping: f32,
        max_iterations: u32,
        tolerance: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = pagerank::pagerank(&self.inner, damping, max_iterations, tolerance)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("ranks", result.ranks)?;
        dict.set_item("iterations", result.iterations)?;
        dict.set_item("converged", result.converged)?;
        dict.set_item("time_seconds", result.time_seconds)?;
        Ok(dict)
    }

    /// Run BFS from a source node on GPU.
    ///
    /// Args:
    ///     source (int): Source node index (0-based).
    ///
    /// Returns:
    ///     dict with keys 'distances' (list[int]), 'reachable' (int),
    ///     'max_depth' (int), 'time_seconds' (float).
    fn bfs<'py>(&self, py: Python<'py>, source: u32) -> PyResult<Bound<'py, PyDict>> {
        let result = bfs::bfs(&self.inner, source)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("distances", result.distances)?;
        dict.set_item("reachable", result.reachable)?;
        dict.set_item("max_depth", result.max_depth)?;
        dict.set_item("time_seconds", result.time_seconds)?;
        Ok(dict)
    }

    /// Extract the k-truss from the graph.
    ///
    /// Args:
    ///     k (int): Truss number (minimum 3).
    ///
    /// Returns:
    ///     dict with keys 'edges' (list[tuple[int,int]]), 'k' (int),
    ///     'iterations' (int), 'time_seconds' (float).
    fn k_truss<'py>(&self, py: Python<'py>, k: u32) -> PyResult<Bound<'py, PyDict>> {
        let result = k_truss::k_truss(&self.inner, k)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("edges", result.edges)?;
        dict.set_item("k", result.k)?;
        dict.set_item("iterations", result.iterations)?;
        dict.set_item("time_seconds", result.time_seconds)?;
        Ok(dict)
    }
}
