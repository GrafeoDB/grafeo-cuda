//! Python bindings for grafeo-cuda.
//!
//! Exposes `GpuDevice` and `GpuGraph` to Python, allowing GPU-accelerated
//! graph algorithms from the Grafeo Python package.

mod gpu_graph;

use pyo3::prelude::*;

use gpu_graph::{PyGpuDevice, PyGpuGraph};

#[pymodule]
fn grafeo_cuda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGpuDevice>()?;
    m.add_class::<PyGpuGraph>()?;
    Ok(())
}
