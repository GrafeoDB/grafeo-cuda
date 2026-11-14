//! GPU triangle counting via sorted neighbor intersection.
//!
//! Three kernels available:
//! - `triangle_count_kernel`: baseline, one thread per node
//! - `triangle_count_warp_kernel`: one warp per node, cooperative binary search
//! - `triangle_count_edge_parallel_kernel`: one warp per forward edge (best load balance)
//!
//! The edge-parallel kernel requires prefix-sum auxiliary buffers.

use std::time::Instant;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use grafeo_cuda_core::error::CudaError;
use grafeo_cuda_core::gpu_graph::GpuGraph;

use crate::{launch, module_cache};

/// Result of GPU triangle counting.
#[derive(Debug, Clone)]
pub struct TriangleCountResult {
    /// Total number of unique triangles in the graph.
    pub total_triangles: u64,
    /// Wall-clock time including kernel launch overhead.
    pub time_seconds: f64,
}

pub(crate) const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ptx/triangle_counting.ptx"));
static MODULE: module_cache::ModuleCache = module_cache::ModuleCache::new();
static MODULE_ORIENTED: module_cache::ModuleCache = module_cache::ModuleCache::new();

/// Count all triangles in the graph on GPU using the edge-parallel kernel.
///
/// Uses a three-step pipeline:
/// 1. Compute forward edge counts per node (v > u)
/// 2. Prefix sum to get global edge-to-warp mapping
/// 3. One warp per forward edge, cooperative intersection
///
/// The graph must be a symmetric CSR with sorted neighbor lists, as built by
/// `export_to_csr(store, true)` or `from_edge_pairs(edges, n, true)`.
pub fn triangle_count(graph: &GpuGraph) -> Result<TriangleCountResult, CudaError> {
    let start = Instant::now();

    // Nothing to launch, and the driver rejects empty grids.
    if graph.num_edges == 0 {
        return Ok(TriangleCountResult {
            total_triangles: 0,
            time_seconds: start.elapsed().as_secs_f64(),
        });
    }

    let ctx = graph.stream.context();
    let funcs = module_cache::get_or_load(
        &MODULE,
        ctx,
        PTX,
        &[
            "triangle_count_kernel",
            "triangle_count_warp_kernel",
            "compute_forward_counts_kernel",
            "prefix_sum_kernel",
            "triangle_count_edge_parallel_kernel",
        ],
    )?;

    let node_config = launch::thread_per_item(graph.num_nodes);

    // Step 1: Compute forward edge counts per node
    let mut fwd_counts = graph.stream.alloc_zeros::<u32>(graph.num_nodes as usize)?;

    // SAFETY: the CSR was validated on transfer; fwd_counts holds num_nodes entries.
    unsafe {
        graph
            .stream
            .launch_builder(&funcs[2]) // compute_forward_counts_kernel
            .arg(&graph.offsets)
            .arg(&graph.targets)
            .arg(&graph.num_nodes)
            .arg(&mut fwd_counts)
            .launch(node_config)
    }
    .map_err(|e| CudaError::Kernel(format!("fwd_counts launch failed: {e}")))?;

    // Step 2: Prefix sum (single-thread kernel, fast enough for <10M nodes)
    let mut fwd_offsets = graph
        .stream
        .alloc_zeros::<u32>(graph.num_nodes as usize + 1)?;

    // SAFETY: fwd_counts holds num_nodes entries and fwd_offsets num_nodes + 1.
    unsafe {
        graph
            .stream
            .launch_builder(&funcs[3]) // prefix_sum_kernel
            .arg(&fwd_counts)
            .arg(&mut fwd_offsets)
            .arg(&graph.num_nodes)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| CudaError::Kernel(format!("prefix_sum launch failed: {e}")))?;

    // Read total forward edges from last element
    graph
        .stream
        .synchronize()
        .map_err(|e| CudaError::Kernel(format!("sync failed: {e}")))?;

    let fwd_offsets_host = graph.stream.clone_dtoh(&fwd_offsets)?;
    let total_fwd_edges = fwd_offsets_host[graph.num_nodes as usize];

    if total_fwd_edges == 0 {
        return Ok(TriangleCountResult {
            total_triangles: 0,
            time_seconds: start.elapsed().as_secs_f64(),
        });
    }

    // Step 3: Edge-parallel triangle counting
    let mut result_buf = graph.stream.clone_htod(&[0u64])?;
    let edge_config = launch::warp_per_item(total_fwd_edges)?; // one warp per forward edge

    // SAFETY: the CSR was validated on transfer (sorted lists, targets in
    // range); fwd_offsets was filled by the prefix sum for this graph.
    unsafe {
        graph
            .stream
            .launch_builder(&funcs[4]) // triangle_count_edge_parallel_kernel
            .arg(&graph.offsets)
            .arg(&graph.targets)
            .arg(&fwd_offsets)
            .arg(&graph.num_nodes)
            .arg(&total_fwd_edges)
            .arg(&mut result_buf)
            .launch(edge_config)
    }
    .map_err(|e| CudaError::Kernel(format!("edge_parallel launch failed: {e}")))?;

    graph
        .stream
        .synchronize()
        .map_err(|e| CudaError::Kernel(format!("sync failed: {e}")))?;

    let result_host = graph.stream.clone_dtoh(&result_buf)?;
    let elapsed = start.elapsed().as_secs_f64();

    Ok(TriangleCountResult {
        total_triangles: result_host[0],
        time_seconds: elapsed,
    })
}

/// Count triangles using a degree-oriented CSR on GPU.
///
/// The input `GpuGraph` must contain a degree-oriented CSR (edges directed from
/// lower-degree to higher-degree vertex). This eliminates the forward-edge
/// counting and prefix-sum steps, launching a single kernel directly.
///
/// Each triangle is counted exactly once: for oriented edge u→v, any
/// w ∈ N_out(u) ∩ N_out(v) forms a unique triangle.
pub fn triangle_count_oriented(graph: &GpuGraph) -> Result<TriangleCountResult, CudaError> {
    let start = Instant::now();

    // Nothing to launch, and the driver rejects empty grids.
    if graph.num_edges == 0 {
        return Ok(TriangleCountResult {
            total_triangles: 0,
            time_seconds: start.elapsed().as_secs_f64(),
        });
    }

    let ctx = graph.stream.context();
    let funcs = module_cache::get_or_load(
        &MODULE_ORIENTED,
        ctx,
        PTX,
        &["triangle_count_oriented_kernel"],
    )?;

    let mut result_buf = graph.stream.clone_htod(&[0u64])?;
    let edge_config = launch::warp_per_item(graph.num_edges)?; // one warp per oriented edge

    // SAFETY: the CSR was validated on transfer (sorted lists, targets in range).
    unsafe {
        graph
            .stream
            .launch_builder(&funcs[0])
            .arg(&graph.offsets)
            .arg(&graph.targets)
            .arg(&graph.num_nodes)
            .arg(&graph.num_edges)
            .arg(&mut result_buf)
            .launch(edge_config)
    }
    .map_err(|e| CudaError::Kernel(format!("oriented TC launch failed: {e}")))?;

    graph
        .stream
        .synchronize()
        .map_err(|e| CudaError::Kernel(format!("sync failed: {e}")))?;

    let result_host = graph.stream.clone_dtoh(&result_buf)?;
    let elapsed = start.elapsed().as_secs_f64();

    Ok(TriangleCountResult {
        total_triangles: result_host[0],
        time_seconds: elapsed,
    })
}
