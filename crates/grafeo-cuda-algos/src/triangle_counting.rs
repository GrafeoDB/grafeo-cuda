//! GPU triangle counting via sorted neighbor intersection.
//!
//! Three kernels available:
//! - `triangle_count_kernel`: baseline, one thread per node
//! - `triangle_count_warp_kernel`: one warp per node, cooperative binary search
//! - `triangle_count_edge_parallel_kernel`: one warp per forward edge (best load balance)
//!
//! The edge-parallel kernel requires prefix-sum auxiliary buffers.

use std::sync::OnceLock;
use std::time::Instant;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use grafeo_cuda_core::error::CudaError;
use grafeo_cuda_core::gpu_graph::GpuGraph;

use crate::module_cache;

/// Result of GPU triangle counting.
#[derive(Debug, Clone)]
pub struct TriangleCountResult {
    /// Total number of unique triangles in the graph.
    pub total_triangles: u64,
    /// Wall-clock time including kernel launch overhead.
    pub time_seconds: f64,
}

const PTX: &str = include_str!("../../../kernels/compiled/triangle_counting.ptx");
static MODULE: OnceLock<module_cache::CachedModule> = OnceLock::new();

/// Count all triangles in the graph on GPU using the edge-parallel kernel.
///
/// Uses a three-step pipeline:
/// 1. Compute forward edge counts per node (v > u)
/// 2. Prefix sum to get global edge-to-warp mapping
/// 3. One warp per forward edge, cooperative intersection
///
/// Falls back to warp-per-node kernel if prefix sum fails.
pub fn triangle_count(graph: &GpuGraph) -> Result<TriangleCountResult, CudaError> {
    let start = Instant::now();

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

    let threads_per_block = 256u32;
    let num_blocks_nodes = graph.num_nodes.div_ceil(threads_per_block);

    // Step 1: Compute forward edge counts per node
    let mut fwd_counts = graph.stream.clone_htod(&vec![0u32; graph.num_nodes as usize])?;

    unsafe {
        graph
            .stream
            .launch_builder(&funcs[2]) // compute_forward_counts_kernel
            .arg(&graph.offsets)
            .arg(&graph.targets)
            .arg(&graph.num_nodes)
            .arg(&mut fwd_counts)
            .launch(LaunchConfig {
                grid_dim: (num_blocks_nodes, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| CudaError::Kernel(format!("fwd_counts launch failed: {e}")))?;

    // Step 2: Prefix sum (single-thread kernel, fast enough for <10M nodes)
    let mut fwd_offsets = graph.stream.clone_htod(&vec![0u32; graph.num_nodes as usize + 1])?;

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
    graph.stream.synchronize()
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

    let total_threads = total_fwd_edges * 32; // one warp per forward edge
    let num_blocks = total_threads.div_ceil(threads_per_block);

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
            .launch(LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .map_err(|e| CudaError::Kernel(format!("edge_parallel launch failed: {e}")))?;

    graph.stream.synchronize()
        .map_err(|e| CudaError::Kernel(format!("sync failed: {e}")))?;

    let result_host = graph.stream.clone_dtoh(&result_buf)?;
    let elapsed = start.elapsed().as_secs_f64();

    Ok(TriangleCountResult {
        total_triangles: result_host[0],
        time_seconds: elapsed,
    })
}
