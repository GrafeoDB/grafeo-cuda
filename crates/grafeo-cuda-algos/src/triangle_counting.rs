//! GPU triangle counting via sorted neighbor intersection.
//!
//! Two kernels available:
//! - `triangle_count_kernel`: baseline, one thread per node
//! - `triangle_count_warp_kernel`: optimized, one warp (32 threads) per node with
//!   cooperative binary-search intersection for load balancing across skewed degrees
//!
//! Complexity: O(m * sqrt(m)) work, parallelized across GPU threads/warps.

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

/// Count all triangles in the graph on GPU.
///
/// Uses the warp-cooperative kernel for better load balancing on skewed graphs.
/// Each warp (32 threads) processes one node, cooperatively intersecting neighbor
/// lists via binary search.
pub fn triangle_count(graph: &GpuGraph) -> Result<TriangleCountResult, CudaError> {
    let start = Instant::now();

    let ctx = graph.stream.context();
    let funcs = module_cache::get_or_load(
        &MODULE,
        ctx,
        PTX,
        &["triangle_count_kernel", "triangle_count_warp_kernel"],
    )?;

    // Allocate result buffer (single u64 counter, initialized to 0)
    let mut result_buf = graph.stream.clone_htod(&[0u64])?;

    // Use the warp kernel: one warp (32 threads) per node
    let func = &funcs[1];
    let warps_needed = graph.num_nodes;
    let threads_per_block = 256u32; // 8 warps per block
    let total_threads = warps_needed * 32;
    let num_blocks = total_threads.div_ceil(threads_per_block);
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: offsets, targets, and result_buf are valid device pointers.
    unsafe {
        graph
            .stream
            .launch_builder(func)
            .arg(&graph.offsets)
            .arg(&graph.targets)
            .arg(&graph.num_nodes)
            .arg(&graph.num_edges)
            .arg(&mut result_buf)
            .launch(config)
    }
    .map_err(|e| CudaError::Kernel(format!("launch failed: {e}")))?;

    graph
        .stream
        .synchronize()
        .map_err(|e| CudaError::Kernel(format!("synchronize failed: {e}")))?;

    let result_host = graph.stream.clone_dtoh(&result_buf)?;
    let elapsed = start.elapsed().as_secs_f64();

    Ok(TriangleCountResult {
        total_triangles: result_host[0],
        time_seconds: elapsed,
    })
}
