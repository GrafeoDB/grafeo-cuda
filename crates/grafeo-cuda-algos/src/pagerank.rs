//! GPU PageRank via iterative sparse matrix-vector multiplication (SpMV).
//!
//! Each iteration: rank_new[v] = (1-d)/N + d * sum(rank[u] / deg[u]) for all in-neighbors u of v.
//! Convergence is checked on-device via a reduction kernel (no per-iteration D2H transfers).
//!
//! Requires symmetric CSR (undirected) or a transpose CSR for directed graphs.

use std::sync::OnceLock;
use std::time::Instant;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use grafeo_cuda_core::error::CudaError;
use grafeo_cuda_core::gpu_graph::GpuGraph;

use crate::module_cache;

/// Result of GPU PageRank computation.
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// Rank values indexed by node (contiguous 0-based).
    pub ranks: Vec<f32>,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
    /// Wall-clock time including all iterations and transfers.
    pub time_seconds: f64,
}

const PTX: &str = include_str!("../../../kernels/compiled/pagerank.ptx");
static MODULE: OnceLock<module_cache::CachedModule> = OnceLock::new();

/// Run PageRank on the GPU.
///
/// Uses iterative SpMV with configurable damping factor, maximum iterations,
/// and convergence tolerance. Convergence is checked entirely on-device via
/// a shared-memory reduction kernel.
pub fn pagerank(
    graph: &GpuGraph,
    damping: f32,
    max_iterations: u32,
    tolerance: f32,
) -> Result<PageRankResult, CudaError> {
    let start = Instant::now();
    let n = graph.num_nodes as usize;

    let ctx = graph.stream.context();
    let funcs = module_cache::get_or_load(
        &MODULE,
        ctx,
        PTX,
        &[
            "pagerank_iteration_kernel",
            "compute_out_degrees_kernel",
            "pagerank_diff_kernel",
        ],
    )?;
    let iter_func = &funcs[0];
    let degree_func = &funcs[1];
    let diff_func = &funcs[2];

    // Initialize uniform ranks
    let initial_rank = 1.0f32 / n as f32;
    let mut ranks_a = graph.stream.clone_htod(&vec![initial_rank; n])?;
    let mut ranks_b = graph.stream.clone_htod(&vec![0.0f32; n])?;

    // Compute out-degrees on GPU
    let mut out_degrees = graph.stream.clone_htod(&vec![0u32; n])?;
    let threads_per_block = 256u32;
    let num_blocks = graph.num_nodes.div_ceil(threads_per_block);
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: all device pointers are valid and correctly sized for num_nodes.
    unsafe {
        graph
            .stream
            .launch_builder(degree_func)
            .arg(&graph.offsets)
            .arg(&graph.num_nodes)
            .arg(&mut out_degrees)
            .launch(config)
    }
    .map_err(|e| CudaError::Kernel(format!("degree kernel failed: {e}")))?;

    // Diff buffer for on-device convergence check (single f32 accumulator)
    let mut diff_buf = graph.stream.clone_htod(&[0.0f32])?;

    let diff_config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: threads_per_block * 4, // sizeof(float) per thread
    };

    let mut converged = false;
    let mut iterations = 0u32;

    for _ in 0..max_iterations {
        // SAFETY: ranks_a (input) and ranks_b (output) are valid device buffers of size n.
        unsafe {
            graph
                .stream
                .launch_builder(iter_func)
                .arg(&graph.offsets)
                .arg(&graph.targets)
                .arg(&ranks_a)
                .arg(&mut ranks_b)
                .arg(&out_degrees)
                .arg(&graph.num_nodes)
                .arg(&damping)
                .launch(config)
        }
        .map_err(|e| CudaError::Kernel(format!("iteration kernel failed: {e}")))?;

        iterations += 1;

        // Compute L1 norm of (ranks_a - ranks_b) entirely on GPU
        graph.stream.memcpy_htod(&[0.0f32], &mut diff_buf)?;

        unsafe {
            graph
                .stream
                .launch_builder(diff_func)
                .arg(&ranks_a)
                .arg(&ranks_b)
                .arg(&graph.num_nodes)
                .arg(&mut diff_buf)
                .launch(diff_config)
        }
        .map_err(|e| CudaError::Kernel(format!("diff kernel failed: {e}")))?;

        // Only transfer 4 bytes (one float) instead of 2*N*4 bytes
        graph
            .stream
            .synchronize()
            .map_err(|e| CudaError::Kernel(format!("synchronize failed: {e}")))?;

        let diff_host = graph.stream.clone_dtoh(&diff_buf)?;
        let diff = diff_host[0];

        if diff < tolerance {
            converged = true;
            std::mem::swap(&mut ranks_a, &mut ranks_b);
            break;
        }

        std::mem::swap(&mut ranks_a, &mut ranks_b);
    }

    graph
        .stream
        .synchronize()
        .map_err(|e| CudaError::Kernel(format!("synchronize failed: {e}")))?;

    let ranks = graph.stream.clone_dtoh(&ranks_a)?;
    let elapsed = start.elapsed().as_secs_f64();

    Ok(PageRankResult {
        ranks,
        iterations,
        converged,
        time_seconds: elapsed,
    })
}
