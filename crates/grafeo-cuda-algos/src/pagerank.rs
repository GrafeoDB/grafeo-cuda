//! GPU PageRank via iterative sparse matrix-vector multiplication (SpMV).
//!
//! Each iteration: rank_new[v] = (1-d)/N + d * (sum(rank[u] / deg[u]) + dangling/N)
//! for all in-neighbors u of v, where `dangling` is the total rank of nodes
//! without outgoing edges. Spreading that rank over all nodes keeps the ranks
//! summing to 1 and matches Grafeo's CPU PageRank.
//! Convergence is checked on-device via a reduction kernel (no per-iteration D2H
//! transfers of the rank vectors).
//!
//! Requires symmetric CSR (undirected) or a transpose CSR for directed graphs.

use std::time::Instant;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use grafeo_cuda_core::error::CudaError;
use grafeo_cuda_core::gpu_graph::GpuGraph;

use crate::{launch, module_cache};

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

pub(crate) const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ptx/pagerank.ptx"));
static MODULE: module_cache::ModuleCache = module_cache::ModuleCache::new();

/// Run PageRank on the GPU.
///
/// Uses iterative SpMV with configurable damping factor, maximum iterations,
/// and convergence tolerance. Convergence is checked entirely on-device via
/// a shared-memory reduction kernel: the run stops once the L1 norm of the
/// change between two iterations drops below `tolerance`. (Grafeo's CPU
/// PageRank compares the largest per-node change instead.)
///
/// Ranks are `f32`. Parallel edges count once, because the CSR export
/// deduplicates them; Grafeo's CPU PageRank weighs them by multiplicity.
pub fn pagerank(
    graph: &GpuGraph,
    damping: f32,
    max_iterations: u32,
    tolerance: f32,
) -> Result<PageRankResult, CudaError> {
    let start = Instant::now();
    let n = graph.num_nodes as usize;

    // No nodes, no ranks; the driver would also reject the empty grid.
    if n == 0 {
        return Ok(PageRankResult {
            ranks: Vec::new(),
            iterations: 0,
            converged: true,
            time_seconds: start.elapsed().as_secs_f64(),
        });
    }

    let ctx = graph.stream.context();
    let funcs = module_cache::get_or_load(
        &MODULE,
        ctx,
        PTX,
        &[
            "pagerank_iteration_kernel",
            "compute_out_degrees_kernel",
            "pagerank_diff_kernel",
            "pagerank_dangling_sum_kernel",
        ],
    )?;
    let iter_func = &funcs[0];
    let degree_func = &funcs[1];
    let diff_func = &funcs[2];
    let dangling_func = &funcs[3];

    // Initialize uniform ranks
    let initial_rank = 1.0f32 / n as f32;
    let mut ranks_a = graph.stream.clone_htod(&vec![initial_rank; n])?;
    let mut ranks_b = graph.stream.alloc_zeros::<f32>(n)?;

    // Compute out-degrees on GPU
    let mut out_degrees = graph.stream.alloc_zeros::<u32>(n)?;
    let config = launch::thread_per_item(graph.num_nodes);

    // SAFETY: offsets has num_nodes + 1 entries and out_degrees num_nodes.
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

    // Single f32 accumulators for the dangling rank and the convergence check.
    let mut dangling_buf = graph.stream.alloc_zeros::<f32>(1)?;
    let mut diff_buf = graph.stream.alloc_zeros::<f32>(1)?;

    let reduce_config = LaunchConfig {
        shared_mem_bytes: launch::THREADS_PER_BLOCK * 4, // sizeof(float) per thread
        ..config
    };

    let mut converged = false;
    let mut iterations = 0u32;

    for _ in 0..max_iterations {
        graph.stream.memset_zeros(&mut dangling_buf)?;

        // SAFETY: ranks_a and out_degrees hold num_nodes entries; dangling_buf one.
        unsafe {
            graph
                .stream
                .launch_builder(dangling_func)
                .arg(&ranks_a)
                .arg(&out_degrees)
                .arg(&graph.num_nodes)
                .arg(&mut dangling_buf)
                .launch(reduce_config)
        }
        .map_err(|e| CudaError::Kernel(format!("dangling kernel failed: {e}")))?;

        // SAFETY: the CSR was validated on transfer, so every target indexes
        // ranks_a and out_degrees. ranks_a (input) and ranks_b (output) are
        // valid device buffers of size n.
        unsafe {
            graph
                .stream
                .launch_builder(iter_func)
                .arg(&graph.offsets)
                .arg(&graph.targets)
                .arg(&ranks_a)
                .arg(&mut ranks_b)
                .arg(&out_degrees)
                .arg(&dangling_buf)
                .arg(&graph.num_nodes)
                .arg(&damping)
                .launch(config)
        }
        .map_err(|e| CudaError::Kernel(format!("iteration kernel failed: {e}")))?;

        iterations += 1;

        // Compute L1 norm of (ranks_a - ranks_b) entirely on GPU
        graph.stream.memset_zeros(&mut diff_buf)?;

        // SAFETY: both rank buffers hold num_nodes entries; diff_buf one.
        unsafe {
            graph
                .stream
                .launch_builder(diff_func)
                .arg(&ranks_a)
                .arg(&ranks_b)
                .arg(&graph.num_nodes)
                .arg(&mut diff_buf)
                .launch(reduce_config)
        }
        .map_err(|e| CudaError::Kernel(format!("diff kernel failed: {e}")))?;

        // Only transfer 4 bytes (one float) instead of 2*N*4 bytes
        graph
            .stream
            .synchronize()
            .map_err(|e| CudaError::Kernel(format!("synchronize failed: {e}")))?;

        let diff_host = graph.stream.clone_dtoh(&diff_buf)?;
        let diff = diff_host[0];

        std::mem::swap(&mut ranks_a, &mut ranks_b);

        if diff < tolerance {
            converged = true;
            break;
        }
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
