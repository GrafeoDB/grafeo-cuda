//! GPU k-truss decomposition via iterative edge peeling.
//!
//! For a given k, the k-truss is the maximal subgraph in which every edge
//! participates in at least k-2 triangles. The algorithm iteratively removes
//! edges with insufficient triangle support until convergence.
//!
//! Two CUDA kernels:
//! 1. `edge_triangle_support_kernel`: compute per-edge triangle support
//! 2. `edge_filter_kernel`: remove edges below the threshold

use std::sync::OnceLock;
use std::time::Instant;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use grafeo_cuda_core::error::CudaError;
use grafeo_cuda_core::gpu_graph::GpuGraph;

use crate::module_cache;

/// Result of GPU k-truss decomposition.
#[derive(Debug, Clone)]
pub struct KTrussResult {
    /// Edges remaining in the k-truss, as (source, target) pairs (0-based indices).
    pub edges: Vec<(u32, u32)>,
    /// The k value used.
    pub k: u32,
    /// Number of iterations until convergence.
    pub iterations: u32,
    /// Wall-clock time.
    pub time_seconds: f64,
}

const PTX: &str = include_str!("../../../kernels/compiled/k_truss.ptx");
static MODULE: OnceLock<module_cache::CachedModule> = OnceLock::new();

/// Extract the k-truss from the graph.
///
/// Returns all edges that are supported by at least k-2 triangles after
/// iterative peeling. The graph must use symmetric CSR (undirected).
///
/// k must be >= 3 (a 3-truss means every edge is in at least 1 triangle).
pub fn k_truss(graph: &GpuGraph, k: u32) -> Result<KTrussResult, CudaError> {
    if k < 3 {
        return Err(CudaError::Kernel(
            "k must be >= 3 for k-truss decomposition".to_string(),
        ));
    }

    let start = Instant::now();
    let threshold = k - 2;

    let ctx = graph.stream.context();
    let funcs = module_cache::get_or_load(
        &MODULE,
        ctx,
        PTX,
        &["edge_triangle_support_kernel", "edge_filter_kernel"],
    )?;
    let support_func = &funcs[0];
    let filter_func = &funcs[1];

    let num_edges = graph.num_edges;

    // Per-edge support count and removed mask (on GPU)
    let mut support = graph.stream.clone_htod(&vec![0u32; num_edges as usize])?;
    let mut removed = graph.stream.clone_htod(&vec![0u8; num_edges as usize])?;
    let mut removed_count = graph.stream.clone_htod(&[0u32])?;

    let node_config = LaunchConfig {
        grid_dim: (graph.num_nodes.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let edge_config = LaunchConfig {
        grid_dim: (num_edges.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut iterations = 0u32;

    loop {
        // Reset support to 0
        graph
            .stream
            .memcpy_htod(&vec![0u32; num_edges as usize], &mut support)?;

        // Compute triangle support for each edge
        unsafe {
            graph
                .stream
                .launch_builder(support_func)
                .arg(&graph.offsets)
                .arg(&graph.targets)
                .arg(&removed)
                .arg(&graph.num_nodes)
                .arg(&mut support)
                .launch(node_config)
        }
        .map_err(|e| CudaError::Kernel(format!("support kernel failed: {e}")))?;

        // Reset removed count
        graph.stream.memcpy_htod(&[0u32], &mut removed_count)?;

        // Filter edges below threshold
        unsafe {
            graph
                .stream
                .launch_builder(filter_func)
                .arg(&support)
                .arg(&mut removed)
                .arg(&num_edges)
                .arg(&threshold)
                .arg(&mut removed_count)
                .launch(edge_config)
        }
        .map_err(|e| CudaError::Kernel(format!("filter kernel failed: {e}")))?;

        graph
            .stream
            .synchronize()
            .map_err(|e| CudaError::Kernel(format!("synchronize failed: {e}")))?;

        iterations += 1;

        let count_host = graph.stream.clone_dtoh(&removed_count)?;
        if count_host[0] == 0 {
            break;
        }
    }

    // Read back the removed mask and targets to build the edge list
    let removed_host = graph.stream.clone_dtoh(&removed)?;
    let offsets_host = graph.stream.clone_dtoh(&graph.offsets)?;
    let targets_host = graph.stream.clone_dtoh(&graph.targets)?;

    let mut edges = Vec::new();
    for u in 0..graph.num_nodes {
        let u_start = offsets_host[u as usize] as usize;
        let u_end = offsets_host[u as usize + 1] as usize;
        for idx in u_start..u_end {
            let v = targets_host[idx];
            // Only emit each undirected edge once (u < v)
            if v > u && removed_host[idx] == 0 {
                edges.push((u, v));
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    Ok(KTrussResult {
        edges,
        k,
        iterations,
        time_seconds: elapsed,
    })
}
