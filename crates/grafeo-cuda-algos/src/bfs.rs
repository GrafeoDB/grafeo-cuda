//! GPU Breadth-First Search via frontier-based expansion.
//!
//! Uses an explicit frontier queue: each iteration only processes nodes discovered
//! in the previous iteration, rather than scanning all nodes. This is O(frontier * degree)
//! per iteration instead of O(V).
//!
//! Uses `atomicCAS` on the distances array for race-free discovery and `atomicAdd`
//! for frontier queue appending.
//!
//! This is the core of the Graph500 TEPS (Traversed Edges Per Second) benchmark.

use std::time::Instant;

use cudarc::driver::PushKernelArg;
use grafeo_cuda_core::error::CudaError;
use grafeo_cuda_core::gpu_graph::GpuGraph;

use crate::{launch, module_cache};

/// Result of GPU BFS.
#[derive(Debug, Clone)]
pub struct BfsResult {
    /// Distance from source for each node. -1 means unreachable.
    /// Indexed by contiguous 0-based node index.
    pub distances: Vec<i32>,
    /// Number of nodes reached from the source.
    pub reachable: u32,
    /// Maximum distance (depth) from source.
    pub max_depth: i32,
    /// Wall-clock time.
    pub time_seconds: f64,
}

pub(crate) const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ptx/bfs.ptx"));
static MODULE: module_cache::ModuleCache = module_cache::ModuleCache::new();

/// Run BFS from a source node on the GPU.
///
/// Uses frontier-based expansion for O(E) total work instead of O(V * depth).
/// The `source` parameter is the 0-based contiguous node index. The graph is
/// traversed along its CSR lists, so a symmetric CSR gives undirected BFS.
pub fn bfs(graph: &GpuGraph, source: u32) -> Result<BfsResult, CudaError> {
    let start = Instant::now();
    let n = graph.num_nodes as usize;

    if source >= graph.num_nodes {
        return Err(CudaError::Kernel(format!(
            "source node {source} out of range (num_nodes = {})",
            graph.num_nodes
        )));
    }

    let ctx = graph.stream.context();
    let funcs = module_cache::get_or_load(
        &MODULE,
        ctx,
        PTX,
        &["bfs_frontier_kernel", "bfs_expand_kernel"],
    )?;
    let expand_func = &funcs[1];

    // Initialize distances: -1 (unvisited), source = 0
    let mut distances_host = vec![-1i32; n];
    distances_host[source as usize] = 0;
    let mut distances_gpu = graph.stream.clone_htod(&distances_host)?;

    // Frontier buffers: double-buffer pattern (ping-pong). Each node enters a
    // frontier at most once, so `n` entries bound every frontier. Both buffers
    // need the full size because they swap roles after every level.
    let mut frontier_a = graph.stream.alloc_zeros::<u32>(n)?;
    graph.stream.memcpy_htod(&[source], &mut frontier_a)?;
    let mut frontier_b = graph.stream.alloc_zeros::<u32>(n)?;
    let mut frontier_size_buf = graph.stream.clone_htod(&[0u32])?;

    let mut current_frontier_size = 1u32;
    let mut depth = 0i32;

    loop {
        // Reset output frontier size to 0
        graph.stream.memcpy_htod(&[0u32], &mut frontier_size_buf)?;

        let config = launch::thread_per_item(current_frontier_size);

        // SAFETY: the CSR was validated on transfer, so every target indexes
        // `distances`. frontier_a holds current_frontier_size entries and
        // frontier_b has room for all n nodes, the most any level can add.
        unsafe {
            graph
                .stream
                .launch_builder(expand_func)
                .arg(&graph.offsets)
                .arg(&graph.targets)
                .arg(&mut distances_gpu)
                .arg(&frontier_a)
                .arg(&current_frontier_size)
                .arg(&mut frontier_b)
                .arg(&mut frontier_size_buf)
                .arg(&depth)
                .launch(config)
        }
        .map_err(|e| CudaError::Kernel(format!("BFS expand failed: {e}")))?;

        graph
            .stream
            .synchronize()
            .map_err(|e| CudaError::Kernel(format!("synchronize failed: {e}")))?;

        let next_size = graph.stream.clone_dtoh(&frontier_size_buf)?;
        current_frontier_size = next_size[0];

        if current_frontier_size == 0 {
            break;
        }

        depth += 1;

        // Swap frontiers (ping-pong)
        std::mem::swap(&mut frontier_a, &mut frontier_b);
    }

    let distances = graph.stream.clone_dtoh(&distances_gpu)?;
    let reachable = distances.iter().filter(|&&d| d >= 0).count() as u32;
    let elapsed = start.elapsed().as_secs_f64();

    Ok(BfsResult {
        distances,
        reachable,
        max_depth: depth,
        time_seconds: elapsed,
    })
}
