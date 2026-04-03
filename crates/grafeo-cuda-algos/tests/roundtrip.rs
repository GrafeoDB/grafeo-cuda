//! Roundtrip integration tests: build CSR on CPU, transfer to GPU, run algorithms, verify results.
//!
//! These tests require an NVIDIA GPU. They will fail with a clear error if no GPU is available.

use grafeo_cuda_algos::{bfs, pagerank, triangle_counting};
use grafeo_cuda_core::{GpuDevice, GpuGraph};

/// Build a complete graph K4 (4 nodes, all pairs connected) as symmetric CSR.
///
/// K4 has 4 nodes, 6 undirected edges (12 directed), 4 triangles.
/// Adjacency (sorted, symmetric):
///   0: [1, 2, 3]
///   1: [0, 2, 3]
///   2: [0, 1, 3]
///   3: [0, 1, 2]
fn k4_csr() -> (Vec<u32>, Vec<u32>) {
    let offsets = vec![0, 3, 6, 9, 12];
    let targets = vec![
        1, 2, 3, // node 0
        0, 2, 3, // node 1
        0, 1, 3, // node 2
        0, 1, 2, // node 3
    ];
    (offsets, targets)
}

/// Build a path graph P5: 0-1-2-3-4 as symmetric CSR.
///
/// 5 nodes, 4 undirected edges (8 directed), 0 triangles.
fn path5_csr() -> (Vec<u32>, Vec<u32>) {
    let offsets = vec![0, 1, 3, 5, 7, 8];
    let targets = vec![
        1, // node 0
        0, 2, // node 1
        1, 3, // node 2
        2, 4, // node 3
        3, // node 4
    ];
    (offsets, targets)
}

fn create_gpu_graph(offsets: &[u32], targets: &[u32]) -> GpuGraph {
    let device = GpuDevice::new(0).expect("No CUDA device found; GPU tests require an NVIDIA GPU");
    GpuGraph::from_csr_slices(offsets, targets, &device).expect("Failed to transfer CSR to GPU")
}

#[test]
fn test_triangle_count_k4() {
    let (offsets, targets) = k4_csr();
    let graph = create_gpu_graph(&offsets, &targets);
    let result = triangle_counting::triangle_count(&graph).expect("triangle_count failed");
    assert_eq!(result.total_triangles, 4, "K4 has exactly 4 triangles");
}

#[test]
fn test_triangle_count_path() {
    let (offsets, targets) = path5_csr();
    let graph = create_gpu_graph(&offsets, &targets);
    let result = triangle_counting::triangle_count(&graph).expect("triangle_count failed");
    assert_eq!(result.total_triangles, 0, "Path graph has no triangles");
}

#[test]
fn test_bfs_k4() {
    let (offsets, targets) = k4_csr();
    let graph = create_gpu_graph(&offsets, &targets);
    let result = bfs::bfs(&graph, 0).expect("BFS failed");

    assert_eq!(result.reachable, 4, "All 4 nodes reachable in K4");
    assert_eq!(result.max_depth, 1, "K4 is fully connected, max depth = 1");
    assert_eq!(result.distances[0], 0, "Source node has distance 0");
    for &d in &result.distances[1..] {
        assert_eq!(d, 1, "All non-source nodes at distance 1 in K4");
    }
}

#[test]
fn test_bfs_path() {
    let (offsets, targets) = path5_csr();
    let graph = create_gpu_graph(&offsets, &targets);
    let result = bfs::bfs(&graph, 0).expect("BFS failed");

    assert_eq!(result.reachable, 5, "All 5 nodes reachable in path");
    assert_eq!(result.max_depth, 4, "Path 0-1-2-3-4 has max depth 4");
    for (i, &d) in result.distances.iter().enumerate() {
        assert_eq!(d, i as i32, "Distance to node {i} should be {i}");
    }
}

#[test]
fn test_pagerank_k4() {
    let (offsets, targets) = k4_csr();
    let graph = create_gpu_graph(&offsets, &targets);
    let result = pagerank::pagerank(&graph, 0.85, 100, 1e-6).expect("PageRank failed");

    assert!(result.converged, "PageRank should converge on K4");
    // Uniform graph: all ranks should be approximately 0.25
    let expected = 0.25f32;
    for (i, &r) in result.ranks.iter().enumerate() {
        assert!(
            (r - expected).abs() < 0.01,
            "Node {i} rank {r} should be ~{expected}"
        );
    }
}
