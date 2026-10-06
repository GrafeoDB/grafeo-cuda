//! Roundtrip integration tests: build CSR on CPU, transfer to GPU, run algorithms, verify results.
//!
//! These tests require an NVIDIA GPU and are ignored by default. Run them with
//! `cargo test -p grafeo-cuda-algos -- --ignored` on a machine with a GPU.

use grafeo_cuda_algos::{bfs, k_truss, pagerank, triangle_counting};
use grafeo_cuda_core::{CudaError, GpuDevice, GpuGraph, from_edge_pairs, from_edge_pairs_oriented};

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

fn device() -> GpuDevice {
    GpuDevice::new(0).expect("No CUDA device found; GPU tests require an NVIDIA GPU")
}

fn create_gpu_graph(offsets: &[u32], targets: &[u32]) -> GpuGraph {
    GpuGraph::from_csr_slices(offsets, targets, &device()).expect("Failed to transfer CSR to GPU")
}

#[test]
#[ignore = "needs a CUDA device"]
fn test_triangle_count_k4() {
    let (offsets, targets) = k4_csr();
    let graph = create_gpu_graph(&offsets, &targets);
    let result = triangle_counting::triangle_count(&graph).expect("triangle_count failed");
    assert_eq!(result.total_triangles, 4, "K4 has exactly 4 triangles");
}

#[test]
#[ignore = "needs a CUDA device"]
fn test_triangle_count_path() {
    let (offsets, targets) = path5_csr();
    let graph = create_gpu_graph(&offsets, &targets);
    let result = triangle_counting::triangle_count(&graph).expect("triangle_count failed");
    assert_eq!(result.total_triangles, 0, "Path graph has no triangles");
}

#[test]
#[ignore = "needs a CUDA device"]
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
#[ignore = "needs a CUDA device"]
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
#[ignore = "needs a CUDA device"]
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

// ---- degenerate graphs ----

/// Regression: zero-node and zero-edge graphs made the kernels launch with an
/// empty grid, which the driver rejects with `CUDA_ERROR_INVALID_VALUE`.
#[test]
#[ignore = "needs a CUDA device"]
fn test_empty_graph() {
    let graph = create_gpu_graph(&[0], &[]);
    assert_eq!((graph.num_nodes, graph.num_edges), (0, 0));

    assert_eq!(
        triangle_counting::triangle_count(&graph)
            .unwrap()
            .total_triangles,
        0
    );
    assert_eq!(
        triangle_counting::triangle_count_oriented(&graph)
            .unwrap()
            .total_triangles,
        0
    );
    let ranks = pagerank::pagerank(&graph, 0.85, 20, 1e-6).unwrap();
    assert!(ranks.ranks.is_empty());
    assert!(ranks.converged);
    assert!(k_truss::k_truss(&graph, 3).unwrap().edges.is_empty());
    assert!(matches!(bfs::bfs(&graph, 0), Err(CudaError::Kernel(_))));
}

#[test]
#[ignore = "needs a CUDA device"]
fn test_edgeless_graph() {
    let graph = create_gpu_graph(&[0, 0, 0, 0], &[]);

    assert_eq!(
        triangle_counting::triangle_count(&graph)
            .unwrap()
            .total_triangles,
        0
    );
    assert_eq!(
        triangle_counting::triangle_count_oriented(&graph)
            .unwrap()
            .total_triangles,
        0
    );
    assert!(k_truss::k_truss(&graph, 3).unwrap().edges.is_empty());

    let result = bfs::bfs(&graph, 1).unwrap();
    assert_eq!(result.distances, vec![-1, 0, -1]);
    assert_eq!((result.reachable, result.max_depth), (1, 0));

    // Every node is dangling: the rank is spread evenly instead of leaking.
    let ranks = pagerank::pagerank(&graph, 0.85, 50, 1e-7).unwrap();
    for (i, &r) in ranks.ranks.iter().enumerate() {
        assert!((r - 1.0 / 3.0).abs() < 1e-5, "node {i} rank {r}");
    }
}

// ---- BFS frontier ----

/// Regression: the first BFS frontier buffer held a single entry and became the
/// output buffer after the first swap, so every later level larger than one
/// node was written past the end of the allocation, over the buffer being read.
#[test]
#[ignore = "needs a CUDA device"]
fn test_bfs_wide_frontier_after_first_level() {
    // 0 - 1, then node 1 fans out to `width` chains of `depth` nodes each, so
    // levels 2 ..= depth + 1 are all `width` nodes wide. The levels must be
    // wider than the number of threads the GPU keeps resident at once, or the
    // overlapping reads and writes happen not to collide.
    let width = 500_000u32;
    let depth = 4u32;
    let node = |level: u32, chain: u32| 2 + (level - 2) * width + chain;
    let mut edges = vec![(0, 1)];
    for chain in 0..width {
        edges.push((1, node(2, chain)));
        for level in 3..=depth + 1 {
            edges.push((node(level - 1, chain), node(level, chain)));
        }
    }
    let num_nodes = 2 + depth * width;
    let mut expected = vec![0i32, 1];
    for level in 2..=depth + 1 {
        expected.extend(std::iter::repeat_n(level as i32, width as usize));
    }

    let csr = from_edge_pairs(&edges, num_nodes, true);
    let graph = GpuGraph::from_csr_arrays(csr, &device()).unwrap();
    for run in 0..5 {
        let result = bfs::bfs(&graph, 0).unwrap();
        let wrong = expected
            .iter()
            .zip(&result.distances)
            .filter(|(e, d)| e != d)
            .count();
        assert_eq!(wrong, 0, "run {run}: {wrong} nodes have a wrong distance");
        assert_eq!(result.reachable, num_nodes, "run {run}");
        assert_eq!(result.max_depth, (depth + 1) as i32, "run {run}");
    }
}

// ---- oriented triangle counting ----

#[test]
#[ignore = "needs a CUDA device"]
fn test_triangle_count_oriented_k5() {
    let edges: Vec<(u32, u32)> = (0..5u32)
        .flat_map(|i| ((i + 1)..5).map(move |j| (i, j)))
        .collect();
    let csr = from_edge_pairs_oriented(&edges, 5);
    let graph = GpuGraph::from_csr_arrays(csr, &device()).unwrap();
    let result = triangle_counting::triangle_count_oriented(&graph).unwrap();
    assert_eq!(result.total_triangles, 10);
}

// ---- input validation ----

/// Regression: malformed CSR from safe code (or Python) reached the kernels,
/// which index `offsets` and `targets` without bounds checks.
#[test]
#[ignore = "needs a CUDA device"]
fn test_invalid_csr_is_rejected_before_transfer() {
    let device = device();
    let cases: [(&[u32], &[u32]); 5] = [
        (&[], &[]),
        (&[0, 5], &[1]),
        (&[0, 1, 2], &[7, 0]),
        (&[0, 2, 1, 2], &[1, 2]),
        (&[0, 2, 2], &[1, 0]),
    ];
    for (offsets, targets) in cases {
        let result = GpuGraph::from_csr_slices(offsets, targets, &device);
        assert!(
            matches!(result, Err(CudaError::InvalidCsr(_))),
            "{offsets:?} / {targets:?} was accepted"
        );
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn test_argument_errors() {
    let (offsets, targets) = k4_csr();
    let graph = create_gpu_graph(&offsets, &targets);
    assert!(matches!(bfs::bfs(&graph, 4), Err(CudaError::Kernel(_))));
    assert!(matches!(
        k_truss::k_truss(&graph, 2),
        Err(CudaError::Kernel(_))
    ));
}
