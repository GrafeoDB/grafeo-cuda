//! Correctness tests: compare GPU algorithm results against Grafeo CPU implementations.
//!
//! These tests create graphs using Grafeo's LpgStore, run algorithms on both CPU and GPU,
//! and verify the results match.

use grafeo_adapters::plugins::algorithms;
use grafeo_core::LpgStore;

use grafeo_cuda_algos::{bfs, k_truss, pagerank, triangle_counting};
use grafeo_cuda_core::{GpuDevice, GpuGraph, export_to_csr};

/// Create a complete graph K_n using Grafeo's LpgStore.
fn create_complete(n: usize) -> LpgStore {
    let store = LpgStore::new().unwrap();
    let nodes: Vec<_> = (0..n).map(|_| store.create_node(&["Node"])).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            store.create_edge(nodes[i], nodes[j], "EDGE");
            store.create_edge(nodes[j], nodes[i], "EDGE");
        }
    }
    store
}

/// Create a path graph 0-1-2-..-(n-1).
fn create_path(n: usize) -> LpgStore {
    let store = LpgStore::new().unwrap();
    let nodes: Vec<_> = (0..n).map(|_| store.create_node(&["Node"])).collect();
    for i in 0..(n - 1) {
        store.create_edge(nodes[i], nodes[i + 1], "EDGE");
        store.create_edge(nodes[i + 1], nodes[i], "EDGE");
    }
    store
}

/// Export a store to GPU via CSR.
fn store_to_gpu(store: &LpgStore) -> GpuGraph {
    let device = GpuDevice::new(0).expect("No CUDA device");
    let csr = export_to_csr(store, true);
    GpuGraph::from_csr_arrays(csr, &device).expect("GPU transfer failed")
}

// ---- Triangle Counting ----

#[test]
fn test_triangles_k4_cpu_vs_gpu() {
    let store = create_complete(4);
    let cpu_count = algorithms::total_triangles(&store);
    let gpu = store_to_gpu(&store);
    let gpu_result = triangle_counting::triangle_count(&gpu).unwrap();
    assert_eq!(
        cpu_count, gpu_result.total_triangles,
        "K4: CPU={cpu_count}, GPU={}",
        gpu_result.total_triangles
    );
}

#[test]
fn test_triangles_k5_cpu_vs_gpu() {
    let store = create_complete(5);
    let cpu_count = algorithms::total_triangles(&store);
    let gpu = store_to_gpu(&store);
    let gpu_result = triangle_counting::triangle_count(&gpu).unwrap();
    assert_eq!(
        cpu_count, gpu_result.total_triangles,
        "K5: CPU={cpu_count}, GPU={}",
        gpu_result.total_triangles
    );
}

#[test]
fn test_triangles_path_cpu_vs_gpu() {
    let store = create_path(10);
    let cpu_count = algorithms::total_triangles(&store);
    let gpu = store_to_gpu(&store);
    let gpu_result = triangle_counting::triangle_count(&gpu).unwrap();
    assert_eq!(cpu_count, 0);
    assert_eq!(gpu_result.total_triangles, 0);
}

// ---- BFS ----

#[test]
fn test_bfs_k4_cpu_vs_gpu() {
    let store = create_complete(4);
    let node_ids = store.node_ids();
    let cpu_layers = algorithms::bfs_layers(&store, node_ids[0]);
    let gpu = store_to_gpu(&store);
    let gpu_result = bfs::bfs(&gpu, 0).unwrap();

    // K4: all nodes reachable at depth 1
    assert_eq!(cpu_layers.len(), 2, "CPU: 2 layers (source + neighbors)");
    assert_eq!(gpu_result.max_depth, 1);
    assert_eq!(gpu_result.reachable, 4);
}

#[test]
fn test_bfs_path_cpu_vs_gpu() {
    let store = create_path(6);
    let node_ids = store.node_ids();
    let cpu_layers = algorithms::bfs_layers(&store, node_ids[0]);
    let gpu = store_to_gpu(&store);
    let gpu_result = bfs::bfs(&gpu, 0).unwrap();

    // Path of 6 nodes: max depth = 5
    assert_eq!(cpu_layers.len(), 6, "CPU: 6 layers for path of 6");
    assert_eq!(gpu_result.max_depth, 5);
    assert_eq!(gpu_result.reachable, 6);
}

// ---- PageRank ----

#[test]
fn test_pagerank_k4_cpu_vs_gpu() {
    let store = create_complete(4);
    let cpu_ranks = algorithms::pagerank(&store, 0.85, 100, 1e-6);
    let gpu = store_to_gpu(&store);
    let gpu_result = pagerank::pagerank(&gpu, 0.85, 100, 1e-6).unwrap();

    assert!(gpu_result.converged, "GPU PageRank should converge on K4");

    // Compare: both should be approximately uniform (0.25)
    let cpu_values: Vec<f64> = cpu_ranks.values().copied().collect();
    for &cpu_r in &cpu_values {
        assert!(
            (cpu_r - 0.25).abs() < 0.01,
            "CPU rank {cpu_r} should be ~0.25"
        );
    }
    for (i, &gpu_r) in gpu_result.ranks.iter().enumerate() {
        assert!(
            (gpu_r - 0.25).abs() < 0.01,
            "GPU rank for node {i}: {gpu_r} should be ~0.25"
        );
    }
}

// ---- K-Truss ----

#[test]
fn test_ktruss_k4_cpu_vs_gpu() {
    let store = create_complete(4);

    // CPU: all edges in K4 have truss number 4
    let cpu_edges_4 = algorithms::k_truss(&store, 4);
    let cpu_edges_5 = algorithms::k_truss(&store, 5);

    let gpu = store_to_gpu(&store);

    // GPU: 4-truss should contain all 6 edges
    let gpu_result_4 = k_truss::k_truss(&gpu, 4).unwrap();
    assert_eq!(
        gpu_result_4.edges.len(),
        cpu_edges_4.len(),
        "4-truss: CPU has {} edges, GPU has {}",
        cpu_edges_4.len(),
        gpu_result_4.edges.len()
    );

    // GPU: 5-truss should be empty for K4
    let gpu_result_5 = k_truss::k_truss(&gpu, 5).unwrap();
    assert_eq!(
        gpu_result_5.edges.len(),
        cpu_edges_5.len(),
        "5-truss: CPU has {} edges, GPU has {}",
        cpu_edges_5.len(),
        gpu_result_5.edges.len()
    );
    assert!(gpu_result_5.edges.is_empty(), "K4 has no 5-truss edges");
}

#[test]
fn test_ktruss_k5_cpu_vs_gpu() {
    let store = create_complete(5);

    // K5: each edge has 3 common neighbors, truss number = 5
    let cpu_edges_5 = algorithms::k_truss(&store, 5);
    let cpu_edges_6 = algorithms::k_truss(&store, 6);

    let gpu = store_to_gpu(&store);

    let gpu_result_5 = k_truss::k_truss(&gpu, 5).unwrap();
    assert_eq!(
        gpu_result_5.edges.len(),
        cpu_edges_5.len(),
        "5-truss of K5: CPU={}, GPU={}",
        cpu_edges_5.len(),
        gpu_result_5.edges.len()
    );

    let gpu_result_6 = k_truss::k_truss(&gpu, 6).unwrap();
    assert_eq!(
        gpu_result_6.edges.len(),
        cpu_edges_6.len(),
        "6-truss of K5: CPU={}, GPU={}",
        cpu_edges_6.len(),
        gpu_result_6.edges.len()
    );
    assert!(gpu_result_6.edges.is_empty(), "K5 has no 6-truss edges");
}

#[test]
fn test_ktruss_path_no_triangles() {
    let store = create_path(5);
    let gpu = store_to_gpu(&store);

    // Path has no triangles, so 3-truss should be empty
    let gpu_result = k_truss::k_truss(&gpu, 3).unwrap();
    assert!(
        gpu_result.edges.is_empty(),
        "Path graph has no triangles, 3-truss should be empty"
    );
}
