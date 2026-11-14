//! Correctness tests: compare GPU algorithm results against Grafeo's CPU
//! implementations (`grafeo_adapters::plugins::algorithms`) on the same graph.
//!
//! Graphs are built in Grafeo's `LpgStore`, exported with `export_to_csr`
//! (symmetric), and CSR index `i` is mapped back to `store.node_ids()[i]`.
//! Undirected edges are stored in both directions because Grafeo's CPU BFS and
//! PageRank follow outgoing edges only.
//!
//! Every test needs a CUDA device and is ignored by default. Run them with
//! `cargo test -p grafeo-cuda-algos -- --ignored` on a machine with a GPU.

use std::collections::{BTreeSet, HashMap};

use grafeo_adapters::plugins::algorithms;
use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::BatchEdge;

struct Db(pub GrafeoDB);

impl Db {
    fn new() -> Self {
        Self(GrafeoDB::new_in_memory())
    }

    fn create_node(&self, labels: &[&str]) -> NodeId {
        self.0.create_node(labels).unwrap()
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        self.0
            .batch_create_edges(vec![BatchEdge::new(src, dst, edge_type)])
            .unwrap()[0]
    }

    fn delete_node_edges(&self, id: NodeId) {
        self.0
            .session()
            .execute_with_params(
                "MATCH (n)-[r]-() WHERE id(n) = $id DELETE r",
                HashMap::from([("id".to_string(), Value::from(id.0 as i64))]),
            )
            .unwrap();
    }

    fn delete_node(&self, id: NodeId) -> bool {
        let result = self
            .0
            .session()
            .execute_with_params(
                "MATCH (n) WHERE id(n) = $id RETURN count(n) AS c",
                HashMap::from([("id".to_string(), Value::from(id.0 as i64))]),
            )
            .unwrap();
        let existed = matches!(
            result.into_rows().first().and_then(|row| row.first()),
            Some(Value::Int64(1))
        );
        if existed {
            self.0
                .session()
                .execute_with_params(
                    "MATCH (n) WHERE id(n) = $id DETACH DELETE n",
                    HashMap::from([("id".to_string(), Value::from(id.0 as i64))]),
                )
                .unwrap();
        }
        existed
    }

    fn node_ids(&self) -> Vec<NodeId> {
        let snap = self.0.snapshot().unwrap();
        snap.store().node_ids(snap.view()).unwrap()
    }

    fn snap(&self) -> grafeo_engine::Snapshot {
        self.0.snapshot().unwrap()
    }

    fn export(&self, symmetric: bool) -> CsrArrays {
        let snap = self.snap();
        export_to_csr(snap.store(), snap.view(), symmetric).unwrap()
    }
}
use proptest::prelude::*;

use grafeo_cuda_algos::{bfs, k_truss, pagerank, triangle_counting};
use grafeo_cuda_core::{CsrArrays, GpuDevice, GpuGraph, export_to_csr, from_edge_pairs_oriented};

// ---- graph builders ----

/// Creates `n` nodes and stores every edge in both directions (a self-loop once).
fn undirected(n: usize, edges: &[(usize, usize)]) -> Db {
    let store = Db::new();
    let ids: Vec<_> = (0..n).map(|_| store.create_node(&["Node"])).collect();
    for &(a, b) in edges {
        store.create_edge(ids[a], ids[b], "EDGE");
        if a != b {
            store.create_edge(ids[b], ids[a], "EDGE");
        }
    }
    store
}

fn complete_edges(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect()
}

fn create_complete(n: usize) -> Db {
    undirected(n, &complete_edges(n))
}

fn create_path(n: usize) -> Db {
    let edges: Vec<_> = (1..n).map(|i| (i - 1, i)).collect();
    undirected(n, &edges)
}

/// Named small graphs with known structure.
fn catalog() -> Vec<(&'static str, Db)> {
    let petersen = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 7),
        (7, 9),
        (9, 6),
        (6, 8),
        (8, 5),
    ];
    let star: Vec<_> = (1..9).map(|leaf| (0, leaf)).collect();
    let wheel: Vec<_> = (1..=6).flat_map(|i| [(0, i), (i, i % 6 + 1)]).collect();
    let mut two_k4 = complete_edges(4);
    two_k4.extend(complete_edges(4).into_iter().map(|(a, b)| (a + 4, b + 4)));
    two_k4.push((3, 4));
    let mut k4_with_tail = complete_edges(4);
    k4_with_tail.extend([(3, 4), (4, 5)]);

    vec![
        ("single node", undirected(1, &[])),
        ("edgeless", undirected(4, &[])),
        ("K4", create_complete(4)),
        ("K5", create_complete(5)),
        ("K7", create_complete(7)),
        ("path", create_path(10)),
        ("petersen", undirected(10, &petersen)),
        ("star", undirected(9, &star)),
        ("wheel", undirected(7, &wheel)),
        (
            "diamond",
            undirected(4, &[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]),
        ),
        (
            "bowtie",
            undirected(5, &[(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)]),
        ),
        (
            "triangle, edge and isolated nodes",
            undirected(7, &[(0, 1), (1, 2), (0, 2), (3, 4)]),
        ),
        ("two K4 joined by a bridge", undirected(8, &two_k4)),
        ("K4 with a tail", undirected(6, &k4_with_tail)),
    ]
}

/// A graph exercising the export edge cases: self-loops, parallel edges and a
/// deleted node. Node 2 is deleted, so the live nodes are 0, 1, 3, 4, 5, 6
/// (CSR indices 0..6). With `detach` false its edges stay behind in the store.
fn messy(self_loops: bool, parallel_edges: bool, detach: bool) -> Db {
    let store = undirected(
        7,
        &[
            (0, 1),
            (1, 3),
            (0, 3),
            (3, 4),
            (4, 5),
            (5, 3),
            (1, 2),
            (2, 6),
            (6, 0),
            (5, 6),
            (4, 6),
        ],
    );
    let ids = store.node_ids();
    if self_loops {
        store.create_edge(ids[0], ids[0], "LOOP");
        store.create_edge(ids[4], ids[4], "LOOP");
    }
    if parallel_edges {
        store.create_edge(ids[0], ids[1], "AGAIN");
        store.create_edge(ids[4], ids[3], "AGAIN");
        store.create_edge(ids[5], ids[6], "AGAIN");
        store.create_edge(ids[6], ids[5], "AGAIN");
    }
    if detach {
        store.delete_node_edges(ids[2]);
    }
    assert!(store.delete_node(ids[2]));
    store
}

// ---- GPU plumbing ----

fn device() -> GpuDevice {
    GpuDevice::new(0).expect("no CUDA device found")
}

fn upload(store: &Db, device: &GpuDevice) -> (CsrArrays, GpuGraph) {
    let csr = store.export(true);
    let gpu = GpuGraph::from_csr_arrays(csr.clone(), device).expect("GPU transfer failed");
    (csr, gpu)
}

/// Maps each node id (from `store.node_ids()`) to its CSR index.
fn index_of<Id: std::hash::Hash + Eq>(ids: Vec<Id>) -> HashMap<Id, u32> {
    ids.into_iter()
        .enumerate()
        .map(|(i, id)| (id, i as u32))
        .collect()
}

// ---- comparisons ----

fn check_triangles(name: &str, cpu_store: &Db, csr: &CsrArrays, gpu: &GpuGraph) {
    let snap = cpu_store.snap();
    let cpu = algorithms::total_triangles(snap.store(), snap.view()).unwrap();
    let edge_parallel = triangle_counting::triangle_count(gpu).unwrap();
    assert_eq!(edge_parallel.total_triangles, cpu, "{name}: triangle_count");

    // Same graph through the degree-oriented builder and kernel.
    let mut pairs = Vec::new();
    for u in 0..csr.num_nodes {
        let range = csr.offsets[u as usize] as usize..csr.offsets[u as usize + 1] as usize;
        pairs.extend(
            csr.targets[range]
                .iter()
                .filter(|&&v| v > u)
                .map(|&v| (u, v)),
        );
    }
    let oriented = from_edge_pairs_oriented(&pairs, csr.num_nodes);
    let oriented_gpu = GpuGraph::from_csr_arrays(oriented, &device()).unwrap();
    let oriented_result = triangle_counting::triangle_count_oriented(&oriented_gpu).unwrap();
    assert_eq!(
        oriented_result.total_triangles, cpu,
        "{name}: triangle_count_oriented"
    );
}

fn check_bfs(name: &str, cpu_store: &Db, gpu: &GpuGraph, source: u32) {
    let ids = cpu_store.node_ids();
    let index = index_of(ids.clone());
    let snap = cpu_store.snap();
    let layers = algorithms::bfs_layers(snap.store(), snap.view(), ids[source as usize]).unwrap();
    let mut expected = vec![-1i32; ids.len()];
    for (depth, layer) in layers.iter().enumerate() {
        for id in layer {
            expected[index[id] as usize] = depth as i32;
        }
    }

    let result = bfs::bfs(gpu, source).unwrap();
    assert_eq!(result.distances, expected, "{name}: BFS from {source}");
    let reachable = expected.iter().filter(|&&d| d >= 0).count() as u32;
    assert_eq!(
        result.reachable, reachable,
        "{name}: reachable from {source}"
    );
    assert_eq!(
        result.max_depth,
        layers.len() as i32 - 1,
        "{name}: max depth from {source}"
    );
}

fn check_pagerank(name: &str, cpu_store: &Db, gpu: &GpuGraph) {
    let ids = cpu_store.node_ids();
    let snap = cpu_store.snap();
    let cpu = algorithms::pagerank(snap.store(), snap.view(), 0.85, 1000, 1e-12, true).unwrap();
    let result = pagerank::pagerank(gpu, 0.85, 1000, 1e-7).unwrap();
    assert_eq!(result.ranks.len(), ids.len(), "{name}: rank count");
    for (i, id) in ids.iter().enumerate() {
        let (gpu_rank, cpu_rank) = (f64::from(result.ranks[i]), cpu[id]);
        assert!(
            (gpu_rank - cpu_rank).abs() < 1e-4,
            "{name}: node {i} GPU rank {gpu_rank} vs CPU rank {cpu_rank}"
        );
    }
    if !ids.is_empty() {
        let total: f64 = result.ranks.iter().map(|&r| f64::from(r)).sum();
        assert!((total - 1.0).abs() < 1e-3, "{name}: ranks sum to {total}");
    }
}

fn check_k_truss(name: &str, cpu_store: &Db, gpu: &GpuGraph) {
    let index = index_of(cpu_store.node_ids());
    for k in 3..=8u32 {
        let snap = cpu_store.snap();
        let cpu: BTreeSet<(u32, u32)> = algorithms::k_truss(snap.store(), snap.view(), k as usize)
            .unwrap()
            .into_iter()
            .map(|(a, b)| {
                let (u, v) = (index[&a], index[&b]);
                (u.min(v), u.max(v))
            })
            .collect();
        let result = k_truss::k_truss(gpu, k).unwrap();
        let gpu_edges: BTreeSet<(u32, u32)> = result.edges.iter().copied().collect();
        assert_eq!(
            result.edges.len(),
            gpu_edges.len(),
            "{name}: duplicate edges"
        );
        assert_eq!(gpu_edges, cpu, "{name}: {k}-truss");
    }
}

// ---- existing spot checks ----

#[test]
#[ignore = "needs a CUDA device"]
fn test_triangles_k4_cpu_vs_gpu() {
    let store = create_complete(4);
    let (_, gpu) = upload(&store, &device());
    let snap = store.snap();
    let cpu_count = algorithms::total_triangles(snap.store(), snap.view()).unwrap();
    let gpu_result = triangle_counting::triangle_count(&gpu).unwrap();
    assert_eq!(cpu_count, 4);
    assert_eq!(gpu_result.total_triangles, cpu_count);
}

#[test]
#[ignore = "needs a CUDA device"]
fn test_triangles_k5_cpu_vs_gpu() {
    let store = create_complete(5);
    let (_, gpu) = upload(&store, &device());
    let snap = store.snap();
    let cpu_count = algorithms::total_triangles(snap.store(), snap.view()).unwrap();
    let gpu_result = triangle_counting::triangle_count(&gpu).unwrap();
    assert_eq!(cpu_count, 10);
    assert_eq!(gpu_result.total_triangles, cpu_count);
}

#[test]
#[ignore = "needs a CUDA device"]
fn test_bfs_path_cpu_vs_gpu() {
    let store = create_path(6);
    let (_, gpu) = upload(&store, &device());
    let gpu_result = bfs::bfs(&gpu, 0).unwrap();
    assert_eq!(gpu_result.max_depth, 5);
    assert_eq!(gpu_result.reachable, 6);
    check_bfs("path", &store, &gpu, 0);
}

#[test]
#[ignore = "needs a CUDA device"]
fn test_pagerank_k4_cpu_vs_gpu() {
    let store = create_complete(4);
    let (_, gpu) = upload(&store, &device());
    let gpu_result = pagerank::pagerank(&gpu, 0.85, 100, 1e-6).unwrap();
    assert!(gpu_result.converged, "GPU PageRank should converge on K4");
    check_pagerank("K4", &store, &gpu);
}

#[test]
#[ignore = "needs a CUDA device"]
fn test_ktruss_k5_cpu_vs_gpu() {
    let store = create_complete(5);
    let (_, gpu) = upload(&store, &device());
    assert_eq!(k_truss::k_truss(&gpu, 5).unwrap().edges.len(), 10);
    assert_eq!(
        k_truss::k_truss(&gpu, 6).unwrap().edges,
        Vec::<(u32, u32)>::new()
    );
    check_k_truss("K5", &store, &gpu);
}

// ---- catalog sweeps ----

#[test]
#[ignore = "needs a CUDA device"]
fn triangles_match_cpu_on_catalog() {
    let device = device();
    for (name, store) in catalog() {
        let (csr, gpu) = upload(&store, &device);
        check_triangles(name, &store, &csr, &gpu);
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn bfs_matches_cpu_on_catalog_from_every_source() {
    let device = device();
    for (name, store) in catalog() {
        let (csr, gpu) = upload(&store, &device);
        for source in 0..csr.num_nodes {
            check_bfs(name, &store, &gpu, source);
        }
    }
}

/// Regression: the GPU kernel dropped the rank of dangling (here: isolated)
/// nodes, so ranks no longer summed to 1 and disagreed with Grafeo's CPU
/// PageRank, which spreads that rank over all nodes.
#[test]
#[ignore = "needs a CUDA device"]
fn pagerank_matches_cpu_on_catalog() {
    let device = device();
    for (name, store) in catalog() {
        let (_, gpu) = upload(&store, &device);
        check_pagerank(name, &store, &gpu);
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn k_truss_matches_cpu_on_catalog() {
    let device = device();
    for (name, store) in catalog() {
        let (_, gpu) = upload(&store, &device);
        check_k_truss(name, &store, &gpu);
    }
}

// ---- export edge cases end to end ----

/// GPU results on a graph with self-loops, parallel edges and a node deleted
/// without its edges, against Grafeo's CPU algorithms on the same graph. The
/// PageRank reference has no parallel edges: the CSR export counts them once,
/// Grafeo's directed CPU PageRank weighs them by multiplicity.
#[test]
#[ignore = "needs a CUDA device"]
fn messy_graph_matches_cpu() {
    let device = device();
    let store = messy(true, true, false);
    let (csr, gpu) = upload(&store, &device);
    assert_eq!(csr.num_nodes, 6);
    let detached = messy(true, true, true).export(true);
    assert_eq!(
        (&csr.offsets, &csr.targets),
        (&detached.offsets, &detached.targets)
    );

    check_triangles("messy", &store, &csr, &gpu);
    for source in 0..csr.num_nodes {
        check_bfs("messy", &store, &gpu, source);
    }
    check_pagerank("messy", &messy(true, false, false), &gpu);
    check_k_truss("messy", &store, &gpu);
}

#[test]
#[ignore = "needs a CUDA device"]
fn grafeo_db_entry_point_matches_cpu() {
    let store = Db::new();
    let ids: Vec<_> = (0..5).map(|_| store.create_node(&["City"])).collect();
    for (a, b) in complete_edges(4) {
        store.create_edge(ids[a], ids[b], "ROAD");
        store.create_edge(ids[b], ids[a], "ROAD");
    }
    let gpu = GpuGraph::from_grafeo(&store.0, &device()).unwrap();
    assert_eq!(gpu.num_nodes, 5);
    assert_eq!(gpu.num_edges, 12);
    check_triangles("GrafeoDB", &store, &store.export(true), &gpu);
    check_bfs("GrafeoDB", &store, &gpu, 0);
    check_bfs("GrafeoDB", &store, &gpu, 4);
    check_pagerank("GrafeoDB", &store, &gpu);
    check_k_truss("GrafeoDB", &store, &gpu);
}

// ---- property tests ----

fn random_edges() -> impl Strategy<Value = (usize, Vec<(usize, usize)>)> {
    (1usize..16).prop_flat_map(|n| (Just(n), prop::collection::vec((0..n, 0..n), 0..48)))
}

/// Drops repeated pairs, keeps one copy of each self-loop.
fn deduplicated_edges(edges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    edges
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    #[ignore = "needs a CUDA device"]
    fn gpu_matches_cpu_on_random_graphs((n, edges) in random_edges()) {
        let device = device();
        let raw = undirected(n, &edges);
        let (csr, gpu) = upload(&raw, &device);

        check_triangles("random", &raw, &csr, &gpu);
        check_bfs("random", &raw, &gpu, 0);
        check_bfs("random", &raw, &gpu, (n - 1) as u32);
        check_pagerank("random", &undirected(n, &deduplicated_edges(&edges)), &gpu);
        check_k_truss("random", &raw, &gpu);
    }
}
