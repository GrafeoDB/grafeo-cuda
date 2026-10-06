//! CPU-only tests for the CSR bridge: Grafeo `GraphStore` export, the edge-pair
//! builders, the TSV parser and CSR validation. None of these need a GPU.

use std::collections::{BTreeSet, HashMap};

use grafeo_core::LpgStore;
use grafeo_cuda_core::{
    CsrArrays, CudaError, export_to_csr, from_edge_pairs, from_edge_pairs_oriented,
    parse_tsv_to_edges, validate_csr,
};
use proptest::prelude::*;

// ---- helpers ----

/// Returns the neighbor list of `node` in `csr`.
fn neighbors(csr: &CsrArrays, node: u32) -> &[u32] {
    let start = csr.offsets[node as usize] as usize;
    let end = csr.offsets[node as usize + 1] as usize;
    &csr.targets[start..end]
}

/// Builds the expected `(offsets, targets)` from per-node neighbor sets.
fn csr_from_sets(sets: &[BTreeSet<u32>]) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32];
    let mut targets = Vec::new();
    for set in sets {
        targets.extend(set.iter().copied());
        offsets.push(targets.len() as u32);
    }
    (offsets, targets)
}

fn assert_csr_eq(csr: &CsrArrays, offsets: &[u32], targets: &[u32]) {
    csr.validate().expect("exported CSR must be valid");
    assert_eq!(csr.offsets, offsets, "offsets");
    assert_eq!(csr.targets, targets, "targets");
    assert_eq!(csr.num_nodes as usize, offsets.len() - 1, "num_nodes");
    assert_eq!(csr.num_edges as usize, targets.len(), "num_edges");
}

/// Asserts that a symmetric CSR really is symmetric.
fn assert_symmetric(csr: &CsrArrays) {
    for u in 0..csr.num_nodes {
        for &v in neighbors(csr, u) {
            assert!(
                neighbors(csr, v).contains(&u),
                "edge {u} -> {v} has no reverse entry"
            );
        }
    }
}

fn person_store(n: usize) -> LpgStore {
    let store = LpgStore::new().unwrap();
    for _ in 0..n {
        store.create_node(&["Person"]);
    }
    store
}

// ---- export_to_csr: edge cases ----

#[test]
fn empty_store_exports_single_offset() {
    let store = LpgStore::new().unwrap();
    for symmetric in [false, true] {
        let csr = export_to_csr(&store, symmetric);
        assert_csr_eq(&csr, &[0], &[]);
    }
}

#[test]
fn isolated_nodes_get_empty_lists() {
    let store = person_store(3);
    for symmetric in [false, true] {
        let csr = export_to_csr(&store, symmetric);
        assert_csr_eq(&csr, &[0, 0, 0, 0], &[]);
    }
}

#[test]
fn directed_export_keeps_only_outgoing_edges() {
    let store = LpgStore::new().unwrap();
    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    let vincent = store.create_node(&["Person"]);
    store.create_edge(alix, gus, "KNOWS");
    store.create_edge(vincent, gus, "KNOWS");

    let directed = export_to_csr(&store, false);
    assert_csr_eq(&directed, &[0, 1, 1, 2], &[1, 1]);

    let symmetric = export_to_csr(&store, true);
    assert_csr_eq(&symmetric, &[0, 1, 3, 4], &[1, 0, 2, 1]);
    assert_symmetric(&symmetric);
}

#[test]
fn self_loops_appear_once() {
    let store = LpgStore::new().unwrap();
    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    store.create_edge(alix, alix, "KNOWS");
    store.create_edge(alix, gus, "KNOWS");

    let directed = export_to_csr(&store, false);
    assert_csr_eq(&directed, &[0, 2, 2], &[0, 1]);

    // The symmetric pass visits the loop twice (as u -> v and v -> u); it must
    // still produce a single entry.
    let symmetric = export_to_csr(&store, true);
    assert_csr_eq(&symmetric, &[0, 2, 3], &[0, 1, 0]);
}

#[test]
fn parallel_edges_collapse() {
    let store = LpgStore::new().unwrap();
    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    store.create_edge(alix, gus, "KNOWS");
    store.create_edge(alix, gus, "LIKES");
    store.create_edge(gus, alix, "KNOWS");

    let directed = export_to_csr(&store, false);
    assert_csr_eq(&directed, &[0, 1, 2], &[1, 0]);

    let symmetric = export_to_csr(&store, true);
    assert_csr_eq(&symmetric, &[0, 1, 2], &[1, 0]);
}

/// Builds Alix -> Gus -> Vincent -> Jules and Alix -> Jules, then deletes Gus.
fn store_with_deleted_middle_node(detach: bool) -> LpgStore {
    let store = LpgStore::new().unwrap();
    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    let vincent = store.create_node(&["Person"]);
    let jules = store.create_node(&["Person"]);
    store.create_edge(alix, gus, "KNOWS");
    store.create_edge(gus, vincent, "KNOWS");
    store.create_edge(vincent, jules, "KNOWS");
    store.create_edge(alix, jules, "KNOWS");
    if detach {
        store.delete_node_edges(gus);
    }
    assert!(store.delete_node(gus));
    store
}

#[test]
fn deleted_node_leaves_no_index_gap() {
    for detach in [false, true] {
        let store = store_with_deleted_middle_node(detach);
        assert_eq!(store.node_ids().len(), 3);

        // Live nodes Alix, Vincent, Jules become 0, 1, 2. Edges touching Gus
        // disappear even when they were not deleted with him.
        let directed = export_to_csr(&store, false);
        assert_csr_eq(&directed, &[0, 1, 2, 2], &[2, 2]);

        let symmetric = export_to_csr(&store, true);
        assert_csr_eq(&symmetric, &[0, 1, 2, 4], &[2, 2, 0, 1]);
    }
}

#[test]
fn deleted_edges_are_skipped() {
    let store = LpgStore::new().unwrap();
    let amsterdam = store.create_node(&["City"]);
    let berlin = store.create_node(&["City"]);
    let paris = store.create_node(&["City"]);
    let gone = store.create_edge(amsterdam, berlin, "ROAD");
    store.create_edge(amsterdam, paris, "ROAD");
    // One of two parallel edges deleted: the other one keeps the pair connected.
    let first = store.create_edge(berlin, paris, "ROAD");
    store.create_edge(berlin, paris, "RAIL");
    assert!(store.delete_edge(gone));
    assert!(store.delete_edge(first));

    let directed = export_to_csr(&store, false);
    assert_csr_eq(&directed, &[0, 1, 2, 2], &[2, 2]);
}

#[test]
fn csr_index_follows_node_ids_order() {
    let store = person_store(5);
    let ids = store.node_ids();
    // Edge from the last node to the first: index 4 -> index 0.
    store.create_edge(ids[4], ids[0], "KNOWS");
    store.delete_node(ids[2]);

    let live = store.node_ids();
    let csr = export_to_csr(&store, false);
    assert_eq!(csr.num_nodes as usize, live.len());
    let from = live.iter().position(|&id| id == ids[4]).unwrap() as u32;
    let to = live.iter().position(|&id| id == ids[0]).unwrap() as u32;
    assert_eq!(neighbors(&csr, from), &[to]);
}

// ---- export_to_csr: property tests against a reference model ----

#[derive(Debug, Clone)]
struct GraphSpec {
    nodes: usize,
    /// `(source, target, deleted)`
    edges: Vec<(usize, usize, bool)>,
    /// `(deleted, detach_edges_first)` per node
    node_deletes: Vec<(bool, bool)>,
}

fn graph_spec() -> impl Strategy<Value = GraphSpec> {
    (1usize..12).prop_flat_map(|nodes| {
        (
            Just(nodes),
            prop::collection::vec((0..nodes, 0..nodes, prop::bool::weighted(0.2)), 0..40),
            prop::collection::vec((prop::bool::weighted(0.2), any::<bool>()), nodes),
        )
            .prop_map(|(nodes, edges, node_deletes)| GraphSpec {
                nodes,
                edges,
                node_deletes,
            })
    })
}

fn build_store(spec: &GraphSpec) -> LpgStore {
    let store = LpgStore::new().unwrap();
    let ids: Vec<_> = (0..spec.nodes)
        .map(|_| store.create_node(&["Person"]))
        .collect();
    let mut created = Vec::new();
    for &(src, dst, deleted) in &spec.edges {
        created.push((store.create_edge(ids[src], ids[dst], "KNOWS"), deleted));
    }
    for (edge, deleted) in created {
        if deleted {
            assert!(store.delete_edge(edge));
        }
    }
    for (node, &(deleted, detach)) in spec.node_deletes.iter().enumerate() {
        if deleted {
            if detach {
                store.delete_node_edges(ids[node]);
            }
            assert!(store.delete_node(ids[node]));
        }
    }
    store
}

/// Expected CSR computed from the spec alone: nodes in creation order minus the
/// deleted ones, edges whose endpoints both survive, set semantics per list.
fn expected_csr(spec: &GraphSpec, symmetric: bool) -> (Vec<u32>, Vec<u32>) {
    let mut index = HashMap::new();
    for (node, &(deleted, _)) in spec.node_deletes.iter().enumerate() {
        if !deleted {
            index.insert(node, index.len() as u32);
        }
    }
    let mut sets = vec![BTreeSet::new(); index.len()];
    for &(src, dst, deleted) in &spec.edges {
        if deleted {
            continue;
        }
        if let (Some(&u), Some(&v)) = (index.get(&src), index.get(&dst)) {
            sets[u as usize].insert(v);
            if symmetric {
                sets[v as usize].insert(u);
            }
        }
    }
    csr_from_sets(&sets)
}

proptest! {
    #[test]
    fn export_matches_reference_model(spec in graph_spec(), symmetric in any::<bool>()) {
        let store = build_store(&spec);
        let csr = export_to_csr(&store, symmetric);
        let (offsets, targets) = expected_csr(&spec, symmetric);
        prop_assert!(csr.validate().is_ok(), "{:?}", csr.validate());
        prop_assert_eq!(&csr.offsets, &offsets);
        prop_assert_eq!(&csr.targets, &targets);
        prop_assert_eq!(csr.num_nodes as usize, offsets.len() - 1);
        prop_assert_eq!(csr.num_edges as usize, targets.len());
        if symmetric {
            assert_symmetric(&csr);
        }
    }

    /// The direct edge-pair builder used by the benchmarks must agree with the
    /// Grafeo export on the same graph.
    #[test]
    fn edge_pairs_agree_with_store_export(
        nodes in 1u32..12,
        raw in prop::collection::vec((0u32..12, 0u32..12), 0..40),
        symmetric in any::<bool>(),
    ) {
        let edges: Vec<(u32, u32)> =
            raw.into_iter().map(|(s, d)| (s % nodes, d % nodes)).collect();
        let store = person_store(nodes as usize);
        let ids = store.node_ids();
        for &(s, d) in &edges {
            store.create_edge(ids[s as usize], ids[d as usize], "KNOWS");
        }
        let from_store = export_to_csr(&store, symmetric);
        let from_pairs = from_edge_pairs(&edges, nodes, symmetric);
        prop_assert!(from_pairs.validate().is_ok(), "{:?}", from_pairs.validate());
        prop_assert_eq!(from_store.offsets, from_pairs.offsets);
        prop_assert_eq!(from_store.targets, from_pairs.targets);
        prop_assert_eq!(from_store.num_edges, from_pairs.num_edges);
    }

    #[test]
    fn oriented_csr_stores_each_edge_once_and_counts_triangles(
        nodes in 1u32..14,
        raw in prop::collection::vec((0u32..14, 0u32..14), 0..60),
    ) {
        let edges: Vec<(u32, u32)> =
            raw.into_iter().map(|(s, d)| (s % nodes, d % nodes)).collect();
        let csr = from_edge_pairs_oriented(&edges, nodes);
        prop_assert!(csr.validate().is_ok(), "{:?}", csr.validate());

        // Undirected simple graph: no loops, no parallel edges.
        let undirected: BTreeSet<(u32, u32)> = edges
            .iter()
            .filter(|(s, d)| s != d)
            .map(|&(s, d)| (s.min(d), s.max(d)))
            .collect();
        prop_assert_eq!(csr.num_edges as usize, undirected.len());

        let mut degree = vec![0u32; nodes as usize];
        for &(u, v) in &undirected {
            degree[u as usize] += 1;
            degree[v as usize] += 1;
        }
        let mut seen = BTreeSet::new();
        for u in 0..nodes {
            for &v in neighbors(&csr, u) {
                let (du, dv) = (degree[u as usize], degree[v as usize]);
                prop_assert!(du < dv || (du == dv && u < v), "{u} -> {v} oriented wrongly");
                prop_assert!(seen.insert((u.min(v), u.max(v))), "edge {u}-{v} stored twice");
            }
        }
        prop_assert_eq!(&seen, &undirected);

        // Triangles via oriented intersection equal a brute-force count.
        let mut oriented_count = 0u64;
        for u in 0..nodes {
            for &v in neighbors(&csr, u) {
                let out_v: BTreeSet<u32> = neighbors(&csr, v).iter().copied().collect();
                oriented_count +=
                    neighbors(&csr, u).iter().filter(|w| out_v.contains(w)).count() as u64;
            }
        }
        let mut brute = 0u64;
        for a in 0..nodes {
            for b in (a + 1)..nodes {
                for c in (b + 1)..nodes {
                    if undirected.contains(&(a, b))
                        && undirected.contains(&(b, c))
                        && undirected.contains(&(a, c))
                    {
                        brute += 1;
                    }
                }
            }
        }
        prop_assert_eq!(oriented_count, brute);
    }
}

// ---- from_edge_pairs ----

#[test]
fn edge_pairs_keep_isolated_trailing_nodes() {
    let csr = from_edge_pairs(&[(0, 1)], 4, true);
    assert_csr_eq(&csr, &[0, 1, 2, 2, 2], &[1, 0]);
}

#[test]
fn edge_pairs_empty() {
    assert_csr_eq(&from_edge_pairs(&[], 0, true), &[0], &[]);
    assert_csr_eq(&from_edge_pairs_oriented(&[], 0), &[0], &[]);
    assert_csr_eq(&from_edge_pairs_oriented(&[(2, 2)], 3), &[0, 0, 0, 0], &[]);
}

// ---- parse_tsv_to_edges ----

struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(name: &str, contents: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("grafeo-cuda-{}-{name}.tsv", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn tsv_parser_remaps_ids_in_order_of_appearance() {
    let file = TempFile::new(
        "remap",
        "# comment\n% matrix-market style comment\n\n19\t88\t1.0\n88 3\nnot numbers\n19\n3\t19\n",
    );
    let (edges, nodes) = parse_tsv_to_edges(&file.0).unwrap();
    // 19 -> 0, 88 -> 1, 3 -> 2. The "19" line has no target and is skipped.
    assert_eq!(nodes, 3);
    assert_eq!(edges, vec![(0, 1), (1, 2), (2, 0)]);
}

#[test]
fn tsv_parser_handles_large_ids_and_empty_files() {
    let file = TempFile::new("large", "18446744073709551615 0\n");
    let (edges, nodes) = parse_tsv_to_edges(&file.0).unwrap();
    assert_eq!((edges, nodes), (vec![(0, 1)], 2));

    let empty = TempFile::new("empty", "# nothing here\n");
    let (edges, nodes) = parse_tsv_to_edges(&empty.0).unwrap();
    assert!(edges.is_empty());
    assert_eq!(nodes, 0);
}

#[test]
fn tsv_parser_reports_missing_file() {
    let missing = std::env::temp_dir().join("grafeo-cuda-definitely-missing.tsv");
    assert!(parse_tsv_to_edges(&missing).is_err());
}

// ---- validate_csr ----

fn invalid(offsets: &[u32], targets: &[u32]) -> String {
    match validate_csr(offsets, targets) {
        Err(CudaError::InvalidCsr(message)) => message,
        other => panic!("expected InvalidCsr for {offsets:?} / {targets:?}, got {other:?}"),
    }
}

#[test]
fn validate_accepts_well_formed_csr() {
    validate_csr(&[0], &[]).unwrap();
    validate_csr(&[0, 0, 0], &[]).unwrap();
    validate_csr(&[0, 3, 6, 9, 12], &[1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]).unwrap();
    // Self-loops are allowed.
    validate_csr(&[0, 2, 3], &[0, 1, 0]).unwrap();
}

#[test]
fn validate_rejects_malformed_csr() {
    assert!(invalid(&[], &[]).contains("empty"));
    assert!(invalid(&[1, 1], &[0]).contains("offsets[0]"));
    assert!(invalid(&[0, 2], &[1]).contains("offsets[1] is 2"));
    assert!(invalid(&[0, 2, 1, 2], &[1, 2]).contains("decrease"));
    assert!(invalid(&[0, 1, 1], &[2]).contains("only 2 nodes"));
    assert!(invalid(&[0, 2, 2], &[1, 0]).contains("not strictly increasing"));
    assert!(invalid(&[0, 2, 2], &[1, 1]).contains("not strictly increasing"));
}

#[test]
fn validate_checks_declared_counts() {
    let mut csr = from_edge_pairs(&[(0, 1)], 2, true);
    csr.validate().unwrap();
    csr.num_nodes = 3;
    assert!(matches!(csr.validate(), Err(CudaError::InvalidCsr(_))));
    csr.num_nodes = 2;
    csr.num_edges = 1;
    assert!(matches!(csr.validate(), Err(CudaError::InvalidCsr(_))));
}
