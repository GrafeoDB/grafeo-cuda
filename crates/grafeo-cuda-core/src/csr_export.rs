use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;

use crate::error::CudaError;

/// CSR (Compressed Sparse Row) arrays ready for GPU transfer.
///
/// Exported from a Grafeo `GraphStore`. Node IDs are remapped to contiguous
/// 0-based u32 indices.
///
/// Every builder in this module produces a CSR that passes [`validate_csr`]:
/// offsets start at 0 and never decrease, every target is a valid node index,
/// and each neighbor list is strictly increasing (sorted, no duplicates).
#[derive(Debug, Clone)]
pub struct CsrArrays {
    /// Row pointer array: `offsets[i]` is the start index in `targets` for node `i`.
    /// Length: `num_nodes + 1`. `offsets[num_nodes]` == `targets.len()`.
    pub offsets: Vec<u32>,

    /// Column index array: concatenated sorted neighbor lists.
    /// Length: `num_edges` (directed) or `2 * num_edges` (undirected/symmetric).
    pub targets: Vec<u32>,

    /// Number of nodes (contiguous 0..num_nodes).
    pub num_nodes: u32,

    /// Number of entries in `targets` (may be > logical edge count for symmetric CSR).
    pub num_edges: u32,
}

impl CsrArrays {
    /// Checks the structural invariants the GPU kernels rely on.
    ///
    /// Equivalent to [`validate_csr`] on `offsets` and `targets`, plus a check
    /// that `num_nodes` and `num_edges` agree with the array lengths.
    pub fn validate(&self) -> Result<(), CudaError> {
        validate_csr(&self.offsets, &self.targets)?;
        let nodes = self.offsets.len() - 1;
        if self.num_nodes as usize != nodes {
            return Err(CudaError::InvalidCsr(format!(
                "num_nodes is {} but offsets describe {nodes} nodes",
                self.num_nodes
            )));
        }
        if self.num_edges as usize != self.targets.len() {
            return Err(CudaError::InvalidCsr(format!(
                "num_edges is {} but targets has {} entries",
                self.num_edges,
                self.targets.len()
            )));
        }
        Ok(())
    }
}

/// Checks that `offsets` and `targets` form a well-formed CSR graph.
///
/// The GPU kernels index both arrays without bounds checks, so a malformed CSR
/// would make them read (and for some kernels write) outside their buffers.
/// [`GpuGraph`](crate::GpuGraph) calls this before every transfer.
///
/// Requirements:
/// - `offsets` is non-empty and `offsets[0] == 0`
/// - `offsets` never decreases and `offsets[n] == targets.len()`
/// - both lengths fit in `u32`
/// - every target is a node index below `offsets.len() - 1`
/// - every neighbor list is strictly increasing (sorted, without duplicates),
///   which the merge and binary-search intersections in triangle counting and
///   k-truss depend on
pub fn validate_csr(offsets: &[u32], targets: &[u32]) -> Result<(), CudaError> {
    let Some((&first, _)) = offsets.split_first() else {
        return Err(CudaError::InvalidCsr(
            "offsets must have num_nodes + 1 entries, got an empty array".to_string(),
        ));
    };
    if u32::try_from(offsets.len()).is_err() || u32::try_from(targets.len()).is_err() {
        return Err(CudaError::InvalidCsr(format!(
            "graph too large for u32 indices: {} offsets, {} targets",
            offsets.len(),
            targets.len()
        )));
    }
    if first != 0 {
        return Err(CudaError::InvalidCsr(format!(
            "offsets[0] must be 0, got {first}"
        )));
    }
    let num_nodes = offsets.len() - 1;
    let last = offsets[num_nodes] as usize;
    if last != targets.len() {
        return Err(CudaError::InvalidCsr(format!(
            "offsets[{num_nodes}] is {last} but targets has {} entries",
            targets.len()
        )));
    }
    for node in 0..num_nodes {
        let (start, end) = (offsets[node], offsets[node + 1]);
        if end < start {
            return Err(CudaError::InvalidCsr(format!(
                "offsets decrease at node {node}: {start} then {end}"
            )));
        }
        let list = &targets[start as usize..end as usize];
        for (position, &target) in list.iter().enumerate() {
            if target as usize >= num_nodes {
                return Err(CudaError::InvalidCsr(format!(
                    "node {node} has neighbor {target}, but there are only {num_nodes} nodes"
                )));
            }
            if position > 0 && list[position - 1] >= target {
                return Err(CudaError::InvalidCsr(format!(
                    "neighbor list of node {node} is not strictly increasing                      ({} then {target})",
                    list[position - 1]
                )));
            }
        }
    }
    Ok(())
}

/// Export a Grafeo `GraphStore` to CSR arrays for GPU transfer.
///
/// Only live nodes take part: CSR index `i` is the node `store.node_ids()[i]`,
/// so results indexed by CSR position map back to Grafeo nodes through that
/// list. Edges that point at a deleted node are dropped.
///
/// Edges are read in the outgoing direction. If `symmetric` is false the CSR
/// holds each edge `u -> v` once, in the list of `u`. If `symmetric` is true
/// the graph is treated as undirected and each edge appears in the lists of
/// both endpoints. Either way, parallel edges collapse into one entry and a
/// self-loop appears once in its node's list.
///
/// Node and CSR entry counts must fit in `u32`.
pub fn export_to_csr(store: &dyn GraphStore, symmetric: bool) -> CsrArrays {
    let node_ids = store.node_ids();
    let num_nodes = node_ids.len();

    // Build mapping from NodeId -> contiguous u32 index
    let mut id_map = std::collections::HashMap::with_capacity(num_nodes);
    for (idx, &node_id) in node_ids.iter().enumerate() {
        id_map.insert(node_id, idx as u32);
    }

    // Collect adjacency lists per node (sorted)
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); num_nodes];

    for (i, &node_id) in node_ids.iter().enumerate() {
        for (neighbor, _edge_id) in store.edges_from(node_id, Direction::Outgoing) {
            if let Some(&j) = id_map.get(&neighbor) {
                adj[i].push(j);
                if symmetric {
                    adj[j as usize].push(i as u32);
                }
            }
        }
    }

    // Sort and deduplicate each adjacency list
    for list in &mut adj {
        list.sort_unstable();
        list.dedup();
    }

    // Build CSR offsets and targets
    let total_edges: usize = adj.iter().map(Vec::len).sum();
    let mut offsets = Vec::with_capacity(num_nodes + 1);
    let mut targets = Vec::with_capacity(total_edges);

    let mut offset = 0u32;
    for list in &adj {
        offsets.push(offset);
        targets.extend_from_slice(list);
        offset += list.len() as u32;
    }
    offsets.push(offset);

    CsrArrays {
        offsets,
        targets,
        num_nodes: num_nodes as u32,
        num_edges: total_edges as u32,
    }
}

/// Build CSR arrays directly from edge pairs, bypassing GraphStore entirely.
///
/// Input: raw `(u32, u32)` edge pairs (external IDs must already be 0-based contiguous).
/// If `symmetric` is true, both directions are included and deduplicated.
/// Adjacency lists are sorted for merge-based intersection.
pub fn from_edge_pairs(edges: &[(u32, u32)], num_nodes: u32, symmetric: bool) -> CsrArrays {
    // Collect all edges (optionally symmetric)
    let capacity = if symmetric {
        edges.len() * 2
    } else {
        edges.len()
    };
    let mut all_edges: Vec<(u32, u32)> = Vec::with_capacity(capacity);
    for &(src, dst) in edges {
        all_edges.push((src, dst));
        if symmetric && src != dst {
            all_edges.push((dst, src));
        }
    }

    // Sort by source, then target (enables sorted adjacency lists)
    all_edges.sort_unstable();
    all_edges.dedup();

    // Build CSR via prefix scan
    let n = num_nodes as usize;
    let mut offsets = vec![0u32; n + 1];
    for &(src, _) in &all_edges {
        offsets[src as usize + 1] += 1;
    }
    for i in 1..=n {
        offsets[i] += offsets[i - 1];
    }

    let targets: Vec<u32> = all_edges.iter().map(|&(_, dst)| dst).collect();

    CsrArrays {
        offsets,
        targets,
        num_nodes,
        num_edges: all_edges.len() as u32,
    }
}

/// Parse a TSV edge list file into raw edge pairs with contiguous 0-based IDs.
///
/// Returns `(edge_pairs, num_nodes)`. IDs are assigned in order of first
/// appearance. Skips blank lines, comment lines starting with '#' or '%', and
/// lines whose first two fields are not unsigned integers.
pub fn parse_tsv_to_edges(path: &std::path::Path) -> std::io::Result<(Vec<(u32, u32)>, u32)> {
    use std::collections::HashMap;
    use std::io::BufRead;

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let mut id_map: HashMap<u64, u32> = HashMap::new();
    let mut next_id = 0u32;
    let mut edges = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let src: u64 = match parts.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let dst: u64 = match parts.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };

        let src_id = *id_map.entry(src).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        let dst_id = *id_map.entry(dst).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        edges.push((src_id, dst_id));
    }

    Ok((edges, next_id))
}

/// Build a degree-oriented CSR from edge pairs.
///
/// Each undirected edge is stored once, directed from the lower-degree endpoint
/// to the higher-degree endpoint (ties broken by vertex ID). This produces a DAG
/// where the maximum out-degree is O(√m), dramatically reducing intersection work
/// for triangle counting.
///
/// The resulting CSR has roughly half the entries of a symmetric CSR.
/// Each triangle in the graph is enumerable exactly once: for edge u→v,
/// any w ∈ N_out(u) ∩ N_out(v) forms a unique triangle (u, v, w).
pub fn from_edge_pairs_oriented(edges: &[(u32, u32)], num_nodes: u32) -> CsrArrays {
    let n = num_nodes as usize;

    // Step 1: Build undirected edge set as canonical (min, max) pairs.
    let mut undirected: Vec<(u32, u32)> = Vec::with_capacity(edges.len());
    for &(src, dst) in edges {
        if src != dst {
            undirected.push((src.min(dst), src.max(dst)));
        }
    }
    undirected.sort_unstable();
    undirected.dedup();

    // Step 2: Compute undirected degree of each node.
    let mut degree = vec![0u32; n];
    for &(u, v) in &undirected {
        degree[u as usize] += 1;
        degree[v as usize] += 1;
    }

    // Step 3: Orient each edge from lower-degree to higher-degree vertex.
    let mut oriented: Vec<(u32, u32)> = Vec::with_capacity(undirected.len());
    for &(u, v) in &undirected {
        let du = degree[u as usize];
        let dv = degree[v as usize];
        if du < dv || (du == dv && u < v) {
            oriented.push((u, v));
        } else {
            oriented.push((v, u));
        }
    }

    // Step 4: Sort by source and build CSR.
    oriented.sort_unstable();

    let mut offsets = vec![0u32; n + 1];
    for &(src, _) in &oriented {
        offsets[src as usize + 1] += 1;
    }
    for i in 1..=n {
        offsets[i] += offsets[i - 1];
    }

    let targets: Vec<u32> = oriented.iter().map(|&(_, dst)| dst).collect();

    CsrArrays {
        offsets,
        targets,
        num_nodes,
        num_edges: oriented.len() as u32,
    }
}
