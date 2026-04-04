use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;

/// CSR (Compressed Sparse Row) arrays ready for GPU transfer.
///
/// Exported from a Grafeo `GraphStore`. Node IDs are remapped to contiguous
/// 0-based u32 indices.
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

/// Export a Grafeo `GraphStore` to CSR arrays for GPU transfer.
///
/// Remaps internal `NodeId` values to contiguous 0-based u32 indices.
/// If `symmetric` is true, each edge (u,v) is stored in both directions.
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
    let capacity = if symmetric { edges.len() * 2 } else { edges.len() };
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
/// Returns (edge_pairs, num_nodes, id_mapping).
/// Skips comment lines starting with '#' or '%'.
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

#[cfg(test)]
mod tests {
    // CSR export correctness is verified via the roundtrip and correctness
    // integration tests in grafeo-cuda-algos.
}
