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

#[cfg(test)]
mod tests {
    // CSR export correctness is verified via the roundtrip and correctness
    // integration tests in grafeo-cuda-algos.
}
