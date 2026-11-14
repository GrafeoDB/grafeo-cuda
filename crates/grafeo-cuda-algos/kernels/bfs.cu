/**
 * Breadth-First Search via iterative frontier expansion.
 *
 * Two kernels:
 * 1. bfs_frontier_kernel: baseline, scans ALL nodes every iteration.
 *    Simple but O(V) per iteration regardless of frontier size.
 *
 * 2. bfs_expand_kernel: optimized, only processes frontier nodes.
 *    O(frontier_size * avg_degree) per iteration.
 *    Combined with bfs_build_frontier_kernel to extract the next frontier.
 *
 * Input: CSR graph (offsets, targets), distances array, current depth.
 * Output: updated distances array, changed flag / next frontier.
 */

/* ---- Baseline: scan all nodes ---- */

extern "C" __global__ void bfs_frontier_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    int* __restrict__ distances,
    unsigned int num_nodes,
    int current_depth,
    unsigned int* __restrict__ changed
) {
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;

    if (distances[u] != current_depth) return;

    unsigned int start = offsets[u];
    unsigned int end = offsets[u + 1];

    for (unsigned int i = start; i < end; i++) {
        unsigned int v = targets[i];
        if (distances[v] == -1) {
            distances[v] = current_depth + 1;
            *changed = 1;
        }
    }
}

/* ---- Optimized: frontier-based expansion ---- */

/**
 * Expand frontier: for each node in the input frontier, mark unvisited
 * neighbors and add them to the output frontier via atomic append.
 *
 * frontier_in: array of node IDs in the current frontier
 * frontier_in_size: number of nodes in current frontier
 * frontier_out: output buffer for next frontier (pre-allocated to num_edges)
 * frontier_out_size: atomic counter for output frontier size
 */
extern "C" __global__ void bfs_expand_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    int* __restrict__ distances,
    const unsigned int* __restrict__ frontier_in,
    unsigned int frontier_in_size,
    unsigned int* __restrict__ frontier_out,
    unsigned int* __restrict__ frontier_out_size,
    int current_depth
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_in_size) return;

    unsigned int u = frontier_in[idx];
    unsigned int start = offsets[u];
    unsigned int end = offsets[u + 1];

    int next_depth = current_depth + 1;

    for (unsigned int i = start; i < end; i++) {
        unsigned int v = targets[i];
        /* atomicCAS: only the first thread to discover v wins */
        int old = atomicCAS(&distances[v], -1, next_depth);
        if (old == -1) {
            /* This thread discovered v: add to output frontier */
            unsigned int pos = atomicAdd(frontier_out_size, 1);
            frontier_out[pos] = v;
        }
    }
}
