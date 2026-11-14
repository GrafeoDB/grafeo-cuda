/**
 * K-truss decomposition via iterative edge peeling.
 *
 * Two kernels:
 * 1. edge_triangle_support_kernel: for each edge (u,v) in the CSR, count
 *    triangles containing it by intersecting sorted neighbor lists of u and v.
 *    Only counts w where (u,w) and (v,w) both exist and neither is removed.
 *
 * 2. edge_filter_kernel: mark edges with support < threshold as removed.
 *    Returns the number of edges removed in this iteration (via atomic counter).
 *
 * Host-side iteration: compute support, filter, repeat until no edges removed.
 * Removed edges are tracked via a boolean mask (0 = active, 1 = removed).
 */

/**
 * Binary search for value in sorted array[lo..hi).
 * Returns 1 if found, 0 if not.
 */
__device__ int binary_search(
    const unsigned int* __restrict__ arr,
    unsigned int lo,
    unsigned int hi,
    unsigned int value
) {
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo) / 2;
        if (arr[mid] == value) return 1;
        if (arr[mid] < value) lo = mid + 1;
        else hi = mid;
    }
    return 0;
}

/**
 * Compute triangle support for each edge.
 *
 * For edge at CSR position idx (connecting u to targets[idx]=v),
 * support[idx] = number of nodes w such that:
 *   - (u,w) exists and is not removed
 *   - (v,w) exists and is not removed
 *
 * Uses merge-intersection of sorted neighbor lists (same as triangle counting).
 * Skips removed edges in the intersection.
 */
extern "C" __global__ void edge_triangle_support_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    const unsigned char* __restrict__ removed,
    unsigned int num_nodes,
    unsigned int* __restrict__ support
) {
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;

    unsigned int u_start = offsets[u];
    unsigned int u_end = offsets[u + 1];

    for (unsigned int idx = u_start; idx < u_end; idx++) {
        if (removed[idx]) continue;

        unsigned int v = targets[idx];
        /* Only process one direction: u < v to avoid double-counting */
        if (v <= u) continue;

        unsigned int v_start = offsets[v];
        unsigned int v_end = offsets[v + 1];

        unsigned int count = 0;

        /* Merge-intersect sorted neighbor lists of u and v */
        unsigned int j = u_start;
        unsigned int k = v_start;

        while (j < u_end && k < v_end) {
            if (removed[j]) { j++; continue; }
            if (removed[k]) { k++; continue; }

            unsigned int nu = targets[j];
            unsigned int nv = targets[k];

            if (nu == nv) {
                count++;
                j++;
                k++;
            } else if (nu < nv) {
                j++;
            } else {
                k++;
            }
        }

        /* Write support for this edge (u->v direction) */
        support[idx] = count;

        /* Also write support for the reverse edge (v->u) via binary search */
        for (unsigned int r = v_start; r < v_end; r++) {
            if (targets[r] == u && !removed[r]) {
                support[r] = count;
                break;
            }
        }
    }
}

/**
 * Mark edges with support below threshold as removed.
 * Returns count of newly removed edges via atomic counter.
 */
extern "C" __global__ void edge_filter_kernel(
    const unsigned int* __restrict__ support,
    unsigned char* __restrict__ removed,
    unsigned int num_edges,
    unsigned int threshold,
    unsigned int* __restrict__ removed_count
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    if (!removed[idx] && support[idx] < threshold) {
        removed[idx] = 1;
        atomicAdd(removed_count, 1);
    }
}
