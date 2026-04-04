/**
 * Triangle counting via sorted neighbor intersection.
 *
 * Two kernels:
 * 1. triangle_count_kernel: baseline node-parallel (one thread per node).
 *    Works well for uniform-degree graphs.
 *
 * 2. triangle_count_warp_kernel: edge-parallel with warp-cooperative intersection.
 *    Assigns one warp (32 threads) per edge (u,v) where u < v, then the warp
 *    cooperatively intersects the sorted neighbor lists of u and v using a
 *    parallel merge-path approach. Much better for skewed-degree graphs.
 *
 * Input: CSR graph (offsets, targets) with sorted adjacency lists.
 * Output: single u64 counter (atomically incremented).
 */

/* ---- Baseline: node-parallel ---- */

extern "C" __global__ void triangle_count_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    unsigned int num_nodes,
    unsigned long long* __restrict__ result
) {
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;

    unsigned int u_start = offsets[u];
    unsigned int u_end = offsets[u + 1];
    unsigned long long local_count = 0;

    for (unsigned int i = u_start; i < u_end; i++) {
        unsigned int v = targets[i];
        if (v <= u) continue;

        unsigned int v_start = offsets[v];
        unsigned int v_end = offsets[v + 1];

        unsigned int j = i + 1;
        unsigned int k = v_start;

        while (j < u_end && k < v_end) {
            unsigned int nu = targets[j];
            unsigned int nv = targets[k];
            if (nu == nv) {
                local_count++;
                j++;
                k++;
            } else if (nu < nv) {
                j++;
            } else {
                k++;
            }
        }
    }

    if (local_count > 0) {
        atomicAdd(result, local_count);
    }
}

/* ---- Optimized: edge-parallel with warp-cooperative intersection ---- */

/**
 * Binary search: find first index in targets[lo..hi) where targets[idx] >= val.
 */
__device__ unsigned int lower_bound(
    const unsigned int* __restrict__ targets,
    unsigned int lo,
    unsigned int hi,
    unsigned int val
) {
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo) / 2;
        if (targets[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/**
 * Warp-cooperative sorted set intersection.
 *
 * Each of the 32 threads in the warp processes a subset of elements from
 * the shorter list, binary-searching into the longer list.
 * The warp then reduces partial counts via __shfl_down_sync.
 */
__device__ unsigned int warp_intersect(
    const unsigned int* __restrict__ a, unsigned int a_len,
    const unsigned int* __restrict__ b, unsigned int b_len,
    unsigned int lane
) {
    /* Ensure a is the shorter list for better work distribution */
    if (a_len > b_len) {
        const unsigned int* tmp_p = a; a = b; b = tmp_p;
        unsigned int tmp_l = a_len; a_len = b_len; b_len = tmp_l;
    }

    unsigned int count = 0;

    /* Each lane handles elements lane, lane+32, lane+64, ... of list a */
    for (unsigned int i = lane; i < a_len; i += 32) {
        unsigned int val = a[i];
        /* Binary search for val in b */
        unsigned int lo = 0, hi = b_len;
        while (lo < hi) {
            unsigned int mid = lo + (hi - lo) / 2;
            if (b[mid] < val) lo = mid + 1;
            else hi = mid;
        }
        if (lo < b_len && b[lo] == val) {
            count++;
        }
    }

    /* Warp reduction */
    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }

    return count;  /* Only lane 0 has the final sum */
}

/**
 * Edge-parallel triangle counting with warp-cooperative intersection.
 *
 * Each warp processes one edge (u,v) where u < v.
 * The warp cooperatively intersects N(u) and N(v) to count common neighbors w > v.
 * Total edges to process = num_edges / 2 (only u < v direction in symmetric CSR).
 *
 * This kernel must be launched with a block size that is a multiple of 32.
 */
extern "C" __global__ void triangle_count_warp_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    unsigned int num_nodes,
    unsigned int num_edges,
    unsigned long long* __restrict__ result
) {
    unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    unsigned int lane = threadIdx.x & 31;

    /* Each warp processes edges by iterating over nodes and their forward neighbors */
    /* We use a flat edge index: warp_id maps to the warp_id-th edge (u,v) with u < v */

    /* To find the edge, we do a virtual mapping: iterate over all edges and pick
       those where u < v. Since the CSR is sorted, for each node u we count only
       neighbors v > u. We linearize this as: edge_idx = offset into the "upper triangle".

       For simplicity and correctness, we iterate: each warp picks node u and then
       iterates over u's forward neighbors assigned by warp stride. */

    /* Node-based warp assignment: each warp takes one node u */
    unsigned int u = warp_id;
    if (u >= num_nodes) return;

    unsigned int u_start = offsets[u];
    unsigned int u_end = offsets[u + 1];
    unsigned long long local_count = 0;

    for (unsigned int i = u_start; i < u_end; i++) {
        unsigned int v = targets[i];
        if (v <= u) continue;  /* Only process u < v */

        unsigned int v_start = offsets[v];
        unsigned int v_end = offsets[v + 1];

        /* Restrict both lists to elements > v to only count w > v */
        unsigned int u_from = lower_bound(targets, i + 1, u_end, v + 1);
        unsigned int v_from = lower_bound(targets, v_start, v_end, v + 1);

        unsigned int a_len = u_end - u_from;
        unsigned int b_len = v_end - v_from;

        if (a_len == 0 || b_len == 0) continue;

        unsigned int count = warp_intersect(
            targets + u_from, a_len,
            targets + v_from, b_len,
            lane
        );

        if (lane == 0 && count > 0) {
            local_count += count;
        }
    }

    if (lane == 0 && local_count > 0) {
        atomicAdd(result, local_count);
    }
}

/* ---- Phase 4: Edge-parallel with prefix-sum mapping ---- */

/**
 * Compute the number of forward edges (v > u) per node.
 * fwd_count[u] = number of neighbors v of u where v > u.
 */
extern "C" __global__ void compute_forward_counts_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    unsigned int num_nodes,
    unsigned int* __restrict__ fwd_count
) {
    unsigned int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;

    unsigned int start = offsets[u];
    unsigned int end = offsets[u + 1];
    unsigned int count = 0;

    for (unsigned int i = start; i < end; i++) {
        if (targets[i] > u) count++;
    }
    fwd_count[u] = count;
}

/**
 * Inclusive prefix sum (scan) on fwd_count to produce fwd_offsets.
 * fwd_offsets[0] = 0, fwd_offsets[u] = sum of fwd_count[0..u).
 * Simple sequential kernel — called once on num_nodes elements.
 */
extern "C" __global__ void prefix_sum_kernel(
    const unsigned int* __restrict__ counts,
    unsigned int* __restrict__ offsets_out,
    unsigned int n
) {
    /* Single-thread prefix sum. Fast enough for n < 10M. */
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int running = 0;
    for (unsigned int i = 0; i < n; i++) {
        offsets_out[i] = running;
        running += counts[i];
    }
    offsets_out[n] = running;
}

/**
 * Edge-parallel triangle counting: one warp per forward edge (u, v) where u < v.
 *
 * Uses fwd_offsets[] to map global warp_id -> (u, edge_index_within_u).
 * Binary search in fwd_offsets to find u, then linear scan to find v.
 * Warp cooperatively intersects N(u) ∩ N(v) for common neighbors w > v.
 */
extern "C" __global__ void triangle_count_edge_parallel_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    const unsigned int* __restrict__ fwd_offsets,
    unsigned int num_nodes,
    unsigned int total_fwd_edges,
    unsigned long long* __restrict__ result
) {
    unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    unsigned int lane = threadIdx.x & 31;

    if (warp_id >= total_fwd_edges) return;

    /* Binary search fwd_offsets to find which node u this edge belongs to */
    unsigned int lo = 0, hi = num_nodes;
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo) / 2;
        if (fwd_offsets[mid + 1] <= warp_id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    unsigned int u = lo;

    /* Find the local edge index within u's forward neighbors */
    unsigned int local_idx = warp_id - fwd_offsets[u];

    /* Walk u's neighbors to find the local_idx-th forward neighbor */
    unsigned int u_start = offsets[u];
    unsigned int u_end = offsets[u + 1];
    unsigned int fwd_seen = 0;
    unsigned int v = u; /* will be overwritten */
    unsigned int edge_pos = u_start;

    for (unsigned int i = u_start; i < u_end; i++) {
        if (targets[i] > u) {
            if (fwd_seen == local_idx) {
                v = targets[i];
                edge_pos = i;
                break;
            }
            fwd_seen++;
        }
    }

    if (v == u) return; /* safety: should not happen */

    /* Intersect N(u) and N(v) for w > v */
    unsigned int v_start = offsets[v];
    unsigned int v_end = offsets[v + 1];

    unsigned int u_from = lower_bound(targets, edge_pos + 1, u_end, v + 1);
    unsigned int v_from = lower_bound(targets, v_start, v_end, v + 1);

    unsigned int a_len = u_end - u_from;
    unsigned int b_len = v_end - v_from;

    if (a_len == 0 || b_len == 0) return;

    unsigned int count = warp_intersect(
        targets + u_from, a_len,
        targets + v_from, b_len,
        lane
    );

    if (lane == 0 && count > 0) {
        atomicAdd(result, (unsigned long long)count);
    }
}
