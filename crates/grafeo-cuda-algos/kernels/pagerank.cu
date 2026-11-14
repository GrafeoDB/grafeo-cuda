/**
 * PageRank via iterative SpMV on CSR graph.
 *
 * Four kernels:
 * 1. compute_out_degrees_kernel: compute out-degree of each node from CSR offsets.
 * 2. pagerank_dangling_sum_kernel: sum the rank held by dangling nodes
 *    (out-degree 0), which have no edges to pass it along.
 * 3. pagerank_iteration_kernel: one PageRank iteration (pull-based).
 *    rank_new[v] = (1-d)/N + d * (sum(rank[u] / deg[u]) + dangling/N)
 *    for in-neighbors u of v. Spreading the dangling rank over all nodes keeps
 *    the ranks summing to 1, as in Grafeo's CPU PageRank.
 * 4. pagerank_diff_kernel: compute L1 norm of rank difference on-device,
 *    avoiding per-iteration D2H transfers for convergence checking.
 *
 * For undirected (symmetric) CSR, in-neighbors == out-neighbors.
 * The reduction kernels assume a power-of-two block size.
 */

extern "C" __global__ void compute_out_degrees_kernel(
    const unsigned int* __restrict__ offsets,
    unsigned int num_nodes,
    unsigned int* __restrict__ out_degrees
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;

    out_degrees[v] = offsets[v + 1] - offsets[v];
}

/**
 * Sum of ranks_in[u] over dangling nodes u, accumulated into *dangling_sum.
 *
 * Shared memory block reduction, then atomicAdd to the global accumulator,
 * which must be zeroed before launch.
 */
extern "C" __global__ void pagerank_dangling_sum_kernel(
    const float* __restrict__ ranks_in,
    const unsigned int* __restrict__ out_degrees,
    unsigned int num_nodes,
    float* __restrict__ dangling_sum
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local = 0.0f;
    if (idx < num_nodes && out_degrees[idx] == 0) {
        local = ranks_in[idx];
    }

    sdata[tid] = local;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && sdata[0] != 0.0f) {
        atomicAdd(dangling_sum, sdata[0]);
    }
}

extern "C" __global__ void pagerank_iteration_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    const float* __restrict__ ranks_in,
    float* __restrict__ ranks_out,
    const unsigned int* __restrict__ out_degrees,
    const float* __restrict__ dangling_sum,
    unsigned int num_nodes,
    float damping
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;

    float sum = 0.0f;
    unsigned int start = offsets[v];
    unsigned int end = offsets[v + 1];

    for (unsigned int i = start; i < end; i++) {
        unsigned int u = targets[i];
        unsigned int deg = out_degrees[u];
        if (deg > 0) {
            sum += ranks_in[u] / (float)deg;
        }
    }

    float spread = *dangling_sum / (float)num_nodes;
    ranks_out[v] = (1.0f - damping) / (float)num_nodes + damping * (sum + spread);
}

/**
 * Compute L1 norm of difference between two rank vectors: sum(|a[i] - b[i]|).
 *
 * Uses shared memory block reduction, then atomicAdd to a global accumulator.
 * The accumulator must be zeroed before launch.
 */
extern "C" __global__ void pagerank_diff_kernel(
    const float* __restrict__ ranks_a,
    const float* __restrict__ ranks_b,
    unsigned int num_nodes,
    float* __restrict__ diff_out
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Each thread computes its local diff */
    float local_diff = 0.0f;
    if (idx < num_nodes) {
        local_diff = fabsf(ranks_a[idx] - ranks_b[idx]);
    }

    sdata[tid] = local_diff;
    __syncthreads();

    /* Block-level reduction in shared memory */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Block leader writes to global accumulator */
    if (tid == 0) {
        atomicAdd(diff_out, sdata[0]);
    }
}
