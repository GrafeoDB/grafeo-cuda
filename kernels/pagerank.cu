/**
 * PageRank via iterative SpMV on CSR graph.
 *
 * Three kernels:
 * 1. compute_out_degrees_kernel: compute out-degree of each node from CSR offsets.
 * 2. pagerank_iteration_kernel: one PageRank iteration (pull-based).
 *    rank_new[v] = (1-d)/N + d * sum(rank[u] / deg[u]) for in-neighbors u of v.
 * 3. pagerank_diff_kernel: compute L1 norm of rank difference on-device,
 *    avoiding per-iteration D2H transfers for convergence checking.
 *
 * For undirected (symmetric) CSR, in-neighbors == out-neighbors.
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

extern "C" __global__ void pagerank_iteration_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ targets,
    const float* __restrict__ ranks_in,
    float* __restrict__ ranks_out,
    const unsigned int* __restrict__ out_degrees,
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

    ranks_out[v] = (1.0f - damping) / (float)num_nodes + damping * sum;
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
