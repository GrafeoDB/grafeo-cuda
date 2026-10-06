# grafeo-cuda

GPU-accelerated graph algorithms for the [Grafeo](https://github.com/GrafeoDB/grafeo) graph database.

## Overview

This crate provides CUDA-accelerated implementations of graph algorithms
(triangle counting, BFS, PageRank, k-truss) that operate on Grafeo's graph
data. It lives in a separate repository to keep GPU dependencies out of the
core Grafeo project.

**Supported hardware**: NVIDIA GPUs with Compute Capability 7.0+ (RTX 20xx and newer).

## Architecture

```
Grafeo (LpgStore) -> CSR export -> GPU transfer -> CUDA kernel -> Results
```

Grafeo's `GraphStore` is exported to CSR (Compressed Sparse Row) format,
transferred to GPU memory via cudarc, and processed by CUDA kernels shipped as
PTX, which the NVIDIA driver compiles for the GPU at load time. Results are
transferred back to host memory.

## Quick Start

### Prerequisites

To build: Rust 1.91.1+. No CUDA toolkit is needed.

To run:

- an NVIDIA GPU with Compute Capability 7.0 or newer (RTX 20xx and newer)
- an NVIDIA driver that supports PTX ISA 8.5, which means a driver for CUDA
  12.5 or newer: 555.42.02+ on Linux, 555.85+ on Windows

Only the driver is needed at run time, not the CUDA toolkit.
`grafeo_cuda_algos::PTX_ISA_VERSION` reports the ISA of the embedded kernels.

To change the kernels: the CUDA Toolkit 12.x (`nvcc` on PATH; on Windows also
the MSVC build tools), see [CUDA kernels and PTX](#cuda-kernels-and-ptx).

### Grafeo dependency

Grafeo comes from crates.io. The workspace depends on the implementation crates
`grafeo-engine`, `grafeo-core` and `grafeo-adapters`, pinned to an exact version
(`=0.5.44`) because their API may change in any Grafeo release. The stable
`grafeo` crate does not yet re-export everything this workspace needs
(`Direction`, used to walk edges through `GraphStore`, and the CPU reference
algorithms used by the tests). Bump all pins together.

To build against a local Grafeo checkout instead, uncomment the
`[patch.crates-io]` section at the end of `Cargo.toml`. The patch only applies
when the checkout has the same version as the pins.

### Build

```bash
cargo build --release
```

The build embeds the PTX committed in `crates/grafeo-cuda-algos/kernels/ptx/`,
so every build, wheel and published crate carries the same kernels whatever
toolkit the build machine has. cudarc is built for a fixed CUDA version (12.6)
and loads the driver at run time. Leave `CUDARC_CUDA_VERSION` unset: a
different value makes cudarc fail to compile.

### CUDA kernels and PTX

The kernel sources are `crates/grafeo-cuda-algos/kernels/*.cu`. Next to them,
`kernels/ptx/` holds the PTX generated from them (target `sm_70`, which the
driver JIT-compiles for every newer GPU) and `manifest.toml`, which records the
SHA-256 of each source and PTX file, the nvcc version and the PTX ISA version.

- Every build checks the manifest and stops if a `.cu` file changed since its
  PTX was generated, or if a PTX file was edited by hand. CI runs the same
  check with `cargo xtask check-ptx`, which needs no toolkit.
- After changing a kernel, run `cargo xtask regenerate-ptx` (needs `nvcc`) and
  commit `kernels/ptx/` together with the `.cu` change.
- To try a kernel change without regenerating, build with
  `GRAFEO_CUDA_BUILD_PTX=1`: the kernels are then compiled with `nvcc` into the
  build directory for that build only. The source tree is never written by the
  build.

The committed PTX was generated with nvcc 12.6, which emits PTX ISA 8.5; that
sets the driver requirement above. Regenerating with a newer toolkit can raise
it, so check the `ptx_isa` line of the manifest before committing.

### Tests

```bash
cargo test --workspace                          # CPU tests; GPU tests are ignored
cargo test --workspace -- --include-ignored     # everything, needs a CUDA device
```

Tests that need a GPU are marked `#[ignore = "needs a CUDA device"]`. The GPU
correctness tests compare each algorithm with Grafeo's CPU implementation on the
same graph.

### Usage (Rust)

```rust
use grafeo_engine::GrafeoDB;
use grafeo_cuda_core::{GpuDevice, GpuGraph};
use grafeo_cuda_algos::{triangle_counting, pagerank, bfs};

let db = GrafeoDB::new_in_memory();
db.import_tsv("graph.tsv", "E", true).unwrap();

let device = GpuDevice::new(0)?;
let gpu = GpuGraph::from_grafeo(&db, &device)?;

let triangles = triangle_counting::triangle_count(&gpu)?;
let ranks = pagerank::pagerank(&gpu, 0.85, 20, 1e-6)?;
let distances = bfs::bfs(&gpu, 0)?;

// Results are indexed by CSR position: index i is node_ids()[i].
let node_ids = db.graph_store().node_ids();
```

`GpuGraph::from_grafeo` exports the graph as undirected (symmetric) CSR. Parallel
edges count once and edges to deleted nodes are dropped.

### Benchmark Binaries

```bash
# Triangle counting
cargo run --release --bin gpu-triangles -- --dataset graph.tsv --save

# PageRank
cargo run --release --bin gpu-pagerank -- --dataset graph.tsv --save

# BFS
cargo run --release --bin gpu-bfs -- --dataset graph.tsv --save

# K-Truss decomposition
cargo run --release --bin gpu-ktruss -- --dataset graph.tsv --k 4 --save
```

### Usage (Python)

```python
from grafeo_cuda import GpuDevice, GpuGraph

device = GpuDevice(0)
gpu = GpuGraph.from_csr(offsets, targets, device)

result = gpu.triangle_count()
print(f"Triangles: {result['total_triangles']}")
print(f"GPU time: {result['time_seconds']:.4f}s")
```

## Crate Structure

| Crate | Purpose |
| ----- | ------- |
| `grafeo-cuda-core` | GPU device management, CSR export, memory transfer |
| `grafeo-cuda-algos` | Algorithm implementations, CUDA kernels with their PTX, benchmark binaries |
| `grafeo-cuda-python` | Python bindings via PyO3 (published to PyPI) |
| `xtask` | Repository tasks: `cargo xtask check-ptx`, `cargo xtask regenerate-ptx` (not published) |

## Algorithms

| Algorithm | Status | Kernel | Benchmark Binary |
| --------- | ------ | ------ | ---------------- |
| Triangle Counting | Implemented | `triangle_counting.cu` | `gpu-triangles` |
| PageRank | Implemented | `pagerank.cu` | `gpu-pagerank` |
| BFS | Implemented | `bfs.cu` | `gpu-bfs` |
| K-Truss | Implemented | `k_truss.cu` | `gpu-ktruss` |

Kernel files are in `crates/grafeo-cuda-algos/kernels/`.

## License

Apache-2.0
