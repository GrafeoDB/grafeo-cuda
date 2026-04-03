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
transferred to GPU memory via cudarc, and processed by CUDA kernels compiled
from PTX. Results are transferred back to host memory.

## Quick Start

### Prerequisites

- Rust 1.91.1+
- CUDA Toolkit 12.x (`nvcc` on PATH)
- NVIDIA GPU (RTX 20xx or newer)

### Build

```bash
cargo build --release
```

The `build.rs` script compiles CUDA kernels in `kernels/` to PTX via `nvcc`.
If `nvcc` is not available, placeholder PTX files are created and GPU features
will return runtime errors.

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
```

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
| `grafeo-cuda-algos` | Algorithm implementations + benchmark binaries |
| `grafeo-cuda-python` | Python bindings via PyO3 |

## Algorithms

| Algorithm | Status | Kernel | Benchmark Binary |
| --------- | ------ | ------ | ---------------- |
| Triangle Counting | Implemented | `kernels/triangle_counting.cu` | `gpu-triangles` |
| PageRank | Implemented | `kernels/pagerank.cu` | `gpu-pagerank` |
| BFS | Implemented | `kernels/bfs.cu` | `gpu-bfs` |
| K-Truss | Implemented | `kernels/k_truss.cu` | `gpu-ktruss` |

## License

Apache-2.0
