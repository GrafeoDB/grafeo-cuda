# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `validate_csr` and `CsrArrays::validate` check the CSR invariants the kernels rely on.
- `CudaError::InvalidCsr`, returned by `GpuGraph::from_csr_slices` and
  `GpuGraph::from_csr_arrays` for malformed input.
- Tests for the CSR bridge that need no GPU: empty graphs, isolated nodes,
  self-loops, parallel edges, edge direction, deleted nodes and edges, the
  edge-pair builders, the TSV parser and CSR validation, plus property tests
  against a reference model.
- GPU tests that compare every algorithm with Grafeo's CPU implementation on a
  catalog of small graphs, on graphs with self-loops, parallel edges and deleted
  nodes, and on random graphs. They are marked
  `#[ignore = "needs a CUDA device"]`; run them with
  `cargo test --workspace -- --include-ignored`.
- Committed PTX for every kernel in `crates/grafeo-cuda-algos/kernels/ptx/`
  (target `sm_70`, PTX ISA 8.5, generated with nvcc 12.6), with a
  `manifest.toml` recording the SHA-256 of each `.cu` and `.ptx` file, the
  nvcc version and the ISA version.
- `cargo xtask check-ptx` (no toolkit needed) and `cargo xtask regenerate-ptx`
  (needs nvcc) to check and rebuild the committed PTX.
- `GRAFEO_CUDA_BUILD_PTX=1` compiles the kernels with nvcc into the build
  directory for one build, to try kernel changes before regenerating.
- `grafeo_cuda_algos::PTX_ISA_VERSION`, the PTX ISA of the embedded kernels.
  Kernel load errors name it.

### Changed

- Grafeo comes from crates.io instead of a sibling `../grafeo` checkout.
  `grafeo-engine`, `grafeo-core` and `grafeo-adapters` are pinned to `=0.5.44`:
  they are implementation crates whose API may change in any Grafeo release, and
  the stable `grafeo` crate does not yet re-export the items this workspace uses
  (`Direction` and the CPU algorithms). `Cargo.toml` has a commented
  `[patch.crates-io]` section for building against a local Grafeo checkout.
- Removed unused dependencies: `grafeo-common` everywhere and `grafeo-engine` in
  the Python binding. `grafeo-core` is now a dev-dependency of
  `grafeo-cuda-algos`.
- `GpuGraph::from_grafeo` reads through `GrafeoDB::graph_store()`.
- The Python binding uses pyo3 0.29 (was 0.27), matching Grafeo. It no longer
  enables pyo3's deprecated `extension-module` feature: maturin 1.9.4 and newer
  (`pyproject.toml` requires 1.11.5) sets `PYO3_BUILD_EXTENSION_MODULE` itself.
- `export_to_csr` documents its index contract: CSR index `i` is
  `store.node_ids()[i]`.
- The CUDA kernels moved from `kernels/` at the repository root into
  `crates/grafeo-cuda-algos/kernels/`, so they ship with the crate.
- Building no longer needs the CUDA toolkit: the build embeds the committed PTX
  and fails with instructions if it does not match the `.cu` sources. It never
  writes into the source tree. At run time only the NVIDIA driver is needed,
  and it must support PTX ISA 8.5 (a CUDA 12.5 or newer driver: 555.42.02+ on
  Linux, 555.85+ on Windows).
- cudarc is built for a fixed CUDA version (`cuda-12060`) instead of asking nvcc,
  so builds do not depend on the build machine. `CUDARC_CUDA_VERSION` is no
  longer needed; setting it to another version makes cudarc fail to compile.
- CI runs on hosted runners without a GPU: formatting, the committed PTX check,
  Clippy for all targets, the CPU tests on Linux and Windows, packaging of the
  crates.io crates, and a check with the minimum supported Rust version
  (1.91.1). The self-hosted GPU runner runs the GPU tests against the committed
  PTX, then regenerates it with its own nvcc and runs them again.
- The release workflow publishes with `cargo publish --workspace`, which
  verifies each crate from its packaged files, instead of
  `--no-verify` and ignoring errors. The Python binding is `publish = false`
  (it is released through PyPI).
- The PyPI workflow checks the committed PTX before building, instead of
  needing nvcc on the runners.

### Removed

- The placeholder PTX that the build wrote when nvcc was missing. Builds without
  nvcc now embed the committed PTX.

### Fixed

- BFS wrote past the end of its one-entry initial frontier buffer from the
  second level on, overwriting device memory and returning wrong distances on
  graphs with wide levels.
- PageRank dropped the rank held by dangling nodes (nodes without outgoing
  edges, such as isolated nodes), so ranks did not sum to 1 and differed from
  Grafeo's CPU PageRank.
- `GpuGraph::from_grafeo` panicked for databases created with
  `GrafeoDB::with_store` or `GrafeoDB::with_read_store`.
- Triangle counting, PageRank and k-truss failed with
  `CUDA_ERROR_INVALID_VALUE` on graphs without nodes or edges. They now return
  empty results.
- Loaded kernels were cached once per process instead of once per CUDA context,
  so a second GPU reused function handles that belong to the first.
- A PTX module that failed to load panicked instead of returning an error.
- The warp-per-edge triangle counting launches computed `edges * 32` without an
  overflow check. Graphs with more than 134,217,727 forward (or oriented) edges
  now get an error instead of a wrong count.
- Malformed CSR passed to `GpuGraph::from_csr_slices` (also reachable from
  Python through `GpuGraph.from_csr`) made the kernels read outside their
  buffers.
- CI had failed since April 2026 because it checked out Grafeo outside the
  workspace. The PyPI and release workflows had the same problem.
- `grafeo-cuda-algos` could not be built from its crates.io package or from the
  Python sdist, because its kernels and their PTX lived outside the crate
  directory. Wheels built without nvcc carried placeholder kernels.
- The build script wrote compiled PTX into the source tree, so builds with and
  without nvcc overwrote each other's kernels.

[Unreleased]: https://github.com/GrafeoDB/grafeo-cuda/compare/v0.1.0...HEAD
