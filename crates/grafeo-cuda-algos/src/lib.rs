//! GPU-accelerated graph algorithms for Grafeo.
//!
//! Each module implements one algorithm, dispatching to a CUDA kernel loaded from
//! PTX embedded at build time. All algorithms operate on `GpuGraph` (CSR in GPU
//! memory).
//!
//! The PTX comes from `kernels/ptx/` in this crate, generated from
//! `kernels/*.cu` and checked against them on every build, so building needs no
//! CUDA toolkit. At run time only the NVIDIA driver is needed; it must support
//! [`PTX_ISA_VERSION`].

// Kernel launches require unsafe: the driver cannot verify argument correctness.
#![allow(unsafe_code)]

pub mod bfs;
pub mod harness;
pub mod k_truss;
mod launch;
mod module_cache;
pub mod pagerank;
pub mod triangle_counting;

/// PTX ISA version of the embedded kernels (their `.version` directive), for
/// example `"8.5"`. The NVIDIA driver JIT-compiles the PTX and must support
/// this ISA: 8.5 needs a driver for CUDA 12.5 or newer.
pub const PTX_ISA_VERSION: &str = env!("GRAFEO_CUDA_PTX_ISA");

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::PTX_ISA_VERSION;

    /// The kernels embedded in this build are real PTX for the reported ISA,
    /// and in a default build exactly the committed files in `kernels/ptx/`.
    #[test]
    fn embedded_ptx_is_the_committed_ptx() {
        let committed = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("ptx");
        for (name, ptx) in [
            ("bfs", crate::bfs::PTX),
            ("k_truss", crate::k_truss::PTX),
            ("pagerank", crate::pagerank::PTX),
            ("triangle_counting", crate::triangle_counting::PTX),
        ] {
            let lines: Vec<&str> = ptx.lines().map(str::trim_end).collect();
            assert!(
                lines.contains(&format!(".version {PTX_ISA_VERSION}").as_str()),
                "{name}: no .version {PTX_ISA_VERSION}"
            );
            assert!(lines.contains(&".target sm_70"), "{name}: not sm_70");
            assert!(ptx.contains(".entry"), "{name}: no kernels");
            if env!("GRAFEO_CUDA_PTX_ORIGIN") == "committed" {
                let file = std::fs::read_to_string(committed.join(format!("{name}.ptx"))).unwrap();
                // Compare line by line so CRLF checkouts compare equal.
                assert!(
                    ptx.lines().eq(file.lines()),
                    "{name}: embedded PTX differs from kernels/ptx/{name}.ptx"
                );
            }
        }
    }
}
