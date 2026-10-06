//! Build script: provide the PTX for the CUDA kernels in `kernels/`.
//!
//! By default the committed PTX in `kernels/ptx/` is used, so building needs no
//! CUDA toolkit and the result does not depend on the build machine. The build
//! stops if `kernels/ptx/manifest.toml` does not match the `.cu` sources: run
//! `cargo xtask regenerate-ptx` (needs nvcc) and commit `kernels/ptx/`.
//!
//! With `GRAFEO_CUDA_BUILD_PTX=1` the kernels are compiled with nvcc instead,
//! for this build only. Either way the PTX goes to `OUT_DIR/ptx/`; this script
//! never writes into the source tree.

use std::path::PathBuf;

#[path = "build_support/ptx_common.rs"]
mod ptx_common;

use ptx_common::BUILD_ENV;

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let kernels_dir = manifest_dir.join("kernels");
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("ptx");
    std::fs::create_dir_all(&out_dir).expect("cannot create OUT_DIR/ptx");

    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=build_support");
    println!("cargo:rerun-if-env-changed={BUILD_ENV}");

    let sources = ptx_common::kernel_sources(&kernels_dir).unwrap_or_else(|e| panic!("{e}"));
    let verified = ptx_common::verify(&kernels_dir);

    let compile = std::env::var(BUILD_ENV).is_ok_and(|v| v == "1");
    let ptx_isa = if compile {
        if let Err(problems) = &verified {
            for problem in problems {
                println!(
                    "cargo:warning=committed PTX is out of date ({problem}); \
                     run `cargo xtask regenerate-ptx` before committing"
                );
            }
        }
        let mut isa = String::new();
        for (name, source) in &sources {
            let target = out_dir.join(format!("{name}.ptx"));
            ptx_common::compile(source, &target)
                .unwrap_or_else(|e| panic!("{BUILD_ENV}=1 is set, but {e}"));
            let ptx = std::fs::read_to_string(&target).expect("cannot read compiled PTX");
            isa = ptx_common::ptx_isa(&ptx)
                .unwrap_or_else(|| panic!("{name}.ptx has no .version directive"))
                .to_string();
        }
        isa
    } else {
        let manifest = verified.unwrap_or_else(|problems| {
            panic!(
                "the committed PTX in kernels/ptx/ does not match the kernel sources:\n  - {}\n\
                 Regenerate it with `cargo xtask regenerate-ptx` (needs nvcc) and commit \
                 kernels/ptx/, or set {BUILD_ENV}=1 to compile the kernels with nvcc for \
                 this build only.",
                problems.join("\n  - ")
            )
        });
        for (name, _) in &sources {
            let file = format!("{name}.ptx");
            std::fs::copy(
                kernels_dir.join(ptx_common::PTX_DIR).join(&file),
                out_dir.join(&file),
            )
            .unwrap_or_else(|e| panic!("cannot copy {file} to OUT_DIR: {e}"));
        }
        manifest.ptx_isa
    };

    println!("cargo:rustc-env=GRAFEO_CUDA_PTX_ISA={ptx_isa}");
    let origin = if compile { "nvcc" } else { "committed" };
    println!("cargo:rustc-env=GRAFEO_CUDA_PTX_ORIGIN={origin}");
}
