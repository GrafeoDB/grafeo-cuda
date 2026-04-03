//! Build script: compile CUDA kernels to PTX via nvcc.
//!
//! Looks for `.cu` files in `kernels/` (at workspace root) and compiles each to
//! `kernels/compiled/<name>.ptx`. If nvcc is not found, emits a warning
//! and creates empty placeholder PTX files (allows building on machines
//! without the CUDA toolkit, but GPU features will fail at runtime).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Workspace root is two levels up from this crate
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    let kernels_dir = workspace_root.join("kernels");
    let compiled_dir = kernels_dir.join("compiled");
    fs::create_dir_all(&compiled_dir).expect("failed to create kernels/compiled/");

    let cu_files: Vec<_> = fs::read_dir(&kernels_dir)
        .expect("failed to read kernels/")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("cu") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    let nvcc_available = Command::new("nvcc").arg("--version").output().is_ok();

    for cu_path in &cu_files {
        let stem = cu_path.file_stem().unwrap().to_str().unwrap();
        let ptx_path = compiled_dir.join(format!("{stem}.ptx"));

        // Rerun if kernel source changes
        println!("cargo:rerun-if-changed={}", cu_path.display());

        if nvcc_available {
            let mut cmd = Command::new("nvcc");
            cmd.args([
                "--ptx",
                "-O3",
                "-arch=sm_70", // Compute Capability 7.0 (RTX 20xx+)
                "-o",
                ptx_path.to_str().unwrap(),
                cu_path.to_str().unwrap(),
            ]);

            // On Windows, nvcc needs cl.exe in PATH. Auto-detect from VS Build Tools.
            if cfg!(target_os = "windows")
                && let Some(cl_dir) = find_msvc_cl_dir()
            {
                cmd.arg(format!("-ccbin={}", cl_dir.display()));
            }

            let status = cmd.status().expect("failed to run nvcc");
            assert!(
                status.success(),
                "nvcc failed to compile {}",
                cu_path.display()
            );
        } else {
            // Create placeholder PTX so include_str!() does not fail at compile time.
            // GPU functions will return errors at runtime.
            println!(
                "cargo:warning=nvcc not found; creating placeholder PTX for {}",
                cu_path.display()
            );
            fs::write(
                &ptx_path,
                format!("// Placeholder: nvcc was not available during build for {stem}.cu\n"),
            )
            .expect("failed to write placeholder PTX");
        }
    }
}

/// Search for the MSVC cl.exe directory (Windows only).
///
/// Checks VS 2022 Build Tools locations in both Program Files directories.
fn find_msvc_cl_dir() -> Option<PathBuf> {
    let candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
    ];

    for base in &candidates {
        let base_path = Path::new(base);
        if !base_path.exists() {
            continue;
        }
        // Pick the latest MSVC version directory
        if let Ok(entries) = fs::read_dir(base_path) {
            let mut versions: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir())
                .collect();
            versions.sort();
            if let Some(latest) = versions.last() {
                let cl_dir = latest.join("bin").join("Hostx64").join("x64");
                if cl_dir.join("cl.exe").exists() {
                    return Some(cl_dir);
                }
            }
        }
    }
    None
}
