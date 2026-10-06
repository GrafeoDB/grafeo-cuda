//! Repository maintenance tasks: `cargo xtask <command>`.
//!
//! - `regenerate-ptx`: compile `crates/grafeo-cuda-algos/kernels/*.cu` with nvcc
//!   into `kernels/ptx/` and rewrite `kernels/ptx/manifest.toml`. Needs the CUDA
//!   toolkit (and on Windows the MSVC build tools), but no GPU.
//! - `check-ptx`: verify that the committed PTX matches the kernel sources.
//!   Needs neither a toolkit nor a GPU.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[path = "../../crates/grafeo-cuda-algos/build_support/ptx_common.rs"]
mod ptx_common;

use ptx_common::{KernelEntry, MANIFEST_FILE, Manifest, PTX_DIR};

const USAGE: &str = "usage: cargo xtask <regenerate-ptx | check-ptx>";

fn main() -> ExitCode {
    let command = std::env::args().nth(1);
    let result = match command.as_deref() {
        Some("regenerate-ptx") => regenerate_ptx(),
        Some("check-ptx") => check_ptx(),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn kernels_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("crates")
        .join("grafeo-cuda-algos")
        .join("kernels")
}

fn check_ptx() -> Result<(), String> {
    let dir = kernels_dir();
    match ptx_common::verify(&dir) {
        Ok(manifest) => {
            println!(
                "committed PTX matches {} kernel sources (nvcc {}, PTX ISA {}, {})",
                manifest.kernels.len(),
                manifest.nvcc,
                manifest.ptx_isa,
                manifest.target
            );
            Ok(())
        }
        Err(problems) => Err(format!(
            "the committed PTX does not match the kernel sources:\n  - {}\n\
             Run `cargo xtask regenerate-ptx` (needs nvcc) and commit kernels/ptx/. \
             To try a kernel change without regenerating, build with {}=1.",
            problems.join("\n  - "),
            ptx_common::BUILD_ENV
        )),
    }
}

fn regenerate_ptx() -> Result<(), String> {
    let dir = kernels_dir();
    let ptx_dir = dir.join(PTX_DIR);
    std::fs::create_dir_all(&ptx_dir)
        .map_err(|e| format!("cannot create {}: {e}", ptx_dir.display()))?;
    let nvcc = nvcc_version()?;
    let sources = ptx_common::kernel_sources(&dir)?;

    let mut kernels = Vec::with_capacity(sources.len());
    let mut isa: Option<String> = None;
    for (name, source) in &sources {
        let ptx_path = ptx_dir.join(format!("{name}.ptx"));
        ptx_common::compile(source, &ptx_path)?;
        // nvcc writes CRLF on Windows; commit LF everywhere.
        let ptx = ptx_common::normalize_newlines(&read(&ptx_path)?);
        std::fs::write(&ptx_path, &ptx)
            .map_err(|e| format!("cannot write {}: {e}", ptx_path.display()))?;
        let text = String::from_utf8_lossy(&ptx);
        let this_isa = ptx_common::ptx_isa(&text)
            .ok_or_else(|| format!("{name}.ptx has no .version directive"))?;
        if ptx_common::ptx_target(&text) != Some(ptx_common::TARGET_ARCH) {
            return Err(format!(
                "{name}.ptx does not target {}",
                ptx_common::TARGET_ARCH
            ));
        }
        match &isa {
            Some(previous) if previous != this_isa => {
                return Err(format!(
                    "kernels disagree on the PTX ISA version: {previous} and {this_isa}"
                ));
            }
            _ => isa = Some(this_isa.to_string()),
        }
        kernels.push(KernelEntry {
            name: name.clone(),
            source_sha256: ptx_common::sha256_hex(&read(source)?),
            ptx_sha256: ptx_common::sha256_hex(&ptx),
        });
        println!("compiled {name}.cu -> ptx/{name}.ptx");
    }

    // Drop PTX whose source is gone.
    for entry in std::fs::read_dir(&ptx_dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let is_ptx = path.extension().and_then(|e| e.to_str()) == Some("ptx");
        if is_ptx && !sources.iter().any(|(name, _)| name == stem) {
            std::fs::remove_file(&path).map_err(|e| e.to_string())?;
            println!("removed stale {}", path.display());
        }
    }

    let manifest = Manifest {
        nvcc,
        ptx_isa: isa.unwrap_or_default(),
        target: ptx_common::TARGET_ARCH.to_string(),
        kernels,
    };
    let manifest_path = ptx_dir.join(MANIFEST_FILE);
    std::fs::write(&manifest_path, render(&manifest))
        .map_err(|e| format!("cannot write {}: {e}", manifest_path.display()))?;
    println!(
        "wrote ptx/{MANIFEST_FILE} (nvcc {}, PTX ISA {})",
        manifest.nvcc, manifest.ptx_isa
    );
    check_ptx()
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

/// Returns the nvcc release, for example `12.6.85` from
/// "Cuda compilation tools, release 12.6, V12.6.85".
fn nvcc_version() -> Result<String, String> {
    let output = ptx_common::nvcc_command()
        .arg("--version")
        .output()
        .map_err(|e| format!("cannot run nvcc ({e}); install the CUDA toolkit"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_nvcc_version(&stdout)
        .map(str::to_string)
        .ok_or_else(|| format!("cannot find the release in `nvcc --version` output:\n{stdout}"))
}

fn parse_nvcc_version(output: &str) -> Option<&str> {
    output
        .lines()
        .find(|line| line.contains("release"))?
        .rsplit(", V")
        .next()
        .map(str::trim)
        .filter(|v| v.chars().next().is_some_and(|c| c.is_ascii_digit()))
}

/// Serializes the manifest in the format [`Manifest::parse`] reads.
fn render(manifest: &Manifest) -> String {
    let mut out = format!(
        "# Generated by `cargo xtask regenerate-ptx`. Do not edit by hand.\n\
         # Hashes are SHA-256 of the files with CRLF line endings normalized to LF.\n\
         nvcc = \"{}\"\n\
         ptx_isa = \"{}\"\n\
         target = \"{}\"\n",
        manifest.nvcc, manifest.ptx_isa, manifest.target
    );
    let mut kernels: Vec<&KernelEntry> = manifest.kernels.iter().collect();
    kernels.sort_by(|a, b| a.name.cmp(&b.name));
    for kernel in kernels {
        // Writing to a String cannot fail.
        let _ = write!(
            out,
            "\n[kernels.{}]\nsource_sha256 = \"{}\"\nptx_sha256 = \"{}\"\n",
            kernel.name, kernel.source_sha256, kernel.ptx_sha256
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Manifest {
        Manifest {
            nvcc: "12.6.85".into(),
            ptx_isa: "8.5".into(),
            target: "sm_70".into(),
            kernels: vec![
                KernelEntry {
                    name: "bfs".into(),
                    source_sha256: "a".repeat(64),
                    ptx_sha256: "b".repeat(64),
                },
                KernelEntry {
                    name: "pagerank".into(),
                    source_sha256: "c".repeat(64),
                    ptx_sha256: "d".repeat(64),
                },
            ],
        }
    }

    #[test]
    fn manifest_round_trips() {
        let manifest = sample();
        assert_eq!(Manifest::parse(&render(&manifest)), Ok(manifest));
    }

    #[test]
    fn manifest_parse_rejects_malformed_input() {
        let text = render(&sample());
        for (broken, reason) in [
            (text.replace("nvcc = ", "compiler = "), "unexpected key"),
            (text.replace("\"8.5\"", "8.5"), "quoted string"),
            (
                text.replace("[kernels.pagerank]", "[kernels.bfs]"),
                "listed twice",
            ),
            (text.replace("[kernels.bfs]", "[bfs]"), "expected [kernels."),
            (
                text.replace(&format!("ptx_sha256 = \"{}\"\n", "b".repeat(64)), ""),
                "kernels.bfs.ptx_sha256",
            ),
            (text.replace("target = \"sm_70\"\n", ""), "missing target"),
        ] {
            let error = Manifest::parse(&broken).unwrap_err();
            assert!(error.contains(reason), "{reason}: got {error}");
        }
    }

    #[test]
    fn hashes_ignore_crlf() {
        assert_eq!(
            ptx_common::sha256_hex(b"a\r\nb\n"),
            ptx_common::sha256_hex(b"a\nb\n")
        );
        assert_ne!(
            ptx_common::sha256_hex(b"a\rb"),
            ptx_common::sha256_hex(b"ab")
        );
        // SHA-256 of the empty input.
        assert_eq!(
            ptx_common::sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn ptx_directives_are_parsed() {
        let ptx = "//\n// Generated by NVIDIA NVVM Compiler\n\n.version 8.5\n.target sm_70\n.address_size 64\n";
        assert_eq!(ptx_common::ptx_isa(ptx), Some("8.5"));
        assert_eq!(ptx_common::ptx_target(ptx), Some("sm_70"));
        assert_eq!(ptx_common::ptx_isa(".versionx 1.0\n"), None);
    }

    #[test]
    fn nvcc_version_is_parsed() {
        let output = "nvcc: NVIDIA (R) Cuda compiler driver\n\
                      Copyright (c) 2005-2024 NVIDIA Corporation\n\
                      Built on Wed_Oct_30_01:18:48_Pacific_Daylight_Time_2024\n\
                      Cuda compilation tools, release 12.6, V12.6.85\n\
                      Build cuda_12.6.r12.6/compiler.35059454_0\n";
        assert_eq!(parse_nvcc_version(output), Some("12.6.85"));
        assert_eq!(parse_nvcc_version("no version here"), None);
    }

    /// The committed PTX must match the sources in this checkout.
    #[test]
    fn committed_ptx_is_current() {
        if let Err(problems) = ptx_common::verify(&kernels_dir()) {
            panic!("{}", problems.join("\n"));
        }
    }
}
