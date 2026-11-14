//! PTX bookkeeping shared by `build.rs` and `cargo xtask` (included with `#[path]`).
//!
//! The PTX committed in `kernels/ptx/` is generated from `kernels/*.cu` by
//! `cargo xtask regenerate-ptx`. `kernels/ptx/manifest.toml` records the
//! SHA-256 of every source and PTX file, the nvcc version and the PTX ISA
//! version, so stale or hand-edited PTX is caught without a GPU or a CUDA
//! toolkit. Hashes are taken after normalizing CRLF line endings to LF, so a
//! Windows checkout with `core.autocrlf = true` hashes the same as Linux.

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

/// Environment variable that makes `build.rs` compile the kernels with nvcc
/// (into `OUT_DIR`) instead of using the committed PTX. Only the value `1`
/// enables it.
pub const BUILD_ENV: &str = "GRAFEO_CUDA_BUILD_PTX";

/// GPU architecture the PTX targets. The driver JIT-compiles `sm_70` PTX for
/// every newer GPU as well.
pub const TARGET_ARCH: &str = "sm_70";

/// Directory below `kernels/` holding the committed PTX and the manifest.
pub const PTX_DIR: &str = "ptx";

/// Manifest file name inside [`PTX_DIR`].
pub const MANIFEST_FILE: &str = "manifest.toml";

/// One kernel source and the PTX generated from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelEntry {
    /// File stem shared by `kernels/<name>.cu` and `kernels/ptx/<name>.ptx`.
    pub name: String,
    /// SHA-256 of the normalized `.cu` source.
    pub source_sha256: String,
    /// SHA-256 of the normalized `.ptx` file.
    pub ptx_sha256: String,
}

/// Contents of `kernels/ptx/manifest.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// nvcc version that generated the PTX, for example `12.6.85`.
    pub nvcc: String,
    /// PTX ISA version from the `.version` directive, for example `8.5`.
    pub ptx_isa: String,
    /// Target architecture from the `.target` directive.
    pub target: String,
    /// One entry per kernel, sorted by name.
    pub kernels: Vec<KernelEntry>,
}

/// Replaces every CRLF with LF.
pub fn normalize_newlines(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut iter = bytes.iter().peekable();
    while let Some(&byte) = iter.next() {
        if byte == b'\r' && iter.peek() == Some(&&b'\n') {
            continue;
        }
        out.push(byte);
    }
    out
}

/// Lowercase hex SHA-256 of `bytes` after [`normalize_newlines`].
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(normalize_newlines(bytes));
    digest
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            use fmt::Write as _;
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// Lists the `.cu` files in `kernels_dir` as `(stem, path)`, sorted by stem.
pub fn kernel_sources(kernels_dir: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let entries = std::fs::read_dir(kernels_dir)
        .map_err(|e| format!("cannot read {}: {e}", kernels_dir.display()))?;
    let mut sources = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| format!("cannot read {}: {e}", kernels_dir.display()))?
            .path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("kernel file name is not UTF-8: {}", path.display()))?
            .to_string();
        sources.push((stem, path));
    }
    sources.sort();
    if sources.is_empty() {
        return Err(format!("no .cu files in {}", kernels_dir.display()));
    }
    Ok(sources)
}

/// Value of the first PTX directive `name` (for example `.version 8.5`).
fn directive<'a>(ptx: &'a str, name: &str) -> Option<&'a str> {
    ptx.lines().find_map(|line| {
        let rest = line.trim().strip_prefix(name)?;
        rest.starts_with(char::is_whitespace)
            .then(|| rest.split_whitespace().next())
            .flatten()
    })
}

/// PTX ISA version from the `.version` directive.
pub fn ptx_isa(ptx: &str) -> Option<&str> {
    directive(ptx, ".version")
}

/// Target architecture from the `.target` directive.
pub fn ptx_target(ptx: &str) -> Option<&str> {
    directive(ptx, ".target").map(|t| t.trim_end_matches(','))
}

impl Manifest {
    /// Parses the manifest format written by `cargo xtask regenerate-ptx`: a
    /// strict subset of TOML with top-level `nvcc`, `ptx_isa` and `target`
    /// strings and one `[kernels.<name>]` table per kernel.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut nvcc = None;
        let mut ptx_isa = None;
        let mut target = None;
        let mut kernels: Vec<(String, Option<String>, Option<String>)> = Vec::new();

        for (number, raw) in text.lines().enumerate() {
            let line = raw.trim();
            let at = |message: String| format!("{MANIFEST_FILE} line {}: {message}", number + 1);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(section) = line.strip_prefix('[') {
                let name = section
                    .strip_suffix(']')
                    .and_then(|s| s.strip_prefix("kernels."))
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| at(format!("expected [kernels.<name>], got {line}")))?;
                if kernels.iter().any(|(n, _, _)| n == name) {
                    return Err(at(format!("kernel {name} listed twice")));
                }
                kernels.push((name.to_string(), None, None));
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| at(format!("expected key = \"value\", got {line}")))?;
            let (key, value) = (key.trim(), value.trim());
            let value = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .filter(|v| !v.contains('"'))
                .ok_or_else(|| at(format!("value of {key} must be a quoted string")))?
                .to_string();
            let slot = match (kernels.last_mut(), key) {
                (None, "nvcc") => &mut nvcc,
                (None, "ptx_isa") => &mut ptx_isa,
                (None, "target") => &mut target,
                (Some((_, source, _)), "source_sha256") => source,
                (Some((_, _, ptx)), "ptx_sha256") => ptx,
                _ => return Err(at(format!("unexpected key {key}"))),
            };
            if slot.replace(value).is_some() {
                return Err(at(format!("{key} set twice")));
            }
        }

        let missing = |key: &str| format!("{MANIFEST_FILE}: missing {key}");
        let kernels = kernels
            .into_iter()
            .map(|(name, source, ptx)| {
                Ok(KernelEntry {
                    source_sha256: source
                        .ok_or_else(|| missing(&format!("kernels.{name}.source_sha256")))?,
                    ptx_sha256: ptx
                        .ok_or_else(|| missing(&format!("kernels.{name}.ptx_sha256")))?,
                    name,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            nvcc: nvcc.ok_or_else(|| missing("nvcc"))?,
            ptx_isa: ptx_isa.ok_or_else(|| missing("ptx_isa"))?,
            target: target.ok_or_else(|| missing("target"))?,
            kernels,
        })
    }
}

/// Checks the committed PTX in `kernels_dir/ptx/` against the `.cu` sources.
///
/// Returns the manifest if every source has an entry with a matching hash,
/// every PTX file exists and matches its hash and directives, and the manifest
/// lists no kernel without a source. Otherwise returns every problem found.
pub fn verify(kernels_dir: &Path) -> Result<Manifest, Vec<String>> {
    let ptx_dir = kernels_dir.join(PTX_DIR);
    let manifest_path = ptx_dir.join(MANIFEST_FILE);
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| vec![format!("cannot read {}: {e}", manifest_path.display())])?;
    let manifest = Manifest::parse(&text).map_err(|e| vec![e])?;
    let sources = kernel_sources(kernels_dir).map_err(|e| vec![e])?;

    let mut problems = Vec::new();
    if manifest.target != TARGET_ARCH {
        problems.push(format!(
            "manifest target is {}, but kernels are built for {TARGET_ARCH}",
            manifest.target
        ));
    }
    if manifest.nvcc.is_empty() {
        problems.push("manifest does not record the nvcc version".to_string());
    }
    for (name, source) in &sources {
        let Some(entry) = manifest.kernels.iter().find(|k| &k.name == name) else {
            problems.push(format!("{name}.cu has no entry in {MANIFEST_FILE}"));
            continue;
        };
        match std::fs::read(source) {
            Ok(bytes) if sha256_hex(&bytes) == entry.source_sha256 => {}
            Ok(_) => problems.push(format!("{name}.cu changed since {name}.ptx was generated")),
            Err(e) => problems.push(format!("cannot read {}: {e}", source.display())),
        }
        let ptx_path = ptx_dir.join(format!("{name}.ptx"));
        let bytes = match std::fs::read(&ptx_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                problems.push(format!("cannot read {}: {e}", ptx_path.display()));
                continue;
            }
        };
        if sha256_hex(&bytes) != entry.ptx_sha256 {
            problems.push(format!(
                "{name}.ptx does not match {MANIFEST_FILE} (edited by hand?)"
            ));
        }
        let ptx = String::from_utf8_lossy(&bytes);
        if ptx_isa(&ptx) != Some(manifest.ptx_isa.as_str()) {
            problems.push(format!(
                "{name}.ptx declares PTX ISA {:?}, manifest says {}",
                ptx_isa(&ptx),
                manifest.ptx_isa
            ));
        }
        if ptx_target(&ptx) != Some(manifest.target.as_str()) {
            problems.push(format!(
                "{name}.ptx targets {:?}, manifest says {}",
                ptx_target(&ptx),
                manifest.target
            ));
        }
    }
    for entry in &manifest.kernels {
        if !sources.iter().any(|(name, _)| name == &entry.name) {
            problems.push(format!(
                "{MANIFEST_FILE} lists {0}, but {0}.cu does not exist",
                entry.name
            ));
        }
    }
    if problems.is_empty() {
        Ok(manifest)
    } else {
        Err(problems)
    }
}

/// An `nvcc` command, with the MSVC host compiler located on Windows when
/// `cl.exe` is not already on PATH.
pub fn nvcc_command() -> Command {
    let mut cmd = Command::new("nvcc");
    if cfg!(target_os = "windows")
        && Command::new("cl").arg("/?").output().is_err()
        && let Some(cl_dir) = find_msvc_cl_dir()
    {
        cmd.arg(format!("-ccbin={}", cl_dir.display()));
    }
    cmd
}

/// Compiles one kernel source to PTX at `output` with the flags used for the
/// committed PTX.
pub fn compile(source: &Path, output: &Path) -> Result<(), String> {
    let result = nvcc_command()
        .args(["--ptx", "-O3", &format!("-arch={TARGET_ARCH}"), "-o"])
        .arg(output)
        .arg(source)
        .output()
        .map_err(|e| {
            format!("cannot run nvcc ({e}); install the CUDA toolkit and put nvcc on PATH")
        })?;
    if !result.status.success() {
        return Err(format!(
            "nvcc failed to compile {}:\n{}{}",
            source.display(),
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    Ok(())
}

/// Searches the Visual Studio 2022 install locations for `cl.exe`.
fn find_msvc_cl_dir() -> Option<PathBuf> {
    let candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
    ];
    candidates.iter().find_map(|base| {
        let mut versions: Vec<PathBuf> = std::fs::read_dir(base)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        versions.sort();
        let cl_dir = versions.last()?.join("bin").join("Hostx64").join("x64");
        cl_dir.join("cl.exe").exists().then_some(cl_dir)
    })
}
