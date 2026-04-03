//! Shared GPU benchmark utilities: graph loading, timing, result output.
//!
//! Mirrors the GraphChallenge harness format for comparable results.

use std::path::Path;
use std::time::Instant;

use grafeo_cuda_core::{GpuDevice, GpuGraph, export_to_csr};
use grafeo_engine::GrafeoDB;
use serde::Serialize;

/// Result of a GPU benchmark run, serialized to JSON.
#[derive(Debug, Serialize)]
pub struct GpuBenchmarkResult {
    pub dataset: String,
    pub algorithm: String,
    pub vertices: u32,
    pub edges: u32,
    /// Best total time (transfer + kernel) across runs (seconds).
    pub time_s: f64,
    /// CSR export + GPU transfer time (seconds).
    pub transfer_time_s: f64,
    /// Best kernel-only time across runs (seconds).
    pub kernel_time_s: f64,
    /// Traversed edges per second (edges / best_kernel_time).
    pub teps: f64,
    /// All kernel-only run times (seconds).
    pub runs: Vec<f64>,
    /// Platform description including GPU.
    pub platform: String,
    /// GPU device name.
    pub gpu_device: String,
    /// Algorithm-specific extra data.
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub extra: serde_json::Value,
}

impl GpuBenchmarkResult {
    /// Save result as JSON to `results/raw/gpu_<algorithm>_<dataset_stem>.json`.
    pub fn save(&self) -> std::io::Result<std::path::PathBuf> {
        let dir = Path::new("results/raw");
        std::fs::create_dir_all(dir)?;
        let stem = Path::new(&self.dataset)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let path = dir.join(format!("gpu_{}_{stem}.json", self.algorithm));
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(path)
    }
}

/// Load a TSV edge list into Grafeo and transfer to GPU.
///
/// Returns the GrafeoDB (for CPU comparison), the GpuGraph, and the transfer time.
pub fn load_and_transfer(dataset: &str, device: &GpuDevice) -> (GrafeoDB, GpuGraph, f64) {
    let db = GrafeoDB::new_in_memory();
    let start = Instant::now();
    let (nodes, edges) = db
        .import_tsv(dataset, "E", true)
        .expect("failed to import TSV");
    let load_time = start.elapsed().as_secs_f64();
    eprintln!("  Loaded {nodes} nodes, {edges} edges in {load_time:.2}s from {dataset}");

    let transfer_start = Instant::now();
    let store = db.store();
    let csr = export_to_csr(&**store, true);
    let gpu_graph =
        GpuGraph::from_csr_arrays(csr, device).expect("Failed to transfer graph to GPU");
    let transfer_time = transfer_start.elapsed().as_secs_f64();
    eprintln!(
        "  Transferred to GPU: {} vertices, {} CSR entries in {transfer_time:.4}s",
        gpu_graph.num_nodes, gpu_graph.num_edges
    );

    (db, gpu_graph, transfer_time)
}

pub fn platform_info(device: &GpuDevice) -> String {
    format!(
        "{} / {} / {} / GPU: {}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        hostname(),
        device.properties().name,
    )
}

fn hostname() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".into())
}
