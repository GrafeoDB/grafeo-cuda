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
    let store = db.graph_store();
    let csr = export_to_csr(&*store, true);
    let gpu_graph =
        GpuGraph::from_csr_arrays(csr, device).expect("Failed to transfer graph to GPU");
    let transfer_time = transfer_start.elapsed().as_secs_f64();
    eprintln!(
        "  Transferred to GPU: {} vertices, {} CSR entries in {transfer_time:.4}s",
        gpu_graph.num_nodes, gpu_graph.num_edges
    );

    (db, gpu_graph, transfer_time)
}

/// Load a TSV edge list directly into GPU memory, bypassing GrafeoDB entirely.
///
/// Parses the TSV into edge pairs, builds CSR on the CPU, and transfers to GPU.
/// Much faster than going through GrafeoDB's LpgStore for benchmark-only workloads.
pub fn load_and_transfer_direct(dataset: &str, device: &GpuDevice) -> (GpuGraph, f64) {
    let start = Instant::now();

    let (edges, num_nodes) =
        grafeo_cuda_core::parse_tsv_to_edges(Path::new(dataset)).expect("failed to parse TSV");
    let parse_time = start.elapsed().as_secs_f64();
    eprintln!(
        "  Parsed {num_nodes} nodes, {} edges in {parse_time:.2}s from {dataset}",
        edges.len()
    );

    let csr_start = Instant::now();
    let csr = grafeo_cuda_core::from_edge_pairs(&edges, num_nodes, true);
    let csr_time = csr_start.elapsed().as_secs_f64();
    eprintln!(
        "  CSR build: {csr_time:.4}s ({} nodes, {} CSR entries)",
        csr.num_nodes, csr.num_edges
    );

    let xfer_start = Instant::now();
    let gpu_graph =
        GpuGraph::from_csr_arrays(csr, device).expect("failed to transfer graph to GPU");
    let xfer_time = xfer_start.elapsed().as_secs_f64();
    eprintln!("  GPU transfer: {xfer_time:.4}s");

    let total = start.elapsed().as_secs_f64();
    eprintln!(
        "  Total (parse+CSR+transfer): {total:.4}s  [{} vertices, {} CSR entries]",
        gpu_graph.num_nodes, gpu_graph.num_edges
    );

    (gpu_graph, total)
}

/// Load a TSV edge list and build a degree-oriented CSR on GPU.
///
/// Returns (oriented GpuGraph, symmetric_edge_count, total_time).
/// The symmetric edge count is needed for TEPS calculation since the oriented
/// graph has roughly half the entries.
pub fn load_and_transfer_direct_oriented(
    dataset: &str,
    device: &GpuDevice,
) -> (GpuGraph, u32, f64) {
    let start = Instant::now();

    let (edges, num_nodes) =
        grafeo_cuda_core::parse_tsv_to_edges(Path::new(dataset)).expect("failed to parse TSV");
    let parse_time = start.elapsed().as_secs_f64();
    eprintln!(
        "  Parsed {num_nodes} nodes, {} edges in {parse_time:.2}s from {dataset}",
        edges.len()
    );

    let csr_start = Instant::now();
    let oriented_csr = grafeo_cuda_core::from_edge_pairs_oriented(&edges, num_nodes);
    let csr_time = csr_start.elapsed().as_secs_f64();
    // Symmetric edge count = 2 * oriented edges (each undirected edge stored once)
    let symmetric_edges = oriented_csr.num_edges * 2;
    eprintln!(
        "  Oriented CSR: {csr_time:.4}s ({} nodes, {} oriented edges, ~{} symmetric)",
        oriented_csr.num_nodes, oriented_csr.num_edges, symmetric_edges
    );

    let xfer_start = Instant::now();
    let gpu_graph =
        GpuGraph::from_csr_arrays(oriented_csr, device).expect("failed to transfer graph to GPU");
    let xfer_time = xfer_start.elapsed().as_secs_f64();
    eprintln!("  GPU transfer: {xfer_time:.4}s");

    let total = start.elapsed().as_secs_f64();
    eprintln!(
        "  Total (parse+CSR+transfer): {total:.4}s  [{} vertices, {} oriented edges]",
        gpu_graph.num_nodes, gpu_graph.num_edges
    );

    (gpu_graph, symmetric_edges, total)
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
