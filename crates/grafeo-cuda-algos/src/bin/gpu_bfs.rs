//! GPU BFS benchmark (Graph500 TEPS metric).

use clap::Parser;
use grafeo_cuda_algos::{bfs, harness};
use grafeo_cuda_core::GpuDevice;

#[derive(Parser)]
#[command(name = "gpu-bfs", about = "GPU BFS benchmark (Graph500 TEPS)")]
struct Args {
    /// Path to TSV dataset.
    #[arg(long)]
    dataset: String,

    /// Number of benchmark runs per source.
    #[arg(long, default_value = "5")]
    runs: usize,

    /// Warmup runs per source.
    #[arg(long, default_value = "1")]
    warmup: usize,

    /// Number of evenly-spaced BFS source nodes.
    #[arg(long, default_value = "4")]
    sources: usize,

    /// Direct TSV->CSR->GPU path (bypasses GrafeoDB).
    #[arg(long)]
    direct: bool,

    /// Save results to JSON.
    #[arg(long)]
    save: bool,
}

fn main() {
    let args = Args::parse();
    let device = GpuDevice::new(0).expect("No CUDA device found");

    let (gpu_graph, transfer_time) = if args.direct {
        harness::load_and_transfer_direct(&args.dataset, &device)
    } else {
        let (_db, graph, time) = harness::load_and_transfer(&args.dataset, &device);
        (graph, time)
    };

    let n = gpu_graph.num_nodes;
    if n == 0 {
        eprintln!("ERROR: Graph is empty");
        std::process::exit(1);
    }

    // Pick evenly spaced source nodes (0-based CSR indices)
    let step = (n as usize) / args.sources.max(1);
    let sources: Vec<u32> = (0..args.sources)
        .map(|i| (i * step.max(1)) as u32)
        .collect();

    let mut all_teps = Vec::new();

    for (i, &src) in sources.iter().enumerate() {
        eprintln!("\n  BFS source {}/{}: node {src}", i + 1, sources.len());

        // Warmup
        for _ in 0..args.warmup {
            let _ = bfs::bfs(&gpu_graph, src);
        }

        let mut times = Vec::with_capacity(args.runs);
        let mut last_result = None;

        for _ in 0..args.runs {
            let result = bfs::bfs(&gpu_graph, src).expect("BFS failed");
            times.push(result.time_seconds);
            last_result = Some(result);
        }

        let br = last_result.unwrap();
        let best_kernel = times.iter().copied().fold(f64::INFINITY, f64::min);
        let teps = if best_kernel > 0.0 {
            gpu_graph.num_edges as f64 / best_kernel
        } else {
            0.0
        };

        eprintln!("    Depth: {}, Reachable: {}", br.max_depth, br.reachable);
        eprintln!("    Transfer: {transfer_time:.6}s");
        eprintln!("    Kernel:   {best_kernel:.6}s");
        eprintln!("    TEPS:     {teps:.0}");

        all_teps.push(teps);

        let result = harness::GpuBenchmarkResult {
            dataset: args.dataset.clone(),
            algorithm: format!("gpu_bfs_source_{i}"),
            vertices: gpu_graph.num_nodes,
            edges: gpu_graph.num_edges,
            time_s: transfer_time + best_kernel,
            transfer_time_s: transfer_time,
            kernel_time_s: best_kernel,
            teps,
            runs: times,
            platform: harness::platform_info(&device),
            gpu_device: device.properties().name.clone(),
            extra: serde_json::json!({
                "source_node": src,
                "max_depth": br.max_depth,
                "reachable_nodes": br.reachable,
            }),
        };

        if args.save {
            let _ = result.save();
        }
    }

    // Harmonic mean of TEPS (Graph500 convention)
    let positive: Vec<f64> = all_teps.iter().copied().filter(|&t| t > 0.0).collect();
    if !positive.is_empty() {
        let harmonic = positive.len() as f64 / positive.iter().map(|t| 1.0 / t).sum::<f64>();
        eprintln!(
            "\n  Harmonic mean TEPS across {} sources: {harmonic:.0}",
            sources.len()
        );
    }
}
