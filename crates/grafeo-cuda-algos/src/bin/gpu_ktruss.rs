//! GPU K-Truss benchmark.

use clap::Parser;
use grafeo_cuda_algos::{harness, k_truss};
use grafeo_cuda_core::GpuDevice;

#[derive(Parser)]
#[command(name = "gpu-ktruss", about = "GPU k-truss decomposition benchmark")]
struct Args {
    /// Path to TSV dataset.
    #[arg(long)]
    dataset: String,

    /// Truss number k (minimum 3).
    #[arg(long, default_value = "3")]
    k: u32,

    /// Number of benchmark runs.
    #[arg(long, default_value = "5")]
    runs: usize,

    /// Number of warmup runs.
    #[arg(long, default_value = "1")]
    warmup: usize,

    /// Save results to JSON.
    #[arg(long)]
    save: bool,
}

fn main() {
    let args = Args::parse();
    let device = GpuDevice::new(0).expect("No CUDA device found");
    let (_db, gpu_graph, transfer_time) = harness::load_and_transfer(&args.dataset, &device);

    // Warmup
    for _ in 0..args.warmup {
        let _ = k_truss::k_truss(&gpu_graph, args.k);
    }

    let mut times = Vec::with_capacity(args.runs);
    let mut last_result = None;

    for _ in 0..args.runs {
        let result = k_truss::k_truss(&gpu_graph, args.k).expect("k_truss failed");
        times.push(result.time_seconds);
        last_result = Some(result);
    }

    let kr = last_result.unwrap();
    let best_kernel = times.iter().copied().fold(f64::INFINITY, f64::min);
    let teps = if best_kernel > 0.0 {
        gpu_graph.num_edges as f64 / best_kernel
    } else {
        0.0
    };

    eprintln!("\n  [gpu_ktruss] on {} (k={})", args.dataset, args.k);
    eprintln!("    Edges in {}-truss: {}", args.k, kr.edges.len());
    eprintln!("    Iterations: {}", kr.iterations);
    eprintln!("    Transfer:   {transfer_time:.6}s");
    eprintln!("    Kernel:     {best_kernel:.6}s");
    eprintln!("    TEPS:       {teps:.0}");

    let result = harness::GpuBenchmarkResult {
        dataset: args.dataset,
        algorithm: "gpu_ktruss".to_string(),
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
            "k": args.k,
            "truss_edges": kr.edges.len(),
            "iterations": kr.iterations,
        }),
    };

    if args.save {
        match result.save() {
            Ok(path) => eprintln!("  Saved to {}", path.display()),
            Err(e) => eprintln!("  Failed to save: {e}"),
        }
    }

    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
