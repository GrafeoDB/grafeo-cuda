//! GPU PageRank benchmark.

use clap::Parser;
use grafeo_cuda_algos::{harness, pagerank};
use grafeo_cuda_core::GpuDevice;

#[derive(Parser)]
#[command(name = "gpu-pagerank", about = "GPU PageRank benchmark")]
struct Args {
    /// Path to TSV dataset.
    #[arg(long)]
    dataset: String,

    /// Damping factor.
    #[arg(long, default_value = "0.85")]
    damping: f32,

    /// Maximum iterations.
    #[arg(long, default_value = "20")]
    max_iterations: u32,

    /// Convergence tolerance.
    #[arg(long, default_value = "1e-6")]
    tolerance: f32,

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
        let _ = pagerank::pagerank(
            &gpu_graph,
            args.damping,
            args.max_iterations,
            args.tolerance,
        );
    }

    let mut times = Vec::with_capacity(args.runs);
    let mut last_result = None;

    for _ in 0..args.runs {
        let result = pagerank::pagerank(
            &gpu_graph,
            args.damping,
            args.max_iterations,
            args.tolerance,
        )
        .expect("pagerank failed");
        times.push(result.time_seconds);
        last_result = Some(result);
    }

    let pr = last_result.unwrap();
    let best_kernel = times.iter().copied().fold(f64::INFINITY, f64::min);
    let teps = if best_kernel > 0.0 {
        gpu_graph.num_edges as f64 * pr.iterations as f64 / best_kernel
    } else {
        0.0
    };

    // Find top 5 nodes by rank
    let mut ranked: Vec<(usize, f32)> = pr.ranks.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top5: Vec<_> = ranked.iter().take(5).collect();

    eprintln!("\n  [gpu_pagerank] on {}", args.dataset);
    eprintln!(
        "    Iterations: {}, Converged: {}",
        pr.iterations, pr.converged
    );
    eprintln!("    Transfer:   {transfer_time:.6}s");
    eprintln!("    Kernel:     {best_kernel:.6}s");
    eprintln!("    TEPS:       {teps:.0}");
    eprintln!("    Top 5:      {top5:?}");

    let result = harness::GpuBenchmarkResult {
        dataset: args.dataset,
        algorithm: "gpu_pagerank".to_string(),
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
            "iterations": pr.iterations,
            "converged": pr.converged,
            "damping": args.damping,
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
