//! GPU Triangle Counting benchmark.

use clap::Parser;
use grafeo_cuda_algos::{harness, triangle_counting};
use grafeo_cuda_core::GpuDevice;

#[derive(Parser)]
#[command(name = "gpu-triangles", about = "GPU triangle counting benchmark")]
struct Args {
    /// Path to TSV dataset.
    #[arg(long)]
    dataset: String,

    /// Number of benchmark runs.
    #[arg(long, default_value = "5")]
    runs: usize,

    /// Number of warmup runs.
    #[arg(long, default_value = "1")]
    warmup: usize,

    /// Direct TSV->CSR->GPU path (bypasses GrafeoDB, much faster transfer).
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

    // Warmup
    for _ in 0..args.warmup {
        let _ = triangle_counting::triangle_count(&gpu_graph);
    }

    let mut times = Vec::with_capacity(args.runs);
    let mut last_count = 0u64;

    for _ in 0..args.runs {
        let result = triangle_counting::triangle_count(&gpu_graph).expect("triangle_count failed");
        times.push(result.time_seconds);
        last_count = result.total_triangles;
    }

    let best_kernel = times.iter().copied().fold(f64::INFINITY, f64::min);
    let teps = if best_kernel > 0.0 {
        gpu_graph.num_edges as f64 / best_kernel
    } else {
        0.0
    };

    eprintln!("\n  [gpu_triangle_counting] on {}", args.dataset);
    eprintln!("    Triangles: {last_count}");
    eprintln!("    Transfer:  {transfer_time:.6}s");
    eprintln!("    Kernel:    {best_kernel:.6}s");
    eprintln!("    TEPS:      {teps:.0}");

    let result = harness::GpuBenchmarkResult {
        dataset: args.dataset,
        algorithm: "gpu_triangle_counting".to_string(),
        vertices: gpu_graph.num_nodes,
        edges: gpu_graph.num_edges,
        time_s: transfer_time + best_kernel,
        transfer_time_s: transfer_time,
        kernel_time_s: best_kernel,
        teps,
        runs: times,
        platform: harness::platform_info(&device),
        gpu_device: device.properties().name.clone(),
        extra: serde_json::json!({ "total_triangles": last_count }),
    };

    if args.save {
        match result.save() {
            Ok(path) => eprintln!("  Saved to {}", path.display()),
            Err(e) => eprintln!("  Failed to save: {e}"),
        }
    }

    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
