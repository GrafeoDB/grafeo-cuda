//! PTX module caching to avoid re-parsing and JIT-compiling on every algorithm call.
//!
//! Each algorithm's PTX is compiled at build time and loaded once per CUDA context.
//! Subsequent calls reuse the cached `CudaModule` and `CudaFunction`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use grafeo_cuda_core::error::CudaError;

/// A cached module with pre-loaded functions.
pub(crate) struct CachedModule {
    _module: Arc<CudaModule>,
    functions: Vec<CudaFunction>,
}

// SAFETY: CudaModule and CudaFunction wrap Arc'd driver handles and raw pointers
// that are safe to share across threads (the CUDA driver is thread-safe).
unsafe impl Send for CachedModule {}
unsafe impl Sync for CachedModule {}

/// Load a PTX module and cache it for the lifetime of the process.
///
/// The `lock` should be a module-level `OnceLock` unique to each algorithm.
/// `ptx_src` is the PTX source (from `include_str!`).
/// `function_names` lists the kernel functions to pre-load.
///
/// Returns cloned `CudaFunction` handles (cheap: just a raw pointer + Arc).
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn get_or_load(
    lock: &OnceLock<CachedModule>,
    ctx: &Arc<CudaContext>,
    ptx_src: &str,
    function_names: &[&str],
) -> Result<Vec<CudaFunction>, CudaError> {
    let cached = lock.get_or_init(|| {
        let module = ctx
            .load_module(Ptx::from_src(ptx_src))
            .expect("failed to load PTX module");

        let mut functions = Vec::with_capacity(function_names.len());
        for &name in function_names {
            let func = module
                .load_function(name)
                .unwrap_or_else(|e| panic!("kernel {name} not found: {e}"));
            functions.push(func);
        }

        CachedModule {
            _module: module,
            functions,
        }
    });

    Ok(cached.functions.clone())
}
