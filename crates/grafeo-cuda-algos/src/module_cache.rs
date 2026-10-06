//! PTX module caching to avoid re-parsing and JIT-compiling on every algorithm call.
//!
//! Each algorithm's PTX is embedded at build time (the committed PTX in
//! `kernels/ptx/`, or nvcc output with `GRAFEO_CUDA_BUILD_PTX=1`), JIT-compiled
//! by the driver and loaded once per CUDA context. Subsequent calls on the same
//! context reuse the cached `CudaFunction` handles.
//!
//! Modules belong to a context: a function loaded in one context cannot be
//! launched on a stream of another (a second GPU, or a non-primary context).
//! The cache is therefore keyed by the raw context handle. Cached functions
//! keep their module, and with it the context, alive, so a cached handle is
//! never reused by a different context.

use std::sync::{Arc, Mutex, PoisonError};

use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc::Ptx;
use grafeo_cuda_core::error::CudaError;

/// A small map from a context key to a value, safe to use as a `static`.
///
/// Lookups are linear: a process has a handful of CUDA contexts at most.
pub(crate) struct PerContext<T> {
    entries: Mutex<Vec<(usize, T)>>,
}

impl<T: Clone> PerContext<T> {
    pub(crate) const fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Returns the value cached for `key`, loading it with `load` on first use.
    ///
    /// A failed load caches nothing, so a later call tries again.
    pub(crate) fn get_or_try_insert<E>(
        &self,
        key: usize,
        load: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        // A panic while holding the lock cannot leave a half-written entry
        // behind (entries are pushed whole), so a poisoned lock is still usable.
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, value)) = entries.iter().find(|(k, _)| *k == key) {
            return Ok(value.clone());
        }
        let value = load()?;
        entries.push((key, value.clone()));
        Ok(value)
    }
}

/// Cache of loaded kernel functions for one PTX module, per CUDA context.
pub(crate) type ModuleCache = PerContext<Vec<CudaFunction>>;

/// Load a PTX module into the context of `ctx` and cache it there.
///
/// `cache` should be a module-level `static` unique to each PTX source and
/// function list. `function_names` lists the kernel functions to pre-load.
///
/// Returns cloned `CudaFunction` handles (cheap: a raw pointer plus an `Arc`).
pub(crate) fn get_or_load(
    cache: &ModuleCache,
    ctx: &Arc<CudaContext>,
    ptx_src: &str,
    function_names: &[&str],
) -> Result<Vec<CudaFunction>, CudaError> {
    let key = ctx.cu_ctx() as usize;
    cache.get_or_try_insert(key, || {
        let module = ctx.load_module(Ptx::from_src(ptx_src)).map_err(|e| {
            CudaError::Kernel(format!(
                "failed to load PTX module (PTX ISA {}; the NVIDIA driver must \
                 support it, see the grafeo-cuda README): {e}",
                crate::PTX_ISA_VERSION
            ))
        })?;
        function_names
            .iter()
            .map(|&name| {
                module
                    .load_function(name)
                    .map_err(|e| CudaError::Kernel(format!("kernel {name} not found: {e}")))
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn loads_once_per_key() {
        let cache: PerContext<u32> = PerContext::new();
        let loads = Cell::new(0);
        let load = |value| {
            loads.set(loads.get() + 1);
            Ok::<_, ()>(value)
        };

        assert_eq!(cache.get_or_try_insert(1, || load(3)), Ok(3));
        assert_eq!(cache.get_or_try_insert(1, || load(19)), Ok(3));
        assert_eq!(loads.get(), 1);

        // A second context gets its own entry instead of the first one's.
        assert_eq!(cache.get_or_try_insert(2, || load(88)), Ok(88));
        assert_eq!(cache.get_or_try_insert(1, || load(0)), Ok(3));
        assert_eq!(cache.get_or_try_insert(2, || load(0)), Ok(88));
        assert_eq!(loads.get(), 2);
    }

    #[test]
    fn failed_load_is_not_cached() {
        let cache: PerContext<u32> = PerContext::new();
        assert_eq!(
            cache.get_or_try_insert(7, || Err("no driver")),
            Err("no driver")
        );
        assert_eq!(cache.get_or_try_insert(7, || Ok::<_, &str>(3)), Ok(3));
    }

    /// Regression: a PTX module that failed to load panicked inside the cache
    /// instead of returning an error.
    #[test]
    #[ignore = "needs a CUDA device"]
    fn invalid_ptx_returns_an_error() {
        let device = grafeo_cuda_core::GpuDevice::new(0).expect("no CUDA device");
        let cache = ModuleCache::new();
        let result = get_or_load(&cache, device.context(), "not ptx at all", &["kernel"]);
        assert!(matches!(result, Err(CudaError::Kernel(_))));
    }
}
