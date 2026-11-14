//! Launch configurations shared by the algorithm modules.

use cudarc::driver::LaunchConfig;
use grafeo_cuda_core::error::CudaError;

/// Threads per block for every kernel in this crate.
pub(crate) const THREADS_PER_BLOCK: u32 = 256;

/// Threads per warp on all supported NVIDIA GPUs.
const WARP_SIZE: u32 = 32;

/// One thread per item. `items` must be non-zero: the driver rejects empty grids.
pub(crate) fn thread_per_item(items: u32) -> LaunchConfig {
    debug_assert!(items > 0, "empty grids are rejected by the driver");
    LaunchConfig {
        grid_dim: (items.div_ceil(THREADS_PER_BLOCK), 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One 32-thread warp per item. `items` must be non-zero.
///
/// The warp kernels compute a `u32` global thread index, so the total thread
/// count must fit in `u32`. Larger inputs are refused instead of silently
/// wrapping around and skipping work.
pub(crate) fn warp_per_item(items: u32) -> Result<LaunchConfig, CudaError> {
    debug_assert!(items > 0, "empty grids are rejected by the driver");
    let threads = items.checked_mul(WARP_SIZE).ok_or_else(|| {
        CudaError::Kernel(format!(
            "{items} work items need more than u32::MAX threads at one warp each; \
             at most {} are supported",
            u32::MAX / WARP_SIZE
        ))
    })?;
    Ok(LaunchConfig {
        grid_dim: (threads.div_ceil(THREADS_PER_BLOCK), 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_per_item_rounds_up_to_whole_blocks() {
        assert_eq!(thread_per_item(1).grid_dim, (1, 1, 1));
        assert_eq!(thread_per_item(256).grid_dim, (1, 1, 1));
        assert_eq!(thread_per_item(257).grid_dim, (2, 1, 1));
        assert_eq!(thread_per_item(u32::MAX).grid_dim, (16_777_216, 1, 1));
        assert_eq!(thread_per_item(3).block_dim, (THREADS_PER_BLOCK, 1, 1));
    }

    #[test]
    fn warp_per_item_covers_every_item() {
        // 8 warps fit in one 256-thread block.
        assert_eq!(warp_per_item(8).unwrap().grid_dim, (1, 1, 1));
        assert_eq!(warp_per_item(9).unwrap().grid_dim, (2, 1, 1));
        let largest = u32::MAX / WARP_SIZE;
        let config = warp_per_item(largest).unwrap();
        assert!(u64::from(config.grid_dim.0) * 256 >= u64::from(largest) * 32);
    }

    /// Regression: `items * 32` was computed unchecked and wrapped in release
    /// builds, launching far too few warps and undercounting triangles.
    #[test]
    fn warp_per_item_refuses_thread_overflow() {
        let too_many = u32::MAX / WARP_SIZE + 1;
        assert!(matches!(warp_per_item(too_many), Err(CudaError::Kernel(_))));
        assert!(matches!(warp_per_item(u32::MAX), Err(CudaError::Kernel(_))));
    }
}
