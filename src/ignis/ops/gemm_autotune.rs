//! GEMM Auto-tuning — T0 compiler integration for optimal GEMM selection.
//!
//! Wraps T0's `auto_select()` to pick the best kernel config per matrix size.

#[cfg(feature = "rocm")]
use crate::gpu_backend::GpuKernel;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use std::sync::Arc;

/// Select and compile the optimal GEMM kernel for given dimensions.
///
/// Uses T0 compiler's `auto_select(M, K, N)` to pick the best tile/split-K
/// configuration, then compiles and caches the kernel.
///
/// # Returns
/// - The compiled GpuKernel ready for dispatch
/// - Grid dimensions [gx, gy]
/// - The GemmConfig used
#[cfg(feature = "rocm")]
pub fn ensure_gemm(
    _runtime: &Arc<GpuRuntime>,
    _m: u32,
    _k: u32,
    _n: u32,
) -> Result<(Arc<GpuKernel>, [u32; 2]), String> {
    // TODO: integrate T0 compiler auto_select when opensource/ is linked
    Err("gemm_autotune: T0 integration pending".to_string())
}
