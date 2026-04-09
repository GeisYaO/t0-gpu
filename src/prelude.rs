//! T0 Prelude — 一行 use 导入常用 DSL + 运行时 API
//!
//! ```ignore
//! use t0_gpu::prelude::*;
//!
//! let rt = GpuRuntime::new()?;
//! let kernel = rt.ensure_kernel_t0("add", || math::elementwise_binary(...), [256,1,1], 0)?;
//! rt.dispatch(&kernel, grid, &ka)?;
//! ```

pub use crate::t0::dsl::{DType, CompiledKernel, KernArgMeta, KernArgType};
pub use crate::t0::ir::Target;
pub use crate::t0::gemm_gen::{GemmConfig, auto_select, compute_grid_auto, build_kernargs};

#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub use crate::ignis::gpu_context::GpuRuntime;
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool, GpuBuffer};

