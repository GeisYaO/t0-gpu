//! T0 内核编译器 — GPU 内核代码生成框架
//!
//! BlockKernel DSL → compile_via_ssa() → TileFunc SSA → T0Kernel → LLVM → HSA ELF
//!
//! ## 使用示例
//! ```ignore
//! use t0_gpu::t0::block_dsl::BlockKernel;
//! use t0_gpu::t0::ir::Target;
//!
//! let mut kb = BlockKernel::new("vadd", 256);
//! let x = kb.arg_ptr("x");
//! // ... build kernel ...
//! let compiled = kb.compile(Target::GFX1100)?;
//! ```

pub mod ir;
pub mod regalloc;
pub mod asm_emitter;
pub mod compile;
pub mod schedule;
pub mod math;
pub mod dsl;
pub mod gemm_gen;
pub mod block_dsl;
pub mod opt_passes;
pub mod isa_probe;
pub mod latency_model;
pub mod cost_model;
pub mod tile_ir;
pub mod tile_ssa;
pub mod tile_ssa_lower;
pub mod block_dsl_to_ssa;

pub mod ssa_ir;
pub mod ssa_regalloc;
pub mod domtree;
pub mod isa_verifier;


pub mod insn_latency;


pub mod auto_gemm;
pub mod softmax_kernels;
pub mod ce_loss_kernels;
pub mod rope_kernels;
pub mod causal_mask_kernels;
pub mod rmsnorm_kernels;
pub mod embedding_kernels;
pub mod adamw_kernels;
pub mod elementwise_kernels;
#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod test_tile_gemm_suite;
#[cfg(test)]
mod precision_vs_torch;
#[cfg(test)]
mod tile_boundary_tests;

pub use ir::*;
pub use compile::T0Kernel;
pub use schedule::{Schedule, GFX1100Schedule};
pub use gemm_gen::{GemmConfig, GemmTranspose, auto_select, compute_grid_auto, build_kernargs,
    auto_select_backward_data, auto_select_backward_weight,
    build_kernargs_backward_data, build_kernargs_backward_weight,
    compute_grid_backward_data, compute_grid_backward_weight};

pub(crate) fn verbose_diagnostics_enabled() -> bool {
    let verbose = std::env::var("T0_VERBOSE_COMPILE").ok();
    matches!(
        verbose.as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("on")
    ) || std::env::var_os("T0_DUMP_ASM").is_some()
        || std::env::var_os("T0_VERIFY_DUMP").is_some()
}
