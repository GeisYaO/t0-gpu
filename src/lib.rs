#![allow(dead_code)]           // Reserved hardware constants/fields
#![allow(non_camel_case_types)] // WmmaFormat: BF16_F32 etc.
#![allow(unreachable_patterns)] // Redundant Op match arms in asm_emitter

//! # T0-GPU
//!
//! RDNA3 (GFX1100) 裸金属 GPU 内核编译器 & KFD 运行时。
//!
//! - **DSL**: 声明式内核定义 → 自动编译到 GFX1100 ISA
//! - **T0 编译器**: 数学 IR → GFX1100 ISA → AMD HSA ELF
//! - **KFD 运行时**: 直接通过 /dev/kfd 驱动接口与 GPU 通信
//!
//! ## 示例 (DSL API)
//! ```ignore
//! use t0_gpu::prelude::*;
//!
//! // GEMM — 自动选择最优配置
//! let kernel = gemm(1024, 1024, 4096).compile()?;
//!
//! // 融合操作
//! let fused = KernelBuilder::new(Target::GFX1100)
//!     .op(Op::SiLU)
//!     .op(Op::Mul)
//!     .compile()?;
//! ```

// ── T0 编译器 ──
pub mod t0;

// ── 便捷导入 ──
pub mod prelude;

// ── ISA 编码器 ──
pub mod rdna3_asm;

// ── Code Object (ELF) 生成器 ──
pub mod rdna3_code_object;

mod llvm_toolchain;

// ── KFD 裸金属运行时 ──
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub mod kfd;

// ── WSL2 DXG 运行时 ──
#[cfg(feature = "wsl_dxg")]
pub mod wsl_dxg;

// ── Unified GPU backend interface ──
// Provides consistent type names regardless of rocm vs wsl_dxg feature
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub mod gpu_backend;

// ── Shared kernargs macro (used by both rocm and wsl_dxg backends) ──
#[macro_export]
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
macro_rules! kernargs {
    ($($val:expr => $ty:ty),* $(,)?) => {{
        let mut _ka = Vec::new();
        $(
            _ka.extend_from_slice(&<$ty>::to_le_bytes($val as $ty));
        )*
        _ka
    }};
}

// ── Ignis — GPU-native autodiff framework ──
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub mod ignis;
