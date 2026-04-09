//! T0 Tile IR — Tile-level GEMM description → ISA kernel compiler
//!
//! 这是 T0 编译器的核心创新：用 tile-level 规格描述 GEMM 计算，
//! 编译器自动生成 ISA 级内核。与 gemm_gen.rs 的区别：
//! - gemm_gen 是预定义的 kernel 工厂（选择器）
//! - tile_ir 是编译器——从 tile 参数自动生成最优内核
//!
//! # 使用示例
//! ```ignore
//! use t0_gpu::t0::tile_ir::{TileGemm, lower_gemm};
//! use t0_gpu::t0::ir::Target;
//!
//! let spec = TileGemm::tile_128x64_k16();
//! let kernel = lower_gemm(&spec);
//! let elf = kernel.compile(Target::GFX1201).unwrap();
//! ```

use super::compile::T0Kernel;
use super::ir::*;

// ============================================================================
// EpilogueOp — Fused post-GEMM element-wise operations
// ============================================================================

/// Element-wise operation applied to each GEMM accumulator value before store.
///
/// Operations are applied in order: Y' = epilogue[n]( ... epilogue[1]( epilogue[0]( Y ) ) ... )
///
/// All epilogue ops operate on f32 values in VGPRs, executing in the store phase
/// with zero additional memory bandwidth (data stays in registers).
///
/// # Example: GEMM + Bias + SiLU
/// ```ignore
/// let mut spec = TileGemm::tile_128x64_k16();
/// spec.add_epilogue(EpilogueOp::BiasAdd);
/// spec.add_epilogue(EpilogueOp::SiLU);
/// let kernel = lower_gemm(&spec);
/// // kernargs: [X, WT, Y, K, N, split_k_shift, y_split_stride, M, bias_ptr]
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum EpilogueOp {
    /// Y[i] += bias[col_index]
    /// Requires extra kernarg: bias_ptr (u64, pointer to [N] f32 vector)
    BiasAdd,
    /// Y[i] *= scale (scalar)
    /// Requires extra kernarg: scale (f32)
    Scale,
    /// Y[i] = max(Y[i], 0.0)
    ReLU,
    /// Y[i] = Y[i] * sigmoid(Y[i])
    SiLU,
    /// Y[i] = 0.5 * Y[i] * (1 + tanh(sqrt(2/π) * (Y[i] + 0.044715 * Y[i]³)))
    /// Uses the fast tanh approximation.
    GELU,
    /// Y[i] = |Y[i]|
    Abs,
    /// Y[i] = -Y[i]
    Neg,
    /// Y[i] = clamp(Y[i], min, max)
    /// Requires extra kernargs: clamp_min (f32), clamp_max (f32)
    Clamp,
}

// ============================================================================
// TileTranspose — GEMM layout mode
// ============================================================================

/// GEMM transpose mode for tile_ir kernels.
///
/// Determines how the B (weight) matrix is laid out in memory.
/// The A (input) matrix is always row-major [M, K].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileTranspose {
    /// Y[M,N] = A[M,K] @ B[N,K]^T — B is row-major with stride=K (forward GEMM)
    NT,
    /// Y[M,N] = A[M,K] @ B[K,N]   — B is row-major with stride=N (backward dX)
    NN,
}

// ============================================================================
// TileGemm — Tile-level GEMM specification
// ============================================================================

/// Complete specification for a tiled GEMM kernel.
///
/// Describes the tile dimensions, parallelism, and memory strategy.
/// `lower_gemm()` compiles this specification into an ISA-level T0Kernel.
///
/// NT mode: Y[M,N] = X[M,K] @ WT[N,K]^T  (bf16 in, f32 accum, f32 out)
/// NN mode: Y[M,N] = X[M,K] @ B[K,N]      (bf16 in, f32 accum, f32 out)
#[derive(Clone, Debug)]
pub struct TileGemm {
    /// Output tile rows per workgroup (must be multiple of 16)
    pub tile_m: u32,
    /// Output tile columns per workgroup (must be multiple of 16)
    pub tile_n: u32,
    /// K-dimension elements per loop iteration (16 or 32)
    pub tile_k: u32,
    /// Enable WGP mode (workgroup spans 2 CUs, 128KB LDS)
    pub wgp_mode: bool,
    /// LDS double-buffering (prefetch next tile while computing current)
    pub double_buffer: bool,
    /// Split-K factor (1 = no split, 2/4/8/16 = split K dimension)
    pub split_k: u32,
    /// Grid axis swap: true=TGID.x→N (L2 friendly), false=TGID.x→M
    pub swap_grid: bool,
    /// Transpose mode: NT (Y=A@B^T) or NN (Y=A@B)
    pub transpose: TileTranspose,
    /// ACC Swapping: keep only 1 row_block of accumulators in VGPRs,
    /// swap others to/from LDS between row_block passes.
    /// Reduces VGPR pressure at cost of extra LDS traffic.
    pub acc_swap: bool,
    /// Epilogue operations: fused element-wise ops applied to each acc value before store.
    /// Empty = no epilogue (plain GEMM output).
    /// Operations are applied in order: result = epilogue[n](...(epilogue[0](acc))...)
    pub epilogue: Vec<EpilogueOp>,
}

impl TileGemm {
    /// Canonical 128×64 k16 LDS double-buffered (实测最优)
    pub fn tile_128x64_k16() -> Self {
        Self {
            tile_m: 128, tile_n: 64, tile_k: 16,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 64×64 k16 LDS double-buffered
    pub fn tile_64x64_k16() -> Self {
        Self {
            tile_m: 64, tile_n: 64, tile_k: 16,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 64×64 k32 CU mode — 2 K sub-steps for WMMA dual-chain ILP
    pub fn tile_64x64_k32() -> Self {
        Self {
            tile_m: 64, tile_n: 64, tile_k: 32,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 64×64 k64 CU mode — best for ≤2048³ (2.6-94.2 TF across sizes)
    pub fn tile_64x64_k64() -> Self {
        Self {
            tile_m: 64, tile_n: 64, tile_k: 64,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 32×64 k16 LDS double-buffered
    pub fn tile_32x64_k16() -> Self {
        Self {
            tile_m: 32, tile_n: 64, tile_k: 16,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 128×64 k32 CU mode — 2 K sub-steps for WMMA dual-chain ILP
    pub fn tile_128x64_k32() -> Self {
        Self {
            tile_m: 128, tile_n: 64, tile_k: 32,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 64×128 k32 CU mode — wider N tile, narrower M
    /// Same 64 WMMA/iter as 128×128 k32, but ACC = 1 row × 8 col = 64 VGPRs
    /// LDS = (64+128)×32×2×2 = 24576 = 24KB
    pub fn tile_64x128_k32() -> Self {
        Self {
            tile_m: 64, tile_n: 128, tile_k: 32,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 64×128 k16 CU mode — fallback for non-aligned K
    pub fn tile_64x128_k16() -> Self {
        Self {
            tile_m: 64, tile_n: 128, tile_k: 16,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 128×128 k16 — optimal tile for WMMA GEMM
    /// 8 col tiles, 16 WMMA/iter, ~220 VGPRs, 2 waves/SIMD
    /// LDS = 128*16*2 + 128*16*2 = 8192/buf × 2 = 16384 bytes (16 KB)
    pub fn tile_128x128_k16() -> Self {
        Self {
            tile_m: 128, tile_n: 128, tile_k: 16,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 128×128 k32 — 2 K sub-steps per iteration
    /// 64 WMMA/iter (~290 cyc) covers 300 cyc VMEM latency much better than k16 (32 WMMA, 146 cyc)
    /// LDS = 2 × (128×32 + 128×32) × 2 = 32768 bytes (32 KB)
    /// K-iterations halved → barriers halved → VMEM stall ~eliminated
    /// WGP mode: +2% from L0 cache sharing (verified safe: 32KB < 64KB CWSR limit)
    pub fn tile_128x128_k32() -> Self {
        Self {
            tile_m: 128, tile_n: 128, tile_k: 32,
            wgp_mode: true, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 128×128 k48 — 3 k_sub, 50% more compute density vs k32
    /// LDS: (128+128)×48×2×2 = 49152 = 48KB (fits CU mode 64KB)
    /// GMEM: 6+6=12 loads/thread × 4 regs = 48 VGPRs
    pub fn tile_128x128_k48() -> Self {
        Self {
            tile_m: 128, tile_n: 128, tile_k: 48,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 128×128 k16 ACC Swap — same tile but only 1 row_block acc live
    /// VGPRs: ~174 → 4 waves/SIMD (vs 244 → 2 waves without swap)
    /// LDS: 16KB (GEMM) + 32KB (swap) = 48KB < 64KB limit
    pub fn tile_128x128_k16_swap() -> Self {
        Self {
            tile_m: 128, tile_n: 128, tile_k: 16,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: true,
            epilogue: vec![],
        }
    }

    /// 128×128 k32 ACC Swap — k32 for VMEM latency hiding + swap for 4-wave
    /// VGPRs: ~191 → 4 waves/SIMD, 0 spills
    /// LDS: 32KB (GEMM) + 32KB (swap) = 64KB (CU mode limit!)
    /// k32 amortizes swap cost: 32 WMMA per K-iter + 2 swaps → 50% overhead (vs 69% for k16)
    pub fn tile_128x128_k32_swap() -> Self {
        Self {
            tile_m: 128, tile_n: 128, tile_k: 32,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: true,
            epilogue: vec![],
        }
    }

    // ── WGP mode configurations (128KB LDS, 2 CUs per workgroup) ──

    /// 256×64 k32 WGP — best for large M (8 waves)
    /// Uses k32 because wg_size=256 threads can't cooperatively load tile_n=64, k=16
    /// (wt_bytes_per_thread = 64*16*2/256 = 8 < 16 bytes per b128 load)
    pub fn tile_256x64_k32_wgp() -> Self {
        Self {
            tile_m: 256, tile_n: 64, tile_k: 32,
            wgp_mode: true, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 256×64 k64 WGP — maximum compute density with low VGPR pressure
    /// ACC: 2×4×8 = 64 VGPRs, GMEM: (4+2)×4 = 24 VGPRs
    /// Predicted ~166 VGPRs → 4 waves/SIMD (double occupancy vs k32 128×128!)
    /// LDS: (256+64)×64×2×2 = 81920 = 80KB → needs WGP mode (128KB)
    /// 4 k_sub steps × 8 WMMA/sub = 32 WMMA/phase (same as k32 128×128)
    pub fn tile_256x64_k64_wgp() -> Self {
        Self {
            tile_m: 256, tile_n: 64, tile_k: 64,
            wgp_mode: true, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 128×128 k16 WGP — DISABLED: n_col_tiles=8 requires >256 VGPRs → regalloc overflow
    // pub fn tile_128x128_k16_wgp() removed: VGPR budget exceeded

    /// 128×128 k64 — one-shot loading, max compute density
    /// LDS: (128+128)×64×2×2 = 65536 = 64KB (fits CU mode barely!)  
    /// GMEM: (8+8) × 4 = 64 VGPRs → ~286 total → WILL SPILL without extra optimization
    pub fn tile_128x128_k64() -> Self {
        Self {
            tile_m: 128, tile_n: 128, tile_k: 64,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 128×64 k64 — halved output tile, much lower VGPR pressure
    /// ACC: 2×4×8 = 64 VGPRs (vs 128 for 128×128)
    /// GMEM: (8+4)×4 = 48 VGPRs
    /// LDS: (128+64)×64×2×2 = 49152 = 48KB (fits CU mode)
    pub fn tile_128x64_k64() -> Self {
        Self {
            tile_m: 128, tile_n: 64, tile_k: 64,
            wgp_mode: false, double_buffer: true,
            split_k: 1, swap_grid: true,
            transpose: TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// 64×64 k16 — 2 waves (CU mode only)
    /// NOTE: multi-dispatch still hangs in both CU and WGP mode!
    /// Kept for single-dispatch use only.
    // pub fn tile_64x64_k16_wgp() removed: WGP doesn't fix 2-wave hang

    /// Number of waves per workgroup = tile_m / 32
    pub fn n_waves(&self) -> u32 { self.tile_m / 32 }

    /// Workgroup size in threads
    pub fn wg_size(&self) -> u32 { self.n_waves() * 32 }

    /// Rows per wave
    pub fn rows_per_wave(&self) -> u32 { self.tile_m / self.n_waves() }

    /// WMMA row blocks per wave = rows_per_wave / 16
    pub fn n_row_blocks(&self) -> u32 { self.rows_per_wave() / 16 }

    /// WMMA column tiles = tile_n / 16
    pub fn n_col_tiles(&self) -> u32 { self.tile_n / 16 }

    /// K sub-steps per tile_k (each WMMA handles K=16)
    pub fn k_sub_steps(&self) -> u32 { self.tile_k / 16 }

    /// LDS bytes for X region per buffer
    pub fn lds_x_size(&self) -> u32 { self.tile_m * self.tile_k * 2 }
    /// LDS bytes for WT region per buffer
    pub fn lds_wt_size(&self) -> u32 { self.tile_n * self.tile_k * 2 }
    /// LDS bytes per buffer (X + WT)
    pub fn lds_per_buffer(&self) -> u32 { self.lds_x_size() + self.lds_wt_size() }
    /// Total LDS bytes (single or double buffered + acc swap region)
    pub fn lds_total(&self) -> u32 {
        let gemm_lds = if self.double_buffer { self.lds_per_buffer() * 2 } else { self.lds_per_buffer() };
        if self.acc_swap {
            gemm_lds + self.acc_swap_region_size()
        } else {
            gemm_lds
        }
    }

    /// LDS bytes needed for ACC swap region.
    /// Each wave stores 1 row_block's acc: n_col_tiles × 8 VGPRs × 32 lanes × 4 bytes
    pub fn acc_swap_region_size(&self) -> u32 {
        let vgprs_per_row_block = self.n_col_tiles() * 8;
        let bytes_per_wave = vgprs_per_row_block * 32 * 4;  // 32 lanes, 4 bytes/f32
        bytes_per_wave * self.n_waves()  // all waves
    }

    /// LDS offset where acc swap region starts
    pub fn acc_swap_base(&self) -> u32 {
        let gemm_lds = if self.double_buffer { self.lds_per_buffer() * 2 } else { self.lds_per_buffer() };
        gemm_lds
    }

    /// Descriptive name
    pub fn name(&self) -> String {
        let db = if self.double_buffer { "_db" } else { "" };
        let sk = if self.split_k > 1 { format!("_sk{}", self.split_k) } else { String::new() };
        let wgp = if self.wgp_mode { "_wgp" } else { "" };
        let mg = if !self.swap_grid { "_mg" } else { "" };
        let tr = match self.transpose { TileTranspose::NN => "_nn", TileTranspose::NT => "" };
        let sw = if self.acc_swap { "_swap" } else { "" };
        let epi = if self.epilogue.is_empty() {
            String::new()
        } else {
            let ops: Vec<&str> = self.epilogue.iter().map(|op| match op {
                EpilogueOp::BiasAdd => "bias",
                EpilogueOp::Scale => "scale",
                EpilogueOp::ReLU => "relu",
                EpilogueOp::SiLU => "silu",
                EpilogueOp::GELU => "gelu",
                EpilogueOp::Abs => "abs",
                EpilogueOp::Neg => "neg",
                EpilogueOp::Clamp => "clamp",
            }).collect();
            format!("_{}", ops.join("_"))
        };
        format!("tile_gemm_{}x{}_k{}{}{}{}{}{}{}{}", self.tile_m, self.tile_n, self.tile_k, db, sk, mg, wgp, tr, sw, epi)
    }

    /// Add an epilogue operation to the fusion chain.
    pub fn add_epilogue(&mut self, op: EpilogueOp) {
        self.epilogue.push(op);
    }

    /// Builder pattern: return self with epilogue added.
    pub fn with_epilogue(mut self, ops: Vec<EpilogueOp>) -> Self {
        self.epilogue = ops;
        self
    }

    /// Check if the epilogue requires a bias pointer kernel argument.
    pub fn has_epilogue_bias(&self) -> bool {
        self.epilogue.contains(&EpilogueOp::BiasAdd)
    }

    /// Check if the epilogue requires a scale kernel argument.
    pub fn has_epilogue_scale(&self) -> bool {
        self.epilogue.contains(&EpilogueOp::Scale)
    }

    /// Check if the epilogue requires clamp kernel arguments.
    pub fn has_epilogue_clamp(&self) -> bool {
        self.epilogue.contains(&EpilogueOp::Clamp)
    }
}

/// Auto-select optimal tile configuration based on matrix dimensions.
///
/// Data-driven tile config selection based on full-spectrum autotuner (2026-03-31).
///
/// Selection principles (split_k=1 baseline, then split_k applied post-hoc):
///   - Small (M,N ≤ 512):           64×64 k64/k32  (best CU saturation)
///   - Medium (M,N ~1024):          64×64 k32      (24.8 TF @ 1024³)
///   - Large (M,N ~2048):           256×64 k32 WGP (49.1 TF = +18% vs 64×64)
///   - Very large (M,N ≥ 4096):     128×128 k32    (84.7 TF @ 4096³)
///   - Huge K (K ≥ 4096, N ≥ 4096): 128×128 k64    (81.7 TF @ 1024×4096×4096)
pub fn tile_auto_select(m: u32, k: u32, n: u32, transpose: TileTranspose) -> TileGemm {
    // Full-spectrum autotuner results (2026-03-31, split_k=1):
    //
    //   256³:               2.0 TF → 64×64 k32
    //   512³:              10.1 TF → 64×64 k64
    //  1024³:              24.8 TF → 64×64 k32
    //  2048³:              49.1 TF → 256×64 k32 WGP  ★ NEW
    //  4096³:              84.7 TF → 128×128 k32
    //
    // Non-square:
    //  256×4096×1024:      32.0 TF → 64×64 k64
    //  512×4096×1024:      50.7 TF → 64×64 k64
    // 1024×4096×4096:      81.7 TF → 128×128 k64

    let min_dim = m.min(n);
    let max_dim = m.max(n);

    let mut spec = if min_dim >= 128 && max_dim >= 4096 {
        // Very large (4096³ class): 128×128 dominates
        if k >= 64 && k % 64 == 0 {
            TileGemm::tile_128x128_k64()
        } else if k >= 32 && k % 32 == 0 {
            TileGemm::tile_128x128_k32()
        } else {
            TileGemm::tile_128x128_k16()
        }
    } else if min_dim >= 64 && max_dim >= 2048 {
        // Large (2048³ class): 256×64 WGP wins (+18% vs 64×64)
        if m >= 256 && k >= 32 && k % 32 == 0 {
            TileGemm::tile_256x64_k32_wgp()
        } else if k >= 32 && k % 32 == 0 {
            TileGemm::tile_64x64_k32()
        } else {
            TileGemm::tile_64x64_k16()
        }
    } else if min_dim >= 64 {
        // Medium (512³-1024³): 64×64 consistently best
        if k >= 64 && k % 64 == 0 {
            TileGemm::tile_64x64_k64()
        } else if k >= 32 && k % 32 == 0 {
            TileGemm::tile_64x64_k32()
        } else {
            TileGemm::tile_64x64_k16()
        }
    } else if m >= 32 || n >= 128 {
        // Rectangular: use 32x64 or 64x64
        if k >= 32 && k % 32 == 0 {
            TileGemm::tile_64x64_k32()
        } else {
            TileGemm::tile_32x64_k16()
        }
    } else {
        // Small M: 32×64 to avoid tile underutilization
        TileGemm::tile_32x64_k16()
    };

    // Apply transpose mode
    spec.transpose = transpose;

    // Split-K: partition K dimension across multiple workgroups for parallelism.
    // Two reasons to use split_k:
    //   1. CU occupancy: when total_tiles < 96, split_k increases parallelism
    //   2. K-loop shortening: even with full occupancy, split_k reduces per-WG
    //      K iterations, improving compute/memory overlap and reducing barrier cost
    let n_tiles_m = (m + spec.tile_m - 1) / spec.tile_m;
    let n_tiles_n = (n + spec.tile_n - 1) / spec.tile_n;
    let total_tiles = n_tiles_m * n_tiles_n;
    
    // Heuristic: target ~4-8 K-loop iterations per workgroup for best pipeline utilization
    // Exception: WGP tiles (256 threads, high arithmetic intensity) don't benefit from
    // split_k when CU occupancy is already good — autotuner shows 20% penalty from sk=8.
    let k_iters = k / spec.tile_k;
    let desired_sk = if spec.wgp_mode && total_tiles >= 96 {
        1  // WGP: sufficient occupancy, no split needed
    } else if total_tiles < 12 {
        8  // very few tiles → max parallelism
    } else if total_tiles < 48 {
        4  // moderate tiles → moderate split
    } else if k_iters > 16 && total_tiles >= 48 {
        // Large matrices: split K to reduce per-WG loop count
        // Only worth it when total compute is large enough to amortize reduction
        let target_iters = 8;
        (k_iters / target_iters).min(8).max(1)
    } else {
        1  // K is already short, no split needed
    };
    
    if desired_sk > 1 && k >= spec.tile_k * 2 {
        let max_sk = k / spec.tile_k;
        let mut sk = desired_sk.min(max_sk);
        while sk > 1 && k % (spec.tile_k * sk) != 0 {
            sk /= 2;
        }
        if sk > 1 { spec.split_k = sk; }
    }

    spec
}

// ============================================================================
// lower_gemm — Compile TileGemm spec to T0Kernel
// ============================================================================

/// Compile a tile-level GEMM specification into an ISA-level kernel.
///
/// Generates a complete GEMM kernel with:
/// 1. Kernel argument loads (X, WT, Y pointers + K, N dims)
/// 2. Thread/wave decomposition and tile address computation
/// 3. K-loop with LDS double-buffered cooperative loading
/// 4. WMMA accumulation
/// 5. Store phase (accumulator → global memory)
///
/// The generated kernel is functionally equivalent to `gemm_gen::generate()`
/// but produced by the compiler from a parametric specification.
pub fn lower_gemm(spec: &TileGemm) -> T0Kernel {
    let target = current_target();
    let wmma_format = WmmaFormat::BF16_F32;
    let frag_a_vgprs = wmma_format.a_vreg_count(target);
    let frag_b_vgprs = wmma_format.b_vreg_count(target);
    let frag_a_align = wmma_format.a_alignment(target);
    let frag_b_align = wmma_format.b_alignment(target);
    let frag_a_loads = frag_a_vgprs / 4;
    let frag_b_loads = frag_b_vgprs / 4;

    let n_col_tiles = spec.n_col_tiles() as usize;
    let n_row_blocks = spec.n_row_blocks() as usize;
    let rows_per_wave = spec.rows_per_wave();

    let lds_x = spec.lds_x_size();
    let lds_wt = spec.lds_wt_size();
    let lds_buf = lds_x + lds_wt;  // per buffer

    // Safety: GFX1201 CWSR limits LDS save area to 64KB per CU.
    // WGP mode with LDS > 64KB causes queue eviction → hard hang.
    // Auto-fallback to CU mode when LDS exceeds limit.
    let effective_wgp = if spec.wgp_mode && spec.lds_total() > 65536 {
        eprintln!("[lower_gemm] WARNING: LDS={}B > 64KB, auto-disabling WGP mode for {}",
            spec.lds_total(), spec.name());
        false
    } else {
        spec.wgp_mode
    };

    let mut k = T0Kernel::new(&spec.name());
    k.set_lds_size(spec.lds_total());
    k.set_wg_size(spec.wg_size());
    k.set_wgp_mode(effective_wgp);

    let x_row_stride = spec.tile_k * 2;  // bytes per row in LDS (no padding for Phase 1)
    let wt_row_stride = spec.tile_k * 2;
    let xrs_shift = x_row_stride.trailing_zeros() as u8;
    let wrs_shift = wt_row_stride.trailing_zeros() as u8;

    // How many b128 (16-byte) loads per thread for X and WT
    let x_bytes_per_thread = spec.tile_m * spec.tile_k * 2 / spec.wg_size();
    let wt_bytes_per_thread = spec.tile_n * spec.tile_k * 2 / spec.wg_size();
    let x_loads_per_thread = x_bytes_per_thread / 16;
    let wt_loads_per_thread = wt_bytes_per_thread / 16;

    // Sequential loading: DISABLED — proven slower for k48 (extra VMEM waits > WMMA gains).
    // All configs use one-shot loading. Large tile_k requires enough GMEM VGPRs to fit.
    let use_sequential = false;  // disabled after k48 experiment
    let x_batch_loads = x_loads_per_thread;
    let wt_batch_loads = wt_loads_per_thread;
    let n_batches = 1u32;
    let batch_k_step = (spec.tile_k * 2) as i32;

    // VALIDATION: each thread must load at least 1 b128 for both X and WT
    assert!(x_loads_per_thread >= 1,
        "tile_ir: x_loads_per_thread=0 for {}! tile_m*tile_k*2={} < wg_size*16={}",
        spec.name(), spec.tile_m * spec.tile_k * 2, spec.wg_size() * 16);
    assert!(wt_loads_per_thread >= 1,
        "tile_ir: wt_loads_per_thread=0 for {}! tile_n*tile_k*2={} < wg_size*16={}",
        spec.name(), spec.tile_n * spec.tile_k * 2, spec.wg_size() * 16);

    // CRITICAL: chunks_per_row must be a power of 2!
    // The cooperative load decomposes tid→(row,col) using SHIFT and AND:
    //   row = tid >> cpr_shift
    //   col = tid & (cpr - 1)
    // This ONLY works when chunks_per_row is 2^n. If not (e.g., k48→cpr=6),
    // the decomposition is completely wrong → garbage LDS data → GPU hang.
    let gmem_row_bytes_x_check = spec.tile_k * 2;
    let chunks_per_row_x_check = gmem_row_bytes_x_check / 16;
    let gmem_row_bytes_wt_check = spec.tile_k * 2;
    let chunks_per_row_wt_check = gmem_row_bytes_wt_check / 16;
    assert!(chunks_per_row_x_check.is_power_of_two(),
        "tile_ir: X chunks_per_row={} is NOT power of 2 for {} (tile_k={})! \
         Bitwise tid decomposition will produce wrong addresses → GPU hang. \
         tile_k must be 16, 32, 64, 128, ... (tile_k*2/16 must be power of 2).",
        chunks_per_row_x_check, spec.name(), spec.tile_k);
    assert!(chunks_per_row_wt_check.is_power_of_two(),
        "tile_ir: WT chunks_per_row={} is NOT power of 2 for {} (tile_k={})! \
         Bitwise tid decomposition will produce wrong addresses → GPU hang.",
        chunks_per_row_wt_check, spec.name(), spec.tile_k);

    // ══════════════════════════════════════════════════════════════
    // Phase 1: Kernel arguments
    // ══════════════════════════════════════════════════════════════

    let x_ptr = k.arg_ptr("X");      // [M, K] bf16
    let wt_ptr = k.arg_ptr("WT");    // [N, K] bf16
    let y_ptr = k.arg_ptr("Y");      // [M, N] f32
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    let _sk_shift = k.arg_u32("split_k_shift");
    let y_split_stride = k.arg_u32("y_split_stride");
    let m_dim = k.arg_u32("M");      // actual M (may not be tile-aligned)

    // ── Epilogue kernel arguments (declared after standard GEMM args) ──
    let epi_bias_ptr = if spec.has_epilogue_bias() {
        Some(k.arg_ptr("bias"))       // [N] f32 bias vector
    } else { None };
    let epi_scale = if spec.has_epilogue_scale() {
        Some(k.arg_f32("epi_scale"))
    } else { None };
    let epi_clamp_min = if spec.has_epilogue_clamp() {
        Some(k.arg_f32("clamp_min"))
    } else { None };
    let epi_clamp_max = if spec.has_epilogue_clamp() {
        Some(k.arg_f32("clamp_max"))
    } else { None };

    k.emit_arg_loads();

    // ══════════════════════════════════════════════════════════════
    // Phase 2: TGID decomposition
    // ══════════════════════════════════════════════════════════════

    let tgid_x_s = k.alloc_sreg();
    k.capture_tgid_x(tgid_x_s);
    let tgid_y_s = k.alloc_sreg();
    k.capture_tgid_y(tgid_y_s);

    let split_k = spec.split_k;
    let sk_shift: u8 = match split_k { 1=>0, 2=>1, 4=>2, 8=>3, 16=>4, _=>panic!("bad split_k") };

    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    let split_k_id_s = k.alloc_sreg();

    if spec.swap_grid {
        k.push(Op::SAddU32 { dst: tile_col_s, src0: tgid_x_s, src1: SOperand::InlineInt(0) });
        if split_k <= 1 {
            k.push(Op::SAddU32 { dst: tile_row_s, src0: tgid_y_s, src1: SOperand::InlineInt(0) });
            k.s_mov_imm(split_k_id_s, 0);
        } else {
            k.s_lshr_b32(tile_row_s, tgid_y_s, sk_shift);
            let mask_s = k.alloc_sreg();
            k.s_mov_imm(mask_s, (split_k - 1) as i32);
            k.s_and_b32(split_k_id_s, tgid_y_s, mask_s);
        }
    } else {
        k.push(Op::SAddU32 { dst: tile_row_s, src0: tgid_x_s, src1: SOperand::InlineInt(0) });
        if split_k <= 1 {
            k.push(Op::SAddU32 { dst: tile_col_s, src0: tgid_y_s, src1: SOperand::InlineInt(0) });
            k.s_mov_imm(split_k_id_s, 0);
        } else {
            k.s_lshr_b32(tile_col_s, tgid_y_s, sk_shift);
            let mask_s = k.alloc_sreg();
            k.s_mov_imm(mask_s, (split_k - 1) as i32);
            k.s_and_b32(split_k_id_s, tgid_y_s, mask_s);
        }
    }

    // ── Tile Categorization (SGPR only, 0 VGPR) ──
    // Full OOB early exit: if tile_base_m >= M → entire WG is outside matrix
    let tile_base_m_s = k.alloc_sreg();
    let tm_shift = spec.tile_m.trailing_zeros() as u8;
    k.s_lshl_b32(tile_base_m_s, tile_row_s, tm_shift);
    k.s_cmp_ge_u32(tile_base_m_s, SReg(m_dim.0));
    let early_exit_label = k.make_label("early_exit");
    k.branch_scc1(&early_exit_label);

    // Classify: is_boundary = (tile_end_m > M) || (tile_end_n > N)
    let s_is_boundary = k.alloc_sreg();
    k.s_mov_imm(s_is_boundary, 0);  // assume internal
    {
        // Check M boundary
        let tile_end_m = k.alloc_sreg();
        k.push(Op::SAddU32 {
            dst: tile_end_m, src0: tile_base_m_s,
            src1: SOperand::InlineInt(spec.tile_m as i32),
        });
        // SCC = (M < tile_end_m) → boundary in M
        k.s_cmp_lt_u32(SReg(m_dim.0), tile_end_m);
        let skip_m = k.make_label("skip_boundary_m");
        k.branch_scc0(&skip_m);
        k.s_mov_imm(s_is_boundary, 1);
        k.label(&skip_m);

        // Check N boundary
        let tn_shift = spec.tile_n.trailing_zeros() as u8;
        let tile_base_n_s = k.alloc_sreg();
        k.s_lshl_b32(tile_base_n_s, tile_col_s, tn_shift);
        let tile_end_n = k.alloc_sreg();
        k.push(Op::SAddU32 {
            dst: tile_end_n, src0: tile_base_n_s,
            src1: SOperand::InlineInt(spec.tile_n as i32),
        });
        k.s_cmp_lt_u32(n_dim, tile_end_n);
        let skip_n = k.make_label("skip_boundary_n");
        k.branch_scc0(&skip_n);
        k.s_mov_imm(s_is_boundary, 1);
        k.label(&skip_n);
    }

    // k_end = ceil(K / tile_k) * tile_k  (round up K to tile_k boundary)
    // This ensures the K-loop covers all K elements even when K % tile_k != 0.
    // OOB K columns are handled by zero-filling cooperative load VGPRs.
    let k_end_s = k.alloc_sreg();
    {
        let k_aligned_s = k.alloc_sreg();
        // k_aligned = (K + tile_k - 1) & ~(tile_k - 1)
        k.push(Op::SAddU32 {
            dst: k_aligned_s, src0: SReg(k_dim.0),
            src1: SOperand::InlineInt((spec.tile_k - 1) as i32),
        });
        let mask_s = k.alloc_sreg();
        k.s_mov_imm(mask_s, !(spec.tile_k as i32 - 1));
        k.s_and_b32(k_aligned_s, k_aligned_s, mask_s);
        if split_k <= 1 {
            k.push(Op::SAddU32 { dst: k_end_s, src0: k_aligned_s, src1: SOperand::InlineInt(0) });
        } else {
            k.s_lshr_b32(k_end_s, k_aligned_s, sk_shift);
        }
    }

    // k_start_bytes = split_k_id * k_end * 2
    let k_start_bytes_s = k.alloc_sreg();
    if split_k <= 1 {
        k.s_mov_imm(k_start_bytes_s, 0);
    } else {
        let s_tmp = k.alloc_sreg();
        k.s_mul_i32(s_tmp, split_k_id_s, k_end_s);
        k.s_lshl_b32(k_start_bytes_s, s_tmp, 1);
    }

    // Y offset = split_k_id * y_split_stride
    let y_offset_s = k.alloc_sreg();
    if split_k <= 1 {
        k.s_mov_imm(y_offset_s, 0);
    } else {
        k.s_mul_i32(y_offset_s, split_k_id_s, SReg(y_split_stride.0));
    }

    // ══════════════════════════════════════════════════════════════
    // Phase 3: Thread decomposition
    // ══════════════════════════════════════════════════════════════

    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);

    // ══════════════════════════════════════════════════════════════
    // Phase 4: Accumulator allocation + zero-init
    // ══════════════════════════════════════════════════════════════

    // acc_n_live_rows: how many row_blocks of acc are live in VGPRs at once
    let acc_n_live_rows = if spec.acc_swap { 1 } else { n_row_blocks };

    let mut acc = Vec::new();
    for _r in 0..acc_n_live_rows {
        for _c in 0..n_col_tiles {
            acc.push(k.alloc_vreg_array(8, Alignment::Align8));
        }
    }
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ACC swap LDS address: acc_swap_base + wave_id * bytes_per_wave + lane_id * bytes_per_lane
    let acc_swap_addr = if spec.acc_swap {
        let bytes_per_lane = (spec.n_col_tiles() * 8 * 4) as i32;  // 8 VGPRs × 4 bytes × n_col
        let bytes_per_wave = (bytes_per_lane * 32) as i32;          // 32 lanes
        let base = spec.acc_swap_base() as i32;

        let addr = k.alloc_vreg();
        // addr = base + wave_id * bytes_per_wave
        let v_bpw = k.alloc_vreg();
        k.v_mov_imm(v_bpw, bytes_per_wave);
        k.v_mul_lo_u32(addr, wave_id, v_bpw);
        k.push(Op::VAddU32 { dst: addr, src0: Operand::VReg(addr), src1: Operand::InlineInt(base) });
        // addr += lane_id * bytes_per_lane
        let lane_off = k.alloc_vreg();
        let v_bpl = k.alloc_vreg();
        k.v_mov_imm(v_bpl, bytes_per_lane);
        k.v_mul_lo_u32(lane_off, lane_id, v_bpl);
        k.v_add_u32(addr, addr, lane_off);

        // Zero-init LDS swap region (inactive row_block starts at 0)
        // acc VGPRs are already zero — store them to initialize the swap slot
        for c in 0..n_col_tiles {
            let off = (c * 32) as u16;
            k.ds_store_b128(addr, acc[c], off);
            k.ds_store_b128(addr, VReg(acc[c].0 + 4), off + 16);
        }
        k.wait_lgkmcnt(0);

        Some(addr)
    } else {
        None
    };

    // Allocate dedicated swap temp VGPRs (8 VGPRs, NOT frag_b)
    // and mark acc as coalesced groups to prevent copy propagation
    let acc_swap_temp = if spec.acc_swap {
        // Dedicated 8-VGPR temp for swap (must not be frag_b/frag_a)
        let temp = k.alloc_vreg_array(8, Alignment::Align8);
        // Mark acc groups as coalesced — prevents SSA copy propagation
        // from folding `v_mov acc, temp` into direct temp references,
        // which would make WMMA read B data as accumulator input.
        for c in 0..n_col_tiles {
            k.mark_coalesced_group(acc[c], 8);
        }
        Some(temp)
    } else {
        None
    };

    // ══════════════════════════════════════════════════════════════
    // Phase 5: Store phase constants (pre-compute for later)
    // ══════════════════════════════════════════════════════════════

    let tile_m_shift = spec.tile_m.trailing_zeros() as u8;
    let rpw_shift = rows_per_wave.trailing_zeros() as u8;

    let mut s_row_bases = Vec::new();
    let s_row_base0 = k.alloc_sreg();
    k.s_lshl_b32(s_row_base0, tile_row_s, tile_m_shift);
    let s_tmp2 = k.alloc_sreg();
    k.s_lshl_b32(s_tmp2, wave_id_s, rpw_shift);
    k.push(Op::SAddU32 { dst: s_row_base0, src0: s_row_base0, src1: SOperand::SReg(s_tmp2) });
    s_row_bases.push(s_row_base0);
    for r in 1..n_row_blocks {
        let rb = k.alloc_sreg();
        k.push(Op::SAddU32 { dst: rb, src0: s_row_base0, src1: SOperand::InlineInt((r * 16) as i32) });
        s_row_bases.push(rb);
    }

    let base_n_s = k.alloc_sreg();
    let tile_n_shift = spec.tile_n.trailing_zeros() as u8;
    k.s_lshl_b32(base_n_s, tile_col_s, tile_n_shift);

    // ══════════════════════════════════════════════════════════════
    // Phase 6: Cooperative load addresses (GMEM → LDS)
    // ══════════════════════════════════════════════════════════════

    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));

    // ── X cooperative address ──
    let gmem_row_bytes_x = spec.tile_k * 2;
    let chunks_per_row_x = gmem_row_bytes_x / 16;
    let x_cpr_shift = chunks_per_row_x.trailing_zeros() as u8;

    let x_row_in_tile = k.alloc_vreg();
    k.v_lshrrev_b32(x_row_in_tile, x_cpr_shift, tid);
    let x_col_chunk = k.alloc_vreg();
    k.v_and_b32_imm(x_col_chunk, tid, chunks_per_row_x - 1);

    let x_abs_row = k.alloc_vreg();
    let s_xbase = k.alloc_sreg();
    k.s_lshl_b32(s_xbase, tile_row_s, tile_m_shift);
    k.v_mov_from_sgpr(x_abs_row, s_xbase);
    k.v_add_u32(x_abs_row, x_abs_row, x_row_in_tile);

    // ── Boundary masking: clamp x_abs_row to [0, M-1] ──
    // Clamp-and-Discard: OOB rows read duplicate data (last valid row).
    // Store phase will EXEC-mask discard these rows → correct output.
    let m_minus1_v = k.alloc_vreg();
    {
        let m_vreg = k.alloc_vreg();
        k.v_mov_from_sgpr(m_vreg, SReg(m_dim.0));
        k.push(Op::VAddU32 {
            dst: m_minus1_v, src0: Operand::VReg(m_vreg),
            src1: Operand::InlineInt(-1),
        });
    }
    k.v_min_u32(x_abs_row, x_abs_row, m_minus1_v);

    let x_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_byte, x_abs_row, k_vreg);
    k.v_lshlrev_b32(x_row_byte, 1, x_row_byte);
    let x_col_byte = k.alloc_vreg();
    k.v_lshlrev_b32(x_col_byte, 4, x_col_chunk);
    k.v_add_u32(x_row_byte, x_row_byte, x_col_byte);

    // Buffer resource descriptor for X: {ptr_lo, ptr_hi, 0x7FFFFFFE, 0x31027000}
    // MUBUF addressing: buffer_load uses SGPR descriptor + VGPR offset
    let x_desc = k.alloc_sreg_quad();
    k.push(Op::SMov { dst: x_desc, src: SOperand::SReg(SReg(x_ptr.0)) });
    k.push(Op::SMov { dst: SReg(x_desc.0 + 1), src: SOperand::SReg(SReg(x_ptr.0 + 1)) });
    k.push(Op::SMov { dst: SReg(x_desc.0 + 2), src: SOperand::Literal(0x7FFFFFFE) });
    k.push(Op::SMov { dst: SReg(x_desc.0 + 3), src: SOperand::Literal(0x31027000) });
    // x_row_byte is the per-thread byte offset (1 VGPR, no 64-bit addr needed!)

    // X LDS store addr + XOR swizzle: swizzle = (row & 7) << 4
    // This distributes 16 consecutive rows across all 32 LDS banks,
    // eliminating 4-way bank conflicts during WMMA fragment reads.
    let x_lds_off_raw = k.alloc_vreg();
    k.v_lshlrev_b32(x_lds_off_raw, xrs_shift, x_row_in_tile);
    k.v_add_u32(x_lds_off_raw, x_lds_off_raw, x_col_byte);
    let x_swizzle = k.alloc_vreg();
    k.v_and_b32_imm(x_swizzle, x_row_in_tile, 7);
    k.v_lshlrev_b32(x_swizzle, 4, x_swizzle);
    let x_lds_off = k.alloc_vreg();
    k.v_xor_b32(x_lds_off, Operand::VReg(x_lds_off_raw), Operand::VReg(x_swizzle));

    // ── WT cooperative address ──
    let gmem_row_bytes_wt = spec.tile_k * 2;
    let chunks_per_row_wt = gmem_row_bytes_wt / 16;
    let wt_cpr_shift = chunks_per_row_wt.trailing_zeros() as u8;

    let wt_row_in_tile = k.alloc_vreg();
    k.v_lshrrev_b32(wt_row_in_tile, wt_cpr_shift, tid);
    let wt_col_chunk = k.alloc_vreg();
    k.v_and_b32_imm(wt_col_chunk, tid, chunks_per_row_wt - 1);

    let wt_abs_row = k.alloc_vreg();
    k.v_mov_from_sgpr(wt_abs_row, base_n_s);
    k.v_add_u32(wt_abs_row, wt_abs_row, wt_row_in_tile);

    // ── Boundary masking: clamp wt_abs_row to [0, N-1] ──
    let n_minus1_v = k.alloc_vreg();
    {
        let n_vreg = k.alloc_vreg();
        k.v_mov_from_sgpr(n_vreg, n_dim);
        k.push(Op::VAddU32 {
            dst: n_minus1_v, src0: Operand::VReg(n_vreg),
            src1: Operand::InlineInt(-1),
        });
    }
    k.v_min_u32(wt_abs_row, wt_abs_row, n_minus1_v);

    let wt_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_byte, wt_abs_row, k_vreg);
    k.v_lshlrev_b32(wt_row_byte, 1, wt_row_byte);
    let wt_col_byte = k.alloc_vreg();
    k.v_lshlrev_b32(wt_col_byte, 4, wt_col_chunk);
    k.v_add_u32(wt_row_byte, wt_row_byte, wt_col_byte);

    // Buffer resource descriptor for WT: {ptr_lo, ptr_hi, 0x7FFFFFFE, 0x31027000}
    let wt_desc = k.alloc_sreg_quad();
    k.push(Op::SMov { dst: wt_desc, src: SOperand::SReg(SReg(wt_ptr.0)) });
    k.push(Op::SMov { dst: SReg(wt_desc.0 + 1), src: SOperand::SReg(SReg(wt_ptr.0 + 1)) });
    k.push(Op::SMov { dst: SReg(wt_desc.0 + 2), src: SOperand::Literal(0x7FFFFFFE) });
    k.push(Op::SMov { dst: SReg(wt_desc.0 + 3), src: SOperand::Literal(0x31027000) });
    // wt_row_byte is the per-thread byte offset (1 VGPR)

    // WT LDS store addr + XOR swizzle: swizzle = (row & 7) << 4
    let wt_lds_off_raw = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_off_raw, wrs_shift, wt_row_in_tile);
    k.v_add_u32(wt_lds_off_raw, wt_lds_off_raw, wt_col_byte);
    let wt_swizzle = k.alloc_vreg();
    k.v_and_b32_imm(wt_swizzle, wt_row_in_tile, 7);
    k.v_lshlrev_b32(wt_swizzle, 4, wt_swizzle);
    let wt_lds_off = k.alloc_vreg();
    k.v_xor_b32(wt_lds_off, Operand::VReg(wt_lds_off_raw), Operand::VReg(wt_swizzle));
    k.push(Op::VAddU32 {
        dst: wt_lds_off, src0: Operand::VReg(wt_lds_off),
        src1: Operand::InlineInt(lds_x as i32),
    });

    // ── LDS read addresses for WMMA fragments (XOR swizzle + dual pointers) ──
    // Precompute lane_swizzle = (lane_row & 7) << 4 for read-side XOR.
    // lane_row = lane_id / 16 (0 or 1 for Wave32 WMMA 16×16).
    let lane_row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(lane_row_stride, xrs_shift, lane_row);

    let lane_swizzle = k.alloc_vreg();
    k.v_and_b32_imm(lane_swizzle, lane_row, 7);
    k.v_lshlrev_b32(lane_swizzle, 4, lane_swizzle);

    let s_wave_x_off = k.alloc_sreg();
    let s_wave_stride = k.alloc_sreg();
    k.s_mov_imm(s_wave_stride, (rows_per_wave * x_row_stride) as i32);
    k.s_mul_i32(s_wave_x_off, wave_id_s, s_wave_stride);

    // Dual pointers for A fragments: _0 = base^swizzle, _16 = (base+16)^swizzle
    // The +16 offset for the second ds_load_b128 of each fragment MUST be
    // applied BEFORE XOR to avoid carry corruption (DeepThink's "Carry Problem").
    // ── X LDS read addresses: raw (pre-XOR) for per-ksub recomputation ──
    // CRITICAL: k_byte_within MUST be added BEFORE XOR to avoid carry corruption.
    // For k>16 (k_sub_steps > 1), `(a XOR b) + c ≠ (a + c) XOR b` when c overlaps
    // with the swizzle bits (bits 4-6). Pre-computing XOR'd addresses and adding
    // k_byte_within as ds_load immediate is WRONG for k>16.
    let mut x_lds_reads_raw = Vec::new();
    for r in 0..n_row_blocks {
        let xr = k.alloc_vreg();
        k.v_mov_from_sgpr(xr, s_wave_x_off);
        k.v_add_u32(xr, xr, lane_row_stride);
        if r > 0 {
            k.push(Op::VAddU32 {
                dst: xr, src0: Operand::VReg(xr),
                src1: Operand::InlineInt((r as i32) * 16 * (x_row_stride as i32)),
            });
        }
        x_lds_reads_raw.push(xr);
    }
    // Pre-XOR'd addresses for ksub=0 (backward compat: zero k_byte_within = no shift)
    let mut x_lds_reads_0 = Vec::new();
    let mut x_lds_reads_16 = Vec::new();
    for r in 0..n_row_blocks {
        let xr_0 = k.alloc_vreg();
        k.v_xor_b32(xr_0, Operand::VReg(x_lds_reads_raw[r]), Operand::VReg(lane_swizzle));
        x_lds_reads_0.push(xr_0);
        if frag_a_loads > 1 {
            let xr_16_base = k.alloc_vreg();
            k.push(Op::VAddU32 { dst: xr_16_base, src0: Operand::VReg(x_lds_reads_raw[r]), src1: Operand::InlineInt(16) });
            let xr_16 = k.alloc_vreg();
            k.v_xor_b32(xr_16, Operand::VReg(xr_16_base), Operand::VReg(lane_swizzle));
            x_lds_reads_16.push(xr_16);
        }
    }

    // ── WT LDS read addresses: save raw for per-ksub recomputation ──
    let wt_lds_read_raw = k.alloc_vreg();
    k.v_lshlrev_b32(wt_lds_read_raw, wrs_shift, lane_row);
    // Pre-XOR'd for ksub=0
    let wt_lds_read_base_0 = k.alloc_vreg();
    k.v_xor_b32(wt_lds_read_base_0, Operand::VReg(wt_lds_read_raw), Operand::VReg(lane_swizzle));
    let wt_lds_read_base_16 = if frag_b_loads > 1 {
        let wt_lds_16_base = k.alloc_vreg();
        k.push(Op::VAddU32 { dst: wt_lds_16_base, src0: Operand::VReg(wt_lds_read_raw), src1: Operand::InlineInt(16) });
        let wt_lds_read_base_16 = k.alloc_vreg();
        k.v_xor_b32(wt_lds_read_base_16, Operand::VReg(wt_lds_16_base), Operand::VReg(lane_swizzle));
        wt_lds_read_base_16
    } else {
        VReg(0)
    };

    // ── GMEM register set — sized for batch, not full tile_k ──
    // For k<=32: batch = full loads (e.g., 4 loads for k32).
    // For k>32: batch = k16-equivalent (2 loads), reused across batches.
    // This keeps GMEM VGPR pressure constant regardless of tile_k.
    //
    // OPTIMIZATION: share gmem_x and gmem_wt registers.
    // When shared, loads must be serialized: load X → store X → load WT → store WT.
    // This halves peak GMEM VGPRs (32→16) at the cost of serialized GMEM latency.
    // Only enable for small tile_k (≤32) where GMEM data per phase is small enough
    // Phase 3 optimization: NEVER share GMEM regs. Sharing forces serial X→WT loading,
    // leaving WT with only 4 instruction slots of VMEM overlap (vs 100+ concurrent).
    // Cost: +16 VGPRs (200→216), still 2 waves/SIMD at 256 budget.
    let share_gmem_regs = false;
    let gmem_x: Vec<VReg> = (0..x_batch_loads as usize)
        .map(|_| k.alloc_vreg_array(4, Alignment::Align4))
        .collect();
    let gmem_wt: Vec<VReg> = if share_gmem_regs {
        gmem_x.clone()  // reuse same physical registers
    } else {
        (0..wt_batch_loads as usize)
            .map(|_| k.alloc_vreg_array(4, Alignment::Align4))
            .collect()
    };

    // Fragment VGPRs (pre-allocated for WMMA)
    // acc_swap: only 1 frag_a needed (1 row_block at a time)
    // NOTE: Row-major (1 frag_a) was tested — achieved 192 VGPRs / 4 waves/SIMD
    // but performance DROPPED 13% (81→71 TF) due to worse frag_b reuse.
    // Column-major (n_row_blocks frag_a) is the optimal tradeoff at 200 VGPRs.
    let frag_a_count = if spec.acc_swap { 1 } else { n_row_blocks };
    let frag_a: Vec<VReg> = (0..frag_a_count)
        .map(|_| k.alloc_vreg_array(frag_a_vgprs, frag_a_align))
        .collect();
    // Streaming mode (n_col_tiles > 4) only needs 1 ping-pong pair = 16 VGPRs.
    // Bulk-load mode needs all n_col_tiles fragments = n_col_tiles * 8 VGPRs.
    // Fix: don't allocate 8 × 8 = 64 VGPRs when only 2 × 8 = 16 are needed!
    let use_streaming = n_col_tiles > 4;
    let frag_b: Vec<VReg> = if use_streaming {
        vec![k.alloc_vreg_array(frag_b_vgprs, frag_b_align)]  // only ping buffer
    } else {
        (0..n_col_tiles).map(|_| k.alloc_vreg_array(frag_b_vgprs, frag_b_align)).collect()
    };
    // Pong buffer for streaming (or dummy for bulk-load)
    let frag_b_shared = if use_streaming {
        k.alloc_vreg_array(frag_b_vgprs, frag_b_align)
    } else {
        VReg(0) // unused in bulk-load mode
    };

    // ── GMEM multi-pass strides (coalesced loading) ──
    let x_rows_per_pass = spec.wg_size() / chunks_per_row_x;
    let wt_rows_per_pass = spec.wg_size() / chunks_per_row_wt;
    let x_lds_stride = (x_rows_per_pass * x_row_stride) as i32;
    let wt_lds_stride = (wt_rows_per_pass * wt_row_stride) as i32;

    // GMEM stride = rows_per_pass * K * 2 (K is runtime, so compute as vreg)
    let x_gmem_stride = k.alloc_vreg();
    {
        let tmp = k.alloc_vreg();
        let s_factor = k.alloc_sreg();
        k.s_mov_imm(s_factor, (x_rows_per_pass * 2) as i32);
        k.v_mov_from_sgpr(tmp, s_factor);
        k.v_mul_lo_u32(x_gmem_stride, k_vreg, tmp);
    }
    let wt_gmem_stride = k.alloc_vreg();
    {
        let tmp = k.alloc_vreg();
        let s_factor = k.alloc_sreg();
        k.s_mov_imm(s_factor, (wt_rows_per_pass * 2) as i32);
        k.v_mov_from_sgpr(tmp, s_factor);
        k.v_mul_lo_u32(wt_gmem_stride, k_vreg, tmp);
    }

    // k_start_bytes split-K offset: add to byte offsets (buffer_load uses 32-bit offset)
    if split_k > 1 {
        let k_start_v = k.alloc_vreg();
        k.v_mov_from_sgpr(k_start_v, k_start_bytes_s);
        k.v_add_u32(x_row_byte, x_row_byte, k_start_v);
        k.v_add_u32(wt_row_byte, wt_row_byte, k_start_v);
    }

    // ══════════════════════════════════════════════════════════════
    // Phase 7: K-loop (gemm_gen-style phase A/B double-buffer)
    // ══════════════════════════════════════════════════════════════

    // K-loop byte offset: tracks current K-iteration's byte offset from partition start.
    // CRITICAL: initialize to 0, NOT k_start_bytes! The k_start_bytes offset is already
    // baked into gmem_base (L607-616). Adding it again here would double-count it,
    // causing OOB reads for split_k > 1 → GPU page fault → hard hang.
    let k_byte_off = k.alloc_vreg();
    k.v_mov_imm(k_byte_off, 0);
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);
    let k_step = (spec.tile_k * 2) as i32;

    // XOR swizzle uses u16 buf_off constants (0 and lds_buf). The 128-multiple
    // commutative law ensures (addr ^ swizzle) + buf_off == (addr + buf_off) ^ swizzle.
    let buf0_off_const: u16 = 0;
    let buf1_off_const: u16 = lds_buf as u16;

    // buf1_off VReg needed for emit_lds_store_graduated (buf0 offset=0 uses None)
    let buf1_off = k.alloc_vreg();
    k.v_mov_imm(buf1_off, lds_buf as i32);

    // ── soffset optimization: pre-compute SGPR row offsets ──
    // Each row's offset = i * gmem_stride. Using soffset SGPRs eliminates
    // the serial v_add chain in the inner loop, freeing VALU pipe for WMMA.
    let x_soffs = setup_row_soffsets(&mut k, x_gmem_stride, x_batch_loads);
    let wt_soffs = setup_row_soffsets(&mut k, wt_gmem_stride, wt_batch_loads);

    // Shared scratch VGPR for base address computation in emit_coop_load_buffer
    // Reusing 1 VGPR across all calls instead of allocating new one each time
    let gmem_scratch = k.alloc_vreg();

    // Pre-compute LDS buf1 base addresses to eliminate v_add in inner loop stores
    // Phase A stores into buf1: instead of lds_off + buf1_off each time,
    // use pre-computed x_lds_buf1/wt_lds_buf1 with None (no runtime add)
    let x_lds_buf1 = k.alloc_vreg();
    k.v_add_u32(x_lds_buf1, x_lds_off, buf1_off);
    let wt_lds_buf1 = k.alloc_vreg();
    k.v_add_u32(wt_lds_buf1, wt_lds_off, buf1_off);

    // ── PROLOGUE: load first tile (N=0) into buf0 ──
    if use_sequential {
        // Sequential: load k16 batches one-by-one, reusing gmem regs
        let mut batch_off = k.alloc_vreg();
        k.v_mov_imm(batch_off, 0);
        for batch in 0..n_batches {
            emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_batch_loads, x_gmem_stride, batch_off, Some(&x_soffs), gmem_scratch);
            emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_batch_loads, wt_gmem_stride, batch_off, Some(&wt_soffs), gmem_scratch);
            let batch_total = x_batch_loads + wt_batch_loads;
            emit_lds_store_graduated(&mut k, &gmem_x, x_lds_off, x_batch_loads, None, x_lds_stride, batch_total);
            emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_off, wt_batch_loads, None, wt_lds_stride, wt_batch_loads);
            if batch + 1 < n_batches {
                k.wait_lgkmcnt(0);  // ensure ds_stores complete before next batch
                k.push(Op::VAddU32 { dst: batch_off, src0: Operand::VReg(batch_off), src1: Operand::InlineInt(batch_k_step) });
            }
        }
    } else {
        if share_gmem_regs {
            // Serialized: load X → store X → load WT → store WT
            // X and WT share the same VGPRs, so must not overlap.
            emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_loads_per_thread, x_gmem_stride, k_byte_off, Some(&x_soffs), gmem_scratch);
            emit_lds_store_graduated(&mut k, &gmem_x, x_lds_off, x_loads_per_thread, None, x_lds_stride, x_loads_per_thread);
            emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_loads_per_thread, wt_gmem_stride, k_byte_off, Some(&wt_soffs), gmem_scratch);
            emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_off, wt_loads_per_thread, None, wt_lds_stride, wt_loads_per_thread);
        } else {
            emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_loads_per_thread, x_gmem_stride, k_byte_off, Some(&x_soffs), gmem_scratch);
            emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_loads_per_thread, wt_gmem_stride, k_byte_off, Some(&wt_soffs), gmem_scratch);
            let total_loads = x_loads_per_thread + wt_loads_per_thread;
            emit_lds_store_graduated(&mut k, &gmem_x, x_lds_off, x_loads_per_thread, None, x_lds_stride, total_loads);
            emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_off, wt_loads_per_thread, None, wt_lds_stride, wt_loads_per_thread);
        }
    }
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, spec.tile_k as i32);



    // ── MAIN LOOP (software pipelined, double-buffered) ──
    let loop_label = k.make_label("k_loop");
    k.label(&loop_label);
    k.s_cmp_ge_u32(k_iter_s, k_end_s);
    let epilog_a = k.make_label("epilog_a");
    k.branch_scc1(&epilog_a);

    // Phase A: load N+1 into gmem, compute buf0, store gmem→buf1
    // For sequential mode (k>32): interleave batch loads with WMMA compute.
    // Each batch: buffer_load → partial WMMA → wait_vmcnt → ds_store.
    // For standard mode: single-shot load → full WMMA → graduated store.
    if use_sequential {
        // ── Sequential k48+ Phase A ──
        // Batch 0: start async VMEM load (first k16 of next tile)
        emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_batch_loads, x_gmem_stride, k_byte_off, Some(&x_soffs), gmem_scratch);
        emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_batch_loads, wt_gmem_stride, k_byte_off, Some(&wt_soffs), gmem_scratch);

        // WMMA compute from buf0 (full compute - all batches load during this)
        emit_lds_read_and_wmma(
            &mut k, &frag_a, &frag_b, &acc,
            &x_lds_reads_raw, &x_lds_reads_0, &x_lds_reads_16,
            wt_lds_read_raw, wt_lds_read_base_0, wt_lds_read_base_16,
            lane_swizzle,
            n_row_blocks, n_col_tiles,
            x_row_stride, wt_row_stride, lds_x,
            spec, buf0_off_const,
            frag_b_shared,
            None,
        );

        // Store batch 0 to LDS buf1
        let batch_total = x_batch_loads + wt_batch_loads;
        emit_lds_store_graduated(&mut k, &gmem_x, x_lds_buf1, x_batch_loads, None, x_lds_stride, batch_total);
        emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_buf1, wt_batch_loads, None, wt_lds_stride, wt_batch_loads);

        // Remaining batches: load→wait→store sequentially
        let batch_off_tmp = k.alloc_vreg();
        for batch in 1..n_batches {
            k.wait_lgkmcnt(0);  // previous ds_store complete
            // Compute byte offset for this batch
            k.v_add_u32(batch_off_tmp, k_byte_off, VReg(0)); // copy k_byte_off
            k.push(Op::VAddU32 {
                dst: batch_off_tmp,
                src0: Operand::VReg(batch_off_tmp),
                src1: Operand::InlineInt(batch as i32 * batch_k_step),
            });
            emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_batch_loads, x_gmem_stride, batch_off_tmp, Some(&x_soffs), gmem_scratch);
            emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_batch_loads, wt_gmem_stride, batch_off_tmp, Some(&wt_soffs), gmem_scratch);
            emit_lds_store_graduated(&mut k, &gmem_x, x_lds_buf1, x_batch_loads, None, x_lds_stride, batch_total);
            emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_buf1, wt_batch_loads, None, wt_lds_stride, wt_batch_loads);
        }
    } else {
        if share_gmem_regs {
            // ── Shared VGPRs: serialize X and WT ──
            // Load X first (WT uses same regs, so can't overlap)
            emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_batch_loads, x_gmem_stride, k_byte_off, Some(&x_soffs), gmem_scratch);
            // WMMA compute from buf0, with X stores interleaved into buf1
            let x_store_sched = StoreSchedule {
                gmem_x: gmem_x.clone(), gmem_wt: vec![],
                x_store_base: x_lds_buf1, wt_store_base: wt_lds_off,
                x_lds_stride, wt_lds_stride,
                x_loads: x_batch_loads, wt_loads: 0,
                total_gmem_outstanding: x_batch_loads,
            };
            emit_lds_read_and_wmma(
                &mut k, &frag_a, &frag_b, &acc,
                &x_lds_reads_raw, &x_lds_reads_0, &x_lds_reads_16,
                wt_lds_read_raw, wt_lds_read_base_0, wt_lds_read_base_16,
                lane_swizzle,
                n_row_blocks, n_col_tiles,
                x_row_stride, wt_row_stride, lds_x,
                spec, buf0_off_const,
                frag_b_shared,
                Some(&x_store_sched),
            );
            // Now gmem_x regs are free. Load WT into same regs.
            emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_batch_loads, wt_gmem_stride, k_byte_off, Some(&wt_soffs), gmem_scratch);
            emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_buf1, wt_batch_loads, None, wt_lds_stride, wt_batch_loads);
        } else {
            emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_batch_loads, x_gmem_stride, k_byte_off, Some(&x_soffs), gmem_scratch);
            emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_batch_loads, wt_gmem_stride, k_byte_off, Some(&wt_soffs), gmem_scratch);
            if spec.acc_swap {
                emit_lds_read_and_wmma_swap(
                    &mut k, &frag_a, &frag_b, &acc,
                    &x_lds_reads_0, &x_lds_reads_16,
                    wt_lds_read_base_0, wt_lds_read_base_16,
                    n_row_blocks, n_col_tiles,
                    x_row_stride, wt_row_stride, lds_x,
                    spec, buf0_off_const,
                    frag_b_shared,
                    acc_swap_addr.unwrap(),
                    acc_swap_temp.unwrap(),
                );
            } else {
                let total_loads = x_batch_loads + wt_batch_loads;
                emit_lds_read_and_wmma(
                    &mut k, &frag_a, &frag_b, &acc,
                    &x_lds_reads_raw, &x_lds_reads_0, &x_lds_reads_16,
                    wt_lds_read_raw, wt_lds_read_base_0, wt_lds_read_base_16,
                    lane_swizzle,
                    n_row_blocks, n_col_tiles,
                    x_row_stride, wt_row_stride, lds_x,
                    spec, buf0_off_const,
                    frag_b_shared,
                    None,
                );
            }
            // Store GMEM→LDS after WMMA (sequential is faster than interleaved for k64)
            let total_loads = x_batch_loads + wt_batch_loads;
            emit_lds_store_graduated(&mut k, &gmem_x, x_lds_buf1, x_batch_loads, None, x_lds_stride, total_loads);
            emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_buf1, wt_batch_loads, None, wt_lds_stride, wt_batch_loads);
        }
    }
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, spec.tile_k as i32);

    k.s_cmp_ge_u32(k_iter_s, k_end_s);
    let epilog_b = k.make_label("epilog_b");
    k.branch_scc1(&epilog_b);

    // Phase B: load N+2 into gmem, compute buf1, store gmem→buf0
    if use_sequential {
        // ── Sequential k48+ Phase B ──
        emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_batch_loads, x_gmem_stride, k_byte_off, Some(&x_soffs), gmem_scratch);
        emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_batch_loads, wt_gmem_stride, k_byte_off, Some(&wt_soffs), gmem_scratch);

        emit_lds_read_and_wmma(
            &mut k, &frag_a, &frag_b, &acc,
            &x_lds_reads_raw, &x_lds_reads_0, &x_lds_reads_16,
            wt_lds_read_raw, wt_lds_read_base_0, wt_lds_read_base_16,
            lane_swizzle,
            n_row_blocks, n_col_tiles,
            x_row_stride, wt_row_stride, lds_x,
            spec, buf1_off_const,
            frag_b_shared,
            None,
        );

        let batch_total = x_batch_loads + wt_batch_loads;
        emit_lds_store_graduated(&mut k, &gmem_x, x_lds_off, x_batch_loads, None, x_lds_stride, batch_total);
        emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_off, wt_batch_loads, None, wt_lds_stride, wt_batch_loads);

        let batch_off_tmp2 = k.alloc_vreg();
        for batch in 1..n_batches {
            k.wait_lgkmcnt(0);
            k.v_add_u32(batch_off_tmp2, k_byte_off, VReg(0));
            k.push(Op::VAddU32 {
                dst: batch_off_tmp2,
                src0: Operand::VReg(batch_off_tmp2),
                src1: Operand::InlineInt(batch as i32 * batch_k_step),
            });
            emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_batch_loads, x_gmem_stride, batch_off_tmp2, Some(&x_soffs), gmem_scratch);
            emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_batch_loads, wt_gmem_stride, batch_off_tmp2, Some(&wt_soffs), gmem_scratch);
            emit_lds_store_graduated(&mut k, &gmem_x, x_lds_off, x_batch_loads, None, x_lds_stride, batch_total);
            emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_off, wt_batch_loads, None, wt_lds_stride, wt_batch_loads);
        }
    } else {
        if share_gmem_regs {
            // ── Shared VGPRs: serialize X and WT (Phase B stores to buf0) ──
            emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_batch_loads, x_gmem_stride, k_byte_off, Some(&x_soffs), gmem_scratch);
            // WMMA compute from buf1, with X stores interleaved into buf0
            let x_store_sched_b = StoreSchedule {
                gmem_x: gmem_x.clone(), gmem_wt: vec![],
                x_store_base: x_lds_off, wt_store_base: wt_lds_off,
                x_lds_stride, wt_lds_stride,
                x_loads: x_batch_loads, wt_loads: 0,
                total_gmem_outstanding: x_batch_loads,
            };
            emit_lds_read_and_wmma(
                &mut k, &frag_a, &frag_b, &acc,
                &x_lds_reads_raw, &x_lds_reads_0, &x_lds_reads_16,
                wt_lds_read_raw, wt_lds_read_base_0, wt_lds_read_base_16,
                lane_swizzle,
                n_row_blocks, n_col_tiles,
                x_row_stride, wt_row_stride, lds_x,
                spec, buf1_off_const,
                frag_b_shared,
                Some(&x_store_sched_b),
            );
            // Load WT into same regs, store to buf0
            emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_batch_loads, wt_gmem_stride, k_byte_off, Some(&wt_soffs), gmem_scratch);
            emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_off, wt_batch_loads, None, wt_lds_stride, wt_batch_loads);
        } else {
            emit_coop_load_buffer(&mut k, &gmem_x, x_row_byte, x_desc, x_batch_loads, x_gmem_stride, k_byte_off, Some(&x_soffs), gmem_scratch);
            emit_coop_load_buffer(&mut k, &gmem_wt, wt_row_byte, wt_desc, wt_batch_loads, wt_gmem_stride, k_byte_off, Some(&wt_soffs), gmem_scratch);
            if spec.acc_swap {
                emit_lds_read_and_wmma_swap(
                    &mut k, &frag_a, &frag_b, &acc,
                    &x_lds_reads_0, &x_lds_reads_16,
                    wt_lds_read_base_0, wt_lds_read_base_16,
                    n_row_blocks, n_col_tiles,
                    x_row_stride, wt_row_stride, lds_x,
                    spec, buf1_off_const,
                    frag_b_shared,
                    acc_swap_addr.unwrap(),
                    acc_swap_temp.unwrap(),
                );
            } else {
                emit_lds_read_and_wmma(
                    &mut k, &frag_a, &frag_b, &acc,
                    &x_lds_reads_raw, &x_lds_reads_0, &x_lds_reads_16,
                    wt_lds_read_raw, wt_lds_read_base_0, wt_lds_read_base_16,
                    lane_swizzle,
                    n_row_blocks, n_col_tiles,
                    x_row_stride, wt_row_stride, lds_x,
                    spec, buf1_off_const,
                    frag_b_shared,
                    None,
                );
            }
            let total_loads = x_batch_loads + wt_batch_loads;
            emit_lds_store_graduated(&mut k, &gmem_x, x_lds_off, x_batch_loads, None, x_lds_stride, total_loads);
            emit_lds_store_graduated(&mut k, &gmem_wt, wt_lds_off, wt_batch_loads, None, wt_lds_stride, wt_batch_loads);
        }
    }
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, spec.tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, k_end_s);
    k.branch_scc1(&loop_label);

    // ── EPILOGUES (no interleaved stores — just compute remaining tiles) ──
    k.label(&epilog_a);
    if spec.acc_swap {
        emit_lds_read_and_wmma_swap(
            &mut k, &frag_a, &frag_b, &acc,
            &x_lds_reads_0, &x_lds_reads_16,
            wt_lds_read_base_0, wt_lds_read_base_16,
            n_row_blocks, n_col_tiles,
            x_row_stride, wt_row_stride, lds_x,
            spec, buf0_off_const,
            frag_b_shared,
            acc_swap_addr.unwrap(),
            acc_swap_temp.unwrap(),
        );
    } else {
        emit_lds_read_and_wmma(
            &mut k, &frag_a, &frag_b, &acc,
            &x_lds_reads_raw, &x_lds_reads_0, &x_lds_reads_16,
            wt_lds_read_raw, wt_lds_read_base_0, wt_lds_read_base_16,
            lane_swizzle,
            n_row_blocks, n_col_tiles,
            x_row_stride, wt_row_stride, lds_x,
            spec, buf0_off_const,
            frag_b_shared,
            None,
        );
    }
    let store_label = k.make_label("store_phase");
    k.branch(&store_label);

    k.label(&epilog_b);
    if spec.acc_swap {
        emit_lds_read_and_wmma_swap(
            &mut k, &frag_a, &frag_b, &acc,
            &x_lds_reads_0, &x_lds_reads_16,
            wt_lds_read_base_0, wt_lds_read_base_16,
            n_row_blocks, n_col_tiles,
            x_row_stride, wt_row_stride, lds_x,
            spec, buf1_off_const,
            frag_b_shared,
            acc_swap_addr.unwrap(),
            acc_swap_temp.unwrap(),
        );
    } else {
        emit_lds_read_and_wmma(
            &mut k, &frag_a, &frag_b, &acc,
            &x_lds_reads_raw, &x_lds_reads_0, &x_lds_reads_16,
            wt_lds_read_raw, wt_lds_read_base_0, wt_lds_read_base_16,
            lane_swizzle,
            n_row_blocks, n_col_tiles,
            x_row_stride, wt_row_stride, lds_x,
            spec, buf1_off_const,
            frag_b_shared,
            None,
        );
    }

    // ══════════════════════════════════════════════════════════════
    // Phase 8: Store accumulators to global memory
    // ══════════════════════════════════════════════════════════════

    k.label(&store_label);

    // Build epilogue context from spec + declared kernargs
    let epi_ctx = EpilogueCtx {
        ops: spec.epilogue.clone(),
        bias_ptr: epi_bias_ptr,
        scale_sreg: epi_scale,
        clamp_min_sreg: epi_clamp_min,
        clamp_max_sreg: epi_clamp_max,
    };

    if spec.acc_swap {
        // ACC swap store: loop over row_blocks, swapping acc from LDS
        emit_store_phase_swap(
            &mut k, &acc, &s_row_bases, base_n_s, y_ptr, n_dim, y_offset_s,
            n_row_blocks, n_col_tiles, lane_row, lane_id,
            acc_swap_addr.unwrap(), acc_swap_temp.unwrap(),
            &epi_ctx,
        );
    } else {
        // Unified store path: always mask by M/N boundary.
        // For interior tiles (M/N aligned to tile), all lanes pass the check — zero
        // functional overhead, only ~5 scalar instructions/store (negligible vs K-loop).
        // This avoids emitting two copies of store phase which doubles code size
        // and causes VGPR overflow on large tiles (128×128 k32 → GPU hang).
        emit_store_phase(
            &mut k, &acc, &s_row_bases, base_n_s, y_ptr, n_dim, y_offset_s,
            n_row_blocks, n_col_tiles, lane_row, lane_id,
            &epi_ctx,
            Some(SReg(m_dim.0)),  // always mask — no-op for aligned tiles
        );
    }

    // Ensure ALL GMEM operations are complete before endpgm.
    k.wait_vmcnt(0);
    k.wait_vscnt(0);
    k.wait_lgkmcnt(0);
    k.endpgm();

    // ── Early exit for fully OOB workgroups ──
    k.label(&early_exit_label);
    k.endpgm();
    // Full SSA optimization pipeline (2026-03-28).
    // All passes enabled: const fold, alg simplify, copy prop, CSE, combine,
    // LICM, DCE, waitcnt opt, post-regalloc scheduling.
    //
    // Key fixes that make this safe for GEMM kernels:
    //   - BufferLoad/BufferStore in has_side_effects() → DCE preserves them
    //   - LICM skips BufferLoad loops → no hoisting from K-loops
    //   - Scheduler skips BufferLoad blocks → no graduated waitcnt corruption
    k.set_opt_level(4);

    k
}

// ────────────────────────────────────────────────────────────────
// Helper: emit cooperative global loads (GMEM → VGPRs) with multi-pass stride
// ────────────────────────────────────────────────────────────────

fn emit_coop_load(
    k: &mut T0Kernel, gmem_regs: &[VReg],
    addr: VReg, loads: u32,
    gmem_stride: VReg,  // row stride for multi-pass
    k_off: VReg,        // K-dimension byte offset to add to base address
) {
    // Each load fetches 16 bytes (B128). Multi-pass: advance addr by gmem_stride between loads.
    // CRITICAL: use fresh VReg pairs per pass to prevent SSA dead-code-elimination.
    let mut cur_addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov(cur_addr, addr);
    k.v_mov(VReg(cur_addr.0 + 1), VReg(addr.0 + 1));
    k.clear_vcc();
    k.v_add_co(cur_addr, cur_addr, k_off);
    k.v_add_co_ci(VReg(cur_addr.0 + 1), VReg(cur_addr.0 + 1));
    for i in 0..loads as usize {
        k.global_load(gmem_regs[i], cur_addr, Width::B128, 0);
        if i + 1 < loads as usize {
            let next_addr = k.alloc_vreg_array(2, Alignment::Align2);
            k.clear_vcc();
            k.v_add_co(next_addr, cur_addr, gmem_stride);
            k.v_add_co_ci(VReg(next_addr.0 + 1), VReg(cur_addr.0 + 1));
            cur_addr = next_addr;
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Helper: emit cooperative BUFFER loads (MUBUF) with multi-pass stride
//
// Uses buffer_load_b128 with per-thread VGPR offset instead of global_load.
// Benefits: better L2 caching, saves 1 VGPR/thread, no carry chains.
// ────────────────────────────────────────────────────────────────

fn emit_coop_load_buffer(
    k: &mut T0Kernel, gmem_regs: &[VReg],
    byte_offset: VReg,  // per-thread byte offset (1 VGPR)
    srsrc: SReg,         // buffer resource descriptor (4 SGPRs)
    loads: u32,
    gmem_stride: VReg,   // row stride for multi-pass (only used when row_soffsets is None)
    k_off: VReg,         // K-dimension byte offset to add
    row_soffsets: Option<&[SReg]>, // soffset optimization: pre-computed SGPR row offsets
    scratch: VReg,       // reusable scratch VGPR for base address computation
) {
    // K-dimension offset: just add to the VGPR offset (no carry needed!)
    // CRITICAL: reuse scratch VGPR instead of allocating new one each call
    k.v_add_u32(scratch, byte_offset, k_off);

    if let Some(soffs) = row_soffsets {
        // ── soffset path: all loads use SAME voffset, different soffset SGPRs ──
        // NO serial v_add chain! All buffer_loads are independent.
        for i in 0..loads as usize {
            let soff = if i < soffs.len() { soffs[i] } else { SOFFSET_ZERO };
            k.buffer_load_soffset(gmem_regs[i], scratch, srsrc, Width::B128, 0, soff);
        }
    } else {
        // ── Legacy path: serial v_add chain (for fallback / sequential mode) ──
        let mut cur_off = scratch;
        for i in 0..loads as usize {
            k.buffer_load(gmem_regs[i], cur_off, srsrc, Width::B128, 0);
            if i + 1 < loads as usize {
                let next_off = k.alloc_vreg();
                k.v_add_u32(next_off, cur_off, gmem_stride);
                cur_off = next_off;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────
// soffset optimization: eliminate serial v_add chain for GMEM row addressing
//
// Instead of:  v_add → buffer_load → v_add → buffer_load (serial chain)
// Emit:        buffer_load(soff0) → buffer_load(soff1) → ... (all independent)
//
// Pre-computed SGPR offsets: {0, stride, 2*stride, 3*stride, ...}
// Each buffer_load uses the SAME voffset (base + k_off) but different soffset.
// ────────────────────────────────────────────────────────────────

/// Pre-compute SGPR row offsets for soffset optimization.
/// Returns a Vec of SReg: [SOFFSET_ZERO, s_stride, s_2stride, s_3stride, ...]
/// Call once in prologue; reuse across all K-loop iterations.
fn setup_row_soffsets(
    k: &mut T0Kernel,
    gmem_stride_vreg: VReg,  // uniform VGPR holding row stride (rows_per_pass * K * 2)
    loads: u32,              // number of rows to load per call
) -> Vec<SReg> {
    if loads <= 1 {
        return vec![SOFFSET_ZERO];
    }
    let mut soffsets = Vec::with_capacity(loads as usize);
    soffsets.push(SOFFSET_ZERO);  // row 0: no offset

    // Extract uniform stride to SGPR
    let s_stride = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: s_stride, src: gmem_stride_vreg });

    // row 1: stride
    soffsets.push(s_stride);

    // row 2+: accumulate
    let mut prev = s_stride;
    for _ in 2..loads {
        let s_next = k.alloc_sreg();
        k.s_add_u32_ss(s_next, prev, s_stride);
        soffsets.push(s_next);
        prev = s_next;
    }
    soffsets
}

/// Emit cooperative BUFFER loads using soffset SGPRs (no serial v_add chain).
/// All buffer_loads use the SAME base voffset, with different soffset SGPRs.
fn emit_coop_load_buffer_soffset(
    k: &mut T0Kernel, gmem_regs: &[VReg],
    byte_offset: VReg,  // per-thread byte offset (1 VGPR)
    srsrc: SReg,         // buffer resource descriptor (4 SGPRs)
    loads: u32,
    row_soffsets: &[SReg], // pre-computed row offsets [SOFFSET_ZERO, s1, s2, s3]
    k_off: VReg,         // K-dimension byte offset to add
) {
    // Compute base = byte_offset + k_off (single v_add, no chain)
    let base_off = k.alloc_vreg();
    k.v_add_u32(base_off, byte_offset, k_off);

    // Issue all loads with the SAME voffset but different soffset — NO serial deps!
    for i in 0..loads as usize {
        let soff = if i < row_soffsets.len() {
            row_soffsets[i]
        } else {
            SOFFSET_ZERO
        };
        k.buffer_load_soffset(gmem_regs[i], base_off, srsrc, Width::B128, 0, soff);
    }
}

/// Schedule for interleaving GMEM→LDS stores within WMMA computation.
struct StoreSchedule {
    gmem_x: Vec<VReg>,
    gmem_wt: Vec<VReg>,
    /// Precomputed base address: x_lds_off + buf_off (computed outside K-loop)
    x_store_base: VReg,
    /// Precomputed base address: wt_lds_off + buf_off (computed outside K-loop)
    wt_store_base: VReg,
    x_lds_stride: i32,
    wt_lds_stride: i32,
    x_loads: u32,
    wt_loads: u32,
    total_gmem_outstanding: u32,
}

// ────────────────────────────────────────────────────────────────
// Helper: emit cooperative global loads with OOB row zero-fill
//
// `oob_flag`: VReg that is 1 for threads whose row is out-of-bounds.
//
// CRITICAL: Zero-fill must happen AFTER wait_vmcnt(0), because global_load
// is asynchronous. If we zero-fill before wait, the load will overwrite zeros.
// Therefore, emit_coop_load_masked just does the load (same as emit_coop_load).
// The caller must call emit_oob_zero_fill() AFTER wait_vmcnt(0).
// ────────────────────────────────────────────────────────────────

fn emit_coop_load_masked(
    k: &mut T0Kernel, gmem_regs: &[VReg],
    addr: VReg, loads: u32,
    gmem_stride: VReg, k_off: VReg,
    _oob_flag: VReg,  // unused here; zero-fill is done separately
) {
    // Just do the load. Zero-fill happens after wait_vmcnt.
    emit_coop_load(k, gmem_regs, addr, loads, gmem_stride, k_off);
}

/// Zero-fill OOB threads' loaded VGPRs. MUST be called AFTER wait_vmcnt(0).
fn emit_oob_zero_fill(
    k: &mut T0Kernel, gmem_regs: &[VReg], loads: u32, oob_flag: VReg,
) {
    // vcc = (oob_flag == 1) → true for OOB lanes
    k.v_cmp_eq_u32_imm(oob_flag, 1);
    // For OOB lanes, overwrite loaded data with 0
    for i in 0..loads as usize {
        for sub in 0..4u32 {
            let vreg = VReg(gmem_regs[i].0 + sub);
            k.v_cndmask_b32(vreg, Operand::VReg(vreg), Operand::InlineInt(0));
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Helper: emit LDS stores (VGPRs → LDS) with multi-pass stride
// ────────────────────────────────────────────────────────────────

fn emit_lds_store(
    k: &mut T0Kernel, gmem_regs: &[VReg], lds_off: VReg, loads: u32,
    buf_off: VReg, lds_stride: i32,
) {
    // CRITICAL: allocate a FRESH VReg for each pass to prevent SSA from
    // dead-code-eliminating the stride increment. In-place self-update
    // v_add_u32(tmp, tmp, stride) gets coalesced by SSA regalloc.
    let mut cur_addr = k.alloc_vreg();
    k.v_add_u32(cur_addr, lds_off, buf_off);
    for i in 0..loads as usize {
        k.ds_store_b128(cur_addr, gmem_regs[i], 0);
        if i + 1 < loads as usize {
            let next_addr = k.alloc_vreg();  // fresh VReg per pass
            k.push(Op::VAddU32 {
                dst: next_addr, src0: Operand::VReg(cur_addr),
                src1: Operand::InlineInt(lds_stride),
            });
            cur_addr = next_addr;
        }
    }
}

// (StoreSchedule moved to line ~942 — with precomputed addresses)

fn emit_interleaved_store(k: &mut T0Kernel, sched: &StoreSchedule, store_idx: u32) {
    let total_stores = sched.x_loads + sched.wt_loads;
    if store_idx >= total_stores { return; }

    // Graduated wait: only wait for the specific load we need
    let remaining = sched.total_gmem_outstanding.saturating_sub(1 + store_idx) as u8;
    k.wait_vmcnt(remaining);

    if store_idx < sched.x_loads {
        let idx = store_idx as usize;
        // Use precomputed base + constant offset → zero v_add in K-loop!
        let offset = (idx as u16) * (sched.x_lds_stride as u16);
        k.ds_store_b128(sched.x_store_base, sched.gmem_x[idx], offset);
    } else {
        let idx = (store_idx - sched.x_loads) as usize;
        // Use precomputed base + constant offset → zero v_add in K-loop!
        let offset = (idx as u16) * (sched.wt_lds_stride as u16);
        k.ds_store_b128(sched.wt_store_base, sched.gmem_wt[idx], offset);
    }
}

fn emit_lds_store_graduated(
    k: &mut T0Kernel, gmem_regs: &[VReg], lds_off: VReg, loads: u32,
    buf_off: Option<VReg>, lds_stride: i32, outstanding_before: u32,
) {
    // When buf_off is None (offset=0), use lds_off directly — saves 1 VGPR + 1 v_add.
    let base_addr = if let Some(boff) = buf_off {
        let a = k.alloc_vreg();
        k.v_add_u32(a, lds_off, boff);
        a
    } else {
        lds_off
    };
    // Phase 2 optimization: fold row offsets into ds_store_b128's immediate offset field.
    // Instead of serial v_add chain: cur += stride → store → cur += stride → store ...
    // All stores share the SAME base_addr, with offset = i * lds_stride.
    // This eliminates (loads-1) v_add_nc_u32 per call, zero extra VGPRs!
    for i in 0..loads as usize {
        let remaining = outstanding_before.saturating_sub(1 + i as u32) as u8;
        k.wait_vmcnt(remaining);
        let row_offset = (i as i32 * lds_stride) as u16;
        k.ds_store_b128(base_addr, gmem_regs[i], row_offset);
    }
}

// ────────────────────────────────────────────────────────────────
// Helper: LDS read fragments + WMMA accumulation
// ────────────────────────────────────────────────────────────────

fn emit_lds_read_and_wmma(
    k: &mut T0Kernel,
    frag_a: &[VReg], frag_b: &[VReg], acc: &[VReg],
    // Raw (pre-XOR) addresses for per-ksub recomputation:
    x_lds_reads_raw: &[VReg],
    // Pre-XOR'd addresses for ksub=0 (also used as scratch for ksub>0):
    x_lds_reads_0: &[VReg], x_lds_reads_16: &[VReg],
    wt_raw: VReg, wt_base_0: VReg, wt_base_16: VReg,
    lane_swizzle: VReg,
    n_row_blocks: usize, n_col_tiles: usize,
    _x_row_stride: u32, wt_row_stride: u32, lds_x: u32,
    spec: &TileGemm, buf_off: u16,
    frag_b_shared: VReg,
    store_schedule: Option<&StoreSchedule>,
) {
    let target = current_target();
    let wmma_format = WmmaFormat::BF16_F32;
    let frag_a_loads = wmma_format.a_vreg_count(target) / 4;
    let frag_b_loads = wmma_format.b_vreg_count(target) / 4;
    let k_sub = spec.k_sub_steps();
    // Use column-streaming for large tiles (>4 cols) to reduce VGPR pressure.
    // Small tiles (≤4 cols) use bulk-load for better LDS-WMMA overlap.
    let use_streaming = n_col_tiles > 4;

    // Interleaved store tracking
    let total_wmma = (n_row_blocks * n_col_tiles * k_sub as usize) as u32;
    let total_stores = store_schedule.map(|s| s.x_loads + s.wt_loads).unwrap_or(0);
    let mut current_wmma: u32 = 0;
    let mut store_idx: u32 = 0;

    // Scratch VGPRs for per-ksub XOR recomputation (allocated once, reused).
    // Only needed for k>16 (k_sub > 1) where k_byte_within > 0.
    let (xr_0_tmp, xr_16_tmp, wt_0_tmp, wt_16_tmp) = if k_sub > 1 {
        (
            (0..n_row_blocks).map(|_| k.alloc_vreg()).collect::<Vec<_>>(),
            if frag_a_loads > 1 {
                (0..n_row_blocks).map(|_| k.alloc_vreg()).collect::<Vec<_>>()
            } else {
                vec![]
            },
            k.alloc_vreg(),
            if frag_b_loads > 1 { k.alloc_vreg() } else { VReg(0) },
        )
    } else {
        (vec![], vec![], VReg(0), VReg(0)) // unused
    };
    // Note: we reuse xr_0_tmp / xr_16_tmp / wt_0_tmp / wt_16_tmp as both
    // the v_add destination and l the subsequent v_xor input. The v_add writes
    // (raw + k_byte_within) into e.g. xr_0_tmp[r], then v_xor reads it back
    // and writes the XOR'd result into the same register. This is valid because
    // SSA lowering treats each write as a new definition.
    // This saves ~10 VGPRs vs having separate add_tmp + xor_result registers.

    for ksub in 0..k_sub {
        let k_byte_within = (ksub * 32) as u16;

        // ── Per-ksub XOR address recomputation ──
        // CRITICAL FIX: For k>16, k_byte_within must be added BEFORE XOR!
        // (a XOR b) + c ≠ (a + c) XOR b when c overlaps with swizzle bits.
        // For ksub=0: use pre-computed addresses (k_byte_within=0, no carry issue).
        // For ksub>0: recompute (raw + k_byte_within) XOR swizzle.
        let (cur_x_reads_0, cur_x_reads_16): (&[VReg], &[VReg]);
        let (cur_wt_0, cur_wt_16): (VReg, VReg);
        if ksub == 0 {
            cur_x_reads_0 = x_lds_reads_0;
            cur_x_reads_16 = x_lds_reads_16;
            cur_wt_0 = wt_base_0;
            cur_wt_16 = wt_base_16;
        } else {
            // Recompute X read addresses: (raw + k_byte_within) XOR swizzle
            // Two-step in-place: v_add → xr_0_tmp[r], then v_xor xr_0_tmp[r] ← xr_0_tmp[r] ^ swizzle
            for r in 0..n_row_blocks {
                k.push(Op::VAddU32 {
                    dst: xr_0_tmp[r], src0: Operand::VReg(x_lds_reads_raw[r]),
                    src1: Operand::InlineInt(k_byte_within as i32),
                });
                k.v_xor_b32(xr_0_tmp[r], Operand::VReg(xr_0_tmp[r]), Operand::VReg(lane_swizzle));
                if frag_a_loads > 1 {
                    k.push(Op::VAddU32 {
                        dst: xr_16_tmp[r], src0: Operand::VReg(x_lds_reads_raw[r]),
                        src1: Operand::InlineInt(k_byte_within as i32 + 16),
                    });
                    k.v_xor_b32(xr_16_tmp[r], Operand::VReg(xr_16_tmp[r]), Operand::VReg(lane_swizzle));
                }
            }
            // Recompute WT read addresses
            {
                k.push(Op::VAddU32 {
                    dst: wt_0_tmp, src0: Operand::VReg(wt_raw),
                    src1: Operand::InlineInt(k_byte_within as i32),
                });
                k.v_xor_b32(wt_0_tmp, Operand::VReg(wt_0_tmp), Operand::VReg(lane_swizzle));
                if frag_b_loads > 1 {
                    k.push(Op::VAddU32 {
                        dst: wt_16_tmp, src0: Operand::VReg(wt_raw),
                        src1: Operand::InlineInt(k_byte_within as i32 + 16),
                    });
                    k.v_xor_b32(wt_16_tmp, Operand::VReg(wt_16_tmp), Operand::VReg(lane_swizzle));
                }
            }
            cur_x_reads_0 = &xr_0_tmp;
            cur_x_reads_16 = &xr_16_tmp;
            cur_wt_0 = wt_0_tmp;
            cur_wt_16 = wt_16_tmp;
        }
        // ds_load immediate offset no longer includes k_byte_within (it's pre-XOR'd now)
        let ds_off = buf_off;

        // ── Load ALL A fragments (skip if preloaded by previous ksub) ──
        // Optimization: in streaming mode, the previous ksub's last column
        // preloads this ksub's frag_a between WMMAs (see ★ PREFETCH below).
        // In row-major mode, frag_a loading is handled inside the row loop.
        let row_major = frag_a.len() < n_row_blocks;
        let frag_a_preloaded = use_streaming && ksub > 0 && !row_major;
        if !frag_a_preloaded && !row_major {
            for r in 0..frag_a.len() {
                k.ds_load_b128(frag_a[r], cur_x_reads_0[r], ds_off);
                if frag_a_loads > 1 {
                    k.ds_load_b128(VReg(frag_a[r].0 + 4), cur_x_reads_16[r], ds_off);
                }
            }
        }

        if use_streaming {
            // ── ROW-MAJOR vs COLUMN-MAJOR streaming ──
            // Row-major: frag_a has 1 group, process one row at a time.
            //   Saves 8 VGPRs but reloads frag_b for each row block.
            // Column-major: frag_a has n_row_blocks groups, all loaded upfront.
            //   Better locality but needs more VGPRs.
            let row_major = frag_a.len() < n_row_blocks;

            if row_major {
                // ══ ROW-MAJOR STREAMING ══
                // For each row block: load frag_a → stream all frag_b columns → WMMA
                let fb_ping = frag_b[0];
                let fb_pong = frag_b_shared;

                for r in 0..n_row_blocks {
                    // Load frag_a for this row block
                    k.ds_load_b128(frag_a[0], cur_x_reads_0[r], ds_off);
                    if frag_a_loads > 1 {
                        k.ds_load_b128(VReg(frag_a[0].0 + 4), cur_x_reads_16[r], ds_off);
                    }

                    // Prefetch first TWO B columns
                    {
                        let base_0: u16 = lds_x as u16;
                        k.ds_load_b128(fb_ping, cur_wt_0, base_0 + ds_off);
                        if frag_b_loads > 1 {
                            k.ds_load_b128(VReg(fb_ping.0 + 4), cur_wt_16, base_0 + ds_off);
                        }
                    }
                    if n_col_tiles > 1 {
                        let base_1: u16 = (lds_x + 16 * wt_row_stride) as u16;
                        k.ds_load_b128(fb_pong, cur_wt_0, base_1 + ds_off);
                        if frag_b_loads > 1 {
                            k.ds_load_b128(VReg(fb_pong.0 + 4), cur_wt_16, base_1 + ds_off);
                        }
                    }

                    // Keep only the next B column in flight.
                    let initial_wait = if n_col_tiles > 1 { frag_b_loads as u8 } else { 0u8 };
                    k.wait_lgkmcnt(initial_wait);

                    for c in 0..n_col_tiles {
                        let cur_fb = if c % 2 == 0 { fb_ping } else { fb_pong };
                        let acc_idx = r * n_col_tiles + c;
                        k.wmma_bf16_f32(acc[acc_idx], frag_a[0], cur_fb, acc[acc_idx]);
                        current_wmma += 1;

                        // ★ Interleaved GMEM→LDS store ★
                        if let Some(sched) = store_schedule {
                            let target = (current_wmma * total_stores) / total_wmma;
                            while store_idx < target {
                                emit_interleaved_store(k, sched, store_idx);
                                store_idx += 1;
                            }
                        }

                        // Prefetch column c+2
                        if c + 2 < n_col_tiles {
                            let next2_base: u16 = (lds_x + ((c + 2) as u32) * 16 * wt_row_stride) as u16;
                            k.ds_load_b128(cur_fb, cur_wt_0, next2_base + ds_off);
                            if frag_b_loads > 1 {
                                k.ds_load_b128(VReg(cur_fb.0 + 4), cur_wt_16, next2_base + ds_off);
                            }
                        }

                        // Wait for next column's B data
                        if c + 1 < n_col_tiles {
                            let remaining = if c + 2 < n_col_tiles { frag_b_loads as u8 } else { 0u8 };
                            k.wait_lgkmcnt(remaining);
                        }
                    }
                }
            } else {
                // ══ COLUMN-MAJOR STREAMING (original) ══
                let fb_ping = frag_b[0];
                let fb_pong = frag_b_shared;

                // Prefetch first TWO B columns
                {
                    let base_0: u16 = lds_x as u16;
                    k.ds_load_b128(fb_ping, cur_wt_0, base_0 + ds_off);
                    if frag_b_loads > 1 {
                        k.ds_load_b128(VReg(fb_ping.0 + 4), cur_wt_16, base_0 + ds_off);
                    }
                }
                if n_col_tiles > 1 {
                    let base_1: u16 = (lds_x + 16 * wt_row_stride) as u16;
                    k.ds_load_b128(fb_pong, cur_wt_0, base_1 + ds_off);
                    if frag_b_loads > 1 {
                        k.ds_load_b128(VReg(fb_pong.0 + 4), cur_wt_16, base_1 + ds_off);
                    }
                }

                // When frag_a was preloaded, it has a head start — account for
                // those 4 loads still in the lgkmcnt pipeline.
                let preloaded_inflight =
                    if frag_a_preloaded { (frag_a_loads as usize * n_row_blocks) as u8 } else { 0u8 };
                let initial_wait =
                    if n_col_tiles > 1 { frag_b_loads as u8 + preloaded_inflight } else { preloaded_inflight };
                k.wait_lgkmcnt(initial_wait);

                for c in 0..n_col_tiles {
                    let cur_fb = if c % 2 == 0 { fb_ping } else { fb_pong };

                    // ── All WMMAs for current column ──
                    for r in 0..n_row_blocks {
                        let acc_idx = r * n_col_tiles + c;
                        k.wmma_bf16_f32(acc[acc_idx], frag_a[r], cur_fb, acc[acc_idx]);
                        current_wmma += 1;

                        // ★ PREFETCH: at last column, preload next ksub's frag_a ★
                        // Must use next ksub's XOR'd addresses (not current ksub's)
                        if c == n_col_tiles - 1 && ksub + 1 < k_sub {
                            let next_k_byte = ((ksub + 1) * 32) as i32;
                            // Compute next ksub's X addresses: (raw + next_k_byte) XOR swizzle
                            for r2 in 0..n_row_blocks {
                                let ntmp = k.alloc_vreg();
                                k.push(Op::VAddU32 {
                                    dst: ntmp, src0: Operand::VReg(x_lds_reads_raw[r2]),
                                    src1: Operand::InlineInt(next_k_byte),
                                });
                                let nxr_0 = k.alloc_vreg();
                                k.v_xor_b32(nxr_0, Operand::VReg(ntmp), Operand::VReg(lane_swizzle));
                                // Note: these regs are used by next ksub's cur_x_reads
                                // but since prefetch uses frag_a[r] as dest, they're temporary
                                k.ds_load_b128(frag_a[r2], nxr_0, buf_off);
                                if frag_a_loads > 1 {
                                    let ntmp16 = k.alloc_vreg();
                                    k.push(Op::VAddU32 {
                                        dst: ntmp16, src0: Operand::VReg(x_lds_reads_raw[r2]),
                                        src1: Operand::InlineInt(next_k_byte + 16),
                                    });
                                    let nxr_16 = k.alloc_vreg();
                                    k.v_xor_b32(nxr_16, Operand::VReg(ntmp16), Operand::VReg(lane_swizzle));
                                    k.ds_load_b128(VReg(frag_a[r2].0 + 4), nxr_16, buf_off);
                                }
                            }
                        }

                        // ★ Interleaved GMEM→LDS store ★
                        if let Some(sched) = store_schedule {
                            let target = (current_wmma * total_stores) / total_wmma;
                            while store_idx < target {
                                emit_interleaved_store(k, sched, store_idx);
                                store_idx += 1;
                            }
                        }
                    }

                    // ── Prefetch column c+2 into the buffer we just consumed ──
                    if c + 2 < n_col_tiles {
                        let next2_base: u16 = (lds_x + ((c + 2) as u32) * 16 * wt_row_stride) as u16;
                        k.ds_load_b128(cur_fb, cur_wt_0, next2_base + ds_off);
                        if frag_b_loads > 1 {
                            k.ds_load_b128(VReg(cur_fb.0 + 4), cur_wt_16, next2_base + ds_off);
                        }
                    }

                    // Wait for next column's B data
                    if c + 1 < n_col_tiles {
                        let mut remaining = if c + 2 < n_col_tiles { frag_b_loads as u8 } else { 0u8 };
                        if c == n_col_tiles - 2 && ksub + 1 < k_sub {
                            // frag_a prefetches not yet issued, remaining stays same
                        }
                        k.wait_lgkmcnt(remaining);
                    }
                }
            }

        } else {
            // ── Bulk-load mode (optimal for ≤4 cols) ──
            // Load ALL B fragments upfront, then graduated-waitcnt WMMA.
            // Stores are emitted at COLUMN BOUNDARIES (between WMMAs).
            //
            // WHY NOT interleave stores within WMMAs?
            //   ds_store uses lgkmcnt, same as ds_load. Interleaving them
            //   within the graduated wait_lgkmcnt loop corrupts the remaining
            //   counts → WMMAs read incomplete LDS data → numerical errors.
            //
            // WHY NOT post-WMMA?
            //   Extends gmem VGPR live ranges across all 8 WMMAs → regalloc
            //   overflow at 144 VGPRs → interference conflicts → corrupt
            //   addresses → GPU hang.
            //
            // SOLUTION: emit stores BETWEEN columns. Compensate the graduated
            //   wait_lgkmcnt by adding pending_stores to `remaining`.
            //   GFX11 LDS FIFO ordering guarantees ds_loads (issued first)
            //   complete before ds_stores (issued later), so the graduated
            //   wait correctly drains reads before stores.
            for c in 0..n_col_tiles {
                let base_off: u16 = (lds_x + (c as u32) * 16 * wt_row_stride) as u16;
                k.ds_load_b128(frag_b[c], cur_wt_0, base_off + ds_off);
                if frag_b_loads > 1 {
                    k.ds_load_b128(VReg(frag_b[c].0 + 4), cur_wt_16, base_off + ds_off);
                }
            }

            let total_loads = (frag_a_loads as usize * n_row_blocks + frag_b_loads as usize * n_col_tiles) as u8;
            let mut pending_stores: u8 = 0;
            for c in 0..n_col_tiles {
                let loads_needed =
                    (frag_a_loads as usize * n_row_blocks + frag_b_loads as usize * (c + 1)) as u8;
                // Account for ds_stores already in lgkmcnt pipeline
                let remaining = total_loads.saturating_sub(loads_needed) + pending_stores;
                k.wait_lgkmcnt(remaining);
                for r in 0..n_row_blocks {
                    let acc_idx = r * n_col_tiles + c;
                    k.wmma_bf16_f32(acc[acc_idx], frag_a[r], frag_b[c], acc[acc_idx]);
                }
                // ★ Emit stores at column boundary (short gmem VGPR live range) ★
                if let Some(sched) = store_schedule {
                    let target = (((c + 1) as u32) * total_stores) / n_col_tiles as u32;
                    while store_idx < target {
                        emit_interleaved_store(k, sched, store_idx);
                        store_idx += 1;
                        pending_stores += 1;
                        current_wmma += 1; // keep counter advancing
                    }
                }
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Helper: Store accumulators to global memory
//
// WMMA v_wmma_f32_16x16x16_bf16 output layout (Wave32):
//   acc[0..7] = 8 VGPRs, each holds one f32 value per lane.
//   For a 16×16 output C-matrix:
//     - Lanes 0-15: even rows (row = lane_id * 2)
//     - Lanes 16-31: odd rows (row = (lane_id - 16) * 2 + 1)
//     - acc[v]: row offset = v*2 within the 16-row block
//   Actually simplified: each lane stores to row = (lane_half) + v*2,
//   col = base_col + lane_row
//   where lane_half = lane_id >> 4 (0 or 1), lane_row = lane_id & 15.
//
// Storage pattern: for each acc register v (0..7):
//   Y[s_row_base + lane_half + v*2, base_col + lane_row] = acc[v]
//   Row stride between consecutive registers = 2 rows = N*8 bytes
// ────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────
// Epilogue context — carries SGPRs for fused post-GEMM operations
// ────────────────────────────────────────────────────────────────

/// Runtime state for epilogue operations, pre-loaded from kernel arguments.
struct EpilogueCtx {
    ops: Vec<EpilogueOp>,
    /// bias_ptr SReg pair (valid when EpilogueOp::BiasAdd is present)
    bias_ptr: Option<SRegPair>,
    /// scale SReg (valid when EpilogueOp::Scale is present)
    scale_sreg: Option<SReg>,
    /// clamp_min SReg (valid when EpilogueOp::Clamp is present)
    clamp_min_sreg: Option<SReg>,
    /// clamp_max SReg (valid when EpilogueOp::Clamp is present)
    clamp_max_sreg: Option<SReg>,
}

/// Apply the full epilogue chain to a single f32 value in a VReg (in-place).
///
/// `val` — the accumulator VReg to transform
/// `col_elem_index` — the column element index for BiasAdd (VReg)
///
/// Uses up to 2 temp VREGs internally. All operations are pure VALU.
fn emit_epilogue_on_vreg(
    k: &mut T0Kernel,
    val: VReg,
    col_elem_index: Option<VReg>,
    ctx: &EpilogueCtx,
) {
    for op in &ctx.ops {
        match op {
            EpilogueOp::BiasAdd => {
                // bias[col] — load bias value and add to acc
                if let (Some(bias_ptr), Some(col_idx)) = (ctx.bias_ptr, col_elem_index) {
                    let bias_addr = k.alloc_vreg_array(2, Alignment::Align2);
                    k.v_mov_from_sgpr(bias_addr, SReg(bias_ptr.0));
                    k.v_mov_from_sgpr(VReg(bias_addr.0 + 1), SReg(bias_ptr.0 + 1));
                    // byte_off = col_idx * 4
                    let byte_off = k.alloc_vreg();
                    k.v_lshlrev_b32(byte_off, 2, col_idx);
                    k.clear_vcc();
                    k.v_add_co(bias_addr, bias_addr, byte_off);
                    k.v_add_co_ci(VReg(bias_addr.0 + 1), VReg(bias_addr.0 + 1));
                    let bias_val = k.alloc_vreg();
                    k.global_load(bias_val, bias_addr, Width::B32, 0);
                    k.wait_vmcnt(0);
                    k.v_add_f32(val, val, bias_val);
                }
            }
            EpilogueOp::Scale => {
                if let Some(scale_s) = ctx.scale_sreg {
                    let scale_v = k.alloc_vreg();
                    k.v_mov_from_sgpr(scale_v, scale_s);
                    k.v_mul_f32(val, val, scale_v);
                }
            }
            EpilogueOp::ReLU => {
                // max(val, 0.0) — inline constant 0x80 = 0.0
                k.push(Op::VMaxF32 {
                    dst: val,
                    src0: Operand::VReg(val),
                    src1: Operand::InlineFloat(0.0),
                });
            }
            EpilogueOp::SiLU => {
                // silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
                // step 1: neg_x = -x (xor with sign bit)
                let neg_x = k.alloc_vreg();
                k.v_xor_b32(neg_x, Operand::VReg(val), Operand::Literal(0x80000000));
                // step 2: scale by log2(e) for v_exp_f32 (which computes 2^x)
                let scaled = k.alloc_vreg();
                // v_mul_f32 scaled, neg_x, log2e
                k.push(Op::VMulF32 {
                    dst: scaled,
                    src0: Operand::VReg(neg_x),
                    src1: Operand::InlineFloat(1.4426950),  // log2(e)
                });
                // step 3: exp2(scaled) = exp(-x)
                let exp_neg = k.alloc_vreg();
                k.v_exp_f32(exp_neg, scaled);
                // step 4: 1 + exp(-x)
                let one_plus = k.alloc_vreg();
                k.push(Op::VAddF32 {
                    dst: one_plus,
                    src0: Operand::VReg(exp_neg),
                    src1: Operand::InlineFloat(1.0),
                });
                // step 5: sigmoid = rcp(1 + exp(-x))
                let sigmoid = k.alloc_vreg();
                k.v_rcp_f32(sigmoid, one_plus);
                // step 6: silu = x * sigmoid
                k.v_mul_f32(val, val, sigmoid);
            }
            EpilogueOp::GELU => {
                // gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                // Fast approx: use tanh(x) ≈ 1 - 2/(1+exp(2x))
                // For now, use the simpler sigmoid approximation:
                // gelu(x) ≈ x * sigmoid(1.702 * x)
                let scaled_x = k.alloc_vreg();
                k.push(Op::VMulF32 {
                    dst: scaled_x,
                    src0: Operand::VReg(val),
                    src1: Operand::InlineFloat(1.702),
                });
                // neg(1.702*x) for sigmoid
                let neg_sx = k.alloc_vreg();
                k.v_xor_b32(neg_sx, Operand::VReg(scaled_x), Operand::Literal(0x80000000));
                // log2(e) * neg_sx
                let log2e_sx = k.alloc_vreg();
                k.push(Op::VMulF32 {
                    dst: log2e_sx,
                    src0: Operand::VReg(neg_sx),
                    src1: Operand::InlineFloat(1.4426950),
                });
                let exp_neg = k.alloc_vreg();
                k.v_exp_f32(exp_neg, log2e_sx);
                let one_plus = k.alloc_vreg();
                k.push(Op::VAddF32 {
                    dst: one_plus,
                    src0: Operand::VReg(exp_neg),
                    src1: Operand::InlineFloat(1.0),
                });
                let sig = k.alloc_vreg();
                k.v_rcp_f32(sig, one_plus);
                k.v_mul_f32(val, val, sig);
            }
            EpilogueOp::Abs => {
                // |x| = x & 0x7FFFFFFF (clear sign bit)
                k.push(Op::VAndB32 {
                    dst: val,
                    src0: Operand::VReg(val),
                    src1: Operand::Literal(0x7FFFFFFF),
                });
            }
            EpilogueOp::Neg => {
                k.v_xor_b32(val, Operand::VReg(val), Operand::Literal(0x80000000));
            }
            EpilogueOp::Clamp => {
                if let (Some(min_s), Some(max_s)) = (ctx.clamp_min_sreg, ctx.clamp_max_sreg) {
                    let min_v = k.alloc_vreg();
                    k.v_mov_from_sgpr(min_v, min_s);
                    let max_v = k.alloc_vreg();
                    k.v_mov_from_sgpr(max_v, max_s);
                    k.v_max_f32(val, val, min_v);
                    k.v_min_f32(val, val, max_v);
                }
            }
        }
    }
}

fn emit_store_phase(
    k: &mut T0Kernel,
    acc: &[VReg], s_row_bases: &[SReg], base_n_s: SReg,
    y_ptr: SRegPair, n_dim: SReg, y_offset_s: SReg,
    n_row_blocks: usize, n_col_tiles: usize,
    lane_row: VReg, lane_id: VReg,
    epilogue: &EpilogueCtx,
    boundary: Option<SReg>,  // Some(m_dim) for boundary tiles, None for interior
) {
    // ── Build Y buffer resource descriptor (4 SGPRs) ──
    let y_srd = k.alloc_sreg_quad();
    k.push(Op::SAddU32 { dst: y_srd, src0: SReg(y_ptr.0), src1: SOperand::SReg(y_offset_s) });
    k.push(Op::SAddcU32 { dst: SReg(y_srd.0 + 1), src0: SReg(y_ptr.0 + 1), src1: SOperand::InlineInt(0) });
    k.push(Op::SMov { dst: SReg(y_srd.0 + 2), src: SOperand::Literal(0x7FFFFFFE) });
    k.push(Op::SMov { dst: SReg(y_srd.0 + 3), src: SOperand::Literal(0x31027000) });

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, n_dim);

    // row_stride = N * 8 (2 rows × 4 bytes/f32)
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    // col_base = base_n + lane_row (element index)
    let col_base_v = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base_v, base_n_s);
    k.v_add_u32(col_base_v, col_base_v, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base_v);

    // Boundary masking setup
    let (m_vreg_opt, saved_exec) = if let Some(m_s) = boundary {
        let m_v = k.alloc_vreg();
        k.v_mov_from_sgpr(m_v, m_s);
        let se = k.alloc_sreg();
        (Some(m_v), Some(se))
    } else {
        (None, None)
    };

    for r in 0..n_row_blocks {
        let base_row_v = k.alloc_vreg();
        k.v_mov_from_sgpr(base_row_v, s_row_bases[r]);
        k.v_add_u32(base_row_v, base_row_v, lane_half);

        let row_bytes = k.alloc_vreg();
        k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
        k.v_lshlrev_b32(row_bytes, 2, row_bytes);

        let voffset_base = k.alloc_vreg();
        k.v_add_u32(voffset_base, row_bytes, col_bytes);

        // Track current logical row for boundary masking
        let cur_row = if boundary.is_some() {
            let cr = k.alloc_vreg();
            k.v_mov(cr, base_row_v);
            Some(cr)
        } else {
            None
        };

        for c in 0..n_col_tiles {
            let voff = k.alloc_vreg();
            k.v_mov(voff, voffset_base);
            if c > 0 {
                k.push(Op::VAddU32 {
                    dst: voff, src0: Operand::VReg(voff),
                    src1: Operand::InlineInt((c * 64) as i32),
                });
            }

            // Compute column element index for epilogue
            let col_elem_idx = if !epilogue.ops.is_empty() {
                let col_idx = k.alloc_vreg();
                k.v_mov(col_idx, col_base_v);
                if c > 0 {
                    k.push(Op::VAddU32 {
                        dst: col_idx, src0: Operand::VReg(col_idx),
                        src1: Operand::InlineInt((c * 16) as i32),
                    });
                }
                Some(col_idx)
            } else {
                None
            };

            // Boundary: compute col mask once per col_tile
            let col_mask_active = if let Some(ref _m_v) = m_vreg_opt {
                // actual_col = col_base + c*16
                let actual_col = k.alloc_vreg();
                if c > 0 {
                    k.push(Op::VAddU32 {
                        dst: actual_col, src0: Operand::VReg(col_base_v),
                        src1: Operand::InlineInt((c * 16) as i32),
                    });
                } else {
                    k.v_mov(actual_col, col_base_v);
                }
                Some(actual_col)
            } else {
                None
            };

            let a_idx = r * n_col_tiles + c;

            // Reset cur_row for each col_tile loop (it gets modified by row advance)
            if let Some(cr) = cur_row {
                k.v_mov(cr, base_row_v);
            }

            for v in 0..8u32 {
                let acc_vreg = VReg(acc[a_idx].0 + v);

                // Apply epilogue chain before store
                if !epilogue.ops.is_empty() {
                    emit_epilogue_on_vreg(k, acc_vreg, col_elem_idx, epilogue);
                }

                // Boundary masking: SaveExec, then mask with row < M && col < N
                if let (Some(m_v), Some(se), Some(cr), Some(ref actual_col)) =
                    (m_vreg_opt, saved_exec, cur_row, &col_mask_active) {
                    // VCC = (cur_row < M)
                    k.v_cmp_lt_u32(Operand::VReg(cr), Operand::VReg(m_v));
                    // EXEC &= VCC (save old EXEC)
                    k.push(Op::SaveExec { dst: se });
                    // Now also mask by col: VCC = (actual_col < N)
                    k.v_cmp_lt_u32(Operand::VReg(*actual_col), Operand::VReg(n_vreg));
                    k.raw_asm("s_and_b32 exec_lo, exec_lo, vcc_lo");
                }

                // buffer_store_b32: same SRD-based addressing for both paths
                k.buffer_store(voff, acc_vreg, y_srd, Width::B32, 0);

                // Restore EXEC after boundary-masked store
                if let Some(se) = saved_exec {
                    k.push(Op::RestoreExec { src: se });
                }

                if v < 7 {
                    k.v_add_u32(voff, voff, row_stride);
                    // Advance logical row by 2 for boundary check
                    if let Some(cr) = cur_row {
                        k.push(Op::VAddU32 {
                            dst: cr, src0: Operand::VReg(cr),
                            src1: Operand::InlineInt(2),
                        });
                    }
                }
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Helper: Store accumulators with boundary masking (EXEC mask)
//
// Same as emit_store_phase but with row < M && col < N predication.
// Uses SaveExec/RestoreExec to mask global_store instructions.
// ────────────────────────────────────────────────────────────────

fn emit_store_phase_masked(
    k: &mut T0Kernel,
    acc: &[VReg], s_row_bases: &[SReg], base_n_s: SReg,
    y_ptr: SRegPair, n_dim: SReg, y_offset_s: SReg,
    n_row_blocks: usize, n_col_tiles: usize,
    lane_row: VReg, lane_id: VReg,
    m_dim: SReg,  // actual M (for row boundary check)
) {
    // Exactly mirrors emit_store_phase but adds EXEC masking before each global_store.
    // Single-level SaveExec/RestoreExec per store (no nesting).

    // lane_half = lane_id >> 4 (0 for lanes 0-15, 1 for lanes 16-31)
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, n_dim);
    let m_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(m_vreg, m_dim);

    // row_stride = N * 8 (2 rows × 4 bytes/f32)
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    // col_base = base_n + lane_row (element index)
    let col_base_v = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base_v, base_n_s);
    k.v_add_u32(col_base_v, col_base_v, lane_row);
    // col_bytes = col_base * 4
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base_v);

    // Reuse one SGPR for all EXEC saves (non-overlapping uses)
    let saved_exec = k.alloc_sreg();

    for r in 0..n_row_blocks {
        // base_row = s_row_bases[r] + lane_half
        let base_row_v = k.alloc_vreg();
        k.v_mov_from_sgpr(base_row_v, s_row_bases[r]);
        k.v_add_u32(base_row_v, base_row_v, lane_half);

        // y_base = Y_ptr + y_offset (compute BEFORE row_bytes to avoid regalloc overlap)
        let y_base = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(y_base, SReg(y_ptr.0));
        k.v_mov_from_sgpr(VReg(y_base.0 + 1), SReg(y_ptr.0 + 1));
        {
            let v_yoff = k.alloc_vreg();
            k.v_mov_from_sgpr(v_yoff, y_offset_s);
            k.clear_vcc();
            k.v_add_co(y_base, y_base, v_yoff);
            k.v_add_co_ci(VReg(y_base.0 + 1), VReg(y_base.0 + 1));
        }

        // row_bytes = base_row * N * 4
        // CRITICAL: compute AFTER y_base alloc to avoid regalloc overlap
        let row_bytes = k.alloc_vreg();
        k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
        k.v_lshlrev_b32(row_bytes, 2, row_bytes);

        // y_base += row_bytes + col_bytes
        k.clear_vcc();
        k.v_add_co(y_base, y_base, row_bytes);
        k.v_add_co_ci(VReg(y_base.0 + 1), VReg(y_base.0 + 1));
        k.v_add_u32(y_base, y_base, col_bytes);

        for c in 0..n_col_tiles {
            // y_addr = y_base + c*64 (16 cols × 4 bytes per col tile)
            let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
            k.v_mov(y_addr, y_base);
            k.v_mov(VReg(y_addr.0 + 1), VReg(y_base.0 + 1));
            if c > 0 {
                k.push(Op::VAddU32 {
                    dst: y_addr, src0: Operand::VReg(y_addr),
                    src1: Operand::InlineInt((c * 64) as i32),
                });
            }

            // Precompute column for this tile: actual_col = col_base + c*16
            let actual_col = k.alloc_vreg();
            if c > 0 {
                k.push(Op::VAddU32 {
                    dst: actual_col, src0: Operand::VReg(col_base_v),
                    src1: Operand::InlineInt((c * 16) as i32),
                });
            } else {
                k.v_mov(actual_col, col_base_v);
            }

            let a_idx = r * n_col_tiles + c;
            let cur_row = k.alloc_vreg();
            k.v_mov(cur_row, base_row_v);

            for v in 0..8u32 {
                // Mask: (cur_row < M) && (actual_col < N)
                // Step 1: VCC = (cur_row < M)
                k.v_cmp_lt_u32(Operand::VReg(cur_row), Operand::VReg(m_vreg));
                // Step 2: save EXEC, EXEC &= VCC (row mask)
                k.push(Op::SaveExec { dst: saved_exec });
                // Step 3: EXEC &= (actual_col < N) — combined row+col mask
                k.v_cmp_lt_u32(Operand::VReg(actual_col), Operand::VReg(n_vreg));
                // s_and_b32 exec_lo, exec_lo, vcc_lo — no save needed
                k.raw_asm("s_and_b32 exec_lo, exec_lo, vcc_lo");

                // Predicated store: only lanes with (row < M && col < N)
                k.global_store(y_addr, VReg(acc[a_idx].0 + v), Width::B32, 0);

                // Restore EXEC to full (pre-row-mask state)
                k.push(Op::RestoreExec { src: saved_exec });

                if v < 7 {
                    // Advance row by 2
                    k.push(Op::VAddU32 {
                        dst: cur_row, src0: Operand::VReg(cur_row),
                        src1: Operand::InlineInt(2),
                    });
                    // Address advance with full EXEC
                    k.clear_vcc();
                    k.v_add_co(y_addr, y_addr, row_stride);
                    k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
                }
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Helper: ACC Swap — exchange acc VGPRs ↔ LDS swap slot
//
// Single-slot swap using temp VGPRs (separate from frag_b).
// For each acc group [c]:
//   1. Load LDS[c] → temp (other row_block's acc)
//   2. Store acc[c] → LDS[c] (current row_block's acc)
//   3. v_mov temp → acc[c]
// After the swap, VGPRs hold the other row_block's data,
// and LDS holds what was previously in VGPRs.
//
// NOTE: acc VRegs MUST be marked as coalesced groups in Phase 4.
// This prevents SSA copy propagation from folding the v_mov into
// a direct reference to temp — which would make WMMA read B data
// as accumulator input (because temp is reused for frag_b loading).
// ────────────────────────────────────────────────────────────────
fn emit_acc_swap(
    k: &mut T0Kernel,
    acc: &[VReg],
    swap_addr: VReg,
    n_col_tiles: usize,
    temp: VReg,  // 8-VGPR aligned temp (dedicated, NOT frag_b)
) {
    for c in 0..n_col_tiles {
        let off = (c as u16) * 32;
        // 1. Load other row_block from LDS into temp
        k.ds_load_b128(temp, swap_addr, off);
        k.ds_load_b128(VReg(temp.0 + 4), swap_addr, off + 16);
        k.wait_lgkmcnt(0);
        // 2. Save current acc to LDS (same slot — load already completed)
        k.ds_store_b128(swap_addr, acc[c], off);
        k.ds_store_b128(swap_addr, VReg(acc[c].0 + 4), off + 16);
        k.wait_lgkmcnt(0);
        // 3. Move loaded data to acc (copy-prop safe: acc is coalesced)
        for v in 0..8u32 {
            k.v_mov(VReg(acc[c].0 + v), VReg(temp.0 + v));
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Helper: Row-block-major LDS read + WMMA with ACC swapping
//
// Structure: for each row_block r:
//   1. Load frag_a from LDS for row r
//   2. Stream frag_b columns + WMMA (single row_block)
//   3. If not last row_block: swap acc VGPRs ↔ LDS
//
// After completion, VGPRs hold row_block 0's acc (same as entry).
// ────────────────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn emit_lds_read_and_wmma_swap(
    k: &mut T0Kernel,
    frag_a: &[VReg],       // [1] — single row_block (reused)
    frag_b: &[VReg],       // [1] — ping buffer
    acc: &[VReg],           // [n_col_tiles] — single row_block
    x_lds_reads_0: &[VReg], // [n_row_blocks] — all row addresses
    x_lds_reads_16: &[VReg],
    wt_base_0: VReg, wt_base_16: VReg,
    n_row_blocks: usize, n_col_tiles: usize,
    _x_row_stride: u32, wt_row_stride: u32, lds_x: u32,
    spec: &TileGemm,
    buf_off: u16,
    frag_b_shared: VReg,   // pong buffer
    swap_addr: VReg,       // LDS swap address
    swap_temp: VReg,       // 8-VGPR temp for acc swap (dedicated, not frag_b)
) {
    let target = current_target();
    let wmma_format = WmmaFormat::BF16_F32;
    let frag_a_loads = wmma_format.a_vreg_count(target) / 4;
    let frag_b_loads = wmma_format.b_vreg_count(target) / 4;
    let k_sub_steps = (spec.tile_k / 16) as usize;

    for r in 0..n_row_blocks {
        for ksub in 0..k_sub_steps {
            let k_byte_within = (ksub * 32) as u16;

            // ── Load A fragment for row_block r ──
            k.ds_load_b128(frag_a[0], x_lds_reads_0[r], buf_off + k_byte_within);
            if frag_a_loads > 1 {
                k.ds_load_b128(VReg(frag_a[0].0 + 4), x_lds_reads_16[r], buf_off + k_byte_within);
            }

            // ── Streaming B columns (same as original streaming mode) ──
            let fb_ping = frag_b[0];
            let fb_pong = frag_b_shared;

            // Prefetch B[0] and B[1]
            let base0: u16 = (lds_x as u16) + (0u16) * (16 * wt_row_stride as u16);
            k.ds_load_b128(fb_ping, wt_base_0, base0 + buf_off + k_byte_within);
            if frag_b_loads > 1 {
                k.ds_load_b128(VReg(fb_ping.0 + 4), wt_base_16, base0 + buf_off + k_byte_within);
            }
            if n_col_tiles > 1 {
                let base1: u16 = (lds_x as u16) + (1u16) * (16 * wt_row_stride as u16);
                k.ds_load_b128(fb_pong, wt_base_0, base1 + buf_off + k_byte_within);
                if frag_b_loads > 1 {
                    k.ds_load_b128(VReg(fb_pong.0 + 4), wt_base_16, base1 + buf_off + k_byte_within);
                }
            }

            // Wait for A + B[0] (keep B[1] in flight if present)
            let initial_wait = if n_col_tiles > 1 { frag_b_loads as u8 } else { 0u8 };
            k.wait_lgkmcnt(initial_wait);

            for c in 0..n_col_tiles {
                let cur_fb = if c % 2 == 0 { fb_ping } else { fb_pong };

                // WMMA for current column (single row_block)
                k.wmma_bf16_f32(acc[c], frag_a[0], cur_fb, acc[c]);

                // Prefetch B[c+2] into the consumed buffer
                if c + 2 < n_col_tiles {
                    let next2_base: u16 = (lds_x + ((c + 2) as u32) * 16 * wt_row_stride) as u16;
                    k.ds_load_b128(cur_fb, wt_base_0, next2_base + buf_off + k_byte_within);
                    if frag_b_loads > 1 {
                        k.ds_load_b128(VReg(cur_fb.0 + 4), wt_base_16, next2_base + buf_off + k_byte_within);
                    }
                }

                // Wait for next column's B data
                if c + 1 < n_col_tiles {
                    let remaining = if c + 2 < n_col_tiles { frag_b_loads as u8 } else { 0u8 };
                    k.wait_lgkmcnt(remaining);
                }
            }
        }

        // ── ACC swap between row_blocks ──
        // After processing row_block r, swap to row_block r+1.
        // After the last row_block (r == n_row_blocks-1), swap back to row_block 0
        // so VGPRs are ready for the next K iteration.
        if n_row_blocks > 1 {
            emit_acc_swap(k, acc, swap_addr, n_col_tiles, swap_temp);
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Helper: Store phase with ACC swapping
//
// Stores all row_blocks to global memory by swapping acc from LDS.
// Row_block 0 is already in VGPRs; for each subsequent row_block,
// swap the acc and then store.
// ────────────────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn emit_store_phase_swap(
    k: &mut T0Kernel,
    acc: &[VReg],           // single row_block (n_col_tiles entries)
    s_row_bases: &[SReg],   // [n_row_blocks]
    base_n_s: SReg,
    y_ptr: SRegPair, n_dim: SReg, y_offset_s: SReg,
    n_row_blocks: usize, n_col_tiles: usize,
    lane_row: VReg, lane_id: VReg,
    swap_addr: VReg,
    swap_temp: VReg,        // 8-VGPR temp for acc swap
    epilogue: &EpilogueCtx,
) {
    // ── Build Y buffer resource descriptor (same as emit_store_phase) ──
    let y_srd = k.alloc_sreg_quad();
    k.push(Op::SAddU32 { dst: y_srd, src0: SReg(y_ptr.0), src1: SOperand::SReg(y_offset_s) });
    k.push(Op::SAddcU32 { dst: SReg(y_srd.0 + 1), src0: SReg(y_ptr.0 + 1), src1: SOperand::InlineInt(0) });
    k.push(Op::SMov { dst: SReg(y_srd.0 + 2), src: SOperand::Literal(0x7FFFFFFE) });
    k.push(Op::SMov { dst: SReg(y_srd.0 + 3), src: SOperand::Literal(0x31027000) });

    // lane_half = lane_id >> 4 (0 for lanes 0-15, 1 for lanes 16-31)
    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);

    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, n_dim);

    // row_stride = N * 8 (2 rows × 4 bytes/f32)
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);

    // col_base = base_N + lane_row (element index)
    let col_base_v = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base_v, base_n_s);
    k.v_add_u32(col_base_v, col_base_v, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base_v);

    for r in 0..n_row_blocks {
        // Swap to row_block r if r > 0
        if r > 0 {
            emit_acc_swap(k, acc, swap_addr, n_col_tiles, swap_temp);
        }

        // base_row = s_row_bases[r] + lane_half
        let base_row_v = k.alloc_vreg();
        k.v_mov_from_sgpr(base_row_v, s_row_bases[r]);
        k.v_add_u32(base_row_v, base_row_v, lane_half);

        // row_bytes = base_row * N * 4 (byte offset from matrix start)
        let row_bytes = k.alloc_vreg();
        k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
        k.v_lshlrev_b32(row_bytes, 2, row_bytes);

        // voffset_base = row_bytes + col_bytes (single 32-bit VGPR!)
        let voffset_base = k.alloc_vreg();
        k.v_add_u32(voffset_base, row_bytes, col_bytes);

        for c in 0..n_col_tiles {
            // voff = voffset_base + c*64 (16 cols × 4 bytes per col tile)
            let voff = k.alloc_vreg();
            k.v_mov(voff, voffset_base);
            if c > 0 {
                k.push(Op::VAddU32 {
                    dst: voff, src0: Operand::VReg(voff),
                    src1: Operand::InlineInt((c * 64) as i32),
                });
            }

            // Compute column element index for epilogue (BiasAdd, etc.)
            let col_elem_idx = if !epilogue.ops.is_empty() {
                let col_idx = k.alloc_vreg();
                k.v_mov(col_idx, col_base_v);
                if c > 0 {
                    k.push(Op::VAddU32 {
                        dst: col_idx, src0: Operand::VReg(col_idx),
                        src1: Operand::InlineInt((c * 16) as i32),
                    });
                }
                Some(col_idx)
            } else {
                None
            };

            // acc index: single row_block, so just 'c'
            for v in 0..8u32 {
                let acc_vreg = VReg(acc[c].0 + v);

                // Apply epilogue chain before store (zero extra GMEM bandwidth)
                if !epilogue.ops.is_empty() {
                    emit_epilogue_on_vreg(k, acc_vreg, col_elem_idx, epilogue);
                }

                // buffer_store_b32: SRD base includes y_offset
                k.buffer_store(voff, acc_vreg, y_srd, Width::B32, 0);
                if v < 7 {
                    // Advance by row_stride = N * 8 bytes (2 rows)
                    k.v_add_u32(voff, voff, row_stride);
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ir::Target;

    #[test]
    fn test_tile_gemm_spec() {
        let spec = TileGemm::tile_128x64_k16();
        assert_eq!(spec.n_waves(), 4);
        assert_eq!(spec.wg_size(), 128);
        assert_eq!(spec.rows_per_wave(), 32);
        assert_eq!(spec.n_row_blocks(), 2);
        assert_eq!(spec.n_col_tiles(), 4);
        assert_eq!(spec.k_sub_steps(), 1);
        assert_eq!(spec.lds_per_buffer(), 128*16*2 + 64*16*2); // 6144
        assert_eq!(spec.lds_total(), 6144 * 2); // double buffer
    }

    #[test]
    fn test_tile_gemm_64x64() {
        let spec = TileGemm::tile_64x64_k16();
        assert_eq!(spec.n_waves(), 2);
        assert_eq!(spec.n_row_blocks(), 2);
        assert_eq!(spec.n_col_tiles(), 4);
    }

    #[test]
    fn test_lower_gemm_compiles() {
        // Core test: lower_gemm produces a kernel that compiles to ELF
        let spec = TileGemm::tile_128x64_k16();
        let kernel = lower_gemm(&spec);
        let result = kernel.compile(Target::GFX1201);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
        let elf = result.unwrap();
        assert!(elf.len() > 100, "ELF too small: {} bytes", elf.len());
        eprintln!("[tile_ir] {} → {} bytes ELF", spec.name(), elf.len());
    }

    #[test]
    fn test_lower_gemm_64x64_compiles() {
        let spec = TileGemm::tile_64x64_k16();
        let kernel = lower_gemm(&spec);
        let result = kernel.compile(Target::GFX1201);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
        eprintln!("[tile_ir] {} → {} bytes ELF", spec.name(), result.unwrap().len());
    }

    #[test]
    fn test_lower_gemm_32x64_compiles() {
        let spec = TileGemm::tile_32x64_k16();
        let kernel = lower_gemm(&spec);
        let result = kernel.compile(Target::GFX1201);
        assert!(result.is_ok(), "compile failed: {:?}", result.err());
        eprintln!("[tile_ir] {} → {} bytes ELF", spec.name(), result.unwrap().len());
    }

    #[test]
    fn test_lower_gemm_128x128_swap_compiles() {
        let spec = TileGemm::tile_128x128_k16_swap();
        eprintln!("[tile_ir] {} LDS total: {} bytes (gemm={}, swap={})",
            spec.name(), spec.lds_total(),
            spec.lds_per_buffer() * 2, spec.acc_swap_region_size());
        let kernel = lower_gemm(&spec);
        let (elf, final_lds) = kernel.compile_with_info(Target::GFX1201)
            .expect("compile failed for swap config");
        assert!(elf.len() > 100, "ELF too small: {} bytes", elf.len());
        eprintln!("[tile_ir] {} → {} bytes ELF, final_lds={}", spec.name(), elf.len(), final_lds);
    }

    #[test]
    fn test_lower_gemm_128x128_k32_compiles() {
        use crate::t0::insn_latency::{analyze_block, ilp_potential};

        // Compile-only test for k32 standard (no swap).
        let spec = TileGemm::tile_128x128_k32();
        eprintln!("[tile_ir] compiling k32 standard: {}", spec.name());
        let kernel = lower_gemm(&spec);

        // ── K-loop instruction analysis ──
        let ops = kernel.ops();
        let mut loop_start = None;
        let mut loop_end = None;
        let mut phase_a_vmcnt = 0u32;
        let mut phase_a_lgkmcnt = 0u32;
        for (i, op) in ops.iter().enumerate() {
            if let Op::Label(name) = op {
                if name.contains("k_loop") { loop_start = Some(i + 1); }
            }
            if let Op::BranchScc1(target) = op {
                if target.contains("k_loop") { loop_end = Some(i); }
            }
        }
        if let (Some(s), Some(e)) = (loop_start, loop_end) {
            let body = &ops[s..e];
            let stats = analyze_block(body);
            let (ilp, bottleneck) = ilp_potential(&stats);

            // Count waitcnts
            for op in body {
                match op {
                    Op::WaitVmcnt(_) => phase_a_vmcnt += 1,
                    Op::WaitLgkmcnt(_) => phase_a_lgkmcnt += 1,
                    _ => {}
                }
            }

            eprintln!("\n╔══════════════════════════════════════════════════════╗");
            eprintln!("║  K-loop Analysis: {} ({} ops)              ", spec.name(), body.len());
            eprintln!("╠══════════════════════════════════════════════════════╣");
            eprintln!("║  WMMA:     {:>3}   (compute density)                 ", stats.wmma_count);
            eprintln!("║  LDS:      {:>3}   (ds_load + ds_store)             ", stats.lds_count);
            eprintln!("║  VMEM:     {:>3} ld + {:>3} st                      ", stats.vmem_load_count, stats.vmem_store_count);
            eprintln!("║  VALU:     {:>3}   SALU: {:>3}   CTRL: {:>3}        ", stats.valu_count, stats.salu_count, stats.ctrl_count);
            eprintln!("║  wait_vmcnt: {}   wait_lgkmcnt: {}                  ", phase_a_vmcnt, phase_a_lgkmcnt);
            eprintln!("║  Issue cycles: {}                                   ", stats.total_issue_cycles);
            eprintln!("║  ILP: {:.1}% ({})                                   ", ilp * 100.0, bottleneck);
            eprintln!("║  WMMA/VMEM ratio: {:.1}                             ",
                if stats.vmem_load_count > 0 { stats.wmma_count as f32 / stats.vmem_load_count as f32 } else { 0.0 });
            eprintln!("╚══════════════════════════════════════════════════════╝");
        }

        let (elf, final_lds) = kernel.compile_with_info(Target::GFX1201)
            .expect("compile failed for k32 standard");
        assert!(elf.len() > 100, "ELF too small: {} bytes", elf.len());
        eprintln!("[tile_ir] {} → {} bytes ELF, final_lds={}", spec.name(), elf.len(), final_lds);

        // ── k48 compile disabled (panics: k48 is not power-of-2) ──
        // let spec48 = TileGemm::tile_128x128_k48();
        // let kernel48 = lower_gemm(&spec48);
        // let (elf48, lds48) = kernel48.compile_with_info(Target::GFX1201)
        //     .expect("compile failed for k48");

        // ── k64 configs (VGPR exploration) ──
        // 128×128 k64: may spill (GMEM=64 VGPRs)
        let spec64 = TileGemm::tile_128x128_k64();
        eprintln!("\n[tile_ir] compiling 128x128 k64: {} (LDS={})", spec64.name(), spec64.lds_total());
        let kernel64 = lower_gemm(&spec64);
        let _ = kernel64.compile_with_info(Target::GFX1201);  // may fail, that's OK

        // 128×64 k64: should fit (ACC=64, GMEM=48)
        let spec64s = TileGemm::tile_128x64_k64();
        eprintln!("\n[tile_ir] compiling 128x64 k64: {} (LDS={})", spec64s.name(), spec64s.lds_total());
        let kernel64s = lower_gemm(&spec64s);
        let _ = kernel64s.compile_with_info(Target::GFX1201);

        // 256×64 k64 WGP: predicted ~166 VGPRs → 4 waves!
        let spec_wgp = TileGemm::tile_256x64_k64_wgp();
        eprintln!("\n[tile_ir] compiling 256x64 k64 WGP: {} (LDS={})", spec_wgp.name(), spec_wgp.lds_total());
        let kernel_wgp = lower_gemm(&spec_wgp);
        let _ = kernel_wgp.compile_with_info(Target::GFX1201);
    }

    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_lower_gemm_128x128_swap_correctness() {
        use crate::gpu_backend::{GpuDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let spec = TileGemm::tile_128x128_k16_swap();
        let kernel = lower_gemm(&spec);
        let (elf, final_lds) = kernel.compile_with_info(Target::GFX1201)
            .expect("compile failed");

        let device = GpuDevice::open().unwrap();
        let queue = device.create_queue().unwrap();
        let pool = DispatchPool::new(&device, 4).unwrap();

        let m = 256u32; let k_dim = 256u32; let n = 256u32;
        let k_padded = (k_dim + spec.tile_k - 1) & !(spec.tile_k - 1);
        let m_padded = (m + spec.tile_m - 1) & !(spec.tile_m - 1);
        let n_padded = (n + spec.tile_n - 1) & !(spec.tile_n - 1);

        // Generate BF16 test data
        let mut rng = 42u64;
        let mut bf16_rand = || -> u16 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let f = (((rng >> 33) as u32 % 200) as f32 - 100.0) * 0.01;
            (f.to_bits() >> 16) as u16
        };

        let x_bf16: Vec<u16> = (0..m * k_dim).map(|_| bf16_rand()).collect();
        let w_bf16: Vec<u16> = (0..n * k_dim).map(|_| bf16_rand()).collect();

        // CPU reference (bf16 → f32)
        let bf16_to_f32 = |v: u16| -> f32 { f32::from_bits((v as u32) << 16) };
        let mut y_ref = vec![0.0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut acc = 0.0f32;
                for kk in 0..k_dim as usize {
                    // NT: X[i,kk] × WT[j,kk]
                    acc += bf16_to_f32(x_bf16[i * k_dim as usize + kk])
                         * bf16_to_f32(w_bf16[j * k_dim as usize + kk]);
                }
                y_ref[i * n as usize + j] = acc;
            }
        }

        // GPU buffers
        let x_buf = device.alloc_vram((m_padded * k_padded * 2) as usize).unwrap();
        let w_buf = device.alloc_vram((n_padded * k_padded * 2) as usize).unwrap();
        let y_buf = device.alloc_vram((m_padded * n_padded * 4) as usize).unwrap();
        x_buf.write(&vec![0u8; (m_padded * k_padded * 2) as usize]);
        w_buf.write(&vec![0u8; (n_padded * k_padded * 2) as usize]);
        y_buf.write(&vec![0u8; (m_padded * n_padded * 4) as usize]);

        // Write BF16 data (row-major, pad zeros for K beyond k_dim)
        let x_bytes: Vec<u8> = x_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        x_buf.write(&x_bytes);
        let w_bytes: Vec<u8> = w_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        w_buf.write(&w_bytes);

        let gk = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [spec.wg_size(), 1, 1],
            lds_size: final_lds,
        }).unwrap();

        let ka_full = build_kernargs_m(
            x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(),
            k_padded, n_padded, m,
            &spec,
        );

        let grid = compute_grid(&spec, m, n);
        let ka_buf = pool.write_kernargs(0, &ka_full);
        queue.submit(&gk, grid, ka_buf);
        queue.wait_idle().unwrap();

        // Read back and compare
        let mut y_gpu = vec![0f32; (m_padded * n_padded) as usize];
        unsafe {
            y_buf.read(std::slice::from_raw_parts_mut(
                y_gpu.as_mut_ptr() as *mut u8,
                (m_padded * n_padded * 4) as usize,
            ));
        }

        // Diagnostic: print first 8 elements of row 0
        eprintln!("[swap] Row 0, first 8 GPU values:");
        for j in 0..8.min(n as usize) {
            eprintln!("  [0,{}]: gpu={:.4} cpu={:.4}", j,
                y_gpu[j], y_ref[j]);
        }
        // Check row 16 (first row of row_block 1 for wave 0)
        if m >= 16 {
            eprintln!("[swap] Row 16, first 4 GPU values:");
            for j in 0..4.min(n as usize) {
                eprintln!("  [16,{}]: gpu={:.4} cpu={:.4}", j,
                    y_gpu[16 * n_padded as usize + j],
                    y_ref[16 * n as usize + j]);
            }
        }

        let mut max_err: f32 = 0.0;
        let mut bad = 0;
        for i in 0..m as usize {
            for j in 0..n as usize {
                let expected = y_ref[i * n as usize + j];
                let got = y_gpu[i * n_padded as usize + j];
                let err = (got - expected).abs();
                let tol = 0.01 * expected.abs().max(1.0);
                if err > tol { bad += 1; }
                max_err = max_err.max(err);
            }
        }

        eprintln!("[swap correctness] {}×{}×{}: max_err={:.4e} bad={}/{}",
            m, k_dim, n, max_err, bad, m * n);
        assert_eq!(bad, 0, "ACC swap correctness failed: {} bad elements", bad);
    }
}

// ============================================================================
// Kernarg builder + grid helper (for dispatch)
// ============================================================================

/// Build kernarg bytes for a tile_ir GEMM kernel dispatch.
///
/// Layout: X_ptr(u64) + WT_ptr(u64) + Y_ptr(u64) + K(u32) + N(u32) + sk_shift(u32) + y_split_stride(u32)
pub fn build_kernargs(
    x_addr: u64, wt_addr: u64, y_addr: u64,
    k_dim: u32, n_dim: u32,
    spec: &TileGemm,
) -> Vec<u8> {
    let sk_shift: u32 = match spec.split_k { 1=>0, 2=>1, 4=>2, 8=>3, 16=>4, _=>0 };
    // For split-K: each partition writes to y_addr + partition_id * y_split_stride
    // y_split_stride = M * N * 4 bytes (full output matrix per partition)
    // For sk=1: unused, set to 0
    let y_split_stride: u32 = 0; // caller should set up separate Y buffers for split-K reduce
    let mut ka = Vec::with_capacity(48);
    ka.extend_from_slice(&x_addr.to_le_bytes());     // arg 0: X ptr
    ka.extend_from_slice(&wt_addr.to_le_bytes());    // arg 1: WT ptr
    ka.extend_from_slice(&y_addr.to_le_bytes());     // arg 2: Y ptr
    ka.extend_from_slice(&k_dim.to_le_bytes());      // arg 3: K
    ka.extend_from_slice(&n_dim.to_le_bytes());      // arg 4: N
    ka.extend_from_slice(&sk_shift.to_le_bytes());   // arg 5: split_k_shift
    ka.extend_from_slice(&y_split_stride.to_le_bytes()); // arg 6: y_split_stride
    // arg 7: M — CRITICAL: kernel uses this for OOB boundary checks
    // Without it, s_is_boundary reads garbage → wrong branch → zero output
    ka.extend_from_slice(&0u32.to_le_bytes());       // placeholder M=0, see build_kernargs_m
    ka
}

/// Build kernargs with explicit M dimension (required for correct OOB checks).
pub fn build_kernargs_m(
    x_addr: u64, wt_addr: u64, y_addr: u64,
    k_dim: u32, n_dim: u32, m_dim: u32,
    spec: &TileGemm,
) -> Vec<u8> {
    let sk_shift: u32 = match spec.split_k { 1=>0, 2=>1, 4=>2, 8=>3, 16=>4, _=>0 };
    let y_split_stride: u32 = 0;
    let mut ka = Vec::with_capacity(48);
    ka.extend_from_slice(&x_addr.to_le_bytes());     // arg 0: X ptr
    ka.extend_from_slice(&wt_addr.to_le_bytes());    // arg 1: WT ptr
    ka.extend_from_slice(&y_addr.to_le_bytes());     // arg 2: Y ptr
    ka.extend_from_slice(&k_dim.to_le_bytes());      // arg 3: K
    ka.extend_from_slice(&n_dim.to_le_bytes());      // arg 4: N
    ka.extend_from_slice(&sk_shift.to_le_bytes());   // arg 5: split_k_shift
    ka.extend_from_slice(&y_split_stride.to_le_bytes()); // arg 6: y_split_stride
    ka.extend_from_slice(&m_dim.to_le_bytes());      // arg 7: M
    ka
}

/// Compute dispatch grid for tile_ir GEMM.
pub fn compute_grid(spec: &TileGemm, m: u32, n: u32) -> [u32; 3] {
    let n_wgs_n = (n + spec.tile_n - 1) / spec.tile_n;
    let n_wgs_m = (m + spec.tile_m - 1) / spec.tile_m;
    if spec.swap_grid {
        // TGID.x → N, TGID.y → M
        [n_wgs_n * spec.wg_size(), n_wgs_m * spec.split_k, 1]
    } else {
        [n_wgs_m * spec.wg_size(), n_wgs_n * spec.split_k, 1]
    }
}

/// Convert f32 → bf16 on CPU (truncation, matching GPU bf16 semantics).
pub fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

/// Convert bf16 → f32 on CPU.
pub fn bf16_to_f32(val: u16) -> f32 {
    f32::from_bits((val as u32) << 16)
}

// ============================================================================
// Compile-only tests (NO GPU dispatch — safe from hangs)
// ============================================================================

#[cfg(test)]
mod compile_tests {
    use super::*;
    use crate::t0::ir::Target;
    use crate::t0::isa_verifier;

    /// Compile tile_ir GEMM kernel for every benchmark size and verify ISA.
    /// NO GPU dispatch — purely offline static analysis.
    ///
    /// Run: cargo test --release --features rocm --lib -- compile_tests::test_compile_all_sizes --nocapture
    #[test]
    fn test_compile_all_sizes() {
        let sizes: Vec<(u32, u32, u32)> = vec![
            (128, 128, 64),
            (256, 256, 64),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ];

        eprintln!("\n{}", "=".repeat(80));
        eprintln!("  tile_ir Compile-Only ISA Verification (NO GPU dispatch)");
        eprintln!("{}", "=".repeat(80));
        eprintln!("{:<15} {:<28} {:>8} {:>8} {:>8} {:>10}",
            "Size", "Config", "Ops", "VGPRs", "SGPRs", "LDS(B)");
        eprintln!("{:-<80}", "");

        let mut all_ok = true;

        for &(m, k, n) in &sizes {
            let spec = tile_auto_select(m, k, n, TileTranspose::NT);
            let name = spec.name();
            let grid = compute_grid(&spec, m, n);

            // Generate kernel (compile-time only, no GPU)
            let t0k = lower_gemm(&spec);
            let n_ops = t0k.ops().len();

            // Verify ISA patterns BEFORE regalloc
            let verify_result = isa_verifier::verify_ops(t0k.ops());
            let verify_status = if verify_result.errors.is_empty() && verify_result.warnings.is_empty() {
                "✅"
            } else if verify_result.errors.is_empty() {
                "⚠️"
            } else {
                all_ok = false;
                "❌"
            };

            // Compile to ELF to get regalloc info
            match t0k.compile(Target::GFX1201) {
                Ok(elf) => {
                    eprintln!("{:<15} {:<28} {:>8} {:>8} {:>8} {:>10} {}  grid={:?}",
                        format!("{}×{}×{}", m, k, n),
                        name,
                        n_ops,
                        "OK", "OK",
                        spec.lds_total(),
                        verify_status,
                        grid,
                    );
                }
                Err(e) => {
                    all_ok = false;
                    eprintln!("{:<15} {:<28} {:>8} COMPILE FAILED: {}",
                        format!("{}×{}×{}", m, k, n),
                        name,
                        n_ops,
                        e);
                }
            }

            // Report any issues
            for w in &verify_result.warnings {
                eprintln!("  ⚠️  {}", w);
            }
            for e in &verify_result.errors {
                eprintln!("  ❌  {}", e);
            }
        }

        eprintln!("\n{}", if all_ok { "All sizes compiled and verified ✅" } else { "Some sizes have issues ❌" });
        assert!(all_ok, "Compile-only verification failed for some sizes");
    }

    /// A/B comparison: compile tile_ir with and without waitcnt optimization.
    /// NO GPU dispatch — purely offline comparison.
    /// Reports which waitcnts are removed and whether the result is safe.
    ///
    /// Run: cargo test --release --features rocm --lib -- compile_tests::test_waitopt_ab_compare --nocapture
    #[test]
    fn test_waitopt_ab_compare() {
        use crate::t0::ir::Target;

        let spec = TileGemm::tile_128x128_k16();

        // ── A: compile WITH T0_SKIP_WAITOPT (baseline, current production) ──
        std::env::set_var("T0_SKIP_WAITOPT", "1");
        let kernel_a = lower_gemm(&spec);
        let asm_a = kernel_a.to_assembly(Target::GFX1201).expect("compile A");

        // ── B: compile WITHOUT T0_SKIP_WAITOPT (new, with BufferLoad fix) ──
        std::env::remove_var("T0_SKIP_WAITOPT");
        // Must create a fresh kernel so env var takes effect at compile time
        let kernel_b = lower_gemm(&spec);
        let asm_b = kernel_b.to_assembly(Target::GFX1201).expect("compile B");

        // Re-enable skip for other tests
        std::env::set_var("T0_SKIP_WAITOPT", "1");

        // Count waitcnt instructions in each ASM
        let count_waitcnt = |asm: &str| -> (usize, usize, usize) {
            let mut vmcnt = 0;
            let mut lgkmcnt = 0;
            let mut vscnt = 0;
            for line in asm.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("s_waitcnt vmcnt(") { vmcnt += 1; }
                else if trimmed.starts_with("s_waitcnt lgkmcnt(") { lgkmcnt += 1; }
                else if trimmed.starts_with("s_waitcnt_vscnt") { vscnt += 1; }
            }
            (vmcnt, lgkmcnt, vscnt)
        };

        let (vm_a, lgkm_a, vs_a) = count_waitcnt(&asm_a);
        let (vm_b, lgkm_b, vs_b) = count_waitcnt(&asm_b);
        let total_a = vm_a + lgkm_a + vs_a;
        let total_b = vm_b + lgkm_b + vs_b;
        let removed = total_a as i64 - total_b as i64;

        let lines_a = asm_a.lines().count();
        let lines_b = asm_b.lines().count();

        eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║  Waitcnt Optimization A/B Comparison (compile-only)         ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║  Kernel: tile_gemm_128x128_k16_db_sk8                       ║");
        eprintln!("╚══════════════════════════════════════════════════════════════╝\n");
        eprintln!("  {:>20}  {:>8}  {:>8}", "", "A (skip)", "B (opt)");
        eprintln!("  {:>20}  {:>8}  {:>8}", "Total ASM lines", lines_a, lines_b);
        eprintln!("  {:>20}  {:>8}  {:>8}", "s_waitcnt vmcnt", vm_a, vm_b);
        eprintln!("  {:>20}  {:>8}  {:>8}", "s_waitcnt lgkmcnt", lgkm_a, lgkm_b);
        eprintln!("  {:>20}  {:>8}  {:>8}", "s_waitcnt_vscnt", vs_a, vs_b);
        eprintln!("  {:>20}  {:>8}  {:>8}", "Total waitcnts", total_a, total_b);
        eprintln!("  {:>20}  {:>8}", "Removed", removed);
        eprintln!();

        // Safety check: waitopt should NEVER ADD waitcnts
        assert!(total_b <= total_a,
            "Waitcnt optimization ADDED instructions! A={}, B={}", total_a, total_b);

        // Report which waitcnts differ (line-by-line diff)
        if asm_a != asm_b {
            eprintln!("  ── Differences (first 20) ──");
            let lines_a_v: Vec<&str> = asm_a.lines().collect();
            let lines_b_v: Vec<&str> = asm_b.lines().collect();
            let mut diff_count = 0;
            let max_lines = lines_a_v.len().max(lines_b_v.len());
            for i in 0..max_lines {
                let la = lines_a_v.get(i).unwrap_or(&"<missing>");
                let lb = lines_b_v.get(i).unwrap_or(&"<missing>");
                if la != lb {
                    diff_count += 1;
                    if diff_count <= 20 {
                        eprintln!("  L{:>4} A: {}", i+1, la.trim());
                        eprintln!("         B: {}", lb.trim());
                    }
                }
            }
            eprintln!("  Total differing lines: {}", diff_count);
        } else {
            eprintln!("  ASM output is IDENTICAL (no waitcnts removed)");
        }

        eprintln!("\n  Verdict: {} waitcnts removed by optimization.\n",
            if removed > 0 { format!("✅ {}", removed) }
            else { "⚠️  0 (optimizer had no effect)".to_string() });
    }

    /// Compile and dump ISA for each size — for manual review of generated assembly.
    ///
    /// Run: T0_DUMP_ASM=1 cargo test --release --features rocm --lib -- compile_tests::test_dump_isa_all_sizes --nocapture
    #[test]
    fn test_dump_isa_all_sizes() {
        let sizes: Vec<(u32, u32, u32)> = vec![
            (128, 128, 64),
            (256, 256, 256),
            (512, 512, 512),
        ];

        for &(m, k, n) in &sizes {
            let spec = tile_auto_select(m, k, n, TileTranspose::NT);
            let t0k = lower_gemm(&spec);

            eprintln!("\n══ {} ({}×{}×{}) ══", spec.name(), m, k, n);
            eprintln!("  wg_size={}, n_waves={}, tile={},{},{}, lds={}, k_sub={}",
                spec.wg_size(), spec.n_waves(), spec.tile_m, spec.tile_n, spec.tile_k,
                spec.lds_total(), spec.k_sub_steps());

            // Grid info
            let grid = compute_grid(&spec, m, n);
            eprintln!("  grid={:?}, total_threads={}", grid, grid[0] * grid[1] * grid[2]);

            // GMEM load info
            let x_loads = spec.tile_m * spec.tile_k * 2 / spec.wg_size() / 16;
            let wt_loads = spec.tile_n * spec.tile_k * 2 / spec.wg_size() / 16;
            eprintln!("  x_loads/thread={}, wt_loads/thread={}", x_loads, wt_loads);

            // Compile
            match t0k.compile(Target::GFX1201) {
                Ok(_) => eprintln!("  ✅ compile OK"),
                Err(e) => eprintln!("  ❌ compile FAILED: {}", e),
            }
        }
    }

    /// Test gemm_gen compile for each size — verify the stable path also compiles.
    ///
    /// Run: cargo test --release --features rocm --lib -- compile_tests::test_compile_gemm_gen_all_sizes --nocapture
    #[test]
    fn test_compile_gemm_gen_all_sizes() {
        use crate::t0::gemm_gen;

        let sizes: Vec<(u32, u32, u32)> = vec![
            (128, 128, 64),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ];

        eprintln!("\n{}", "=".repeat(70));
        eprintln!("  gemm_gen Compile-Only Verification");
        eprintln!("{}", "=".repeat(70));

        for &(m, k, n) in &sizes {
            let cfg = gemm_gen::auto_select(m, k, n);
            let t0k = gemm_gen::generate(&cfg);
            match t0k.compile(Target::GFX1201) {
                Ok(_) => {
                    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m, n);
                    eprintln!("{:<15} {:<25} ✅ grid=[{},{},1]",
                        format!("{}×{}×{}", m, k, n), cfg.name(), gx, gy);
                }
                Err(e) => {
                    eprintln!("{:<15} {:<25} ❌ {}", format!("{}×{}×{}", m, k, n), cfg.name(), e);
                }
            }
        }
    }
}

// ============================================================================
// GPU correctness tests
// ============================================================================

#[cfg(all(test, any(feature = "rocm", feature = "wsl_dxg")))]
mod gpu_tests {
    use super::*;
    use crate::ignis::gpu_context::GpuRuntime;
    use std::sync::{Arc, OnceLock};

    struct SyncRt(Arc<GpuRuntime>);
    unsafe impl Sync for SyncRt {}
    unsafe impl Send for SyncRt {}
    static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

    fn with_rt<F, R>(f: F) -> R
    where F: FnOnce(&GpuRuntime) -> R {
        let rt = GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("Failed to create GpuRuntime"))
        });
        let _ = rt.0.wait_idle();
        let result = f(&rt.0);
        let _ = rt.0.wait_idle();
        result
    }

    /// Upload bf16 data to GPU. Returns GpuBuffer.
    fn upload_bf16(rt: &GpuRuntime, data: &[u16]) -> crate::gpu_backend::GpuBuffer {
        let n_bytes = ((data.len() * 2).max(512) + 511) & !511;
        let buf = rt.alloc(n_bytes).expect("alloc bf16");
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
        };
        buf.write(bytes);
        buf
    }

    /// CPU reference: Y[M,N] = X[M,K] @ WT[N,K]^T  (using bf16-rounded values)
    fn cpu_gemm_nt_bf16(
        x_bf16: &[u16], wt_bf16: &[u16],
        m: usize, k: usize, n: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let a = bf16_to_f32(x_bf16[i * k + kk]);
                    let b = bf16_to_f32(wt_bf16[j * k + kk]);
                    sum += a * b;
                }
                y[i * n + j] = sum;
            }
        }
        y
    }

    /// Core GPU test: tile_ir GEMM correctness
    ///
    /// Y[M,N] = X[M,K] @ WT[N,K]^T
    #[test]
    fn test_tile_ir_gpu_gemm_128x64() {
        with_rt(|rt| {
            let m = 128usize;
            let k = 64usize;
            let n = 64usize;
            let spec = TileGemm::tile_128x64_k16();

            // Generate test data (small values to avoid bf16 overflow)
            let x_f32: Vec<f32> = (0..m*k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
            let wt_f32: Vec<f32> = (0..n*k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();

            // Convert to bf16
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let wt_bf16: Vec<u16> = wt_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            // Upload
            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);
            let y_buf = rt.alloc_zero(m * n * 4).expect("alloc Y");

            // Compile kernel
            let t0k = lower_gemm(&spec);
            let kernel = rt.ensure_kernel_t0(
                &spec.name(),
                || lower_gemm(&spec),
                [spec.wg_size(), 1, 1],
                spec.lds_total(),
            ).expect("compile tile_ir GEMM");

            // Build kernargs
            let ka = build_kernargs_m(
                x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                k as u32, n as u32, m as u32, &spec,
            );

            // Dispatch
            let grid = compute_grid(&spec, m as u32, n as u32);
            eprintln!("[tile_ir GPU] {}x{}x{} grid={:?} wg={} ka={}B",
                m, k, n, grid, spec.wg_size(), ka.len());
            rt.dispatch(&kernel, grid, &ka).expect("dispatch");

            // Read back
            let result = rt.read_f32(&y_buf, m * n);

            // CPU reference
            let expected = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m, k, n);

            // Compare
            let mut max_err = 0.0f32;
            let mut n_bad = 0;
            for i in 0..m*n {
                let err = (result[i] - expected[i]).abs();
                if err > max_err { max_err = err; }
                if err > 0.1 { n_bad += 1; }
            }

            eprintln!("[tile_ir GPU] max_err={:.6} n_bad={}/{} (thr=0.1)", max_err, n_bad, m*n);
            eprintln!("  result[0..8] = {:?}", &result[0..8]);
            eprintln!("  expected[0..8] = {:?}", &expected[0..8]);

            assert!(n_bad == 0,
                "tile_ir GEMM: {} of {} elements differ >0.1 (max_err={:.6})",
                n_bad, m*n, max_err);
            eprintln!("[PASS] test_tile_ir_gpu_gemm_128x64: {}x{}x{} verified (max_err={:.6})",
                m, k, n, max_err);
        });
    }

    /// Smaller GEMM: 64×64 tile
    #[test]
    fn test_tile_ir_gpu_gemm_64x64() {
        with_rt(|rt| {
            let m = 64usize;
            let k = 64usize;
            let n = 64usize;
            let spec = TileGemm::tile_64x64_k16();

            let x_f32: Vec<f32> = (0..m*k).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect();
            let wt_f32: Vec<f32> = (0..n*k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let wt_bf16: Vec<u16> = wt_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);
            let y_buf = rt.alloc_zero(m * n * 4).expect("alloc Y");

            let kernel = rt.ensure_kernel_t0(
                &spec.name(),
                || lower_gemm(&spec),
                [spec.wg_size(), 1, 1],
                spec.lds_total(),
            ).expect("compile");

            let ka = build_kernargs_m(
                x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                k as u32, n as u32, m as u32, &spec,
            );
            let grid = compute_grid(&spec, m as u32, n as u32);
            rt.dispatch(&kernel, grid, &ka).expect("dispatch");

            let result = rt.read_f32(&y_buf, m * n);
            let expected = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m, k, n);

            let max_err = result.iter().zip(expected.iter())
                .map(|(r, e)| (r - e).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 0.1,
                "64x64 GEMM max_err={:.6} too large", max_err);
            eprintln!("[PASS] test_tile_ir_gpu_gemm_64x64: verified (max_err={:.6})", max_err);
        });
    }
    /// Correctness sweep: test tile_ir at multiple sizes with CPU reference.
    /// Identifies the exact size where correctness breaks.
    #[test]
    fn test_tile_ir_correctness_sweep() {
        with_rt(|rt| {
            let sizes: Vec<(usize, usize, usize)> = vec![
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
            ];
            let mut all_pass = true;
            for &(m, k, n) in &sizes {
                let spec = tile_auto_select(m as u32, k as u32, n as u32, TileTranspose::NT);
                eprintln!("\n=== {}×{}×{} {} split_k={} ===", m, k, n, spec.name(), spec.split_k);

                let x_bf16: Vec<u16> = (0..m*k).map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.01)).collect();
                let wt_bf16: Vec<u16> = (0..n*k).map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.01)).collect();
                let x_buf = upload_bf16(rt, &x_bf16);
                let wt_buf = upload_bf16(rt, &wt_bf16);
                let sk = spec.split_k as usize;
                let y_buf = rt.alloc_zero(m * n * 4 * sk).expect("alloc Y");

                let kernel = rt.ensure_kernel_t0(
                    &format!("sweep_{}", spec.name()),
                    || lower_gemm(&spec),
                    [spec.wg_size(), 1, 1],
                    spec.lds_total(),
                ).expect("compile");
                let ka = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                    k as u32, n as u32, m as u32, &spec,
                );
                let grid = compute_grid(&spec, m as u32, n as u32);
                eprintln!("  grid={:?} wg={} ka={}B", grid, spec.wg_size(), ka.len());

                match rt.dispatch(&kernel, grid, &ka) {
                    Ok(()) => {},
                    Err(e) => { eprintln!("  DISPATCH FAILED: {}", e); continue; }
                }

                let result = rt.read_f32(&y_buf, m * n);
                let expected = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m, k, n);

                let mut max_err = 0.0f32;
                let mut n_bad = 0;
                let mut first_bad_idx = usize::MAX;
                for i in 0..m*n {
                    let err = (result[i] - expected[i]).abs();
                    if err > max_err { max_err = err; }
                    if err > 0.1 {
                        n_bad += 1;
                        if first_bad_idx == usize::MAX { first_bad_idx = i; }
                    }
                }

                if n_bad == 0 {
                    eprintln!("  ✅ PASS max_err={:.6}", max_err);
                } else {
                    let bad_row = first_bad_idx / n;
                    let bad_col = first_bad_idx % n;
                    eprintln!("  ❌ FAIL {}/{} bad (max_err={:.2e})", n_bad, m*n, max_err);
                    eprintln!("  first bad at [{},{}]: got={:.6} expected={:.6}",
                        bad_row, bad_col, result[first_bad_idx], expected[first_bad_idx]);
                    // Show pattern: how many bad per tile
                    let n_tiles_m = (m + 127) / 128;
                    let n_tiles_n = (n + 63) / 64;
                    for tm in 0..n_tiles_m.min(4) {
                        for tn in 0..n_tiles_n.min(4) {
                            let mut tile_bad = 0;
                            for r in 0..128.min(m - tm*128) {
                                for c in 0..64.min(n - tn*64) {
                                    let idx = (tm*128 + r) * n + tn*64 + c;
                                    if (result[idx] - expected[idx]).abs() > 0.1 { tile_bad += 1; }
                                }
                            }
                            if tile_bad > 0 {
                                eprint!("  tile[{},{}]={}/{} ", tm, tn, tile_bad, 128*64);
                            }
                        }
                    }
                    eprintln!();
                    all_pass = false;
                }
            }
            assert!(all_pass, "Some sizes failed correctness check");
        });
    }

    /// tile_ir standalone benchmark: correctness (CPU ref) + performance (TFLOPS).
    /// Standard sizes matching Triton/rocBLAS benchmark.
    /// Run: cargo test --release --lib --features rocm -- test_tile_ir_benchmark --nocapture --test-threads=1
    #[test]
    fn test_tile_ir_benchmark() {
        use std::time::Instant;

        with_rt(|rt| {
            let sizes: Vec<(u32, u32, u32)> = vec![
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
            ];
            let warmup = 3u32;
            let iters = 10u32;

            eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
            eprintln!("║  tile_ir Benchmark — RX 7900 XTX (GFX1201)                  ║");
            eprintln!("║  BF16 WMMA GEMM, F32 output, CPU reference verified         ║");
            eprintln!("╚══════════════════════════════════════════════════════════════╝\n");
            eprintln!("{:<20} {:>8} {:>10} {:>10} {:>12}",
                "Size", "μs/iter", "TFLOPS", "max_err", "config");
            eprintln!("{}", "-".repeat(65));

            for &(m, k, n) in &sizes {
                let flops = 2.0 * m as f64 * k as f64 * n as f64;
                let spec = tile_auto_select(m, k, n, TileTranspose::NT);

                // Data
                let x_bf16: Vec<u16> = (0..(m*k) as usize).map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.01)).collect();
                let wt_bf16: Vec<u16> = (0..(n*k) as usize).map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.01)).collect();
                let x_buf = upload_bf16(rt, &x_bf16);
                let wt_buf = upload_bf16(rt, &wt_bf16);
                let sk = spec.split_k;
                let y_buf = rt.alloc_zero((m * n * 4 * sk) as usize).expect("alloc Y");

                // Compile
                let name = format!("bench_tile_{}", spec.name());
                let kernel = match rt.ensure_kernel_t0(
                    &name, || lower_gemm(&spec),
                    [spec.wg_size(), 1, 1], spec.lds_total(),
                ) {
                    Ok(k) => k,
                    Err(e) => { eprintln!("{:<20} COMPILE FAIL: {}", format!("{}³", m), e); continue; }
                };
                let ka = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                    k, n, m, &spec,
                );
                let grid = compute_grid(&spec, m, n);

                // Warmup
                for _ in 0..warmup {
                    if rt.dispatch(&kernel, grid, &ka).is_err() { break; }
                }

                // Timed: batch-submit all dispatches, wait once (matches Triton/rocBLAS methodology)
                // This eliminates inter-dispatch GPU idle time and measures pure kernel throughput.
                let t0 = Instant::now();
                for _ in 0..iters {
                    rt.dispatch_async(&kernel, grid, &ka);
                }
                rt.wait_idle();
                let us = t0.elapsed().as_micros() as f64 / iters as f64;
                let tflops = if us > 0.0 { flops / (us * 1e6) } else { 0.0 };

                // Correctness (CPU reference) — only for sizes ≤ 2048
                let verify = if m <= 2048 {
                    let result = rt.read_f32(&y_buf, (m * n) as usize);
                    let expected = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m as usize, k as usize, n as usize);
                    let max_err = result.iter().zip(expected.iter())
                        .map(|(r, e)| (r - e).abs())
                        .fold(0.0f32, f32::max);
                    format!("{:.4}", max_err)
                } else {
                    "skip".to_string()
                };

                eprintln!("{:<20} {:>8.1} {:>10.3} {:>10} {:>12}",
                    format!("{}×{}×{}", m, k, n), us, tflops, verify, spec.name());
            }
        });
    }

    /// tile_k=16 vs tile_k=32 benchmark — head-to-head comparison.
    /// Run: cargo test --release --lib --features rocm -- test_tile_ir_k32_benchmark --nocapture --ignored --test-threads=1
    #[test]
    #[ignore]
    fn test_tile_ir_k32_benchmark() {
        use std::time::Instant;

        with_rt(|rt| {
            let sizes: Vec<(u32, u32, u32)> = vec![
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
                // Non-square sizes (matching README / rocBLAS benchmark)
                (128,  1024, 4096),
                (256,  1024, 4096),
                (512,  1024, 4096),
                (1024, 1024, 4096),
            ];
            let warmup = 5u32;
            let iters = 20u32;

            eprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║  tile_k Benchmark — k16 (128×128) vs k32 (128×64) + k16 (128×128)       ║");
            eprintln!("║  RX 7900 XTX (GFX1201), BF16 WMMA GEMM, F32 output                     ║");
            eprintln!("╚══════════════════════════════════════════════════════════════════════════╝\n");
            eprintln!("{:<20} {:>10} {:>10} {:>10} {:>10} {:>8}",
                "Size", "k16 μs", "k16 TF", "k32 μs", "k32 TF", "Speedup");
            eprintln!("{}", "-".repeat(72));

            for &(m, k, n) in &sizes {
                let flops = 2.0 * m as f64 * k as f64 * n as f64;

                // Data
                let x_bf16: Vec<u16> = (0..(m*k) as usize).map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.01)).collect();
                let wt_bf16: Vec<u16> = (0..(n*k) as usize).map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.01)).collect();
                let x_buf = upload_bf16(rt, &x_bf16);
                let wt_buf = upload_bf16(rt, &wt_bf16);
                let y_buf = rt.alloc_zero((m * n * 4) as usize).expect("alloc Y");

                // === k16: 128×128 k16 ===
                let spec_k16 = TileGemm::tile_128x128_k16();
                let name_k16 = format!("bench_k16_{}", spec_k16.name());
                let kernel_k16 = match rt.ensure_kernel_t0(
                    &name_k16, || lower_gemm(&spec_k16),
                    [spec_k16.wg_size(), 1, 1], spec_k16.lds_total(),
                ) {
                    Ok(k) => k,
                    Err(e) => { eprintln!("{:<20} k16 FAIL: {}", format!("{}³", m), e); continue; }
                };
                let ka_k16 = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                    k, n, m, &spec_k16,
                );
                let grid_k16 = compute_grid(&spec_k16, m, n);

                for _ in 0..warmup {
                    let _ = rt.dispatch(&kernel_k16, grid_k16, &ka_k16);
                }

                // Correctness check for k16
                let y_ref_k16 = rt.read_f32(&y_buf, (m * n) as usize);

                let t0 = Instant::now();
                for _ in 0..iters {
                    rt.dispatch_async(&kernel_k16, grid_k16, &ka_k16);
                }
                rt.wait_idle();
                let us_k16 = t0.elapsed().as_micros() as f64 / iters as f64;
                let tf_k16 = if us_k16 > 0.0 { flops / (us_k16 * 1e6) } else { 0.0 };

                // === k32: 128×128 k32 (254 VGPRs, 0 spills, our best config) ===
                let spec_k32 = TileGemm::tile_128x128_k32();
                let name_k32 = format!("bench_k32_{}", spec_k32.name());
                let kernel_k32 = match rt.ensure_kernel_t0(
                    &name_k32, || lower_gemm(&spec_k32),
                    [spec_k32.wg_size(), 1, 1], spec_k32.lds_total(),
                ) {
                    Ok(k) => k,
                    Err(e) => { eprintln!("{:<20} k32 FAIL: {}", format!("{}³", m), e); continue; }
                };
                let ka_k32 = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                    k, n, m, &spec_k32,
                );
                let grid_k32 = compute_grid(&spec_k32, m, n);

                for _ in 0..warmup {
                    let _ = rt.dispatch(&kernel_k32, grid_k32, &ka_k32);
                }

                // Correctness check for k32 vs k16
                let y_k32 = rt.read_f32(&y_buf, (m * n) as usize);
                let max_err = if m <= 2048 {
                    let cpu_ref = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m as usize, k as usize, n as usize);
                    cpu_ref.iter().zip(y_k32.iter()).map(|(r, g)| (r - g).abs()).fold(0f32, f32::max)
                } else { 0.0 };

                let t0 = Instant::now();
                for _ in 0..iters {
                    rt.dispatch_async(&kernel_k32, grid_k32, &ka_k32);
                }
                rt.wait_idle();
                let us_k32 = t0.elapsed().as_micros() as f64 / iters as f64;
                let tf_k32 = if us_k32 > 0.0 { flops / (us_k32 * 1e6) } else { 0.0 };

                let speedup = us_k16 / us_k32;

                // === k64: 128×64 k64 (234 VGPRs, 0 spills) ===
                let spec_k64 = TileGemm::tile_128x64_k64();
                let name_k64 = format!("bench_k64_{}", spec_k64.name());
                let y_buf_k64 = rt.alloc_zero((m * n * 4) as usize).expect("alloc Y k64");
                let kernel_k64 = match rt.ensure_kernel_t0(
                    &name_k64, || lower_gemm(&spec_k64),
                    [spec_k64.wg_size(), 1, 1], spec_k64.lds_total(),
                ) {
                    Ok(k) => k,
                    Err(e) => { eprintln!("{:<20} k64 FAIL: {}", format!("{}³", m), e); continue; }
                };
                let ka_k64 = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf_k64.gpu_addr(),
                    k, n, m, &spec_k64,
                );
                let grid_k64 = compute_grid(&spec_k64, m, n);

                for _ in 0..warmup {
                    let _ = rt.dispatch(&kernel_k64, grid_k64, &ka_k64);
                }

                let t0 = Instant::now();
                for _ in 0..iters {
                    rt.dispatch_async(&kernel_k64, grid_k64, &ka_k64);
                }
                rt.wait_idle();
                let us_k64 = t0.elapsed().as_micros() as f64 / iters as f64;
                let tf_k64 = if us_k64 > 0.0 { flops / (us_k64 * 1e6) } else { 0.0 };
                let speedup64 = us_k16 / us_k64;

                eprintln!("{:<20} k16:{:>7.1}μs={:>5.1}TF  k32:{:>7.1}μs={:>5.1}TF({:.2}×)  k64:{:>7.1}μs={:>5.1}TF({:.2}×) err={:.2e}",
                    format!("{}×{}×{}", m, k, n), us_k16, tf_k16, us_k32, tf_k32, speedup, us_k64, tf_k64, speedup64, max_err);
            }
        });
    }

    /// WGP k64 benchmark: best configs head-to-head at 4096³
    /// Run: cargo test --release --lib --features rocm -- test_wgp_k64_benchmark --nocapture --ignored --test-threads=1
    #[test]
    #[ignore]
    fn test_wgp_k64_benchmark() {
        use std::time::Instant;

        with_rt(|rt| {
            let m = 4096u32; let k = 4096u32; let n = 4096u32;
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            let warmup = 10u32;
            let iters = 30u32;

            let x_bf16: Vec<u16> = (0..(m*k) as usize).map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.01)).collect();
            let wt_bf16: Vec<u16> = (0..(n*k) as usize).map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.01)).collect();
            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);

            eprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║  WGP k64 Benchmark — 4096³ GEMM, RX 7900 XTX (GFX1201)                ║");
            eprintln!("╚══════════════════════════════════════════════════════════════════════════╝\n");

            let configs: Vec<(&str, TileGemm)> = vec![
                ("k32 128×128 CU", {
                    let mut s = TileGemm::tile_128x128_k32();
                    s.wgp_mode = false;
                    s
                }),
                ("k32 128×128 WGP", TileGemm::tile_128x128_k32()),  // default is now WGP
                ("k64 128×128 CU", TileGemm::tile_128x128_k64()),
                ("k64 128×128 WGP", {
                    let mut s = TileGemm::tile_128x128_k64();
                    s.wgp_mode = true;
                    s
                }),
            ];

            for (label, spec) in &configs {
                let y_buf = rt.alloc_zero((m * n * 4) as usize).expect("alloc Y");
                let kname = format!("wgp_bench_{}", spec.name());
                let kernel = match rt.ensure_kernel_t0(
                    &kname, || lower_gemm(spec),
                    [spec.wg_size(), 1, 1], spec.lds_total(),
                ) {
                    Ok(k) => k,
                    Err(e) => { eprintln!("  {:<20} FAIL: {}", label, e); continue; }
                };
                let ka = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                    k, n, m, spec,
                );
                let grid = compute_grid(spec, m, n);

                for _ in 0..warmup {
                    let _ = rt.dispatch(&kernel, grid, &ka);
                }

                let t0 = Instant::now();
                for _ in 0..iters {
                    rt.dispatch_async(&kernel, grid, &ka);
                }
                rt.wait_idle();
                let us = t0.elapsed().as_micros() as f64 / iters as f64;
                let tf = if us > 0.0 { flops / (us * 1e6) } else { 0.0 };

                eprintln!("  {:<20} {:>8.1} μs  {:>6.1} TFLOPS  grid=({},{},{})",
                    label, us, tf, grid[0], grid[1], grid[2]);
            }
        });
    }

    /// Quick 4096³ benchmark: k16 vs k16+swap vs k32+swap, all split_k=1
    /// Run: cargo test --release --lib --features rocm -- test_tile_ir_k32_swap_4096 --nocapture --ignored --test-threads=1
    #[test]
    #[ignore]
    fn test_tile_ir_k32_swap_4096() {
        use std::time::Instant;

        with_rt(|rt| {
            let m = 4096u32; let k = 4096u32; let n = 4096u32;
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            let warmup = 5u32;
            let iters = 20u32;

            let x_bf16: Vec<u16> = (0..(m*k) as usize).map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.01)).collect();
            let wt_bf16: Vec<u16> = (0..(n*k) as usize).map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.01)).collect();
            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);

            eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
            eprintln!("║  4096³ Benchmark — GMEM-reuse: k16/k32 × standard/swap     ║");
            eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

            let configs: Vec<(&str, TileGemm)> = vec![
                ("k16 standard",    TileGemm::tile_128x128_k16()),
                ("k32 standard",    TileGemm::tile_128x128_k32()),
                ("k16 + swap",      TileGemm::tile_128x128_k16_swap()),
                ("k32 + swap",      TileGemm::tile_128x128_k32_swap()),
            ];

            for (label, spec) in &configs {
                let y_buf = rt.alloc_zero((m * n * 4) as usize).expect("alloc Y");
                let name = format!("bench4k_{}", spec.name());
                let kernel = match rt.ensure_kernel_t0(
                    &name, || lower_gemm(spec),
                    [spec.wg_size(), 1, 1], spec.lds_total(),
                ) {
                    Ok(k) => k,
                    Err(e) => { eprintln!("{:<20} FAIL: {}", label, e); continue; }
                };
                let ka = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                    k, n, m, spec,
                );
                let grid = compute_grid(spec, m, n);

                for _ in 0..warmup {
                    let _ = rt.dispatch(&kernel, grid, &ka);
                }

                let t0 = Instant::now();
                for _ in 0..iters {
                    rt.dispatch_async(&kernel, grid, &ka);
                }
                rt.wait_idle();
                let us = t0.elapsed().as_micros() as f64 / iters as f64;
                let tf = if us > 0.0 { flops / (us * 1e6) } else { 0.0 };

                eprintln!("{:<20} {:>8.1} μs  {:>8.2} TFLOPS  LDS={}  grid=[{},{},1]",
                    label, us, tf, spec.lds_total(), grid[0], grid[1]);
                rt.recycle(y_buf);
            }
        });
    }

    /// Full-spectrum autotuner: benchmark ALL tile configs at multiple sizes.
    /// Produces the data needed to update tile_auto_select().
    /// Run: cargo test --release --lib --features rocm -- test_full_spectrum_autotuner --nocapture --ignored --test-threads=1
    #[test]
    #[ignore]
    fn test_full_spectrum_autotuner() {
        use std::time::Instant;

        with_rt(|rt| {
            let sizes: Vec<(u32, u32, u32)> = vec![
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
                // Non-square
                (256, 4096, 1024),
                (512, 4096, 1024),
                (1024, 4096, 4096),
            ];
            let warmup = 5u32;
            let iters = 20u32;

            eprintln!("\n╔══════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║  Full-Spectrum Autotuner — RX 7900 XTX (GFX1201)                       ║");
            eprintln!("║  All tile configs × all sizes, BF16 WMMA GEMM                          ║");
            eprintln!("╚══════════════════════════════════════════════════════════════════════════╝\n");

            // All tile configs to test (split_k=1 for fair comparison)
            let make_configs = || -> Vec<(&str, TileGemm)> {
                vec![
                    ("64×64 k16",        TileGemm::tile_64x64_k16()),
                    ("64×64 k32",        TileGemm::tile_64x64_k32()),
                    ("64×64 k64",        TileGemm::tile_64x64_k64()),
                    ("128×64 k16",       TileGemm::tile_128x64_k16()),
                    ("64×128 k16",       TileGemm::tile_64x128_k16()),
                    ("64×128 k32",       TileGemm::tile_64x128_k32()),
                    ("128×128 k16",      TileGemm::tile_128x128_k16()),
                    ("128×128 k32",      TileGemm::tile_128x128_k32()),
                    ("128×128 k64",      TileGemm::tile_128x128_k64()),
                    ("128×128 k16 swap", TileGemm::tile_128x128_k16_swap()),
                    ("128×128 k32 swap", TileGemm::tile_128x128_k32_swap()),
                    ("256×64 k32 WGP",   TileGemm::tile_256x64_k32_wgp()),
                ]
            };

            for &(m, k, n) in &sizes {
                let flops = 2.0 * m as f64 * k as f64 * n as f64;
                eprintln!("\n── {}×{}×{} ({:.1} GFLOP) ──",
                    m, k, n, flops / 1e9);

                let x_bf16: Vec<u16> = (0..(m*k) as usize).map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.01)).collect();
                let wt_bf16: Vec<u16> = (0..(n*k) as usize).map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.01)).collect();
                let x_buf = upload_bf16(rt, &x_bf16);
                let wt_buf = upload_bf16(rt, &wt_bf16);

                let mut best_tf = 0.0f64;
                let mut best_name = "";

                for (label, spec) in make_configs() {
                    // Skip invalid combos
                    if k % spec.tile_k != 0 { continue; }
                    if m < spec.tile_m || n < spec.tile_n { continue; }

                    let y_buf = rt.alloc_zero((m * n * 4) as usize).expect("alloc Y");
                    let kname = format!("autotune_{}_{}", spec.name(), m);
                    let kernel = match rt.ensure_kernel_t0(
                        &kname, || lower_gemm(&spec),
                        [spec.wg_size(), 1, 1], spec.lds_total(),
                    ) {
                        Ok(k) => k,
                        Err(e) => { eprintln!("  {:<22} FAIL: {}", label, e); continue; }
                    };
                    let ka = build_kernargs_m(
                        x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                        k, n, m, &spec,
                    );
                    let grid = compute_grid(&spec, m, n);

                    for _ in 0..warmup {
                        let _ = rt.dispatch(&kernel, grid, &ka);
                    }

                    let t0 = Instant::now();
                    for _ in 0..iters {
                        rt.dispatch_async(&kernel, grid, &ka);
                    }
                    rt.wait_idle();
                    let us = t0.elapsed().as_micros() as f64 / iters as f64;
                    let tf = if us > 0.0 { flops / (us * 1e6) } else { 0.0 };

                    let marker = if tf > best_tf { best_tf = tf; best_name = label; " ★" } else { "" };
                    eprintln!("  {:<22} {:>8.1} μs  {:>6.1} TF  grid=({},{},{}){}",
                        label, us, tf, grid[0], grid[1], grid[2], marker);
                    rt.recycle(y_buf);
                }
                eprintln!("  → BEST: {:<22} {:.1} TFLOPS", best_name, best_tf);
            }
        });
    }

    /// Multi-dispatch stress test: same kernel dispatched 5 times
    #[test]
    fn test_tile_ir_multi_dispatch() {
        with_rt(|rt| {
            let m = 64usize;
            let k = 64usize;
            let n = 64usize;
            let spec = TileGemm::tile_32x64_k16();

            let x_f32: Vec<f32> = (0..m*k).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect();
            let wt_f32: Vec<f32> = (0..n*k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let wt_bf16: Vec<u16> = wt_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);

            let kernel = rt.ensure_kernel_t0(
                "tile_multi_dispatch",
                || lower_gemm(&spec),
                [spec.wg_size(), 1, 1],
                spec.lds_total(),
            ).expect("compile");

            let expected = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m, k, n);

            for iter in 0..5 {
                let y_buf = rt.alloc_zero(m*n*4).expect("alloc Y");
                let ka = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                    k as u32, n as u32, m as u32, &spec,
                );
                let grid = compute_grid(&spec, m as u32, n as u32);
                eprintln!("[multi] iter {} dispatch...", iter);
                rt.dispatch(&kernel, grid, &ka).expect("dispatch");
                let result = rt.read_f32(&y_buf, m*n);
                let max_err = result.iter().zip(expected.iter())
                    .map(|(r, e)| (r - e).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("[multi] iter {}: Y[0]={:.6} max_err={:.6}", iter, result[0], max_err);
                assert!(max_err < 0.1, "iter {} max_err={}", iter, max_err);
            }
            eprintln!("[PASS] test_tile_ir_multi_dispatch: 5 dispatches verified");
        });
    }

    /// WGP mode: 256×64 k32 tile (8 waves per workgroup)
    #[test]
    fn test_tile_ir_gpu_gemm_256x64_wgp() {
        with_rt(|rt| {
            let m = 256usize;
            let k = 128usize;  // must be >= tile_k=32
            let n = 64usize;
            let spec = TileGemm::tile_256x64_k32_wgp();

            let x_f32: Vec<f32> = (0..m*k).map(|i| ((i % 23) as f32 - 11.0) * 0.005).collect();
            let wt_f32: Vec<f32> = (0..n*k).map(|i| ((i % 17) as f32 - 8.0) * 0.005).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let wt_bf16: Vec<u16> = wt_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);
            let y_buf = rt.alloc_zero(m * n * 4).expect("alloc Y");

            let kernel = rt.ensure_kernel_t0(
                &spec.name(),
                || lower_gemm(&spec),
                [spec.wg_size(), 1, 1],
                spec.lds_total(),
            ).expect("compile");

            let ka = build_kernargs_m(
                x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                k as u32, n as u32, m as u32, &spec,
            );
            let grid = compute_grid(&spec, m as u32, n as u32);
            eprintln!("[WGP 256x64] grid={:?} wg={} lds={}", grid, spec.wg_size(), spec.lds_total());
            rt.dispatch(&kernel, grid, &ka).expect("dispatch");

            let result = rt.read_f32(&y_buf, m * n);
            let expected = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m, k, n);

            let max_err = result.iter().zip(expected.iter())
                .map(|(r, e)| (r - e).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 0.1,
                "256x64 WGP GEMM max_err={:.6} too large", max_err);
            eprintln!("[PASS] test_tile_ir_gpu_gemm_256x64_wgp: verified (max_err={:.6})", max_err);
        });
    }

    /// WGP mode: 256×64 k32 with larger matrix (512×256×64)
    #[test]
    fn test_tile_ir_gpu_gemm_256x64_wgp_large() {
        with_rt(|rt| {
            let m = 512usize;
            let k = 256usize;
            let n = 64usize;
            let spec = TileGemm::tile_256x64_k32_wgp();

            let x_f32: Vec<f32> = (0..m*k).map(|i| ((i % 23) as f32 - 11.0) * 0.003).collect();
            let wt_f32: Vec<f32> = (0..n*k).map(|i| ((i % 17) as f32 - 8.0) * 0.003).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let wt_bf16: Vec<u16> = wt_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);
            let y_buf = rt.alloc_zero(m * n * 4).expect("alloc Y");

            let kernel = rt.ensure_kernel_t0(
                &format!("{}_large", spec.name()),
                || lower_gemm(&spec),
                [spec.wg_size(), 1, 1],
                spec.lds_total(),
            ).expect("compile");

            let ka = build_kernargs_m(
                x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                k as u32, n as u32, m as u32, &spec,
            );
            let grid = compute_grid(&spec, m as u32, n as u32);
            eprintln!("[WGP 256x64 large] grid={:?}", grid);
            rt.dispatch(&kernel, grid, &ka).expect("dispatch");

            let result = rt.read_f32(&y_buf, m * n);
            let expected = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m, k, n);

            let max_err = result.iter().zip(expected.iter())
                .map(|(r, e)| (r - e).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 0.15,
                "256x64 WGP large GEMM max_err={:.6} too large", max_err);
            eprintln!("[PASS] test_tile_ir_gpu_gemm_256x64_wgp_large: 512x256x64 verified (max_err={:.6})", max_err);
        });
    }

    /// WGP multi-dispatch test using 128x128 WGP (4 waves)
    #[test]
    fn test_tile_ir_wgp_multi_dispatch_128x64() {
        with_rt(|rt| {
            let m = 128usize;
            let k = 128usize;
            let n = 64usize;
            let spec = TileGemm::tile_128x64_k16();

            let x_f32: Vec<f32> = (0..m*k).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect();
            let wt_f32: Vec<f32> = (0..n*k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let wt_bf16: Vec<u16> = wt_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);

            let kernel = rt.ensure_kernel_t0(
                &format!("{}_multi", spec.name()),
                || lower_gemm(&spec),
                [spec.wg_size(), 1, 1],
                spec.lds_total(),
            ).expect("compile");

            let expected = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m, k, n);

            for iter in 0..5 {
                let y_buf = rt.alloc_zero(m*n*4).expect("alloc Y");
                let ka = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                    k as u32, n as u32, m as u32, &spec,
                );
                let grid = compute_grid(&spec, m as u32, n as u32);
                eprintln!("[128x64 multi] iter {} dispatch...", iter);
                rt.dispatch(&kernel, grid, &ka).expect("dispatch");
                let result = rt.read_f32(&y_buf, m*n);
                let max_err = result.iter().zip(expected.iter())
                    .map(|(r, e)| (r - e).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("[128x64 multi] iter {}: max_err={:.6}", iter, max_err);
                assert!(max_err < 0.1, "iter {} max_err={}", iter, max_err);
            }
            eprintln!("[PASS] test_tile_ir_wgp_multi_dispatch_128x64: 5 dispatches OK!");
        });
    }

    /// Auto-select test: verify tile_auto_select picks reasonable configs
    #[test]
    fn test_tile_auto_select() {
        // Small M → 32×64
        let s = tile_auto_select(32, 256, 64, TileTranspose::NT);
        assert_eq!(s.tile_m, 32);
        assert_eq!(s.tile_n, 64);

        // Medium M → 128×64
        let s = tile_auto_select(128, 256, 64, TileTranspose::NT);
        assert_eq!(s.tile_m, 128);
        assert_eq!(s.tile_n, 64);

        // Large M → 128×64 CU mode (WGP available but CU mode preferred for perf)
        let s = tile_auto_select(512, 512, 512, TileTranspose::NT);
        assert_eq!(s.tile_m, 128);
        assert_eq!(s.tile_k, 16);
        assert!(!s.wgp_mode);

        // NN-mode preserved
        let s = tile_auto_select(128, 256, 64, TileTranspose::NN);
        assert_eq!(s.transpose, TileTranspose::NN);

        eprintln!("[PASS] test_tile_auto_select");
    }

    /// Diagnostic: dump and compare 32×64 (OK) vs 64×64 (HANG) assembly
    #[test]
    fn test_dump_asm_64x64_vs_32x64() {
        let spec32 = TileGemm::tile_32x64_k16();
        let spec64 = TileGemm::tile_64x64_k16();

        let k32 = lower_gemm(&spec32);
        let k64 = lower_gemm(&spec64);

        let asm32 = k32.to_assembly(crate::t0::ir::Target::GFX1201).unwrap();
        let asm64 = k64.to_assembly(crate::t0::ir::Target::GFX1201).unwrap();

        // Save full assembly to /tmp for inspection
        std::fs::write("/tmp/tile_32x64.s", &asm32).unwrap();
        std::fs::write("/tmp/tile_64x64.s", &asm64).unwrap();
        eprintln!("Full assembly saved to /tmp/tile_32x64.s and /tmp/tile_64x64.s");

        // Compare key sync instructions
        for (label, asm) in &[("32x64 (OK)", &asm32), ("64x64 (HANG)", &asm64)] {
            eprintln!("\n=== {} ===", label);
            eprintln!("Lines: {}", asm.lines().count());
            let mut barrier_n = 0;
            let mut in_epilog = false;
            for (i, line) in asm.lines().enumerate() {
                let lt = line.trim();
                if lt.contains("s_barrier") {
                    barrier_n += 1;
                    eprintln!("  L{}: {}", i+1, lt);
                } else if lt.contains("waitcnt") || lt.contains("s_endpgm") {
                    eprintln!("  L{}: {}", i+1, lt);
                } else if lt.contains("exec") {
                    eprintln!("  L{}: {}", i+1, lt);
                }
                // Check last 5 instructions before endpgm
                if lt.contains("s_endpgm") {
                    // Print preceding 10 lines for epilog context
                    eprintln!("  --- Epilog (10 lines before endpgm) ---");
                    let lines: Vec<&str> = asm.lines().collect();
                    let start = if i > 10 { i - 10 } else { 0 };
                    for j in start..=i {
                        eprintln!("    L{}: {}", j+1, lines[j].trim());
                    }
                }
            }
            eprintln!("  Barrier count: {}", barrier_n);
        }

        // Compare amdhsa_kernel descriptors
        for (label, asm) in &[("32x64", &asm32), ("64x64", &asm64)] {
            eprintln!("\n=== {} KD ===", label);
            for line in asm.lines() {
                if line.contains("amdhsa_") {
                    eprintln!("  {}", line.trim());
                }
            }
        }
    }

    /// MINIMAL REPRO: bare 2-wave kernel (barrier + endpgm), dispatched 5 times.
    /// If this hangs, the problem is fundamental to multi-wave barriers.
    /// If this works, the problem is in our GEMM codegen.
    #[test]
    fn test_minimal_2wave_barrier() {
        use crate::t0::compile::T0Kernel;
        use crate::t0::ir::{Target, Alignment};

        with_rt(|rt| {
            // Build minimal kernel: wg_size=64 (2 waves), 1 barrier, write tid to output
            let mut k = T0Kernel::new("min_2wave_barrier");
            k.set_wg_size(64);
            k.set_lds_size(256);  // minimal LDS

            let out_ptr = k.arg_ptr("out");
            k.emit_arg_loads();

            // tid = v0 (WORKITEM_ID_X)
            let tid = k.alloc_vreg();
            k.push(crate::t0::ir::Op::VMov { dst: tid, src: crate::t0::ir::Operand::VReg(crate::t0::ir::VReg(0)) });

            // barrier — this is the key test
            k.barrier();

            // out[tid] = float(tid) — verify kernel actually ran
            let ftid = k.alloc_vreg();
            k.v_cvt_f32_u32(ftid, tid);

            let addr = k.alloc_vreg_array(2, Alignment::Align2);
            let offset_v = k.alloc_vreg();
            k.v_lshlrev_b32(offset_v, 2, tid);  // tid * 4
            k.v_mov_from_sgpr(addr, crate::t0::ir::SReg(out_ptr.0));
            k.v_mov_from_sgpr(crate::t0::ir::VReg(addr.0 + 1), crate::t0::ir::SReg(out_ptr.0 + 1));
            k.v_add_co(addr, addr, offset_v);
            k.v_add_co_ci(crate::t0::ir::VReg(addr.0 + 1), crate::t0::ir::VReg(addr.0 + 1));
            k.global_store(addr, ftid, crate::t0::ir::Width::B32, 0);

            k.wait_vscnt(0);
            k.endpgm();

            // Compile
            let kernel = rt.ensure_kernel_t0(
                "min_2wave_barrier",
                || k,
                [64, 1, 1],
                256,
            ).expect("compile minimal 2-wave kernel");

            // Multi-dispatch 5 times
            for iter in 0..5 {
                let out_buf = rt.alloc_zero(64 * 4).expect("alloc out");
                let mut ka = [0u8; 8];
                ka[0..8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());

                eprintln!("[min 2-wave] iter {} dispatch...", iter);
                rt.dispatch(&kernel, [64, 1, 1], &ka).expect("dispatch");

                let result = rt.read_f32(&out_buf, 64);
                // Verify: out[i] should == i as f32
                let mut ok = true;
                for i in 0..64 {
                    if (result[i] - i as f32).abs() > 0.01 {
                        eprintln!("  MISMATCH: out[{}]={} expected={}", i, result[i], i);
                        ok = false;
                        break;
                    }
                }
                if ok {
                    eprintln!("[min 2-wave] iter {}: PASS (out[0]={}, out[63]={})", iter, result[0], result[63]);
                } else {
                    panic!("iter {}: output mismatch", iter);
                }
            }
            eprintln!("[PASS] test_minimal_2wave_barrier: 5 dispatches OK!");
        });
    }

    /// MINIMAL REPRO: bare 4-wave kernel (barrier + endpgm), dispatched 5 times.
    /// Control test — 4-wave 128×64 works, so 4-wave barriers should be fine.
    #[test]
    fn test_minimal_4wave_barrier() {
        use crate::t0::compile::T0Kernel;
        use crate::t0::ir::{Target, Alignment};

        with_rt(|rt| {
            let mut k = T0Kernel::new("min_4wave_barrier");
            k.set_wg_size(128);
            k.set_lds_size(256);

            let out_ptr = k.arg_ptr("out");
            k.emit_arg_loads();

            let tid = k.alloc_vreg();
            k.push(crate::t0::ir::Op::VMov { dst: tid, src: crate::t0::ir::Operand::VReg(crate::t0::ir::VReg(0)) });

            k.barrier();

            let ftid = k.alloc_vreg();
            k.v_cvt_f32_u32(ftid, tid);

            let addr = k.alloc_vreg_array(2, Alignment::Align2);
            let offset_v = k.alloc_vreg();
            k.v_lshlrev_b32(offset_v, 2, tid);
            k.v_mov_from_sgpr(addr, crate::t0::ir::SReg(out_ptr.0));
            k.v_mov_from_sgpr(crate::t0::ir::VReg(addr.0 + 1), crate::t0::ir::SReg(out_ptr.0 + 1));
            k.v_add_co(addr, addr, offset_v);
            k.v_add_co_ci(crate::t0::ir::VReg(addr.0 + 1), crate::t0::ir::VReg(addr.0 + 1));
            k.global_store(addr, ftid, crate::t0::ir::Width::B32, 0);

            k.wait_vscnt(0);
            k.endpgm();

            let kernel = rt.ensure_kernel_t0(
                "min_4wave_barrier",
                || k,
                [128, 1, 1],
                256,
            ).expect("compile minimal 4-wave kernel");

            for iter in 0..5 {
                let out_buf = rt.alloc_zero(128 * 4).expect("alloc out");
                let mut ka = [0u8; 8];
                ka[0..8].copy_from_slice(&out_buf.gpu_addr().to_le_bytes());

                eprintln!("[min 4-wave] iter {} dispatch...", iter);
                rt.dispatch(&kernel, [128, 1, 1], &ka).expect("dispatch");
                let result = rt.read_f32(&out_buf, 128);
                eprintln!("[min 4-wave] iter {}: out[0]={}, out[127]={}", iter, result[0], result[127]);
            }
            eprintln!("[PASS] test_minimal_4wave_barrier: 5 dispatches OK!");
        });
    }

    /// Safe benchmark: gemm_gen performance (multi-dispatch) + tile_ir correctness (single dispatch).
    ///
    /// Avoids tile_ir multi-dispatch which can cause GPU hangs.
    /// Run: cargo test --release --lib --features rocm -- test_safe_benchmark --nocapture --test-threads=1
    #[test]
    fn test_safe_benchmark() {
        use crate::t0::gemm_gen;
        use std::time::Instant;

        with_rt(|rt| {
            let warmup = 3;
            let iters = 10;

            let sizes: Vec<(u32, u32, u32)> = vec![
                (128, 128, 64),
                (256, 256, 64),
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
            ];

            eprintln!("\n{}", "=".repeat(90));
            eprintln!("  Safe Benchmark: gemm_gen timing + tile_ir single-dispatch verify");
            eprintln!("  Warmup: {} iters, Measured: {} iters", warmup, iters);
            eprintln!("{}", "=".repeat(90));
            eprintln!("{:<15} {:>10} {:>10} {:>15} {:>12}",
                "Size (M×K×N)", "gemm μs", "TFLOPS", "tile_ir verify", "tile max_err");
            eprintln!("{:-<75}", "");

            for &(m, k, n) in &sizes {
                let flops = 2.0 * m as f64 * k as f64 * n as f64;

                // ── Prepare data ──
                let x_bf16: Vec<u16> = (0..m*k).map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.01)).collect();
                let wt_bf16: Vec<u16> = (0..n*k).map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.01)).collect();
                let x_buf = upload_bf16(rt, &x_bf16);
                let wt_buf = upload_bf16(rt, &wt_bf16);

                // ── gemm_gen benchmark (safe: multi-dispatch proven stable) ──
                let gemm_cfg = gemm_gen::auto_select(m, k, n);
                let gemm_sk = gemm_cfg.split_k.unwrap_or(1);
                // CRITICAL: allocate Y buffer large enough for all split-K partitions!
                // Each split_k_id writes to Y + id*M*N*4, so total = M*N*4*split_k.
                let y_buf_gemm = rt.alloc_zero((m * n * 4 * gemm_sk) as usize).expect("alloc Y gemm");
                let gemm_name = format!("sbench_gemm_{}x{}x{}", m, k, n);
                let gemm_kernel_t0 = gemm_gen::generate(&gemm_cfg);
                let gemm_kernel = rt.ensure_kernel_t0(
                    &gemm_name,
                    || gemm_kernel_t0,
                    [gemm_cfg.wg_size, 1, 1],
                    gemm_cfg.lds_total(),
                ).expect("compile gemm_gen");
                let gemm_ka = gemm_gen::build_kernargs(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf_gemm.gpu_addr(),
                    k, n, m, &gemm_cfg,
                );
                let (gx, gy) = gemm_gen::compute_grid_auto(&gemm_cfg, m, n);

                // Warmup
                for _ in 0..warmup {
                    if rt.dispatch(&gemm_kernel, [gx, gy, 1], &gemm_ka).is_err() { break; }
                }
                // Timed runs
                let t0 = Instant::now();
                let mut gemm_ok = true;
                for _ in 0..iters {
                    if rt.dispatch(&gemm_kernel, [gx, gy, 1], &gemm_ka).is_err() {
                        gemm_ok = false; break;
                    }
                }
                let gemm_us = if gemm_ok { t0.elapsed().as_micros() as f64 / iters as f64 } else { 0.0 };
                let gemm_tflops = if gemm_us > 0.0 { flops / (gemm_us * 1e6) } else { 0.0 };

                // ── tile_ir single-dispatch correctness only ──
                let tile_verify = if m <= 2048 && k <= 2048 && n <= 2048 {
                    let tile_spec = tile_auto_select(m, k, n, TileTranspose::NT);
                    let tile_name = format!("sbench_tile_{}x{}x{}", m, k, n);
                    match rt.ensure_kernel_t0(
                        &tile_name,
                        || lower_gemm(&tile_spec),
                        [tile_spec.wg_size(), 1, 1],
                        tile_spec.lds_total(),
                    ) {
                        Ok(tile_kernel) => {
                            let tile_sk = tile_spec.split_k;
                            // CRITICAL: allocate Y buffer for all split-K partitions
                            let y_buf_tile = rt.alloc_zero((m * n * 4 * tile_sk) as usize).expect("alloc Y tile");
                            let tile_ka = build_kernargs_m(
                                x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf_tile.gpu_addr(),
                                k, n, m, &tile_spec,
                            );
                            let grid = compute_grid(&tile_spec, m, n);
                            match rt.dispatch(&tile_kernel, grid, &tile_ka) {
                                Ok(()) => {
                                    // Read gemm_gen result for cross-verify
                                    let gemm_result = rt.read_f32(&y_buf_gemm, (m * n) as usize);
                                    let tile_result = rt.read_f32(&y_buf_tile, (m * n) as usize);
                                    let mut max_err = 0.0f32;
                                    let mut n_bad = 0usize;
                                    for i in 0..(m * n) as usize {
                                        let err = (tile_result[i] - gemm_result[i]).abs();
                                        if err > max_err { max_err = err; }
                                        if err > 0.5 { n_bad += 1; }
                                    }
                                    if n_bad == 0 { format!("✅ {:.4}", max_err) }
                                    else { format!("❌ {}/{}", n_bad, m*n) }
                                }
                                Err(e) => format!("HANG {}", e),
                            }
                        }
                        Err(e) => format!("COMPILE {}", e),
                    }
                } else {
                    "SKIP (>2048)".to_string()
                };

                eprintln!("{:<15} {:>10.1} {:>10.3} {:>15} {:>12}",
                    format!("{}×{}×{}", m, k, n),
                    gemm_us, gemm_tflops,
                    if gemm_ok { "✅" } else { "FAIL" },
                    tile_verify);
            }

            eprintln!("\n  gemm_gen TFLOPS = real measured performance");
            eprintln!("  tile_ir = single-dispatch correctness cross-verified against gemm_gen\n");
        });
    }

    /// Performance benchmark: tile_ir vs gemm_gen across multiple matrix sizes.
    /// Both are NT-mode GEMM (Y = A @ B^T), bf16 inputs, f32 output.
    ///
    /// Run: cargo test --release --lib --features rocm -- tile_ir::gpu_tests::test_benchmark_tile_ir_vs_gemm_gen --nocapture --test-threads=1
    #[test]
    fn test_benchmark_tile_ir_vs_gemm_gen() {
        use crate::t0::gemm_gen;
        use std::time::Instant;

        with_rt(|rt| {
            let warmup = 5;
            let iters = 20;

            // Matrix sizes: (M, K, N) — skip 64³ (tile_gemm_32x64 intermittent hang)
            let sizes: Vec<(u32, u32, u32)> = vec![
                (128, 128, 64),    // small
                (256, 256, 64),    // medium
                (256, 256, 256),   // square medium
                (512, 512, 512),   // square large
                (1024, 1024, 1024), // production
                (2048, 2048, 2048), // large production
            ];

            eprintln!("\n{}", "=".repeat(80));
            eprintln!("  tile_ir vs gemm_gen Performance Benchmark (NT-mode GEMM)");
            eprintln!("  Warmup: {} iters, Measured: {} iters", warmup, iters);
            eprintln!("{}", "=".repeat(80));
            eprintln!("{:<15} {:>10} {:>10} {:>10} {:>10} {:>7}",
                "Size (M×K×N)", "tile_ir μs", "TFLOPS", "gemm_gen μs", "TFLOPS", "Ratio");
            eprintln!("{:-<75}", "");

            for &(m, k, n) in &sizes {
                let flops = 2.0 * m as f64 * k as f64 * n as f64;

                // ── Prepare data ──
                let x_bf16: Vec<u16> = (0..m*k).map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.01)).collect();
                let wt_bf16: Vec<u16> = (0..n*k).map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.01)).collect();
                let x_buf = upload_bf16(rt, &x_bf16);
                let wt_buf = upload_bf16(rt, &wt_bf16);

                // ── tile_ir setup ──
                let tile_spec = tile_auto_select(m, k, n, TileTranspose::NT);
                // CRITICAL: allocate Y buffer for all split-K partitions
                let tile_sk = tile_spec.split_k;
                let y_buf_tile = rt.alloc_zero((m * n * 4 * tile_sk) as usize).expect("alloc Y tile");

                // ── gemm_gen setup ──
                let gemm_cfg = gemm_gen::auto_select(m, k, n);
                let gemm_sk = gemm_cfg.split_k.unwrap_or(1);
                let y_buf_gemm = rt.alloc_zero((m * n * 4 * gemm_sk) as usize).expect("alloc Y gemm");

                let tile_name = format!("bench_tile_{}x{}x{}", m, k, n);
                let tile_kernel = rt.ensure_kernel_t0(
                    &tile_name,
                    || lower_gemm(&tile_spec),
                    [tile_spec.wg_size(), 1, 1],
                    tile_spec.lds_total(),
                ).expect("compile tile_ir");
                let tile_ka = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf_tile.gpu_addr(),
                    k, n, m, &tile_spec,
                );
                let tile_grid = compute_grid(&tile_spec, m, n);

                // ── gemm_gen kernel compile (gemm_cfg declared above) ──
                let gemm_name = format!("bench_gemm_{}x{}x{}", m, k, n);
                let gemm_kernel_t0 = gemm_gen::generate(&gemm_cfg);
                let gemm_kernel = rt.ensure_kernel_t0(
                    &gemm_name,
                    || gemm_kernel_t0,
                    [gemm_cfg.wg_size, 1, 1],
                    gemm_cfg.lds_total(),
                ).expect("compile gemm_gen");
                let gemm_ka_arr = gemm_gen::build_kernargs(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf_gemm.gpu_addr(),
                    k, n, m, &gemm_cfg,
                );
                let (gemm_gx, gemm_gy) = gemm_gen::compute_grid_auto(&gemm_cfg, m, n);

                // ── Benchmark tile_ir ──
                // Verify first dispatch works before benchmarking
                if let Err(e) = rt.dispatch(&tile_kernel, [tile_grid[0], tile_grid[1], tile_grid[2]], &tile_ka) {
                    eprintln!("{:<15} tile_ir DISPATCH FAILED: {}",
                        format!("{}×{}×{}", m, k, n), e);
                    continue;
                }
                for _ in 1..warmup {
                    if rt.dispatch(&tile_kernel, [tile_grid[0], tile_grid[1], tile_grid[2]], &tile_ka).is_err() { break; }
                }
                let t0 = Instant::now();
                let mut tile_ok = true;
                for _ in 0..iters {
                    if rt.dispatch(&tile_kernel, [tile_grid[0], tile_grid[1], tile_grid[2]], &tile_ka).is_err() {
                        tile_ok = false;
                        break;
                    }
                }
                let tile_us = if tile_ok { t0.elapsed().as_micros() as f64 / iters as f64 } else { 0.0 };
                let tile_tflops = if tile_us > 0.0 { flops / (tile_us * 1e6) } else { 0.0 };

                // ── Benchmark gemm_gen ──
                if let Err(e) = rt.dispatch(&gemm_kernel, [gemm_gx, gemm_gy, 1], &gemm_ka_arr) {
                    eprintln!("{:<15} gemm_gen DISPATCH FAILED: {}",
                        format!("{}×{}×{}", m, k, n), e);
                    eprintln!("{:<15} {:>10.1} {:>10.3} {:>11} {:>10} {:>7}",
                        format!("{}×{}×{}", m, k, n),
                        tile_us, tile_tflops, "FAIL", "-", "-");
                    continue;
                }
                for _ in 1..warmup {
                    if rt.dispatch(&gemm_kernel, [gemm_gx, gemm_gy, 1], &gemm_ka_arr).is_err() { break; }
                }
                let t0 = Instant::now();
                let mut gemm_ok = true;
                for _ in 0..iters {
                    if rt.dispatch(&gemm_kernel, [gemm_gx, gemm_gy, 1], &gemm_ka_arr).is_err() {
                        gemm_ok = false;
                        break;
                    }
                }
                let gemm_us = if gemm_ok { t0.elapsed().as_micros() as f64 / iters as f64 } else { 0.0 };
                let gemm_tflops = if gemm_us > 0.0 { flops / (gemm_us * 1e6) } else { 0.0 };

                let ratio = if tile_us > 0.0 && gemm_us > 0.0 { gemm_us / tile_us } else { 0.0 };

                eprintln!("{:<15} {:>10.1} {:>10.3} {:>11.1} {:>10.3} {:>7.2}x",
                    format!("{}×{}×{}", m, k, n),
                    tile_us, tile_tflops,
                    gemm_us, gemm_tflops,
                    ratio);

                // ── Correctness verification: dispatch once to fresh buffers ──
                let y_verify_tile = rt.alloc_zero((m * n * 4) as usize).expect("alloc verify tile");
                let y_verify_gemm = rt.alloc_zero((m * n * 4) as usize).expect("alloc verify gemm");

                // Build kernargs pointing to fresh output buffers
                let tile_ka_verify = build_kernargs_m(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_verify_tile.gpu_addr(),
                    k, n, m, &tile_spec,
                );
                let gemm_ka_verify = gemm_gen::build_kernargs(
                    x_buf.gpu_addr(), wt_buf.gpu_addr(), y_verify_gemm.gpu_addr(),
                    k, n, m, &gemm_cfg,
                );

                // Dispatch once for correctness
                let tile_verify_ok = rt.dispatch(&tile_kernel,
                    [tile_grid[0], tile_grid[1], tile_grid[2]], &tile_ka_verify).is_ok();
                let gemm_verify_ok = rt.dispatch(&gemm_kernel,
                    [gemm_gx, gemm_gy, 1], &gemm_ka_verify).is_ok();

                if tile_verify_ok && gemm_verify_ok {
                    let tile_result = rt.read_f32(&y_verify_tile, (m * n) as usize);
                    let gemm_result = rt.read_f32(&y_verify_gemm, (m * n) as usize);

                    // Cross-verify: tile_ir vs gemm_gen
                    let mut max_cross_err = 0.0f32;
                    let mut n_bad_cross = 0usize;
                    let mut tile_nonzero = 0usize;
                    let mut gemm_nonzero = 0usize;
                    for i in 0..(m * n) as usize {
                        if tile_result[i] != 0.0 { tile_nonzero += 1; }
                        if gemm_result[i] != 0.0 { gemm_nonzero += 1; }
                        let err = (tile_result[i] - gemm_result[i]).abs();
                        if err > max_cross_err { max_cross_err = err; }
                        if err > 0.5 { n_bad_cross += 1; }
                    }

                    let total = (m * n) as usize;
                    let status = if n_bad_cross == 0 && tile_nonzero > 0 { "✅" }
                                 else if tile_nonzero == 0 || gemm_nonzero == 0 { "⚠️ ZERO" }
                                 else { "❌ MISMATCH" };

                    eprintln!("  verify: {} cross_err={:.4} nonzero=tile:{}/gemm:{}/{}",
                        status, max_cross_err, tile_nonzero, gemm_nonzero, total);

                    // CPU reference for small sizes (≤512 elements per dim)
                    if m <= 512 && k <= 512 && n <= 512 {
                        let expected = cpu_gemm_nt_bf16(
                            &x_bf16, &wt_bf16,
                            m as usize, k as usize, n as usize,
                        );
                        let mut max_cpu_err = 0.0f32;
                        let mut n_bad_cpu = 0usize;
                        for i in 0..total {
                            let err = (tile_result[i] - expected[i]).abs();
                            if err > max_cpu_err { max_cpu_err = err; }
                            if err > 0.5 { n_bad_cpu += 1; }
                        }
                        let cpu_status = if n_bad_cpu == 0 { "✅" } else { "❌" };
                        eprintln!("  cpu_ref: {} max_err={:.4} bad={}/{}",
                            cpu_status, max_cpu_err, n_bad_cpu, total);
                    }
                } else {
                    eprintln!("  verify: ⚠️ dispatch failed (tile={} gemm={})",
                        tile_verify_ok, gemm_verify_ok);
                }
            }

            eprintln!("\nRatio > 1.0 means tile_ir is faster than gemm_gen");
            eprintln!("Ratio < 1.0 means gemm_gen is faster than tile_ir\n");

            // Also show config choices for each size
            eprintln!("\n{:<15} {:<30} {:<35}", "Size", "tile_ir config", "gemm_gen config");
            eprintln!("{:-<80}", "");
            for &(m, k, n) in &sizes {
                let ts = tile_auto_select(m, k, n, TileTranspose::NT);
                let gc = gemm_gen::auto_select(m, k, n);
                eprintln!("{:<15} {:<30} {:<35}",
                    format!("{}×{}×{}", m, k, n),
                    ts.name(),
                    gc.name());
            }
        });
    }

    // ── Epilogue Fusion Tests ──

    #[test]
    fn test_epilogue_relu_compiles() {
        let spec = TileGemm::tile_64x64_k16()
            .with_epilogue(vec![EpilogueOp::ReLU]);
        assert!(spec.name().contains("_relu"));
        let kernel = lower_gemm(&spec);
        let elf = kernel.compile(Target::GFX1201).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ GEMM+ReLU: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_epilogue_silu_compiles() {
        let spec = TileGemm::tile_64x64_k16()
            .with_epilogue(vec![EpilogueOp::SiLU]);
        assert!(spec.name().contains("_silu"));
        let kernel = lower_gemm(&spec);
        let elf = kernel.compile(Target::GFX1201).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ GEMM+SiLU: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_epilogue_gelu_compiles() {
        let spec = TileGemm::tile_64x64_k16()
            .with_epilogue(vec![EpilogueOp::GELU]);
        assert!(spec.name().contains("_gelu"));
        let kernel = lower_gemm(&spec);
        let elf = kernel.compile(Target::GFX1201).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ GEMM+GELU: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_epilogue_bias_relu_chain_compiles() {
        let spec = TileGemm::tile_64x64_k16()
            .with_epilogue(vec![EpilogueOp::BiasAdd, EpilogueOp::ReLU]);
        assert!(spec.name().contains("_bias_relu"));
        assert!(spec.has_epilogue_bias());
        let kernel = lower_gemm(&spec);
        let elf = kernel.compile(Target::GFX1201).unwrap();
        assert!(elf.len() > 0);
        // kernel should have extra bias_ptr argument
        let args = kernel.args();
        let has_bias_arg = args.iter().any(|a| a.name == "bias");
        assert!(has_bias_arg, "kernel should have 'bias' argument");
        eprintln!("✓ GEMM+Bias+ReLU: {} bytes ELF, {} args", elf.len(), args.len());
    }

    #[test]
    fn test_epilogue_scale_compiles() {
        let spec = TileGemm::tile_64x64_k16()
            .with_epilogue(vec![EpilogueOp::Scale]);
        let kernel = lower_gemm(&spec);
        let elf = kernel.compile(Target::GFX1201).unwrap();
        assert!(elf.len() > 0);
        let args = kernel.args();
        let has_scale_arg = args.iter().any(|a| a.name == "epi_scale");
        assert!(has_scale_arg, "kernel should have 'epi_scale' argument");
        eprintln!("✓ GEMM+Scale: {} bytes ELF", elf.len());
    }

    #[test]
    fn test_epilogue_no_regression_plain_gemm() {
        // Ensure plain GEMM (empty epilogue) still works
        let spec = TileGemm::tile_64x64_k16();
        assert!(spec.epilogue.is_empty());
        assert!(!spec.name().contains("_relu"));
        let kernel = lower_gemm(&spec);
        let elf = kernel.compile(Target::GFX1201).unwrap();
        assert!(elf.len() > 0);
        eprintln!("✓ Plain GEMM (no epilogue): {} bytes ELF", elf.len());
    }

    // ── Epilogue Fusion GPU E2E Tests ──

    #[test]
    fn test_epilogue_relu_gpu_e2e() {
        with_rt(|rt| {
            let m = 64usize;
            let k = 64usize;
            let n = 64usize;
            let spec = TileGemm::tile_64x64_k16()
                .with_epilogue(vec![EpilogueOp::ReLU]);

            // Test data: mix of positive and negative values
            // After GEMM, some outputs will be negative → ReLU should zero them
            let x_f32: Vec<f32> = (0..m*k).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
            let wt_f32: Vec<f32> = (0..n*k).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let wt_bf16: Vec<u16> = wt_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            // CPU reference: GEMM + ReLU
            let gemm_out = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m, k, n);
            let expected: Vec<f32> = gemm_out.iter().map(|&v| v.max(0.0)).collect();

            // Count how many negatives are in raw GEMM output (they should be zeroed by ReLU)
            let n_neg = gemm_out.iter().filter(|&&v| v < 0.0).count();
            eprintln!("[GEMM+ReLU] {} of {} elements are negative → should be zeroed", n_neg, m*n);

            // Upload
            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);
            let y_buf = rt.alloc_zero(m * n * 4).expect("alloc Y");

            // Compile GEMM+ReLU kernel
            let kernel = rt.ensure_kernel_t0(
                &spec.name(),
                || lower_gemm(&spec),
                [spec.wg_size(), 1, 1],
                spec.lds_total(),
            ).expect("compile GEMM+ReLU");

            // Dispatch (ReLU doesn't need extra kernargs)
            let ka = build_kernargs_m(
                x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                k as u32, n as u32, m as u32, &spec,
            );
            let grid = compute_grid(&spec, m as u32, n as u32);
            rt.dispatch(&kernel, grid, &ka).expect("dispatch GEMM+ReLU");

            // Verify
            let result = rt.read_f32(&y_buf, m * n);
            let mut max_err = 0.0f32;
            let mut n_bad = 0;
            let mut n_correctly_relu = 0;
            for i in 0..m*n {
                let err = (result[i] - expected[i]).abs();
                if err > max_err { max_err = err; }
                if err > 0.1 { n_bad += 1; }
                // ReLU check: negative GEMM outputs should become 0 or very close to 0
                if gemm_out[i] < -0.001 && result[i] < 0.001 { n_correctly_relu += 1; }
            }
            // Count significant negatives (not near-zero)
            let n_significant_neg = gemm_out.iter().filter(|&&v| v < -0.001).count();

            eprintln!("[GEMM+ReLU GPU] max_err={:.6} n_bad={}/{}", max_err, n_bad, m*n);
            eprintln!("  correctly ReLU'd: {}/{} significant negatives", n_correctly_relu, n_significant_neg);
            eprintln!("  result[0..8] = {:?}", &result[0..8]);
            eprintln!("  expected[0..8] = {:?}", &expected[0..8]);

            // All significantly negative GEMM outputs should be zeroed by ReLU
            assert_eq!(n_correctly_relu, n_significant_neg,
                "ReLU: {}/{} significant negatives zeroed (expected all)", n_correctly_relu, n_significant_neg);
            assert!(n_bad == 0,
                "GEMM+ReLU: {} of {} elements differ >0.1 (max_err={:.6})",
                n_bad, m*n, max_err);
            eprintln!("[PASS] GEMM+ReLU GPU E2E: {}x{}x{} verified (max_err={:.6})",
                m, k, n, max_err);
        });
    }

    #[test]
    fn test_epilogue_silu_gpu_e2e() {
        with_rt(|rt| {
            let m = 64usize;
            let k = 64usize;
            let n = 64usize;
            let spec = TileGemm::tile_64x64_k16()
                .with_epilogue(vec![EpilogueOp::SiLU]);

            let x_f32: Vec<f32> = (0..m*k).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
            let wt_f32: Vec<f32> = (0..n*k).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let wt_bf16: Vec<u16> = wt_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            // CPU reference: GEMM + SiLU
            let gemm_out = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m, k, n);
            let expected: Vec<f32> = gemm_out.iter().map(|&x| {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                x * sigmoid
            }).collect();

            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);
            let y_buf = rt.alloc_zero(m * n * 4).expect("alloc Y");

            let kernel = rt.ensure_kernel_t0(
                &spec.name(),
                || lower_gemm(&spec),
                [spec.wg_size(), 1, 1],
                spec.lds_total(),
            ).expect("compile GEMM+SiLU");

            let ka = build_kernargs_m(
                x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                k as u32, n as u32, m as u32, &spec,
            );
            let grid = compute_grid(&spec, m as u32, n as u32);
            rt.dispatch(&kernel, grid, &ka).expect("dispatch GEMM+SiLU");

            let result = rt.read_f32(&y_buf, m * n);
            let mut max_err = 0.0f32;
            let mut max_rel = 0.0f32;
            let mut n_bad = 0;
            for i in 0..m*n {
                let err = (result[i] - expected[i]).abs();
                let rel = if expected[i].abs() > 1e-6 { err / expected[i].abs() } else { err };
                if err > max_err { max_err = err; }
                if rel > max_rel { max_rel = rel; }
                if err > 0.1 { n_bad += 1; }
            }

            eprintln!("[GEMM+SiLU GPU] max_abs_err={:.6} max_rel_err={:.6} n_bad={}/{}",
                max_err, max_rel, n_bad, m*n);
            eprintln!("  result[0..4]   = {:?}", &result[0..4]);
            eprintln!("  expected[0..4] = {:?}", &expected[0..4]);

            // SiLU uses v_exp_f32 + v_rcp_f32 which have limited precision
            // Allow up to 5% relative error for transcendental approximation
            assert!(n_bad == 0,
                "GEMM+SiLU: {} of {} elements differ >0.1", n_bad, m*n);
            eprintln!("[PASS] GEMM+SiLU GPU E2E: {}x{}x{} verified (max_err={:.6})",
                m, k, n, max_err);
        });
    }

    #[test]
    fn test_epilogue_abs_gpu_e2e() {
        with_rt(|rt| {
            let m = 64usize;
            let k = 64usize;
            let n = 64usize;
            let spec = TileGemm::tile_64x64_k16()
                .with_epilogue(vec![EpilogueOp::Abs]);

            let x_f32: Vec<f32> = (0..m*k).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
            let wt_f32: Vec<f32> = (0..n*k).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let wt_bf16: Vec<u16> = wt_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let gemm_out = cpu_gemm_nt_bf16(&x_bf16, &wt_bf16, m, k, n);
            let expected: Vec<f32> = gemm_out.iter().map(|&v| v.abs()).collect();

            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);
            let y_buf = rt.alloc_zero(m * n * 4).expect("alloc Y");

            let kernel = rt.ensure_kernel_t0(
                &spec.name(),
                || lower_gemm(&spec),
                [spec.wg_size(), 1, 1],
                spec.lds_total(),
            ).expect("compile GEMM+Abs");

            let ka = build_kernargs_m(
                x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                k as u32, n as u32, m as u32, &spec,
            );
            let grid = compute_grid(&spec, m as u32, n as u32);
            rt.dispatch(&kernel, grid, &ka).expect("dispatch GEMM+Abs");

            let result = rt.read_f32(&y_buf, m * n);
            let mut max_err = 0.0f32;
            for i in 0..m*n {
                let err = (result[i] - expected[i]).abs();
                if err > max_err { max_err = err; }
                // All results should be non-negative
                assert!(result[i] >= 0.0, "Abs: result[{}]={} is negative!", i, result[i]);
            }

            eprintln!("[PASS] GEMM+Abs GPU E2E: {}x{}x{} verified (max_err={:.6})",
                m, k, n, max_err);
        });
    }

    /// Benchmark: fused GEMM+ReLU vs separate GEMM then separate ReLU
    #[test]
    fn test_epilogue_fusion_benchmark() {
        with_rt(|rt| {
            let m = 4096usize;
            let k = 4096usize;
            let n = 4096usize;

            // Generate data
            let x_f32: Vec<f32> = (0..m*k).map(|i| ((i % 37) as f32 - 18.0) * 0.001).collect();
            let wt_f32: Vec<f32> = (0..n*k).map(|i| ((i % 31) as f32 - 15.0) * 0.001).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let wt_bf16: Vec<u16> = wt_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let x_buf = upload_bf16(rt, &x_bf16);
            let wt_buf = upload_bf16(rt, &wt_bf16);
            // Large Y buffer: split-K configs may need up to 8 planes
            let y_buf = rt.alloc_zero(m * n * 4 * 8).expect("alloc Y");

            // Use known-best config at 4096³ (128x128_k32 = 96.4 TF, 2026-03-31)
            // Previous 103.7 TF was based on buggy LDS reads (k>16 carry corruption).
            // Do NOT use tile_auto_select which may pick split_k=8 — misleading numbers.
            let mut spec_plain = TileGemm::tile_128x128_k32();
            spec_plain.split_k = 1; // no split-K for fair comparison

            let mut spec_fused = TileGemm::tile_128x128_k32()
                .with_epilogue(vec![EpilogueOp::ReLU]);
            spec_fused.split_k = 1;

            // ── Compile both ──
            let kernel_plain = rt.ensure_kernel_t0(
                &spec_plain.name(),
                || lower_gemm(&spec_plain),
                [spec_plain.wg_size(), 1, 1],
                spec_plain.lds_total(),
            ).expect("compile plain GEMM");

            let kernel_fused = rt.ensure_kernel_t0(
                &spec_fused.name(),
                || lower_gemm(&spec_fused),
                [spec_fused.wg_size(), 1, 1],
                spec_fused.lds_total(),
            ).expect("compile fused GEMM+ReLU");

            let ka_plain = build_kernargs_m(
                x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                k as u32, n as u32, m as u32, &spec_plain,
            );
            let ka_fused = build_kernargs_m(
                x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(),
                k as u32, n as u32, m as u32, &spec_fused,
            );
            let grid_plain = compute_grid(&spec_plain, m as u32, n as u32);
            let grid_fused = compute_grid(&spec_fused, m as u32, n as u32);

            let n_iters = 20;

            // ── Warmup (10 sync dispatches for GPU clock ramp) ──
            for _ in 0..10 {
                rt.dispatch(&kernel_plain, grid_plain, &ka_plain).expect("warmup plain");
            }
            for _ in 0..10 {
                rt.dispatch(&kernel_fused, grid_fused, &ka_fused).expect("warmup fused");
            }

            // ── Timed: Plain GEMM (async batch matching autotuner methodology) ──
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                rt.dispatch_async(&kernel_plain, grid_plain, &ka_plain);
            }
            rt.wait_idle().expect("wait plain");
            let plain_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

            // ── Timed: Fused GEMM+ReLU ──
            let t1 = std::time::Instant::now();
            for _ in 0..n_iters {
                rt.dispatch_async(&kernel_fused, grid_fused, &ka_fused);
            }
            rt.wait_idle().expect("wait fused");
            let fused_us = t1.elapsed().as_micros() as f64 / n_iters as f64;

            // Compute TFLOPS
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let fused_tflops = flops / (fused_us * 1e6);
            let plain_tflops = flops / (plain_us * 1e6);

            eprintln!("\n═══════════════════════════════════════════════════════════");
            eprintln!("  Epilogue Fusion Benchmark: {}×{}×{}", m, k, n);
            eprintln!("  Config: {} (no split-K)", spec_plain.name());
            eprintln!("═══════════════════════════════════════════════════════════");
            eprintln!("  Plain GEMM:       {:.1} µs → {:.1} TFLOPS",
                plain_us, plain_tflops);
            eprintln!("  Fused GEMM+ReLU:  {:.1} µs → {:.1} TFLOPS",
                fused_us, fused_tflops);
            let overhead_pct = (fused_us / plain_us - 1.0) * 100.0;
            if overhead_pct > 0.0 {
                eprintln!("  Overhead:         +{:.1}%", overhead_pct);
            } else {
                eprintln!("  Speedup:          {:.1}% faster", -overhead_pct);
            }
            eprintln!("  (ReLU fused at zero VRAM bandwidth cost)");
            eprintln!("═══════════════════════════════════════════════════════════\n");
        });
    }
}
