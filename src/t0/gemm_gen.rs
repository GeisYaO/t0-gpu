//! Parameterized GEMM kernel generator (T0-Tensile)
//!
//! A single `generate()` function replaces all hand-written GEMM variants
//! by accepting a `GemmConfig` that controls tile sizes, K-unroll factor,
//! LDS strategy, and pipeline depth.
//!
//! # Example
//! ```rust,no_run
//! use t0_gpu::t0::gemm_gen::{GemmConfig, generate};
//! let config = GemmConfig::tile_64x64_k16();
//! let kernel = generate(&config);
//! let elf = kernel.compile(t0_gpu::t0::ir::Target::GFX1100).unwrap();
//! ```

use super::ir::*;
use super::compile::T0Kernel;

// ============================================================================
// Configuration
// ============================================================================

/// GEMM transpose mode — determines how input matrices are laid out in memory.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum GemmTranspose {
    /// Y[M,N] = A[M,K] @ B[N,K]^T — both row-major with stride=K (default, forward GEMM)
    NT,
    /// Y[M,N] = A[M,K] @ B[K,N]   — A stride=K, B stride=N (backward dX = dY @ W)
    NN,
}

impl Default for GemmTranspose {
    fn default() -> Self { GemmTranspose::NT }
}

/// GEMM epilogue operation — fused into the store phase after WMMA accumulation.
///
/// Each variant determines what happens to each accumulator value before
/// it is written to global memory. Bias is loaded from a separate pointer
/// (passed as kernarg), broadcast across rows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum EpilogueOp {
    /// Store f32 accumulators directly (no fusion). Default behavior.
    #[default]
    StoreF32,
    /// acc += bias[col]; store f32
    BiasAddStoreF32,
    /// acc += bias[col]; acc = max(acc, 0); store f32
    BiasAddReluStoreF32,
}

/// GEMM kernel configuration — each combination produces a different optimized kernel.
///
/// NT mode: Y[M,N] = X[M,K] @ WT[N,K]^T  (bf16 in, f32 accumulate, f32 out)
/// NN mode: Y[M,N] = X[M,K] @ B[K,N]     (bf16 in, f32 accumulate, f32 out)
#[derive(Clone, Debug)]
pub struct GemmConfig {
    /// Output tile rows per workgroup (must be multiple of 32).
    pub tile_m: u32,
    /// Output tile columns per workgroup (must be multiple of 64).
    pub tile_n: u32,
    /// K-dimension tile size per loop iteration (16 or 32).
    pub tile_k: u32,
    /// Workgroup size (threads). Must be 64 × (tile_m / 32).
    pub wg_size: u32,
    /// Use LDS cooperative loading (true) or direct GMEM (false).
    pub use_lds: bool,
    /// Double-buffer LDS (requires use_lds=true).
    pub double_buffer: bool,
    /// Split-K factor (None or Some(1) = no split, Some(2/4/8) = split K dimension)
    pub split_k: Option<u32>,
    /// LDS row padding in bytes (eliminates bank conflicts).
    pub lds_pad: u32,
    /// Number of column passes (host-level dispatch hint, not used in kernel gen).
    pub n_col_passes: u32,
    /// Grid axis swap: true=TGID.x→N, TGID.y→M (L2 friendly for square);
    /// false=TGID.x→M, TGID.y→N (better for rectangular M<N).
    pub swap_grid: bool,
    /// WGP mode: workgroup spans 2 CUs = 128KB LDS + 4 SIMDs + doubled VGPR pool.
    pub wgp_mode: bool,
    /// Transpose mode: NT (default, Y=A@B^T) or NN (Y=A@B).
    pub transpose: GemmTranspose,
    /// Epilogue operation fused into the store phase.
    pub epilogue: EpilogueOp,
}

impl GemmConfig {
    /// 16×64, K=16, LDS double-buffered (small-M: 1 wave, max M-parallelism)
    pub fn tile_16x64_k16() -> Self {
        Self { tile_m: 16, tile_n: 64, tile_k: 16, wg_size: 32, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 32×64, K=16, no LDS (equivalent to `matmul`)
    pub fn tile_32x64_direct() -> Self {
        Self { tile_m: 32, tile_n: 64, tile_k: 16, wg_size: 64, use_lds: false, double_buffer: false, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 32×64, K=16, LDS double-buffered (equivalent to `matmul_lds_db`)
    pub fn tile_32x64_k16() -> Self {
        Self { tile_m: 32, tile_n: 64, tile_k: 16, wg_size: 64, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 32×64, K=32, LDS double-buffered (small-M optimized with deeper K unroll)
    pub fn tile_32x64_k32() -> Self {
        Self { tile_m: 32, tile_n: 64, tile_k: 32, wg_size: 64, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 32×128, K=16, LDS double-buffered (small-M: wide N for more compute per WG)
    pub fn tile_32x128_k16() -> Self {
        Self { tile_m: 32, tile_n: 128, tile_k: 16, wg_size: 64, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 64×64, K=16, LDS double-buffered (equivalent to `matmul_64x64_lds_db`)
    pub fn tile_64x64_k16() -> Self {
        Self { tile_m: 64, tile_n: 64, tile_k: 16, wg_size: 64, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 64×64, K=32, LDS double-buffered (equivalent to `matmul_64x64_k32`)
    pub fn tile_64x64_k32() -> Self {
        Self { tile_m: 64, tile_n: 64, tile_k: 32, wg_size: 64, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 128×64, K=16, LDS double-buffered (higher compute density)
    pub fn tile_128x64_k16() -> Self {
        Self { tile_m: 128, tile_n: 64, tile_k: 16, wg_size: 128, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 64×128, K=16, LDS double-buffered (wider N tiles)
    pub fn tile_64x128_k16() -> Self {
        Self { tile_m: 64, tile_n: 128, tile_k: 16, wg_size: 64, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 128×64, K=32 (max compute density)
    pub fn tile_128x64_k32() -> Self {
        Self { tile_m: 128, tile_n: 64, tile_k: 32, wg_size: 128, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 128×128, K=16, LDS double-buffered (highest AI = 64 FLOP/byte)
    /// Acc VGPRs: 4 row_blocks × 8 col_tiles × 8 = 256 — BUT each wave only uses
    /// n_row_blocks(2) × n_col_tiles(8) × 8 = 128 VGPRs. Tight but feasible.
    pub fn tile_128x128_k16() -> Self {
        Self { tile_m: 128, tile_n: 128, tile_k: 16, wg_size: 128, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 64×64, K=64, LDS double-buffered (4× fewer loop iterations)
    pub fn tile_64x64_k64() -> Self {
        Self { tile_m: 64, tile_n: 64, tile_k: 64, wg_size: 64, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 1, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 128×128 via 2×(128×64) column passes. Effective AI=64.
    /// X data reused from L2 cache on second pass.
    pub fn tile_128x128_2pass() -> Self {
        Self { tile_m: 128, tile_n: 64, tile_k: 16, wg_size: 128, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 2, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }
    /// 64×128 via 2×(64×64) column passes. Effective AI=43.
    pub fn tile_64x128_2pass() -> Self {
        Self { tile_m: 64, tile_n: 64, tile_k: 16, wg_size: 64, use_lds: true, double_buffer: true, split_k: None, lds_pad: 0, n_col_passes: 2, swap_grid: true, wgp_mode: true, transpose: GemmTranspose::NT, epilogue: EpilogueOp::default() }
    }

    /// Number of WMMA column tiles = tile_n / 16
    pub fn n_col_tiles(&self) -> usize { (self.tile_n / 16) as usize }
    /// Number of row blocks per wave = (tile_m / n_waves) / 16
    pub fn n_row_blocks(&self) -> usize {
        let n_waves = self.wg_size / 32;
        let rows_per_wave = self.tile_m / n_waves;
        (rows_per_wave / 16) as usize
    }
    /// Number of waves per workgroup
    pub fn n_waves(&self) -> u32 { self.wg_size / 32 }
    /// Rows per wave = tile_m / n_waves (must be multiple of 16)
    pub fn rows_per_wave(&self) -> u32 { self.tile_m / self.n_waves() }
    /// K sub-steps per tile_k (each WMMA handles K=16)
    pub fn k_sub_steps(&self) -> u32 { self.tile_k / 16 }
    /// LDS bytes per buffer (X + WT region)
    pub fn lds_per_buffer(&self) -> u32 {
        let x_bytes = self.tile_m * self.tile_k * 2;  // M rows × K cols × bf16
        let wt_bytes = self.tile_n * self.tile_k * 2;
        x_bytes + wt_bytes
    }
    /// Total LDS bytes (single or double buffered)
    pub fn lds_total(&self) -> u32 {
        if self.double_buffer { self.lds_per_buffer() * 2 } else { self.lds_per_buffer() }
    }
    /// LDS X region size per buffer (with padding)
    pub fn lds_x_size(&self) -> u32 { self.tile_m * (self.tile_k * 2 + self.lds_pad) }
    /// LDS WT region size per buffer (with padding)
    pub fn lds_wt_size(&self) -> u32 { self.tile_n * (self.tile_k * 2 + self.lds_pad) }
    /// LDS row stride in bytes for X (tile_k × 2 + pad)
    pub fn lds_x_row_stride(&self) -> u32 { self.tile_k * 2 + self.lds_pad }
    /// LDS row stride in bytes for WT (tile_k × 2 + pad)
    pub fn lds_wt_row_stride(&self) -> u32 { self.tile_k * 2 + self.lds_pad }
    /// Bytes per global load per thread for X
    pub fn x_bytes_per_thread(&self) -> u32 { self.tile_m * self.tile_k * 2 / self.wg_size }
    /// Bytes per global load per thread for WT
    pub fn wt_bytes_per_thread(&self) -> u32 { self.tile_n * self.tile_k * 2 / self.wg_size }
    /// Total WMMA operations per K-tile
    pub fn wmma_per_k_tile(&self) -> usize {
        self.n_row_blocks() * self.n_col_tiles() * self.k_sub_steps() as usize
    }
    /// Estimated VGPR usage for LDS double-buffer kernel on a specific target.
    pub fn estimated_vgprs_for_target(&self, target: Target) -> u32 {
        let nrb = self.n_row_blocks() as u32;
        let nct = self.n_col_tiles() as u32;
        let x_lpt = self.x_bytes_per_thread() / 16; // b128 loads for X
        let wt_lpt = self.wt_bytes_per_thread() / 16; // b128 loads for WT
        let wmma = WmmaFormat::BF16_F32;
        let acc = nrb * nct * wmma.dst_vreg_count(target);
        let x_frag = nrb * wmma.a_vreg_count(target);
        let wt_frag = nct * wmma.b_vreg_count(target);
        let gmem_x = x_lpt * 4;            // GMEM load regs for X (b128 = 4 VGPRs)
        let gmem_wt = wt_lpt * 4;          // GMEM load regs for WT
        let addr_temps = 49;               // address computation, LDS offsets, store temps
        acc + x_frag + wt_frag + gmem_x + gmem_wt + addr_temps
    }

    /// Estimated VGPR usage for the active target context.
    pub fn estimated_vgprs(&self) -> u32 {
        self.estimated_vgprs_for_target(current_target())
    }

    /// Check if this config is feasible on a specific target.
    pub fn is_feasible_for_target(&self, target: Target) -> bool {
        self.estimated_vgprs_for_target(target) <= 256
    }

    /// Check if this config is feasible on the active target context.
    pub fn is_feasible(&self) -> bool {
        self.is_feasible_for_target(current_target())
    }
    /// Descriptive name
    pub fn name(&self) -> String {
        let lds_tag = if self.use_lds {
            if self.double_buffer { "_ldsdb" } else { "_lds" }
        } else { "_direct" };
        let sk_tag = match self.split_k {
            Some(sk) if sk > 1 => format!("_sk{}", sk),
            _ => String::new(),
        };
        let pass_tag = if self.n_col_passes > 1 {
            format!("_{}p", self.n_col_passes)
        } else { String::new() };
        let grid_tag = if !self.swap_grid { "_mg" } else { "" }; // mg = M-on-grid-X
        let wgp_tag = if self.wgp_mode { "_wgp" } else { "" };
        let trans_tag = if self.transpose == GemmTranspose::NN { "_nn" } else { "" };
        let epi_tag = match self.epilogue {
            EpilogueOp::StoreF32 => "",
            EpilogueOp::BiasAddStoreF32 => "_bias",
            EpilogueOp::BiasAddReluStoreF32 => "_bias_relu",
        };
        format!("t0_gemm_{}x{}_k{}{}{}{}{}{}{}{}", self.tile_m, self.tile_n, self.tile_k, lds_tag, sk_tag, pass_tag, grid_tag, wgp_tag, trans_tag, epi_tag)
    }
}

/// Predefined sweep search space — configurations to benchmark
pub fn sweep_configs() -> Vec<GemmConfig> {
    vec![
        GemmConfig::tile_16x64_k16(),
        GemmConfig::tile_32x64_k16(),
        GemmConfig::tile_32x64_k32(),
        GemmConfig::tile_32x128_k16(),
        GemmConfig::tile_64x64_k16(),
        GemmConfig::tile_64x64_k32(),
        GemmConfig::tile_128x64_k16(),
        GemmConfig::tile_64x128_k16(),
        GemmConfig::tile_128x64_k32(),
    ]
}

// ============================================================================
// Generator
// ============================================================================

/// Generate a GEMM kernel from configuration.
///
/// Returns (kernel, lds_size, workgroup_size, grid_fn) where grid_fn
/// computes grid dimensions for a given (M, N).
pub fn generate(cfg: &GemmConfig) -> T0Kernel {
    let target = current_target();
    let wmma = WmmaFormat::BF16_F32;
    let max_vgprs = 256u32;
    let est_vgprs = cfg.estimated_vgprs_for_target(target);
    if est_vgprs > max_vgprs {
        panic!(
            "[gemm_gen] Config '{}' on {:?} requires ~{} VGPRs (max {}). \
             Use n_col_passes=2 or smaller tile. Breakdown: acc={}, x_frag={}, wt_frag={}, gmem={}, temps=49",
            cfg.name(), target, est_vgprs, max_vgprs,
            cfg.n_row_blocks() as u32 * cfg.n_col_tiles() as u32 * wmma.dst_vreg_count(target),
            cfg.n_row_blocks() as u32 * wmma.a_vreg_count(target),
            cfg.n_col_tiles() as u32 * wmma.b_vreg_count(target),
            cfg.x_bytes_per_thread() / 16 * 4 + cfg.wt_bytes_per_thread() / 16 * 4,
        );
    }
    let mut k = if cfg.use_lds && cfg.double_buffer {
        generate_lds_db(cfg)
    } else {
        generate_direct(cfg)
    };
    // GEMM kernels use carefully hand-crafted instruction sequences
    // (cooperative loads, barriers, LDS double-buffering, WMMA scheduling).
    // The optimization passes (DCE, CSE, instruction scheduling, etc.) are
    // designed for DSL/tile-IR kernels and will break these patterns.
    k.set_skip_optimize(true);
    // SSA regalloc is safe for GEMM: insert_spill_reloads() places spill slots
    // at existing_lds + offset, avoiding overlap with the GEMM double-buffer.
    // The spill infrastructure (compile.rs L917-920) correctly passes lds_size.
    k
}

/// Returns (grid_x, grid_y) for a given (M, N) based on this config.
/// Uses ceiling division — caller must ensure output buffer is large enough
/// for the padded tile dimensions (m_padded * n_padded elements).
pub fn compute_grid(cfg: &GemmConfig, m: u32, n: u32) -> (u32, u32) {
    let effective_n = cfg.tile_n * cfg.n_col_passes;
    let tiles_m = (m + cfg.tile_m - 1) / cfg.tile_m;
    let tiles_n = (n + effective_n - 1) / effective_n;
    if cfg.swap_grid {
        (tiles_n * cfg.wg_size, tiles_m)
    } else {
        (tiles_m * cfg.wg_size, tiles_n)
    }
}

/// Grid with split-K. Uses ceiling division for M/N.
pub fn compute_grid_split_k(cfg: &GemmConfig, m: u32, n: u32, split_k: u32) -> (u32, u32) {
    let effective_n = cfg.tile_n * cfg.n_col_passes;
    assert!(split_k > 0 && split_k.is_power_of_two(),
        "split_k={} must be power of 2", split_k);
    let tiles_m = (m + cfg.tile_m - 1) / cfg.tile_m;
    let tiles_n = (n + effective_n - 1) / effective_n;
    if cfg.swap_grid {
        (tiles_n * cfg.wg_size, tiles_m * split_k)
    } else {
        (tiles_m * cfg.wg_size, tiles_n * split_k)
    }
}

// ============================================================================
// Auto-Select: Pick optimal GemmConfig for given matrix dimensions
// ============================================================================

/// Select the optimal GEMM kernel configuration for given matrix dimensions.
///
/// Uses the cost_model exhaustive search (400+ candidates × hardware model)
/// to pick the best config. Falls back to the hand-tuned heuristic if the
/// cost model finds no feasible solution.
pub fn auto_select(m: u32, k: u32, n: u32) -> GemmConfig {
    super::cost_model::predict_best(m, k, n)
}

/// Legacy hand-tuned heuristic (preserved as fallback and for A/B comparison).
///
/// Based on empirical sweep data (RX 7900 XTX, GFX1100):
/// - Tiny squares (≤512): split_k=4 k32 for CU fill
/// - Medium squares (1024): 64×64 k16
/// - Large squares (≥2048): k32 split2 or 128×64 k32
/// - Rectangular (small M, big K): split_k=8 k16
pub fn auto_select_legacy(m: u32, k: u32, n: u32) -> GemmConfig {
    let mn = (m as u64) * (n as u64);

    // Optimal configs from benchmark sweep (2026-03-21, post-WGP optimization).
    // Helper: clamp split-K so K is divisible by (tile_k * sk).
    let clamp_sk = |tile_k: u32, desired_sk: u32| -> Option<u32> {
        let max_sk = k / tile_k;  // maximum splits before k_chunk < tile_k
        let mut sk = desired_sk.min(max_sk);
        // Round down to nearest power of 2
        while sk > 1 && k % (tile_k * sk) != 0 {
            sk /= 2;
        }
        if sk > 1 { Some(sk) } else { None }
    };

    // Choose tile_m: must divide M. Prefer 128, fallback to 64.
    let use_128m = m % 128 == 0 && m >= 128;
    let use_64m  = m % 64 == 0;

    if m <= 256 && k >= 512 && n >= 1024 && use_128m {
        // Thin: small M, large K×N → M-on-X + WGP, sk=4
        GemmConfig { swap_grid: false, wgp_mode: true, split_k: clamp_sk(16, 4),
            ..GemmConfig::tile_128x64_k16() }
    } else if m <= 512 && k >= 512 && n >= 1024 && use_128m {
        // Medium-thin → M-on-X + WGP, sk=8
        GemmConfig { swap_grid: false, wgp_mode: true, split_k: clamp_sk(16, 8),
            ..GemmConfig::tile_128x64_k16() }
    } else if mn <= 512 * 512 {
        // Small square → 64×64, clamp sk
        GemmConfig { swap_grid: false, split_k: clamp_sk(16, 8),
            ..GemmConfig::tile_64x64_k16() }
    } else if mn <= 1024 * 1024 && use_128m {
        // Medium square → WGP, sk=2
        GemmConfig { swap_grid: false, wgp_mode: true, split_k: clamp_sk(16, 2),
            ..GemmConfig::tile_128x64_k16() }
    } else if use_128m {
        // Large/very large → WGP, sk=8
        GemmConfig { wgp_mode: true, split_k: clamp_sk(16, 8),
            ..GemmConfig::tile_128x64_k16() }
    } else if use_64m {
        // M not divisible by 128 → use 64×64 with WGP + split-K
        GemmConfig { wgp_mode: true, split_k: clamp_sk(16, 4),
            ..GemmConfig::tile_64x64_k16() }
    } else {
        // Fallback: 32×64 (tile_m=32 divides most sizes)
        GemmConfig { split_k: clamp_sk(16, 4),
            ..GemmConfig::tile_32x64_k16() }
    }
}

/// Standard set of pre-selected configs covering all common use cases.
pub fn standard_configs() -> Vec<GemmConfig> {
    vec![
        GemmConfig::tile_16x64_k16(),
        GemmConfig::tile_32x64_k16(),
        GemmConfig::tile_32x64_k32(),
        GemmConfig::tile_32x128_k16(),
        GemmConfig::tile_64x64_k16(),
        GemmConfig::tile_64x64_k32(),
        GemmConfig::tile_128x64_k32(),
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_32x64_k32() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_16x64_k16() },
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_64x64_k32() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k32() },
        GemmConfig { split_k: Some(8), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_128x64_k32() },
    ]
}

/// Compute grid dimensions for a config, handling split-K automatically.
pub fn compute_grid_auto(cfg: &GemmConfig, m: u32, n: u32) -> (u32, u32) {
    let sk = cfg.split_k.unwrap_or(1);
    if sk > 1 {
        compute_grid_split_k(cfg, m, n, sk)
    } else {
        compute_grid(cfg, m, n)
    }
}

/// Build 40-byte kernarg buffer for a GEMM dispatch.
///
/// Layout: [X:u64, WT:u64, Y:u64, K:u32, N:u32, split_k_shift:u32, y_split_stride:u32]
pub fn build_kernargs(
    x_addr: u64, wt_addr: u64, y_addr: u64,
    k: u32, n: u32, m: u32,
    cfg: &GemmConfig,
) -> [u8; 40] {
    let sk = cfg.split_k.unwrap_or(1);
    let y_stride = if sk > 1 { m * n * 4 } else { 0 };
    let mut ka = [0u8; 40];
    ka[0..8].copy_from_slice(&x_addr.to_le_bytes());
    ka[8..16].copy_from_slice(&wt_addr.to_le_bytes());
    ka[16..24].copy_from_slice(&y_addr.to_le_bytes());
    ka[24..28].copy_from_slice(&k.to_le_bytes());
    ka[28..32].copy_from_slice(&n.to_le_bytes());
    ka[32..36].copy_from_slice(&0u32.to_le_bytes());
    ka[36..40].copy_from_slice(&y_stride.to_le_bytes());
    ka
}

/// Build 48-byte kernarg buffer for a GEMM dispatch with epilogue fusion.
///
/// Layout (extends 40-byte layout):
/// [X:u64, WT:u64, Y:u64, K:u32, N:u32, split_k_shift:u32, y_split_stride:u32, bias:u64]
///
/// Set bias_addr=0 for no bias (EpilogueOp::StoreF32).
pub fn build_kernargs_with_bias(
    x_addr: u64, wt_addr: u64, y_addr: u64,
    k: u32, n: u32, m: u32,
    cfg: &GemmConfig,
    bias_addr: u64,
) -> [u8; 48] {
    let sk = cfg.split_k.unwrap_or(1);
    let y_stride = if sk > 1 { m * n * 4 } else { 0 };
    let mut ka = [0u8; 48];
    ka[0..8].copy_from_slice(&x_addr.to_le_bytes());
    ka[8..16].copy_from_slice(&wt_addr.to_le_bytes());
    ka[16..24].copy_from_slice(&y_addr.to_le_bytes());
    ka[24..28].copy_from_slice(&k.to_le_bytes());
    ka[28..32].copy_from_slice(&n.to_le_bytes());
    ka[32..36].copy_from_slice(&0u32.to_le_bytes());
    ka[36..40].copy_from_slice(&y_stride.to_le_bytes());
    ka[40..48].copy_from_slice(&bias_addr.to_le_bytes());
    ka
}

// ============================================================================
// GEMM Backward Helpers
// ============================================================================
//
// Given forward: Y[M,N] = X[M,K] @ W[N,K]^T   (NT GEMM)
//
// Backward data:   dX[M,K] = dY[M,N] @ W[N,K]^T
//   → This IS an NT GEMM!  A=dY, B=W, M_dim=M, K_dim=N, N_dim=K
//   → W is already in [N,K] layout from forward. Zero transpose needed.
//
// Backward weight: dW[N,K] = dY^T[N,M] @ X[M,K]^T
//   → Need: transpose dY[M,N]→dY_T[N,M], transpose X[M,K]→X_T[K,M]
//   → Then NT GEMM: A=dY_T[N,M], B=X_T[K,M], M_dim=N, K_dim=M, N_dim=K
//   → Result: dW[N,K]
//
// Both use the SAME NT kernel generated by `generate()`. Only dimensions differ.

/// Select config for backward-data GEMM: dX[M,K] = dY[M,N] @ WT,
/// where forward was Y = X @ WT^T, so WT is stored as [N,K].
///
/// **Caller must transpose WT[N,K] → W[K,N] first!**
/// Then NT GEMM: A=dY[M,N], B=W[K,N], output=dX[M,K]
///   → GEMM dimensions: M_gemm=M, K_gemm=N (contraction), N_gemm=K (output)
///   → B has [K, N] layout with stride=N per row, K_gemm=N ✓
pub fn auto_select_backward_data(m: u32, n_orig: u32, k_orig: u32) -> GemmConfig {
    // dX = dY[M, N_orig] @ W[K_orig, N_orig]^T
    // GEMM: M=M, K=N_orig (contraction), N=K_orig (output cols)
    auto_select(m, n_orig, k_orig)
}

/// Build kernargs for backward-data: dX = dY @ W^T
///
/// **`w_addr` must point to the TRANSPOSED weight W[K_orig, N_orig], NOT WT[N_orig, K_orig]!**
///
/// - `dy_addr`: GPU addr of dY[M, N_orig] (bf16, row-major, stride=N_orig)
/// - `w_addr`:  GPU addr of W[K_orig, N_orig] = WT^T (bf16, row-major, stride=N_orig)
/// - `dx_addr`: GPU addr of dX[M, K_orig] (f32, output)
pub fn build_kernargs_backward_data(
    dy_addr: u64, w_addr: u64, dx_addr: u64,
    m: u32, n_orig: u32, k_orig: u32,
    cfg: &GemmConfig,
) -> [u8; 40] {
    // NT GEMM: A=dY[M,N], B=W[K,N] (transposed WT), K_contract=N, N_out=K
    build_kernargs(dy_addr, w_addr, dx_addr, n_orig, k_orig, m, cfg)
}

/// Grid dimensions for backward-data GEMM.
pub fn compute_grid_backward_data(cfg: &GemmConfig, m: u32, k_orig: u32) -> (u32, u32) {
    // NT GEMM output is [M, K_orig], grid tiles over M and K_orig
    compute_grid_auto(cfg, m, k_orig)
}

/// Select config for backward-weight GEMM: dW[N,K] = dY_T[N,M] @ X_T[K,M]^T
///
/// **Caller must pre-transpose dY and X before dispatching!**
///   - dY[M,N]   → dY_T[N,M]   (use T0 transpose kernel)
///   - X[M,K]    → X_T[K,M]    (use T0 transpose kernel)
///
/// Then: NT GEMM A=dY_T, B=X_T, M_gemm=N, K_gemm=M, N_gemm=K
pub fn auto_select_backward_weight(m: u32, n_orig: u32, k_orig: u32) -> GemmConfig {
    // dW = dY_T[N_orig, M] @ X_T[K_orig, M]^T
    // GEMM: M=N_orig, K=M (contraction), N=K_orig (output cols)
    auto_select(n_orig, m, k_orig)
}

/// Build kernargs for backward-weight: dW = dY_T @ X_T^T
///
/// **Inputs must already be transposed!**
/// - `dy_t_addr`: GPU addr of dY^T[N_orig, M] (bf16, row-major, stride=M)
/// - `x_t_addr`:  GPU addr of X^T[K_orig, M] (bf16, row-major, stride=M)
/// - `dw_addr`:   GPU addr of dW[N_orig, K_orig] (f32, output)
pub fn build_kernargs_backward_weight(
    dy_t_addr: u64, x_t_addr: u64, dw_addr: u64,
    m: u32, n_orig: u32, k_orig: u32,
    cfg: &GemmConfig,
) -> [u8; 40] {
    // NT GEMM: A=dY_T[N,M], B=X_T[K,M], K_contract=M, N_out=K_orig
    build_kernargs(dy_t_addr, x_t_addr, dw_addr, m, k_orig, n_orig, cfg)
}

/// Grid dimensions for backward-weight GEMM.
pub fn compute_grid_backward_weight(cfg: &GemmConfig, n_orig: u32, k_orig: u32) -> (u32, u32) {
    // NT GEMM output is [N_orig, K_orig]
    compute_grid_auto(cfg, n_orig, k_orig)
}

#[cfg(test)]
mod auto_select_tests {
    use super::*;

    #[test]
    fn test_auto_select_sizes() {
        // Small square: should produce a valid config
        let c = auto_select(256, 256, 256);
        assert!(c.tile_m >= 16 && c.tile_m <= 256, "tile_m={}", c.tile_m);
        assert!(c.tile_n >= 32 && c.tile_n <= 128, "tile_n={}", c.tile_n);
        assert!(c.wg_size >= 32, "wg_size={}", c.wg_size);
        eprintln!("256×256: {}", c.name());

        // Medium square: should use >=64 tile_m
        let c = auto_select(1024, 1024, 1024);
        assert!(c.tile_m >= 64, "1024² tile_m should be >= 64: {}", c.tile_m);
        assert_eq!(c.tile_k, 16);
        eprintln!("1024×1024: {}", c.name());

        // Very large: should use large tiles
        let c = auto_select(8192, 8192, 8192);
        assert!(c.tile_m >= 64, "8192³ tile_m should be >= 64: {}", c.tile_m);
        eprintln!("8192×8192: {}", c.name());

        // Thin M: tile_m should divide M
        let c = auto_select(128, 4096, 4096);
        assert!(128 % c.tile_m == 0, "tile_m={} should divide M=128", c.tile_m);
        eprintln!("128×4096: {}", c.name());
    }

    /// Verify auto_select_legacy still works (preserved for A/B comparison)
    #[test]
    fn test_auto_select_legacy_preserved() {
        let c = auto_select_legacy(256, 256, 256);
        assert!(c.split_k.unwrap_or(1) > 1, "legacy small should use split-K");
        assert_eq!(c.tile_m, 64);

        let c = auto_select_legacy(1024, 1024, 1024);
        assert_eq!(c.tile_m, 128);
        assert!(c.wgp_mode);
    }
}

#[cfg(test)]
mod epilogue_tests {
    use super::*;

    #[test]
    fn test_epilogue_name_tags() {
        let mut cfg = GemmConfig::tile_32x64_k16();
        assert!(cfg.name().contains("t0_gemm_32x64_k16"));
        assert!(!cfg.name().contains("_bias"));

        cfg.epilogue = EpilogueOp::BiasAddStoreF32;
        assert!(cfg.name().ends_with("_bias"), "name={}", cfg.name());

        cfg.epilogue = EpilogueOp::BiasAddReluStoreF32;
        assert!(cfg.name().ends_with("_bias_relu"), "name={}", cfg.name());
    }

    #[test]
    fn test_epilogue_bias_generates() {
        let mut cfg = GemmConfig::tile_32x64_k16();
        cfg.epilogue = EpilogueOp::BiasAddStoreF32;
        // generate() should succeed without panic
        let _kernel = generate(&cfg);
    }

    #[test]
    fn test_epilogue_bias_relu_generates() {
        let mut cfg = GemmConfig::tile_32x64_k16();
        cfg.epilogue = EpilogueOp::BiasAddReluStoreF32;
        let _kernel = generate(&cfg);
    }

    #[test]
    fn test_kernargs_with_bias_layout() {
        let cfg = GemmConfig::tile_32x64_k16();
        let ka = build_kernargs_with_bias(
            0x1000, 0x2000, 0x3000,
            64, 64, 128, &cfg, 0x4000,
        );
        assert_eq!(ka.len(), 48);
        // Verify bias_ptr at bytes 40..48
        let bias = u64::from_le_bytes(ka[40..48].try_into().unwrap());
        assert_eq!(bias, 0x4000);
    }
}

// ============================================================================
// LDS Double-Buffered Generator (handles any tile_m × tile_n × tile_k)
// ============================================================================

fn generate_lds_db(cfg: &GemmConfig) -> T0Kernel {
    let mut k = T0Kernel::new(&cfg.name());
    let n_col_tiles = cfg.n_col_tiles();
    let n_row_blocks = cfg.n_row_blocks();
    let _n_waves = cfg.n_waves();
    let rows_per_wave = cfg.rows_per_wave();

    let lds_x = cfg.lds_x_size();
    let lds_wt = cfg.lds_wt_size();
    let lds_buf = lds_x + lds_wt;  // per buffer
    k.set_lds_size(lds_buf * 2);   // double buffer
    k.set_wgp_mode(cfg.wgp_mode);

    let x_row_stride = cfg.lds_x_row_stride();   // tile_k * 2 bytes
    let wt_row_stride = cfg.lds_wt_row_stride();

    // How many b128 loads per thread for X and WT
    let x_loads_per_thread = cfg.x_bytes_per_thread() / 16;  // 16 bytes per b128
    let wt_loads_per_thread = cfg.wt_bytes_per_thread() / 16;

    // ── Args ──
    let x_ptr = k.arg_ptr("X");
    let wt_ptr = k.arg_ptr("WT");
    let y_ptr = k.arg_ptr("Y");
    let k_dim = k.arg_u32("K");
    let n_dim = k.arg_u32("N");
    let _split_k_shift_arg = k.arg_u32("split_k_shift");  // reserved (unused, kept for layout)
    let y_split_stride_arg = k.arg_u32("y_split_stride"); // M*N*4 for split, 0 for normal

    // Epilogue: bias pointer (only declared when epilogue needs it)
    let bias_ptr = if cfg.epilogue != EpilogueOp::StoreF32 {
        let bp = k.arg_ptr("bias");
        Some(bp)
    } else {
        None
    };
    k.emit_arg_loads();

    // ── TGIDs with split-K support ──
    // swap_grid=true:  TGID.x → tile_col (N), TGID.y → tile_row (M)
    // swap_grid=false: TGID.x → tile_row (M), TGID.y → tile_col (N)
    let tgid_x_s = k.alloc_sreg();
    k.capture_tgid_x(tgid_x_s);
    let tgid_y_s = k.alloc_sreg();
    k.capture_tgid_y(tgid_y_s);

    let split_k = cfg.split_k.unwrap_or(1);
    let split_k_shift: u8 = match split_k { 1 => 0, 2 => 1, 4 => 2, 8 => 3, 16 => 4, _ => panic!("unsupported split_k") };

    // Assign tile_col / tile_row based on swap_grid
    let tile_col_s = k.alloc_sreg();
    let tile_row_s = k.alloc_sreg();
    let split_k_id_s = k.alloc_sreg();

    // The "raw" fast-axis TGID and slow-axis TGID
    let (fast_tgid, slow_tgid) = if cfg.swap_grid {
        // fast = TGID.x = tile_col, slow = TGID.y = tile_row [+ split_k]
        (tgid_x_s, tgid_y_s)
    } else {
        // fast = TGID.x = tile_row, slow = TGID.y = tile_col [+ split_k]
        (tgid_x_s, tgid_y_s)
    };

    if cfg.swap_grid {
        // TGID.x = tile_col directly
        k.push(Op::SAddU32 { dst: tile_col_s, src0: fast_tgid, src1: SOperand::InlineInt(0) });
        if split_k <= 1 {
            k.push(Op::SAddU32 { dst: tile_row_s, src0: slow_tgid, src1: SOperand::InlineInt(0) });
            k.s_mov_imm(split_k_id_s, 0);
        } else {
            // tile_row = TGID.y >> shift, split_k_id = TGID.y & mask
            k.s_lshr_b32(tile_row_s, slow_tgid, split_k_shift);
            let mask_s = k.alloc_sreg();
            k.s_mov_imm(mask_s, (split_k - 1) as i32);
            k.s_and_b32(split_k_id_s, slow_tgid, mask_s);
        }
    } else {
        // TGID.x = tile_row directly
        k.push(Op::SAddU32 { dst: tile_row_s, src0: fast_tgid, src1: SOperand::InlineInt(0) });
        if split_k <= 1 {
            k.push(Op::SAddU32 { dst: tile_col_s, src0: slow_tgid, src1: SOperand::InlineInt(0) });
            k.s_mov_imm(split_k_id_s, 0);
        } else {
            // tile_col = TGID.y >> shift, split_k_id = TGID.y & mask
            k.s_lshr_b32(tile_col_s, slow_tgid, split_k_shift);
            let mask_s = k.alloc_sreg();
            k.s_mov_imm(mask_s, (split_k - 1) as i32);
            k.s_and_b32(split_k_id_s, slow_tgid, mask_s);
        }
    }

    // k_end = K >> split_k_shift (= K when no split)
    let k_end_s = k.alloc_sreg();
    if split_k <= 1 {
        k.push(Op::SAddU32 { dst: k_end_s, src0: SReg(k_dim.0), src1: SOperand::InlineInt(0) });
    } else {
        k.s_lshr_b32(k_end_s, SReg(k_dim.0), split_k_shift);
    }

    // k_start_bytes = split_k_id * k_end * 2
    let k_start_bytes_s = k.alloc_sreg();
    if split_k <= 1 {
        k.s_mov_imm(k_start_bytes_s, 0);
    } else {
        let s_tmp = k.alloc_sreg();
        k.s_mul_i32(s_tmp, split_k_id_s, k_end_s);
        k.s_lshl_b32(k_start_bytes_s, s_tmp, 1); // * 2 for bf16
    }

    // Y offset = split_k_id * y_split_stride
    let y_offset_s = k.alloc_sreg();
    if split_k <= 1 {
        k.s_mov_imm(y_offset_s, 0);
    } else {
        k.s_mul_i32(y_offset_s, split_k_id_s, SReg(y_split_stride_arg.0));
    }

    // ── Thread decomposition ──
    let tid = VReg(0);
    let lane_id = k.alloc_vreg();
    k.v_and_b32_imm(lane_id, tid, 31);
    let wave_id = k.alloc_vreg();
    k.v_lshrrev_b32(wave_id, 5, tid);
    let wave_id_s = k.alloc_sreg();
    k.push(Op::VReadfirstlane { dst: wave_id_s, src: wave_id });
    let lane_row = k.alloc_vreg();
    k.v_and_b32_imm(lane_row, lane_id, 15);

    // ── Accumulators: n_row_blocks × n_col_tiles × 8 VGPRs ──
    let mut acc = Vec::new();
    for _r in 0..n_row_blocks {
        for _c in 0..n_col_tiles {
            acc.push(k.alloc_vreg_array(8, Alignment::Align8));
        }
    }
    for a in &acc {
        for i in 0..8u32 { k.v_mov_imm(VReg(a.0 + i), 0); }
    }

    // ── Store phase constants ──
    // s_row_base[r] for each row block
    let tile_m_shift = match cfg.tile_m {
        16 => 4, 32 => 5, 64 => 6, 128 => 7, _ => panic!("unsupported tile_m"),
    };
    let rpw_shift = match rows_per_wave {
        16 => 4, 32 => 5, 64 => 6, _ => panic!("unsupported rows_per_wave"),
    };
    let mut s_row_bases = Vec::new();
    let s_row_base0 = k.alloc_sreg();
    k.s_lshl_b32(s_row_base0, tile_row_s, tile_m_shift);
    let s_tmp = k.alloc_sreg();
    k.s_lshl_b32(s_tmp, wave_id_s, rpw_shift);
    k.push(Op::SAddU32 { dst: s_row_base0, src0: s_row_base0, src1: SOperand::SReg(s_tmp) });
    s_row_bases.push(s_row_base0);
    for r in 1..n_row_blocks {
        let rb = k.alloc_sreg();
        k.push(Op::SAddU32 {
            dst: rb, src0: s_row_base0,
            src1: SOperand::InlineInt((r * 16) as i32),
        });
        s_row_bases.push(rb);
    }

    let base_n_s = k.alloc_sreg();
    let tile_n_shift = match cfg.tile_n {
        32 => 5, 64 => 6, 128 => 7, _ => panic!("unsupported tile_n"),
    };
    k.s_lshl_b32(base_n_s, tile_col_s, tile_n_shift);

    // ══════════════════════════════════════════════════════════════════
    // COOPERATIVE LOAD ADDRESSES
    // ══════════════════════════════════════════════════════════════════

    let k_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(k_vreg, SReg(k_dim.0));

    // ── X cooperative load address (COALESCED) ──
    // CRITICAL: GMEM rows are tile_k*2 bytes wide (no padding).
    // LDS rows are x_row_stride bytes wide (with padding).
    // Thread decomposition uses GMEM width; LDS store uses padded stride.
    let gmem_row_bytes_x = cfg.tile_k * 2;  // actual data bytes per row
    let gmem_row_bytes_wt = cfg.tile_k * 2;
    // LDS row stride may not be power-of-2 (e.g., 36 = 32+4 pad)
    // Use shift when possible, multiply otherwise
    let xrs_is_pow2 = x_row_stride.is_power_of_two();
    let wrs_is_pow2 = wt_row_stride.is_power_of_two();
    let xrs_shift_val = if xrs_is_pow2 { x_row_stride.trailing_zeros() as u8 } else { 0 };
    let wrs_shift_val = if wrs_is_pow2 { wt_row_stride.trailing_zeros() as u8 } else { 0 };
    // Thread t loads from byte offset (t * 16) within the tile's GMEM footprint.
    // Using GMEM row width for decomposition:
    //   row_in_tile = (t * 16) / gmem_row_bytes
    //   col_byte    = (t * 16) % gmem_row_bytes
    let chunks_per_row_x = gmem_row_bytes_x / 16;  // 2 for k16, 4 for k32
    let x_cpr_shift = match chunks_per_row_x { 2 => 1, 4 => 2, 8 => 3, _ => panic!("cpr_x") };

    let x_row_in_tile = k.alloc_vreg();
    k.v_lshrrev_b32(x_row_in_tile, x_cpr_shift, tid);  // tid / chunks_per_row
    let x_col_chunk = k.alloc_vreg();
    k.v_and_b32_imm(x_col_chunk, tid, chunks_per_row_x - 1);  // tid % chunks_per_row

    // GMEM address: X_ptr + (tile_row * tile_m + x_row_in_tile) * K * 2 + k_byte_off + x_col_chunk * 16
    let x_abs_row = k.alloc_vreg();
    let s_xbase = k.alloc_sreg();
    k.s_lshl_b32(s_xbase, tile_row_s, tile_m_shift);
    k.v_mov_from_sgpr(x_abs_row, s_xbase);
    k.v_add_u32(x_abs_row, x_abs_row, x_row_in_tile);

    let x_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(x_row_byte, x_abs_row, k_vreg);
    k.v_lshlrev_b32(x_row_byte, 1, x_row_byte);
    // Add column offset: x_col_chunk * 16
    let x_col_byte = k.alloc_vreg();
    k.v_lshlrev_b32(x_col_byte, 4, x_col_chunk);  // col_chunk * 16
    k.v_add_u32(x_row_byte, x_row_byte, x_col_byte);

    let x_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(x_gmem_base, SReg(x_ptr.0));
    k.v_mov_from_sgpr(VReg(x_gmem_base.0 + 1), SReg(x_ptr.0 + 1));
    k.v_add_co(x_gmem_base, x_gmem_base, x_row_byte);
    k.v_add_co_ci(VReg(x_gmem_base.0 + 1), VReg(x_gmem_base.0 + 1));

    // X LDS store addr: row_in_tile * x_row_stride(padded) + col_chunk * 16
    let x_lds_off = k.alloc_vreg();
    if xrs_is_pow2 {
        k.v_lshlrev_b32(x_lds_off, xrs_shift_val, x_row_in_tile);
    } else {
        let s_xrs = k.alloc_sreg();
        k.s_mov_imm(s_xrs, x_row_stride as i32);
        let v_xrs = k.alloc_vreg();
        k.v_mov_from_sgpr(v_xrs, s_xrs);
        k.v_mul_lo_u32(x_lds_off, x_row_in_tile, v_xrs);
    }
    k.v_add_u32(x_lds_off, x_lds_off, x_col_byte);          // + col_chunk * 16

    // ── WT cooperative load address (COALESCED) ──
    let chunks_per_row_wt = gmem_row_bytes_wt / 16;  // use GMEM width, not LDS stride
    let wt_cpr_shift = match chunks_per_row_wt { 2 => 1, 4 => 2, 8 => 3, _ => panic!("cpr_wt") };

    let wt_row_in_tile = k.alloc_vreg();
    k.v_lshrrev_b32(wt_row_in_tile, wt_cpr_shift, tid);
    let wt_col_chunk = k.alloc_vreg();
    k.v_and_b32_imm(wt_col_chunk, tid, chunks_per_row_wt - 1);

    let wt_abs_row = k.alloc_vreg();
    k.v_mov_from_sgpr(wt_abs_row, base_n_s);
    k.v_add_u32(wt_abs_row, wt_abs_row, wt_row_in_tile);

    let wt_row_byte = k.alloc_vreg();
    k.v_mul_lo_u32(wt_row_byte, wt_abs_row, k_vreg);
    k.v_lshlrev_b32(wt_row_byte, 1, wt_row_byte);
    let wt_col_byte = k.alloc_vreg();
    k.v_lshlrev_b32(wt_col_byte, 4, wt_col_chunk);
    k.v_add_u32(wt_row_byte, wt_row_byte, wt_col_byte);

    let wt_gmem_base = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_from_sgpr(wt_gmem_base, SReg(wt_ptr.0));
    k.v_mov_from_sgpr(VReg(wt_gmem_base.0 + 1), SReg(wt_ptr.0 + 1));
    k.v_add_co(wt_gmem_base, wt_gmem_base, wt_row_byte);
    k.v_add_co_ci(VReg(wt_gmem_base.0 + 1), VReg(wt_gmem_base.0 + 1));

    // WT LDS store addr: lds_x + row_in_tile * wt_row_stride + col_chunk * 16
    let wt_lds_off = k.alloc_vreg();
    if wrs_is_pow2 {
        k.v_lshlrev_b32(wt_lds_off, wrs_shift_val, wt_row_in_tile);
    } else {
        let s_wrs = k.alloc_sreg();
        k.s_mov_imm(s_wrs, wt_row_stride as i32);
        let v_wrs = k.alloc_vreg();
        k.v_mov_from_sgpr(v_wrs, s_wrs);
        k.v_mul_lo_u32(wt_lds_off, wt_row_in_tile, v_wrs);
    }
    k.v_add_u32(wt_lds_off, wt_lds_off, wt_col_byte);
    k.push(Op::VAddU32 {
        dst: wt_lds_off, src0: Operand::VReg(wt_lds_off),
        src1: Operand::InlineInt(lds_x as i32),
    });

    // ── LDS read addresses for WMMA fragments ──
    // X frag[r]: (wave_id * rows_per_wave + r*16 + lane_row) * x_row_stride
    let lane_row_stride = k.alloc_vreg();
    if xrs_is_pow2 {
        k.v_lshlrev_b32(lane_row_stride, xrs_shift_val, lane_row);
    } else {
        let s_xrs2 = k.alloc_sreg();
        k.s_mov_imm(s_xrs2, x_row_stride as i32);
        let v_xrs2 = k.alloc_vreg();
        k.v_mov_from_sgpr(v_xrs2, s_xrs2);
        k.v_mul_lo_u32(lane_row_stride, lane_row, v_xrs2);
    }

    let s_wave_x_off = k.alloc_sreg();
    let s_wave_stride = k.alloc_sreg();
    k.s_mov_imm(s_wave_stride, (rows_per_wave * x_row_stride) as i32);
    k.s_mul_i32(s_wave_x_off, wave_id_s, s_wave_stride);

    let mut x_lds_reads = Vec::new();
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
        x_lds_reads.push(xr);
    }

    // WT: lane_row * wt_row_stride (tile offset as LDS_X + col*16*wt_row_stride)
    let wt_lds_read_base = k.alloc_vreg();
    if wrs_is_pow2 {
        k.v_lshlrev_b32(wt_lds_read_base, wrs_shift_val, lane_row);
    } else {
        let s_wrs2 = k.alloc_sreg();
        k.s_mov_imm(s_wrs2, wt_row_stride as i32);
        let v_wrs2 = k.alloc_vreg();
        k.v_mov_from_sgpr(v_wrs2, s_wrs2);
        k.v_mul_lo_u32(wt_lds_read_base, lane_row, v_wrs2);
    }

    // ── Temp VGPRs (pre-allocated, reused) ──
    let n_x_gmem_regs = x_loads_per_thread as usize;
    let n_wt_gmem_regs = wt_loads_per_thread as usize;
    let gmem_x: Vec<VReg> = (0..n_x_gmem_regs)
        .map(|_| k.alloc_vreg_array(4, Alignment::Align4))
        .collect();
    let gmem_wt: Vec<VReg> = (0..n_wt_gmem_regs)
        .map(|_| k.alloc_vreg_array(4, Alignment::Align4))
        .collect();

    // WMMA fragment registers
    let x_frags: Vec<VReg> = (0..n_row_blocks)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();
    let wt_frags: Vec<VReg> = (0..n_col_tiles)
        .map(|_| k.alloc_vreg_array(8, Alignment::Align8))
        .collect();

    // Pre-allocated temp addresses (reuse across all macro invocations!)
    let tmp_xa = k.alloc_vreg_array(2, Alignment::Align2);
    let tmp_wa = k.alloc_vreg_array(2, Alignment::Align2);
    let tmp_lds_x = k.alloc_vreg();
    let tmp_lds_w = k.alloc_vreg();
    let tmp_frag_addr = k.alloc_vreg();  // reused for all fragment reads

    // ── GMEM multi-pass strides (coalesced loading) ──
    let x_rows_per_pass = cfg.wg_size / chunks_per_row_x;
    let wt_rows_per_pass = cfg.wg_size / chunks_per_row_wt;
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

    // ── K-loop state (with split-K support) ──
    // k_byte_off starts at k_start_bytes (0 for non-split)
    let k_byte_off = k.alloc_vreg();
    k.v_mov_from_sgpr(k_byte_off, k_start_bytes_s);  // start at k_start_bytes
    let k_iter_s = k.alloc_sreg();
    k.s_mov_imm(k_iter_s, 0);
    let k_step = (cfg.tile_k * 2) as i32;

    // k_end_s already computed above from split-K logic

    let buf0_off = k.alloc_vreg();
    k.v_mov_imm(buf0_off, 0);
    let buf1_off = k.alloc_vreg();
    k.v_mov_imm(buf1_off, lds_buf as i32);

    // ══════════════════════════════════════════════════════════════════
    // HELPER CLOSURES (via macros to avoid borrow issues)
    // ══════════════════════════════════════════════════════════════════

    macro_rules! coop_gmem_load {
        ($k:expr, $koff:expr) => {{
            // X loads (coalesced, multi-pass)
            $k.v_mov(tmp_xa, x_gmem_base);
            $k.v_mov(VReg(tmp_xa.0 + 1), VReg(x_gmem_base.0 + 1));
            $k.v_add_co(tmp_xa, tmp_xa, $koff);
            $k.v_add_co_ci(VReg(tmp_xa.0 + 1), VReg(tmp_xa.0 + 1));
            for i in 0..n_x_gmem_regs {
                $k.global_load(gmem_x[i], tmp_xa, Width::B128, 0);
                if i + 1 < n_x_gmem_regs {
                    // Advance to next row block
                    $k.v_add_co(tmp_xa, tmp_xa, x_gmem_stride);
                    $k.v_add_co_ci(VReg(tmp_xa.0 + 1), VReg(tmp_xa.0 + 1));
                }
            }
            // WT loads (coalesced, multi-pass)
            $k.v_mov(tmp_wa, wt_gmem_base);
            $k.v_mov(VReg(tmp_wa.0 + 1), VReg(wt_gmem_base.0 + 1));
            $k.v_add_co(tmp_wa, tmp_wa, $koff);
            $k.v_add_co_ci(VReg(tmp_wa.0 + 1), VReg(tmp_wa.0 + 1));
            for i in 0..n_wt_gmem_regs {
                $k.global_load(gmem_wt[i], tmp_wa, Width::B128, 0);
                if i + 1 < n_wt_gmem_regs {
                    $k.v_add_co(tmp_wa, tmp_wa, wt_gmem_stride);
                    $k.v_add_co_ci(VReg(tmp_wa.0 + 1), VReg(tmp_wa.0 + 1));
                }
            }
        }};
    }

    macro_rules! coop_lds_store {
        ($k:expr, $buf_off:expr) => {{
            $k.v_add_u32(tmp_lds_x, x_lds_off, $buf_off);
            for i in 0..n_x_gmem_regs {
                $k.ds_store_b128(tmp_lds_x, gmem_x[i], 0);
                if i + 1 < n_x_gmem_regs {
                    $k.push(Op::VAddU32 {
                        dst: tmp_lds_x, src0: Operand::VReg(tmp_lds_x),
                        src1: Operand::InlineInt(x_lds_stride),
                    });
                }
            }
            $k.v_add_u32(tmp_lds_w, wt_lds_off, $buf_off);
            for i in 0..n_wt_gmem_regs {
                $k.ds_store_b128(tmp_lds_w, gmem_wt[i], 0);
                if i + 1 < n_wt_gmem_regs {
                    $k.push(Op::VAddU32 {
                        dst: tmp_lds_w, src0: Operand::VReg(tmp_lds_w),
                        src1: Operand::InlineInt(wt_lds_stride),
                    });
                }
            }
        }};
    }

    macro_rules! lds_read_and_wmma {
        ($k:expr, $buf_off:expr) => {{
            for ks in 0..cfg.k_sub_steps() as usize {
                let k_byte_within = (ks * 32) as u16;

                if n_row_blocks >= 2 {
                    // ── INTERLEAVED SCHEDULE with refined waitcnt + WMMA ILP ──
                    //
                    // Load order (C = 2 + n_col_tiles*2 total ds_load_b128):
                    //   [0,1]: x_frags[0] (lo, hi)
                    //   [2,3]: wt_frags[0] (lo, hi)
                    //   [4,5]: wt_frags[1] ...
                    //   [2c+2, 2c+3]: wt_frags[c]
                    //
                    // LGKM counter is FIFO: lgkmcnt(N) means "≤N outstanding"
                    // → first (C-N) loads completed.

                    // ── Issue ALL Phase-1 ds_loads ──
                    $k.v_add_u32(tmp_frag_addr, x_lds_reads[0], $buf_off);
                    $k.ds_load_b128(x_frags[0], tmp_frag_addr, k_byte_within);
                    $k.ds_load_b128(VReg(x_frags[0].0 + 4), tmp_frag_addr, k_byte_within + 16);
                    for c in 0..n_col_tiles {
                        $k.v_add_u32(tmp_frag_addr, wt_lds_read_base, $buf_off);
                        let base_off: u16 = (lds_x + (c as u32) * 16 * wt_row_stride) as u16;
                        $k.ds_load_b128(wt_frags[c], tmp_frag_addr, base_off + k_byte_within);
                        $k.ds_load_b128(VReg(wt_frags[c].0 + 4), tmp_frag_addr, base_off + k_byte_within + 16);
                    }
                    // Total in-flight: C = 2 + n_col_tiles * 2

                    // ── Graduated waitcnt: dispatch WMMA as operands become ready ──
                    // For WMMA x_frags[0] × wt_frags[c], we need loads [0..2c+3].
                    // lgkmcnt(C - (2c+4)) ensures those loads are done.
                    let total_loads = (2 + n_col_tiles * 2) as u8;
                    for c in 0..n_col_tiles {
                        let loads_needed = (2 * c + 4) as u8; // x[0](2) + wt[0..c](2*(c+1))
                        let remaining = total_loads.saturating_sub(loads_needed);
                        $k.wait_lgkmcnt(remaining);
                        let a_idx = 0 * n_col_tiles + c;
                        $k.wmma_bf16_f32(acc[a_idx], x_frags[0], wt_frags[c], acc[a_idx]);
                    }

                    // ── Phase 2: load X[1], then WMMA X[1]×WT[all] ──
                    // X[1] loads overlap with last WMMA(s) above (WMMA ~32 cyc each)
                    $k.v_add_u32(tmp_frag_addr, x_lds_reads[1], $buf_off);
                    $k.ds_load_b128(x_frags[1], tmp_frag_addr, k_byte_within);
                    $k.ds_load_b128(VReg(x_frags[1].0 + 4), tmp_frag_addr, k_byte_within + 16);
                    // Wait for X[1] — only 2 loads in flight
                    $k.wait_lgkmcnt(0);
                    // WMMA X[1]×WT — wt_frags still valid from Phase 1
                    for c in 0..n_col_tiles {
                        let a_idx = 1 * n_col_tiles + c;
                        $k.wmma_bf16_f32(acc[a_idx], x_frags[1], wt_frags[c], acc[a_idx]);
                    }

                    // ── Phase 3: remaining row blocks (r >= 2) ──
                    for r in 2..n_row_blocks {
                        // Prefetch X[r] while computing WMMA X[r-1] (already done above for r=1)
                        $k.v_add_u32(tmp_frag_addr, x_lds_reads[r], $buf_off);
                        $k.ds_load_b128(x_frags[r], tmp_frag_addr, k_byte_within);
                        $k.ds_load_b128(VReg(x_frags[r].0 + 4), tmp_frag_addr, k_byte_within + 16);
                        $k.wait_lgkmcnt(0);
                        for c in 0..n_col_tiles {
                            let a_idx = r * n_col_tiles + c;
                            $k.wmma_bf16_f32(acc[a_idx], x_frags[r], wt_frags[c], acc[a_idx]);
                        }
                    }
                } else {
                    // ── SIMPLE SCHEDULE (n_row_blocks == 1) ──
                    // Refined: issue x_frag[0] first, then wt_frags with graduated wait
                    $k.v_add_u32(tmp_frag_addr, x_lds_reads[0], $buf_off);
                    $k.ds_load_b128(x_frags[0], tmp_frag_addr, k_byte_within);
                    $k.ds_load_b128(VReg(x_frags[0].0 + 4), tmp_frag_addr, k_byte_within + 16);
                    // Issue all wt_frag loads
                    for c in 0..n_col_tiles {
                        $k.v_add_u32(tmp_frag_addr, wt_lds_read_base, $buf_off);
                        let base_off: u16 = (lds_x + (c as u32) * 16 * wt_row_stride) as u16;
                        $k.ds_load_b128(wt_frags[c], tmp_frag_addr, base_off + k_byte_within);
                        $k.ds_load_b128(VReg(wt_frags[c].0 + 4), tmp_frag_addr, base_off + k_byte_within + 16);
                    }
                    // Graduated waitcnt: dispatch WMMA as each wt_frag pair completes
                    let total_loads = (2 + n_col_tiles * 2) as u8;
                    for c in 0..n_col_tiles {
                        let loads_needed = (2 * c + 4) as u8;
                        let remaining = total_loads.saturating_sub(loads_needed);
                        $k.wait_lgkmcnt(remaining);
                        $k.wmma_bf16_f32(acc[c], x_frags[0], wt_frags[c], acc[c]);
                    }
                }
            }
        }};
    }



    // ══════════════════════════════════════════════════════════════════
    // ══════════════════════════════════════════════════════════════════
    // PROLOGUE
    // ══════════════════════════════════════════════════════════════════
    coop_gmem_load!(k, k_byte_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf0_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, cfg.tile_k as i32);

    // ══════════════════════════════════════════════════════════════════
    // MAIN LOOP
    // ══════════════════════════════════════════════════════════════════
    let loop_label = k.make_label("ggen_loop");
    k.label(&loop_label);
    k.s_cmp_ge_u32(k_iter_s, k_end_s);
    let epilog_a = k.make_label("ggen_ea");
    k.branch_scc1(&epilog_a);

    // Phase A: prefetch→buf1, compute buf0
    coop_gmem_load!(k, k_byte_off);
    lds_read_and_wmma!(k, buf0_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf1_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, cfg.tile_k as i32);

    k.s_cmp_ge_u32(k_iter_s, k_end_s);
    let epilog_b = k.make_label("ggen_eb");
    k.branch_scc1(&epilog_b);

    // Phase B: prefetch→buf0, compute buf1
    coop_gmem_load!(k, k_byte_off);
    lds_read_and_wmma!(k, buf1_off);
    k.wait_vmcnt(0);
    coop_lds_store!(k, buf0_off);
    k.wait_lgkmcnt(0);
    k.s_barrier();
    k.push(Op::VAddU32 { dst: k_byte_off, src0: Operand::VReg(k_byte_off), src1: Operand::InlineInt(k_step) });
    k.s_add_u32(k_iter_s, k_iter_s, cfg.tile_k as i32);
    k.s_cmp_lt_u32(k_iter_s, k_end_s);
    k.branch_scc1(&loop_label);

    // ══════════════════════════════════════════════════════════════════
    // EPILOGUES
    // ══════════════════════════════════════════════════════════════════
    k.label(&epilog_a);
    lds_read_and_wmma!(k, buf0_off);
    let store_label = k.make_label("ggen_store");
    k.s_mov_imm(k_iter_s, 0);
    k.s_cmp_eq_u32_imm(k_iter_s, 0);
    k.branch_scc1(&store_label);

    k.label(&epilog_b);
    lds_read_and_wmma!(k, buf1_off);

    // ══════════════════════════════════════════════════════════════════
    // STORE PHASE: f32 → global memory
    // ══════════════════════════════════════════════════════════════════
    k.label(&store_label);

    let lane_half = k.alloc_vreg();
    k.v_lshrrev_b32(lane_half, 4, lane_id);
    let n_vreg = k.alloc_vreg();
    k.v_mov_from_sgpr(n_vreg, SReg(n_dim.0));
    let row_stride = k.alloc_vreg();
    k.v_lshlrev_b32(row_stride, 3, n_vreg);  // N * 8 (2 rows × 4 bytes)

    let col_base_v = k.alloc_vreg();
    k.v_mov_from_sgpr(col_base_v, base_n_s);
    k.v_add_u32(col_base_v, col_base_v, lane_row);
    let col_bytes = k.alloc_vreg();
    k.v_lshlrev_b32(col_bytes, 2, col_base_v);  // col * 4 bytes

    for r in 0..n_row_blocks {
        let base_row_v = k.alloc_vreg();
        k.v_mov_from_sgpr(base_row_v, s_row_bases[r]);
        k.v_add_u32(base_row_v, base_row_v, lane_half);

        let row_bytes = k.alloc_vreg();
        k.v_mul_lo_u32(row_bytes, base_row_v, n_vreg);
        k.v_lshlrev_b32(row_bytes, 2, row_bytes);

        // Compute row_block y_base = y_ptr + y_offset + row_bytes + col_bytes (once per row_block)
        let y_base = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(y_base, SReg(y_ptr.0));
        k.v_mov_from_sgpr(VReg(y_base.0 + 1), SReg(y_ptr.0 + 1));
        {
            let v_yoff = k.alloc_vreg();
            k.v_mov_from_sgpr(v_yoff, y_offset_s);
            k.v_add_co(y_base, y_base, v_yoff);
            k.v_add_co_ci(VReg(y_base.0 + 1), VReg(y_base.0 + 1));
        }
        k.v_add_co(y_base, y_base, row_bytes);
        k.v_add_co_ci(VReg(y_base.0 + 1), VReg(y_base.0 + 1));
        k.v_add_u32(y_base, y_base, col_bytes);

        for c in 0..n_col_tiles {
            // ── Epilogue: load bias[col] for this col_tile ──
            // bias layout: bias[N] f32 row-vector, index = base_n + c*16 + lane_row
            // One f32 per lane, broadcast across all 8 acc rows.
            let bias_val = if cfg.epilogue != EpilogueOp::StoreF32 {
                if let Some(bp) = bias_ptr {
                    let bv = k.alloc_vreg();
                    let bias_addr = k.alloc_vreg_array(2, Alignment::Align2);
                    k.v_mov_from_sgpr(bias_addr, SReg(bp.0));
                    k.v_mov_from_sgpr(VReg(bias_addr.0 + 1), SReg(bp.0 + 1));
                    // bias byte offset = (base_n + c*16 + lane_row) * 4
                    let bias_col = k.alloc_vreg();
                    k.v_mov_from_sgpr(bias_col, base_n_s);
                    k.v_add_u32(bias_col, bias_col, lane_row);
                    if c > 0 {
                        k.push(Op::VAddU32 {
                            dst: bias_col, src0: Operand::VReg(bias_col),
                            src1: Operand::InlineInt((c * 16) as i32),
                        });
                    }
                    let bias_byte_off = k.alloc_vreg();
                    k.v_lshlrev_b32(bias_byte_off, 2, bias_col); // * 4
                    k.v_add_co(bias_addr, bias_addr, bias_byte_off);
                    k.v_add_co_ci(VReg(bias_addr.0 + 1), VReg(bias_addr.0 + 1));
                    k.global_load(bv, bias_addr, Width::B32, 0);
                    k.wait_vmcnt(0);
                    Some(bv)
                } else { None }
            } else { None };

            // y_addr = y_base + c*64 (16 cols × 4 bytes)
            let y_addr = k.alloc_vreg_array(2, Alignment::Align2);
            k.v_mov(y_addr, y_base);
            k.v_mov(VReg(y_addr.0 + 1), VReg(y_base.0 + 1));
            if c > 0 {
                k.push(Op::VAddU32 {
                    dst: y_addr, src0: Operand::VReg(y_addr),
                    src1: Operand::InlineInt((c * 64) as i32),
                });
            }
            let a_idx = r * n_col_tiles + c;
            for vk in 0..8u32 {
                let acc_reg = VReg(acc[a_idx].0 + vk);

                // Apply epilogue: bias add + optional relu
                if let Some(bv) = bias_val {
                    k.push(Op::VAddF32 {
                        dst: acc_reg,
                        src0: Operand::VReg(acc_reg),
                        src1: Operand::VReg(bv),
                    });
                }
                if cfg.epilogue == EpilogueOp::BiasAddReluStoreF32 {
                    k.push(Op::VMaxF32 {
                        dst: acc_reg,
                        src0: Operand::VReg(acc_reg),
                        src1: Operand::InlineFloat(0.0),
                    });
                }

                k.global_store(y_addr, acc_reg, Width::B32, 0);
                if vk < 7 {
                    k.v_add_co(y_addr, y_addr, row_stride);
                    k.v_add_co_ci(VReg(y_addr.0 + 1), VReg(y_addr.0 + 1));
                }
            }
        }
    }

    k.wait_vscnt(0);
    k.endpgm();
    k
}

// ============================================================================
// Direct (no-LDS) Generator — for reference/small matrices
// ============================================================================

fn generate_direct(cfg: &GemmConfig) -> T0Kernel {
    // Delegate to the LDS version with single-buffer for now
    // TODO: implement zero-LDS variant
    let mut lds_cfg = cfg.clone();
    lds_cfg.use_lds = true;
    lds_cfg.double_buffer = true;
    generate_lds_db(&lds_cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimated_vgprs_respects_gfx1201_frag_sizes() {
        let cfg = GemmConfig::tile_128x64_k16();
        assert_eq!(cfg.estimated_vgprs_for_target(Target::GFX1100), 265);
        assert_eq!(cfg.estimated_vgprs_for_target(Target::GFX1201), 233);
    }

    #[test]
    fn test_feasibility_differs_by_target() {
        let cfg = GemmConfig::tile_128x64_k16();
        assert!(!cfg.is_feasible_for_target(Target::GFX1100));
        assert!(cfg.is_feasible_for_target(Target::GFX1201));
    }
}
