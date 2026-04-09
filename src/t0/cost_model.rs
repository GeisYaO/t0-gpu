//! GFX1100 Cost Model for Auto-Scheduling
//!
//! Models hardware constraints of AMD RX 7900 XTX (RDNA3, GFX1100)
//! and provides cost estimation for GEMM tile parameter selection.
//!
//! # Usage
//! ```rust
//! let best = auto_schedule_gemm(4096, 4096, 512, DataFormat::BF16);
//! // Returns optimal tile_m, tile_n, tile_k, workgroup_size
//! ```

// ============================================================================
// GFX1100 Hardware Limits
// ============================================================================

/// Hardware specifications for GFX1100 (Navi 31, RDNA3).
#[derive(Clone, Debug)]
pub struct GFX1100Limits {
    /// VGPRs per SIMD unit (Wave32 mode)
    pub max_vgprs: u32,
    /// SGPRs per wavefront  
    pub max_sgprs: u32,
    /// LDS per workgroup (bytes)
    pub lds_per_wg: u32,
    /// Total LDS per CU (bytes, shared by all WGs on CU)
    pub lds_per_cu: u32,
    /// Number of CUs on die
    pub n_cus: u32,
    /// WMMA throughput: VALU-normalized cycles per v_wmma instruction
    /// (empirical: ~43 shader cycles / 18 ≈ 2.4, rounded to 4)
    pub wmma_cycles: u32,
    /// WMMA tile size: always 16×16×16 on GFX11
    pub wmma_mn: u32,
    pub wmma_k: u32,
    /// Peak VMEM bandwidth (GB/s)
    pub vmem_bandwidth_gbps: f64,
    /// Peak LDS bandwidth (GB/s)
    pub lds_bandwidth_gbps: f64,
    /// GPU clock (GHz, for cycle estimation)
    pub clock_ghz: f64,
    /// WGP mode: each WGP = 2 CUs sharing LDS.
    /// In WGP mode, a workgroup can use both SIMDs in the WGP,
    /// effectively doubling the concurrent wave capacity.
    pub wgp_mode: bool,
    /// SIMDs per CU (RDNA3 = 2)
    pub simds_per_cu: u32,
    /// L2 cache size in bytes (RX 7900 XTX = 6 MB)
    pub l2_cache_bytes: u64,
}

impl Default for GFX1100Limits {
    fn default() -> Self {
        GFX1100Limits {
            max_vgprs: 256,
            max_sgprs: 106,
            lds_per_wg: 65536,    // 64 KB
            lds_per_cu: 131072,   // 128 KB (shared across WGP in WGP mode)
            n_cus: 96,
            wmma_cycles: 4,       // probe-calibrated: ~36 shader cycles ≈ 3.4→4 VALU-norm
            wmma_mn: 16,
            wmma_k: 16,
            vmem_bandwidth_gbps: 960.0,
            lds_bandwidth_gbps: 3700.0,
            clock_ghz: 2.5,
            wgp_mode: true,       // enable WGP mode by default on GFX1100
            simds_per_cu: 2,
            l2_cache_bytes: 6 * 1024 * 1024, // 6 MB L2 on RX 7900 XTX
        }
    }
}

// ============================================================================
// Data format
// ============================================================================

/// Data format for GEMM operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataFormat {
    BF16,   // 2 bytes, WMMA native
    F16,    // 2 bytes, WMMA native
    F32,    // 4 bytes, no WMMA (scalar mul-add)
}

impl DataFormat {
    pub fn bytes(&self) -> u32 {
        match self {
            DataFormat::BF16 | DataFormat::F16 => 2,
            DataFormat::F32 => 4,
        }
    }

    pub fn has_wmma(&self) -> bool {
        matches!(self, DataFormat::BF16 | DataFormat::F16)
    }
}

// ============================================================================
// Tile configuration
// ============================================================================

/// A candidate tile configuration for GEMM.
#[derive(Clone, Debug)]
pub struct TileConfig {
    pub tile_m: u32,        // rows per workgroup
    pub tile_n: u32,        // cols per workgroup  
    pub tile_k: u32,        // K-dimension per loop iteration
    pub waves_per_wg: u32,  // waves in workgroup (1, 2, or 4)
    pub use_lds: bool,      // whether to use LDS for tile staging
    pub split_k: u32,       // split-K factor (1 = no split, 2/4/8/16)
    pub wgp_mode: bool,     // WGP mode (2 CUs share LDS)
    pub swap_grid: bool,    // true=N-on-X (L2 friendly), false=M-on-X (rectangular)
    pub lds_pad: u32,       // LDS row padding in bytes (0/4/8, eliminates bank conflicts)
}

impl TileConfig {
    /// Workgroup size in threads.
    pub fn wg_threads(&self) -> u32 {
        self.waves_per_wg * 32 // Wave32
    }

    /// Number of WMMA tiles in N direction.
    pub fn n_wmma_tiles(&self) -> u32 {
        self.tile_n / 16
    }

    /// Number of WMMA tiles in M direction (per wave).
    pub fn m_wmma_per_wave(&self) -> u32 {
        (self.tile_m / self.waves_per_wg) / 16
    }

    /// Short name for display.
    pub fn name(&self) -> String {
        let sk = if self.split_k > 1 { format!("_sk{}", self.split_k) } else { String::new() };
        let wgp = if self.wgp_mode { "_wgp" } else { "" };
        let mg = if !self.swap_grid { "_mg" } else { "" };
        let pad = if self.lds_pad > 0 { format!("_pad{}", self.lds_pad) } else { String::new() };
        format!("{}x{}_k{}{}{}{}{}", self.tile_m, self.tile_n, self.tile_k, sk, mg, wgp, pad)
    }
}

/// Cost estimation result.
#[derive(Clone, Debug)]
pub struct TileCost {
    pub config: TileConfig,
    /// Estimated VGPRs needed per wave
    pub vgprs: u32,
    /// Estimated SGPRs needed
    pub sgprs: u32,
    /// LDS bytes needed per workgroup
    pub lds_bytes: u32,
    /// Occupancy: waves per SIMD unit
    pub occupancy: u32,
    /// Number of workgroups to cover the full M×N output
    pub n_wgs: u32,
    /// Whether this config fits in hardware limits
    pub feasible: bool,
    /// Performance score (higher is better): TFLOPS estimate
    pub score: f64,
    /// Bottleneck description
    pub bottleneck: &'static str,
}

// ============================================================================
// Cost estimation
// ============================================================================

/// Estimate cost for a given tile configuration on GFX1100.
pub fn estimate_tile_cost(
    config: &TileConfig,
    m: u32, n: u32, k: u32,
    fmt: DataFormat,
    hw: &GFX1100Limits,
) -> TileCost {
    let tile_m = config.tile_m;
    let tile_n = config.tile_n;
    let tile_k = config.tile_k;

    // ── VGPR estimation ──
    // Accumulators: n_wmma_tiles × m_wmma_per_wave × 8 VGPRs (f32 accumulator)
    let n_wmma = config.n_wmma_tiles();
    let m_wmma = config.m_wmma_per_wave();
    let acc_vgprs = n_wmma * m_wmma * 8;

    // Fragments: A fragment (8 VGPRs) + B fragments (n_wmma × 8 VGPRs)
    let frag_a_vgprs = m_wmma * 8;
    let frag_b_vgprs = n_wmma * 8;

    // Address registers: ~6 (x_base, wt_base, k_byte_off, etc.)
    let addr_vgprs = 6u32;

    // Thread ID / lane decomposition: ~4
    let misc_vgprs = 4u32;

    let vgprs = acc_vgprs + frag_a_vgprs + frag_b_vgprs + addr_vgprs + misc_vgprs;

    // ── SGPR estimation ──
    // Kernargs (5 pointers/dims × 2) + TGID (3) + loop counter + misc
    let sgprs = 5u32 + 10 + 5; // ~20 SGPRs typical

    // ── LDS estimation (with bank conflict padding) ──
    let lds_pad = config.lds_pad;
    let lds_bytes = if config.use_lds {
        // A tile: tile_m rows × (tile_k × elem_bytes + lds_pad) per row
        // B tile: tile_n rows × (tile_k × elem_bytes + lds_pad) per row
        let a_bytes = tile_m * (tile_k * fmt.bytes() + lds_pad);
        let b_bytes = tile_n * (tile_k * fmt.bytes() + lds_pad);
        // Double-buffered: × 2
        (a_bytes + b_bytes) * 2
    } else {
        0u32
    };

    // ── Feasibility check ──
    let feasible = vgprs <= hw.max_vgprs
        && sgprs <= hw.max_sgprs
        && lds_bytes <= hw.lds_per_wg;

    // ── Occupancy ──
    // RDNA3 Wave32: 256 VGPRs / SIMD, max 16 waves per SIMD
    let occ_per_simd = if vgprs == 0 { 0 } else {
        let by_vgpr = if vgprs <= 64 { 16u32 }
            else if vgprs <= 96 { 10 }
            else if vgprs <= 128 { 8 }
            else if vgprs <= 192 { 4 }
            else if vgprs <= 256 { 2 }
            else { 0 };
        let by_lds = if lds_bytes == 0 { 16 }
            else { (hw.lds_per_cu / lds_bytes.max(1)).min(16) };
        by_vgpr.min(by_lds)
    };

    // WGP mode: per-config (not hw-global).
    // Each WGP = 2 CUs with simds_per_cu SIMDs each.
    let simds_per_wgp = if config.wgp_mode { hw.simds_per_cu * 2 } else { hw.simds_per_cu };
    let max_waves_per_wgp = occ_per_simd * simds_per_wgp;

    let occupancy = occ_per_simd;

    // Hard constraint: WG's waves must fit in one WGP's capacity.
    let feasible = feasible && config.waves_per_wg <= max_waves_per_wgp;

    // ── Workgroup count (with split-K) ──
    let n_wgs_m = (m + tile_m - 1) / tile_m;
    let n_wgs_n = (n + tile_n - 1) / tile_n;
    let n_wgs = n_wgs_m * n_wgs_n * config.split_k; // split-K multiplies WGs
    let k_per_chunk = k / config.split_k; // each WG processes K/split_k elements

    // Feasibility: K must be divisible by split_k * tile_k
    let feasible = feasible && (k % (config.split_k * tile_k) == 0);

    // ── Performance model (with latency_model integration) ──
    let k_iters = (k_per_chunk + tile_k - 1) / tile_k; // K-iters per WG (split-K aware)

    // Output tiles per WG = (tile_m/16) × (tile_n/16)
    let output_wmma_tiles = (tile_m / 16) * (tile_n / 16);
    let total_wmma_per_wg = output_wmma_tiles * k_iters;

    // Total FLOPs = M × N × K × 2 (fixed for problem size)
    let total_flops: f64 = 2.0 * m as f64 * n as f64 * k as f64;

    // ── Compute cycle estimation ──
    // Try instruction-level analysis first (accurate but requires code generation).
    // Fall back to macro-level model if kernel generation fails.
    let compute_cycles_per_wg: u32;
    
    // Use a thread-local cache to avoid regenerating kernels for the same config
    let kloop = analyze_kloop(config);
    
    if let Some(ref analysis) = kloop {
        // Instruction-level model: use refined cycles from actual K-loop analysis.
        // cycles_per_iter accounts for WMMA pipeline depth and LDS/VALU overlap.
        let cycles_per_iter = analysis.cycles_per_iter;
        // The K-loop body contains TWO phases (double-buffer ping-pong),
        // so each "iteration" of the outer loop covers 2 K-tiles.
        // k_iters counts individual K-tiles, so adjust:
        let effective_iters = (k_iters + 1) / 2; // outer loop iterations
        compute_cycles_per_wg = (cycles_per_iter * effective_iters as f64) as u32;
    } else {
        // Fallback: macro-level model (original formula)
        let wmma_per_wave = total_wmma_per_wg / config.waves_per_wg;
        let wmma_cycles = wmma_per_wave * hw.wmma_cycles;
        let loads_per_wave_per_iter = (tile_m / config.waves_per_wg + tile_n + 31) / 32;
        let overhead_per_iter = loads_per_wave_per_iter + 6u32;
        let overhead_cycles = overhead_per_iter * k_iters;
        compute_cycles_per_wg = wmma_cycles + overhead_cycles / 2;
    }

    // Effective scheduling units
    let n_scheduling_units = if hw.wgp_mode { hw.n_cus / 2 } else { hw.n_cus };
    let wgs_per_unit = max_waves_per_wgp / config.waves_per_wg;
    let total_concurrent_wgs = n_scheduling_units * wgs_per_unit;
    let compute_time_cycles = if total_concurrent_wgs > 0 {
        let batches = (n_wgs as f64 / total_concurrent_wgs as f64).ceil();
        compute_cycles_per_wg as f64 * batches
    } else {
        f64::MAX
    };

    // ── Memory bound: Roofline Model ──
    //
    // Per-WG arithmetic intensity (DRAM-level):
    //   AI = 2 × tile_m × tile_n / ((tile_m + tile_n) × bytes)
    //   This tells us the FLOPs/byte ratio at the DRAM level.
    //
    // Note: AI is independent of K — this is the data reuse property of tiling.
    // The roofline predicts a LOWER BOUND on performance from DRAM BW alone.
    // In practice, L2 cache provides additional reuse that we model as
    // effective_bw = DRAM_bw × (1 + l2_boost_factor).
    //
    // GFX1100 L2: 6 MB across 6 shader arrays. For large GEMMs, L2 provides
    // ~50-70% effective bandwidth boost due to tile-strip sharing.
    let ai_wg = 2.0 * tile_m as f64 * tile_n as f64
        / ((tile_m + tile_n) as f64 * fmt.bytes() as f64);
    
    // Occupancy-dependent DRAM BW efficiency + L2 cache boost
    let dram_bw_eff = match occupancy {
        0 => 0.0,
        1 => 0.55,
        2 => 0.75,
        3..=4 => 0.85,
        5..=8 => 0.92,
        _ => 0.95,
    };
    
    // L2 boost: larger tiles and higher occupancy get more L2 reuse.
    // Empirically, L2 can boost effective BW by 50-80% for large GEMMs on GFX1100.
    // Model: l2_boost increases with tile area (more data sharing per WG)
    let tile_area = tile_m * tile_n;
    let l2_boost = if tile_area >= 128 * 128 {
        0.7  // large tiles: substantial L2 reuse
    } else if tile_area >= 64 * 128 {
        0.5  // medium tiles
    } else if tile_area >= 64 * 64 {
        0.3  // small tiles
    } else {
        0.1  // tiny tiles: minimal reuse
    };
    
    let effective_bw = hw.vmem_bandwidth_gbps * dram_bw_eff * (1.0 + l2_boost); // GB/s
    
    // Roofline ceiling: achievable TFLOPS from memory bandwidth
    let mem_tflops = ai_wg * effective_bw / 1000.0;
    
    // Peak compute ceiling (from WMMA throughput)
    // 96 CUs × 2 SIMDs × 1 WMMA per 4 cycles × 8192 FLOPs per WMMA × 2.5 GHz / 1e12
    // = 96 × 2 × 2.5e9 / 4 × 8192 / 1e12 ≈ 98.3 TFLOPS theoretical
    // Published peak: 123 TFLOPS (AMD spec)
    let peak_tflops = 123.0;
    
    // Also account for compute time from K-loop analysis
    let compute_tflops = if compute_time_cycles > 0.0 {
        total_flops / (compute_time_cycles / (hw.clock_ghz * 1e9)) / 1e12
    } else {
        peak_tflops
    };
    
    // Effective TFLOPS = min(compute, memory roofline)
    let effective_tflops = compute_tflops.min(mem_tflops).min(peak_tflops);
    
    // Convert back to time for bottleneck analysis
    let time_sec = if effective_tflops > 0.0 {
        total_flops / (effective_tflops * 1e12)
    } else {
        f64::MAX
    };
    let _mem_time_cycles = time_sec * hw.clock_ghz * 1e9;

    // Bottleneck: which ceiling is binding?
    let bottleneck = if effective_tflops >= peak_tflops - 0.1 {
        "peak"
    } else if mem_tflops < compute_tflops {
        "memory"
    } else {
        "compute"
    };

    // Score = predicted achievable TFLOPS (from roofline)
    let score = if feasible { effective_tflops } else { 0.0 };

    TileCost {
        config: config.clone(),
        vgprs,
        sgprs,
        lds_bytes,
        occupancy,
        n_wgs,
        feasible,
        score,
        bottleneck,
    }
}

// ============================================================================
// Auto-scheduling: exhaustive search
// ============================================================================

/// Auto-schedule GEMM: search tile space and return the best configuration.
///
/// Searches over:
/// - tile_m ∈ {32, 64, 128}
/// - tile_n ∈ {64, 128}
/// - tile_k ∈ {16, 32, 48, 64}
/// - waves_per_wg ∈ {2, 4, 8}
/// - split_k ∈ {1, 2, 4, 8}
/// - wgp_mode ∈ {false, true}
/// - swap_grid ∈ {false, true}
///
/// Returns ordered list of feasible configs, best first.
pub fn auto_schedule_gemm(
    m: u32, n: u32, k: u32,
    fmt: DataFormat,
) -> Vec<TileCost> {
    let hw = GFX1100Limits::default();

    let tile_m_candidates = [32u32, 64, 128];
    let tile_n_candidates = [64u32, 128]; // 128 enables larger output tiles (needs tile_ir path)
    let tile_k_candidates = [16u32, 32, 48, 64]; // k64 shows +5-60% gains vs k32 in benchmarks
    let waves_candidates = [2u32, 4, 8];
    let split_k_candidates = [1u32, 2, 4, 8];

    let mut results = Vec::new();
    let lds_pad_candidates = [0u32, 4, 8];

    for &tile_m in &tile_m_candidates {
        for &tile_n in &tile_n_candidates {
            for &tile_k in &tile_k_candidates {
                for &waves in &waves_candidates {
                    // Constraint: tile_m must be divisible by waves × 16 (WMMA rows)
                    if tile_m < waves * 16 { continue; }
                    if tile_m % (waves * 16) != 0 { continue; }
                    if tile_n % 16 != 0 { continue; }
                    if fmt.has_wmma() && tile_k < hw.wmma_k { continue; }

                    for &sk in &split_k_candidates {
                        // K must be divisible by split_k * tile_k
                        if k % (sk * tile_k) != 0 { continue; }
                        if sk > k / tile_k { continue; }

                        for &wgp in &[false, true] {
                            for &swap in &[true, false] {
                                for &pad in &lds_pad_candidates {
                                    let config = TileConfig {
                                        tile_m, tile_n, tile_k,
                                        waves_per_wg: waves,
                                        use_lds: true,
                                        split_k: sk,
                                        wgp_mode: wgp,
                                        swap_grid: swap,
                                        lds_pad: pad,
                                    };
                                    let cost = estimate_tile_cost(&config, m, n, k, fmt, &hw);
                                    if cost.feasible {
                                        // Verify final GemmConfig fits in 256 VGPRs
                                        // (to_gemm_config may split tile_n into multi-pass)
                                        let gc = config.to_gemm_config();
                                        if !gc.is_feasible() { continue; }
                                        results.push(cost);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort by score descending (best first)
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Get the single best GEMM tile configuration.
/// Returns None if no feasible configuration exists.
pub fn best_gemm_config(m: u32, n: u32, k: u32, fmt: DataFormat) -> Option<TileCost> {
    auto_schedule_gemm(m, n, k, fmt).into_iter().next()
}

/// Print a comparison table of top N tile configurations.
pub fn print_schedule_report(m: u32, n: u32, k: u32, fmt: DataFormat, top_n: usize) {
    let results = auto_schedule_gemm(m, n, k, fmt);

    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Auto-schedule GEMM: M={}, N={}, K={}, {:?}{}║",
        m, n, k, fmt, " ".repeat(64usize.saturating_sub(48 + format!("{m}{n}{k}").len())));
    eprintln!("╠══════════════════════════════════════════════════════════════════╣");
    eprintln!("║ {:>3} {:>3} {:>3} {:>4}  {:>4} {:>4} {:>5} {:>4} {:>6} {:>8} ║",
        "tM", "tN", "tK", "WPG", "VGPR", "LDS", "Occ", "WGs", "Score", "Bound");
    eprintln!("╟──────────────────────────────────────────────────────────────────╢");

    for (i, cost) in results.iter().take(top_n).enumerate() {
        let c = &cost.config;
        eprintln!("║ {:>3} {:>3} {:>3} {:>3}w sk{:<2} {:>4} {:>4} {:>4}w {:>5} {:>5.1}T {:>8} ║{}",
            c.tile_m, c.tile_n, c.tile_k, c.waves_per_wg, c.split_k,
            cost.vgprs,
            if cost.lds_bytes > 0 { format!("{}K", cost.lds_bytes / 1024) } else { "-".into() },
            cost.occupancy, cost.n_wgs,
            cost.score, cost.bottleneck,
            if i == 0 { format!(" ← BEST ({})", c.name()) } else { String::new() });
    }

    eprintln!("╚══════════════════════════════════════════════════════════════════╝");
    eprintln!("Searched {} feasible configs", results.len());
}

// ============================================================================
// TileConfig → GemmConfig conversion (Autotune bridge)
// ============================================================================

impl TileConfig {
    /// Convert cost-model TileConfig to gemm_gen GemmConfig.
    ///
    /// This bridges the gap between the cost_model exhaustive search and
    /// the gemm_gen code generator.
    pub fn to_gemm_config(&self) -> super::gemm_gen::GemmConfig {
        let wg_size = self.waves_per_wg * 32;
        // When tile_n > 64, use multi-pass (n_col_passes=2) to stay within
        // GFX1100's 256-VGPR limit. Single-pass tile_n=128 needs ~273 VGPRs
        // (acc=128 + wt_frag=64 + x_frag=16 + gmem=16 + temps=49).
        // Multi-pass: tile_n=64 per pass → acc=64, wt_frag=32 → ~177 VGPRs.
        let (effective_tile_n, n_col_passes) = if self.tile_n > 64 {
            (64, (self.tile_n / 64) as u32)
        } else {
            (self.tile_n, 1)
        };
        super::gemm_gen::GemmConfig {
            tile_m: self.tile_m,
            tile_n: effective_tile_n,
            tile_k: self.tile_k,
            wg_size,
            use_lds: self.use_lds,
            double_buffer: self.use_lds,  // always double-buffer when using LDS
            split_k: if self.split_k > 1 { Some(self.split_k) } else { None },
            lds_pad: self.lds_pad,
            n_col_passes,
            swap_grid: self.swap_grid,
            wgp_mode: self.wgp_mode,
            transpose: super::gemm_gen::GemmTranspose::NT,  // default
            epilogue: super::gemm_gen::EpilogueOp::StoreF32,
        }
    }

    /// Convert cost-model TileConfig to tile_ir TileGemm.
    ///
    /// tile_ir handles tile_n=128 natively (streaming mode) without multi-pass.
    /// Only valid for tile_k values that are powers of 2 (16, 32, 64, 128).
    pub fn to_tile_gemm(&self) -> super::tile_ir::TileGemm {
        super::tile_ir::TileGemm {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            tile_k: self.tile_k,
            wgp_mode: self.wgp_mode,
            double_buffer: self.use_lds,
            split_k: self.split_k,
            swap_grid: true,  // tile_ir presets always use swap_grid=true (L2 friendly + proven safe)
            transpose: super::tile_ir::TileTranspose::NT,
            acc_swap: false,
            epilogue: vec![],
        }
    }

    /// Whether this config can be compiled and dispatched by tile_ir.
    ///
    /// Safety constraints (violating any of these causes GPU hang):
    /// - tile_k must be power-of-2 (bitwise tid decomposition in cooperative load)
    /// - waves_per_wg must match tile_ir's derived n_waves = tile_m / 32
    ///   (grid dimensions use wg_size, which tile_ir computes as n_waves * 32)
    /// - swap_grid must be true (all tile_ir presets use true; false is
    ///   untested in autotuner dispatch and has caused hangs)
    /// - split_k must be 1 (split_k > 1 hangs: y_split_stride=0 bug causes
    ///   all partitions to write same Y address → GPU write conflict hang)
    pub fn can_use_tile_ir(&self) -> bool {
        let tile_ir_waves = self.tile_m / 32;
        self.tile_k.is_power_of_two()
            && self.tile_m >= 32
            && self.tile_n >= 32
            && self.waves_per_wg == tile_ir_waves
            && self.swap_grid  // only proven-safe grid layout
            && self.split_k <= 1  // split_k > 1 hangs (y_split_stride=0 bug)
    }
}

/// CPU-only: predict the best GemmConfig for given dimensions using cost model.
///
/// This is the primary autotune entry point — no GPU needed.
/// Internally runs `auto_schedule_gemm` and converts the top result.
///
/// # Example
/// ```rust,ignore
/// let cfg = cost_model::predict_best(1024, 1024, 1024);
/// let kernel = gemm_gen::generate(&cfg);
/// ```
pub fn predict_best(m: u32, k: u32, n: u32) -> super::gemm_gen::GemmConfig {
    let results = auto_schedule_gemm(m, n, k, DataFormat::BF16);
    if let Some(best) = results.first() {
        best.config.to_gemm_config()
    } else {
        // Fallback: use gemm_gen's hand-tuned heuristic
        super::gemm_gen::auto_select_legacy(m, k, n)
    }
}



// ============================================================================
// Instruction-level K-loop Analysis (insn_latency integration)
// ============================================================================

/// Instruction-level K-loop statistics from actual code generation.
///
/// Unlike the macro-level `estimate_tile_cost`, this generates the REAL
/// GEMM kernel for a given config and analyzes the K-loop body instruction
/// by instruction using the `insn_latency` model.
#[derive(Clone, Debug)]
pub struct KLoopAnalysis {
    pub total_ops: usize,
    pub valu: u32,
    pub vtrans: u32,
    pub wmma: u32,
    pub vmem_loads: u32,
    pub vmem_stores: u32,
    pub lds: u32,
    pub salu: u32,
    pub ctrl: u32,
    /// Total issue cycles (sum of all instruction issue costs)
    pub issue_cycles: u32,
    /// Estimated critical path through the K-loop body
    pub critical_path: u32,
    /// ILP potential: fraction of WMMA latency that can be hidden [0.0, 1.0]
    pub ilp_potential: f32,
    /// Primary bottleneck
    pub bottleneck: &'static str,
    /// Refined cycles per K-loop iteration (accounting for overlap)
    pub cycles_per_iter: f64,
}

/// Generate a GEMM kernel from a TileConfig and analyze its K-loop body.
///
/// Returns None if the kernel cannot be generated or no K-loop is found.
pub fn analyze_kloop(config: &TileConfig) -> Option<KLoopAnalysis> {
    use super::insn_latency;
    use super::ir::Op;

    let gemm_cfg = config.to_gemm_config();
    // Skip configs that exceed VGPR limit (would panic in generate)
    if !gemm_cfg.is_feasible() { return None; }
    let kernel = super::gemm_gen::generate(&gemm_cfg);
    let ops = kernel.ops();

    // Find the K-loop body between "ggen_loop" label and back-edge branch
    let mut loop_start = None;
    let mut loop_end = None;
    for (i, op) in ops.iter().enumerate() {
        if let Op::Label(name) = op {
            if name.starts_with("ggen_loop") { loop_start = Some(i + 1); }
        }
        if let Op::Branch(target) = op {
            if target.starts_with("ggen_loop") { loop_end = Some(i); }
        }
        if let Op::BranchScc1(target) = op {
            if target.starts_with("ggen_loop") { loop_end = Some(i); }
        }
    }

    let (start, end) = match (loop_start, loop_end) {
        (Some(s), Some(e)) if s < e => (s, e),
        _ => return None,
    };

    let loop_body = &ops[start..end];
    let stats = insn_latency::analyze_block(loop_body);
    let (ilp, bottleneck) = insn_latency::ilp_potential(&stats);

    // ── RDNA3 K-loop Pipeline Model ──
    //
    // RDNA3 CU has 3 concurrent execution pipelines for GEMM:
    //   1. WMMA/Matrix unit: issue rate = 1 per 4 cycles (shared with VALU pipe)
    //   2. Vector ALU (VALU+LDS): issue rate = 1 per cycle
    //   3. VMEM (global loads): issue rate = 1 per cycle
    //
    // In a well-scheduled double-buffered K-loop:
    //   Phase A: {LDS read buf0 → WMMA compute} overlapped with {VMEM prefetch buf1}
    //   Phase B: {LDS read buf1 → WMMA compute} overlapped with {VMEM prefetch buf0}
    //
    // The critical path per K-loop iteration is:
    //   max(WMMA_issue_time, VALU_issue_time + LDS_issue_time, VMEM_issue_time)
    //
    // WMMA issue: 4 cycles per WMMA op (they share the VALU pipe but have 4x throughput)
    let wmma_issue = stats.wmma_count as f64 * 4.0;
    
    // LDS issue: 1 cycle per ds_read_b128 operation
    let lds_issue = stats.lds_count as f64;
    
    // VALU + SALU issue: 1 cycle per op (VALU and SALU share same issue slot)
    let alu_issue = stats.valu_count as f64 + stats.salu_count as f64;
    
    // VMEM issue: 1 cycle per global_load_b128 (fully overlapped with compute)
    let vmem_issue = (stats.vmem_load_count + stats.vmem_store_count) as f64;
    
    // Control flow overhead (waitcnt, barriers, loop counter): ~1 cycle each
    let ctrl_issue = stats.ctrl_count as f64;
    
    // WMMA and VALU share the vector issue port — they CANNOT overlap.
    // The total vector pipe demand = WMMA_issue + VALU_issue
    // LDS uses a separate pipe but depends on VALU for address computation.
    // Model: vector_pipe = WMMA_issue + max(LDS_issue, ALU_issue)
    // because LDS reads and VALU interleave in the same phase.
    let vector_pipe = wmma_issue + lds_issue.max(alu_issue);
    
    // VMEM prefetches overlap with compute (double-buffering).
    // Only add VMEM issue cycles if they EXCEED the compute time.
    let compute_time = vector_pipe + ctrl_issue;
    let cycles_per_iter = compute_time.max(vmem_issue);

    Some(KLoopAnalysis {
        total_ops: loop_body.len(),
        valu: stats.valu_count,
        vtrans: stats.vtrans_count,
        wmma: stats.wmma_count,
        vmem_loads: stats.vmem_load_count,
        vmem_stores: stats.vmem_store_count,
        lds: stats.lds_count,
        salu: stats.salu_count,
        ctrl: stats.ctrl_count,
        issue_cycles: stats.total_issue_cycles,
        critical_path: stats.estimated_critical_path,
        ilp_potential: ilp,
        bottleneck,
        cycles_per_iter,
    })
}

/// Print K-loop analysis report for a TileConfig.
pub fn print_kloop_analysis(config: &TileConfig) {
    if let Some(a) = analyze_kloop(config) {
        eprintln!("╔═══════════════════════════════════════════════════╗");
        eprintln!("║  K-loop insn analysis: {:>28}  ║", config.name());
        eprintln!("╠═══════════════════════════════════════════════════╣");
        eprintln!("║  Total: {} ops ({} issue cycles)             ", a.total_ops, a.issue_cycles);
        eprintln!("║  VALU:{:>3}  VTRANS:{:>3}  WMMA:{:>3}              ", a.valu, a.vtrans, a.wmma);
        eprintln!("║  VMEM:{:>3} ld + {:>3} st  LDS:{:>3}               ", a.vmem_loads, a.vmem_stores, a.lds);
        eprintln!("║  SALU:{:>3}  CTRL:{:>3}                           ", a.salu, a.ctrl);
        eprintln!("║  ILP potential: {:.1}% ({})                      ", a.ilp_potential * 100.0, a.bottleneck);
        eprintln!("║  Cycles/iter (refined): {:.0}                    ", a.cycles_per_iter);
        eprintln!("╚═══════════════════════════════════════════════════╝");
    } else {
        eprintln!("[kloop] No K-loop found for {}", config.name());
    }
}

// ============================================================================
// Tile-IR GPU Autotuner (independent of gemm_gen)
// ============================================================================

/// Result of a tile_ir GPU autotuning run.
#[derive(Clone, Debug)]
pub struct TileIrTuneResult {
    /// Best tile_ir config found
    pub best: super::tile_ir::TileGemm,
    /// TFLOPS measured for the best config
    pub best_tflops: f64,
    /// All (config_name, tflops) sorted by performance
    pub all: Vec<(String, f64)>,
    /// Whether result came from disk cache
    pub from_cache: bool,
}

/// Tune tile_ir GEMM for given (M, N, K) dimensions using actual GPU benchmarks.
///
/// Pipeline:
///   1. `auto_schedule_gemm()` generates exhaustive candidate set
///   2. `can_use_tile_ir()` filters to tile_ir-compatible configs
///   3. Each candidate is compiled and benchmarked on the GPU
///   4. Best result cached to `~/.t0_autotune/tile_ir_MxNxK.json`
///
/// # Safety
/// - GpuRuntime must be provided by caller (no internal creation)
/// - Only tile_ir kernels are dispatched (no gemm_gen mixing)
/// - Each candidate wrapped in catch_unwind for compilation safety
/// - LDS ≤ 65536 and VGPR ≤ 256 enforced
///
/// # Example
/// ```rust,no_run
/// let result = tune_tile_ir(&rt, 4096, 4096, 4096)?;
/// eprintln!("Best: {} ({:.1} TF)", result.best.name(), result.best_tflops);
/// ```
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub fn tune_tile_ir(
    rt: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
    m: u32, n: u32, k: u32,
) -> Result<TileIrTuneResult, String> {
    // 1. Check disk cache first
    let cache_path = tile_ir_cache_path(m, n, k);
    if let Some(cached) = load_tile_ir_cache(&cache_path) {
        eprintln!("[tile_tune] Cache hit: {}×{}×{} → {} ({:.1} TF)",
            m, n, k, cached.best.name(), cached.best_tflops);
        return Ok(cached);
    }

    // 2. Build candidate list: PROVEN PRESETS FIRST, then cost_model discoveries.
    // This ensures we always benchmark the best-known configs even if queue
    // gets poisoned partway through (proven configs run before risky ones).
    let mut candidates: Vec<super::tile_ir::TileGemm> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // 2a. Proven-good presets (ordered by expected performance: k64 > k32 > k16)
    //     SAFETY: force wgp_mode=false on all candidates.
    //     WGP mode switch (RSRC1.WGP_MODE flip) between dispatches on the same
    //     KFD queue causes GPU hang (confirmed 2026-03-30: k64 wgp=false → k32 wgp=true → hang).
    let presets = [
        super::tile_ir::TileGemm::tile_128x128_k64(),
        super::tile_ir::TileGemm::tile_128x128_k32(),
        super::tile_ir::TileGemm::tile_128x128_k16(),
        super::tile_ir::TileGemm::tile_128x64_k32(),
        super::tile_ir::TileGemm::tile_128x64_k16(),
        super::tile_ir::TileGemm::tile_64x64_k16(),
        super::tile_ir::TileGemm::tile_32x64_k16(),
    ];
    for preset in &presets {
        let mut p = preset.clone();
        p.wgp_mode = false; // SAFETY: force consistent WGP mode
        if p.lds_total() > 65536 { continue; }
        if k % p.tile_k != 0 { continue; }
        if p.tile_m > m && m >= 32 { continue; }
        if p.tile_n > n && n >= 32 { continue; }
        let name = p.name();
        if seen.contains(&name) { continue; }
        seen.insert(name);
        candidates.push(p);
    }

    // 2b. Cost-model discoveries (may find novel configs the presets miss)
    //     SAFETY: force wgp_mode=false on all cost-model candidates too.
    let cost_results = auto_schedule_gemm(m, n, k, DataFormat::BF16);
    for cost in &cost_results {
        if !cost.config.can_use_tile_ir() { continue; }
        let mut spec = cost.config.to_tile_gemm();
        spec.wgp_mode = false; // SAFETY: consistent WGP mode
        if spec.lds_total() > 65536 { continue; }
        let name = spec.name();
        if seen.contains(&name) { continue; }
        seen.insert(name);
        candidates.push(spec);
    }

    if candidates.is_empty() {
        return Err("tile_tune: no feasible tile_ir candidates".into());
    }

    eprintln!("[tile_tune] Benchmarking {} tile_ir candidates for {}×{}×{} ...",
        candidates.len(), m, n, k);

    // 3. Pre-compile ALL candidates, keeping GpuKernels alive in a Vec.
    //
    // ROOT CAUSE FIX: The previous code created GpuKernel inside benchmark_tile_ir_one(),
    // which dropped (freeing code_buf VRAM) when the function returned. The next kernel's
    // GpuKernel::load could receive the same VA from Linux mmap recycling. But the GPU's
    // SQ instruction cache / TLB may still reference the old VA→physical mapping. When the
    // CP fetches the new kernel's code at the recycled VA, it reads stale or unmapped data
    // → GPU hang (confirmed: k64 desc_va=0x...22C0, k32 desc_va=0x...32C0, diff = 0x1000).
    //
    // Fix: keep ALL kernels alive until the entire tune session completes. This matches
    // the pattern used by ensure_kernel_t0 (HashMap cache), which never drops code_bufs
    // and never hangs when switching between kernel configs.
    use crate::gpu_backend::{GpuKernel, KernelLoadConfig};

    struct CompiledCandidate {
        spec: super::tile_ir::TileGemm,
        kernel: GpuKernel,
        lds_size: u32,
    }
    let mut compiled: Vec<CompiledCandidate> = Vec::new();

    for spec in &candidates {
        let spec_clone = spec.clone();
        let compile_result = std::panic::catch_unwind(move || -> Result<(Vec<u8>, u32, u32), String> {
            let kernel_ir = super::tile_ir::lower_gemm(&spec_clone);
            let base_lds = kernel_ir.lds_size();
            let (elf, final_lds) = kernel_ir.compile_with_info(super::ir::Target::GFX1100)?;
            Ok((elf, base_lds, final_lds))
        });
        let (elf, base_lds, final_lds) = match compile_result {
            Ok(Ok((elf, base, fin))) => (elf, base, fin),
            Ok(Err(e)) => {
                eprintln!("[tile_tune]   {} → FAIL: {}", spec.name(), e);
                continue;
            }
            Err(_) => {
                eprintln!("[tile_tune]   {} → FAIL: compilation panic", spec.name());
                continue;
            }
        };
        // Skip spilled kernels: LDS spill region means final_lds > base_lds.
        // Spilled kernels are 40-50x slower (confirmed: 64x128_k64 = 2.3 TF vs 90+ TF).
        if final_lds > base_lds {
            eprintln!("[tile_tune]   {} → SKIP: VGPR spill detected (LDS {} → {})",
                spec.name(), base_lds, final_lds);
            continue;
        }
        let lds_size = final_lds;
        match GpuKernel::load(
            &rt.device, &elf,
            &KernelLoadConfig {
                workgroup_size: [spec.wg_size(), 1, 1],
                lds_size,
            },
        ) {
            Ok(kernel) => {
                compiled.push(CompiledCandidate {
                    spec: spec.clone(),
                    kernel,
                    lds_size,
                });
            }
            Err(e) => {
                eprintln!("[tile_tune]   {} → FAIL: load: {}", spec.name(), e);
            }
        }
    }

    if compiled.is_empty() {
        return Err("tile_tune: all candidates failed to compile".into());
    }

    // 4. Allocate shared buffers ONCE (reused across all benchmarks)
    let max_tile_k = compiled.iter().map(|c| c.spec.tile_k).max().unwrap_or(64);
    let k_pad_max = (k + max_tile_k - 1) & !(max_tile_k - 1);
    let a_buf = rt.alloc((m as usize * k_pad_max as usize * 2).max(4096))?;
    let b_buf = rt.alloc((n as usize * k_pad_max as usize * 2).max(4096))?;
    let c_buf = rt.alloc((m as usize * n as usize * 4).max(4096))?;
    a_buf.zero(); b_buf.zero(); c_buf.zero();

    // 5. Benchmark each pre-compiled candidate
    let mut results: Vec<(super::tile_ir::TileGemm, f64)> = Vec::new();

    for cc in &compiled {
        let k_pad = (k + cc.spec.tile_k - 1) & !(cc.spec.tile_k - 1);
        let ka = super::tile_ir::build_kernargs_m(
            a_buf.gpu_addr(), b_buf.gpu_addr(), c_buf.gpu_addr(),
            k_pad, n, m, &cc.spec,
        );
        let grid = super::tile_ir::compute_grid(&cc.spec, m, n);

        // Warmup: 10 sync dispatches for GPU clock ramp-up
        let mut warmup_ok = true;
        for _ in 0..10 {
            if let Err(e) = rt.dispatch(&cc.kernel, grid, &ka) {
                eprintln!("[tile_tune]   {} → FAIL: warmup: {}", cc.spec.name(), e);
                warmup_ok = false;
                break;
            }
        }
        if !warmup_ok {
            if rt.is_poisoned() {
                eprintln!("[tile_tune] Queue poisoned, stopping benchmark");
                break;
            }
            continue;
        }

        // Timed: batch-submit 20 async dispatches, wait once
        let n_iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            rt.dispatch_async(&cc.kernel, grid, &ka);
        }
        match rt.wait_idle() {
            Ok(()) => {},
            Err(e) => {
                eprintln!("[tile_tune]   {} → FAIL: timed: {}", cc.spec.name(), e);
                if e.contains("hung") || e.contains("TIMEOUT") {
                    eprintln!("[tile_tune] Queue poisoned, stopping benchmark");
                    break;
                }
                continue;
            }
        }
        let elapsed_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        let total_flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = if elapsed_us > 0.0 { total_flops / (elapsed_us * 1e6) } else { 0.0 };
        eprintln!("[tile_tune]   {} → {:.1} TF", cc.spec.name(), tflops);
        results.push((cc.spec.clone(), tflops));
    }

    // Recycle shared buffers
    rt.recycle(a_buf);
    rt.recycle(b_buf);
    rt.recycle(c_buf);
    // compiled Vec drops here → all GpuKernels freed AFTER benchmarking is done

    if results.is_empty() {
        return Err("tile_tune: all candidates failed".into());
    }

    // 6. Sort by TFLOPS descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let best = &results[0];
    eprintln!("[tile_tune] ✓ Best: {} ({:.1} TF)", best.0.name(), best.1);

    let tune_result = TileIrTuneResult {
        best: best.0.clone(),
        best_tflops: best.1,
        all: results.iter().map(|(s, t)| (s.name(), *t)).collect(),
        from_cache: false,
    };

    // 5. Save to disk cache
    save_tile_ir_cache(&cache_path, &tune_result);

    Ok(tune_result)
}


// ── Tile IR cache persistence ──

fn tile_ir_cache_dir() -> std::path::PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        std::path::PathBuf::from(home).join(".t0_autotune")
    } else {
        std::path::PathBuf::from("/tmp/.t0_autotune")
    }
}

fn tile_ir_cache_path(m: u32, n: u32, k: u32) -> std::path::PathBuf {
    tile_ir_cache_dir().join(format!("tile_ir_{}x{}x{}.json", m, n, k))
}

fn load_tile_ir_cache(path: &std::path::Path) -> Option<TileIrTuneResult> {
    let content = std::fs::read_to_string(path).ok()?;
    // Parse: {"tile_m":128,"tile_n":128,"tile_k":64,"tflops":87.1,...}
    let tile_m = parse_json_u32(&content, "tile_m")?;
    let tile_n = parse_json_u32(&content, "tile_n")?;
    let tile_k = parse_json_u32(&content, "tile_k")?;
    let tflops = parse_json_f64(&content, "tflops")?;
    let split_k = parse_json_u32(&content, "split_k").unwrap_or(1);
    let wgp = content.contains("\"wgp\":true");

    let spec = super::tile_ir::TileGemm {
        tile_m, tile_n, tile_k,
        wgp_mode: wgp,
        double_buffer: true,
        split_k,
        swap_grid: true,
        transpose: super::tile_ir::TileTranspose::NT,
        acc_swap: false,
        epilogue: vec![],
    };

    Some(TileIrTuneResult {
        best: spec,
        best_tflops: tflops,
        all: Vec::new(),
        from_cache: true,
    })
}

fn save_tile_ir_cache(path: &std::path::Path, result: &TileIrTuneResult) {
    let _ = std::fs::create_dir_all(tile_ir_cache_dir());
    let s = &result.best;
    let all_json: Vec<String> = result.all.iter()
        .map(|(name, tf)| format!("[\"{}\",{:.2}]", name, tf))
        .collect();
    let json = format!(
        concat!(
            "{{\"tile_m\":{},\"tile_n\":{},\"tile_k\":{},",
            "\"split_k\":{},\"wgp\":{},\"tflops\":{:.2},",
            "\"all\":[{}]}}"
        ),
        s.tile_m, s.tile_n, s.tile_k,
        s.split_k, s.wgp_mode, result.best_tflops,
        all_json.join(","),
    );
    if let Err(e) = std::fs::write(path, &json) {
        eprintln!("[tile_tune] Cache write failed: {}", e);
    }
}

fn parse_json_u32(json: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    if end == 0 { return None; }
    rest[..end].parse().ok()
}

fn parse_json_f64(json: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(rest.len());
    if end == 0 { return None; }
    rest[..end].parse().ok()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_limits_default() {
        let hw = GFX1100Limits::default();
        assert_eq!(hw.max_vgprs, 256);
        assert_eq!(hw.n_cus, 96);
        assert_eq!(hw.lds_per_wg, 65536);
    }

    #[test]
    fn test_tile_config_derived() {
        let cfg = TileConfig {
            tile_m: 32, tile_n: 64, tile_k: 16, waves_per_wg: 2, use_lds: true,
            split_k: 1, wgp_mode: false, swap_grid: true, lds_pad: 0,
        };
        assert_eq!(cfg.wg_threads(), 64);
        assert_eq!(cfg.n_wmma_tiles(), 4);
        assert_eq!(cfg.m_wmma_per_wave(), 1);
    }

    #[test]
    fn test_current_default_is_feasible() {
        let config = TileConfig {
            tile_m: 32, tile_n: 64, tile_k: 16, waves_per_wg: 2, use_lds: true,
            split_k: 1, wgp_mode: false, swap_grid: true, lds_pad: 0,
        };
        let hw = GFX1100Limits::default();
        let cost = estimate_tile_cost(&config, 4096, 4096, 512, DataFormat::BF16, &hw);
        assert!(cost.feasible, "hand-tuned config should be feasible");
        assert!(cost.vgprs <= 256, "VGPRs should fit: {}", cost.vgprs);
        assert!(cost.occupancy >= 2, "should have at least 2 waves: {}", cost.occupancy);
        assert!(cost.score > 0.0, "should have positive score: {}", cost.score);
        eprintln!("Hand-tuned config: {} VGPRs, {} occ, {:.1} TFLOPS, {}",
            cost.vgprs, cost.occupancy, cost.score, cost.bottleneck);
    }

    #[test]
    fn test_auto_schedule_returns_results() {
        let results = auto_schedule_gemm(4096, 4096, 512, DataFormat::BF16);
        assert!(!results.is_empty(), "should find at least 1 feasible config");
        let best = &results[0];
        assert!(best.feasible);
        assert!(best.score > 0.0);
        eprintln!("Best: {} waves={} {:.1} TFLOPS ({})",
            best.config.name(), best.config.waves_per_wg, best.score, best.bottleneck);
    }

    #[test]
    fn test_auto_schedule_respects_limits() {
        let results = auto_schedule_gemm(4096, 4096, 512, DataFormat::BF16);
        let hw = GFX1100Limits::default();

        for cost in &results {
            assert!(cost.vgprs <= hw.max_vgprs, "VGPRs {} exceeds {}", cost.vgprs, hw.max_vgprs);
            assert!(cost.lds_bytes <= hw.lds_per_wg, "LDS {} exceeds {}", cost.lds_bytes, hw.lds_per_wg);
        }
    }

    #[test]
    fn test_auto_schedule_small_gemm() {
        // Small GEMM should still find configs
        let results = auto_schedule_gemm(128, 128, 64, DataFormat::BF16);
        assert!(!results.is_empty());

        let best = &results[0];
        eprintln!("Small GEMM best: tile_m={}, tile_n={}, tile_k={}, {:.1} TFLOPS",
            best.config.tile_m, best.config.tile_n, best.config.tile_k, best.score);
    }

    #[test]
    fn test_print_schedule_report() {
        // Smoke test: just ensure it doesn't panic
        print_schedule_report(2048, 2048, 512, DataFormat::BF16, 5);
    }

    #[test]
    fn test_infeasible_config() {
        let config = TileConfig {
            tile_m: 64, tile_n: 128, tile_k: 32, waves_per_wg: 1, use_lds: true,
            split_k: 1, wgp_mode: false, swap_grid: true, lds_pad: 0,
        };
        let hw = GFX1100Limits::default();
        let cost = estimate_tile_cost(&config, 4096, 4096, 512, DataFormat::BF16, &hw);
        if cost.feasible {
            assert!(cost.occupancy <= 4, "large tile should have low occupancy: {}", cost.occupancy);
        }
        eprintln!("Large tile: feasible={}, {} VGPRs, {} occ", cost.feasible, cost.vgprs, cost.occupancy);
    }

    // ── P1-B: LDS Bank Conflict tests ──

    #[test]
    fn test_lds_padding_increases_lds_usage() {
        let hw = GFX1100Limits::default();
        let base = TileConfig {
            tile_m: 128, tile_n: 64, tile_k: 16, waves_per_wg: 4, use_lds: true,
            split_k: 1, wgp_mode: false, swap_grid: true, lds_pad: 0,
        };
        let padded = TileConfig { lds_pad: 4, ..base.clone() };

        let cost_base = estimate_tile_cost(&base, 4096, 4096, 512, DataFormat::BF16, &hw);
        let cost_pad = estimate_tile_cost(&padded, 4096, 4096, 512, DataFormat::BF16, &hw);

        assert!(cost_pad.lds_bytes > cost_base.lds_bytes,
            "padded LDS ({}) should be larger than unpadded ({})",
            cost_pad.lds_bytes, cost_base.lds_bytes);
        eprintln!("LDS: base={} pad4={} (delta=+{})",
            cost_base.lds_bytes, cost_pad.lds_bytes, cost_pad.lds_bytes - cost_base.lds_bytes);
    }

    #[test]
    fn test_search_includes_lds_padded_configs() {
        let results = auto_schedule_gemm(4096, 4096, 512, DataFormat::BF16);
        let has_padded = results.iter().any(|r| r.config.lds_pad > 0);
        assert!(has_padded, "search should include lds_pad > 0 configs");

        let has_pad4 = results.iter().any(|r| r.config.lds_pad == 4);
        let has_pad8 = results.iter().any(|r| r.config.lds_pad == 8);
        eprintln!("Search: total={}, pad4={}, pad8={}", results.len(), has_pad4, has_pad8);
    }

    // ── P1-A: Autotune tests ──

    #[test]
    fn test_tile_config_to_gemm_config() {
        let tc = TileConfig {
            tile_m: 128, tile_n: 64, tile_k: 16, waves_per_wg: 4, use_lds: true,
            split_k: 4, wgp_mode: true, swap_grid: false, lds_pad: 4,
        };
        let gc = tc.to_gemm_config();
        assert_eq!(gc.tile_m, 128);
        assert_eq!(gc.tile_n, 64);
        assert_eq!(gc.tile_k, 16);
        assert_eq!(gc.wg_size, 128);
        assert_eq!(gc.split_k, Some(4));
        assert_eq!(gc.lds_pad, 4);
        assert!(gc.wgp_mode);
        assert!(!gc.swap_grid);
        assert!(gc.use_lds);
        assert!(gc.double_buffer);
    }

    #[test]
    fn test_predict_best_returns_valid_config() {
        let cfg = predict_best(1024, 1024, 1024);
        assert!(cfg.tile_m >= 32, "tile_m should be >= 32: {}", cfg.tile_m);
        assert!(cfg.tile_n >= 32, "tile_n should be >= 32: {}", cfg.tile_n);
        assert!(cfg.tile_k >= 16, "tile_k should be >= 16: {}", cfg.tile_k);
        assert!(cfg.wg_size >= 32, "wg_size should be >= 32: {}", cfg.wg_size);
        eprintln!("predict_best(1024,1024,1024): {}", cfg.name());
    }

    #[test]
    fn test_predict_best_various_sizes() {
        for &(m, k, n) in &[(128, 256, 64), (512, 512, 512), (2048, 2048, 2048), (4096, 4096, 512)] {
            let cfg = predict_best(m, k, n);
            eprintln!("predict_best({},{},{}): {}", m, k, n, cfg.name());
            assert!(cfg.tile_m > 0 && cfg.tile_n > 0 && cfg.tile_k > 0);
        }
    }

    /// GPU E2E: tune tile_ir for 4096³ GEMM
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[ignore] // Run explicitly: cargo test --release --features rocm -- test_tune_tile_ir_4096 --ignored --nocapture
    fn test_tune_tile_ir_4096() {
        let rt = crate::ignis::gpu_context::GpuRuntime::new()
            .expect("GpuRuntime::new");

        // Clear cache for fresh benchmark
        let cache = tile_ir_cache_path(4096, 4096, 4096);
        let _ = std::fs::remove_file(&cache);

        let result = tune_tile_ir(&rt, 4096, 4096, 4096)
            .expect("tune_tile_ir failed");

        eprintln!("\n╔══════════════════════════════════════════╗");
        eprintln!("║  tile_ir tune: 4096³ GEMM BF16           ║");
        eprintln!("╠══════════════════════════════════════════╣");
        eprintln!("║  Best: {:>30} ║", result.best.name());
        eprintln!("║  TFLOPS: {:>8.1}                        ║", result.best_tflops);
        eprintln!("╠──────────────────────────────────────────╣");
        for (name, tf) in &result.all {
            eprintln!("║  {:>35} {:.1} TF ║", name, tf);
        }
        eprintln!("╚══════════════════════════════════════════╝");

        assert!(result.best_tflops > 50.0,
            "Expected > 50 TFLOPS, got {:.1}", result.best_tflops);

        // Verify cache was written
        assert!(cache.exists(), "disk cache should be saved");

        // Verify cache hit
        let cached = tune_tile_ir(&rt, 4096, 4096, 4096).unwrap();
        assert!(cached.from_cache, "second call should be cache hit");
        assert!((cached.best_tflops - result.best_tflops).abs() < 0.01);
    }

    /// GPU E2E: tune tile_ir for 9 standard sizes
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    #[ignore]
    fn test_tune_tile_ir_all_sizes() {
        let rt = crate::ignis::gpu_context::GpuRuntime::new()
            .expect("GpuRuntime::new");

        let sizes: Vec<(u32, u32, u32)> = vec![
            // Square matrices (power-of-2)
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            // Non-square (typical transformer shapes)
            (128, 4096, 1024),
            (256, 4096, 1024),
            (512, 4096, 1024),
            (1024, 4096, 1024),
        ];

        eprintln!("\n{:>20}  {:>10}  {:>30}", "Size", "TFLOPS", "Best Config");
        eprintln!("{}", "─".repeat(65));

        for &(m, k, n) in &sizes {
            // Clear cache for each size
            let _ = std::fs::remove_file(tile_ir_cache_path(m, n, k));

            match tune_tile_ir(&rt, m, n, k) {
                Ok(result) => {
                    eprintln!("{:>4}×{:<4}×{:<4}  {:>8.1} TF  {:>30}",
                        m, k, n, result.best_tflops, result.best.name());
                }
                Err(e) => {
                    eprintln!("{:>4}×{:<4}×{:<4}  FAILED: {}", m, k, n, e);
                }
            }
        }
    }
}
