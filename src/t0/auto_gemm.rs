//! Auto-GEMM — Runtime autotuning for GEMM kernel configurations.
//!
//! Replaces static cost_model::predict_best() with a benchmark-validated
//! config selection pipeline:
//!
//! 1. **Candidate generation**: cost_model → top-N feasible configs
//! 2. **Benchmark**: compile + dispatch each candidate on actual GPU
//! 3. **Selection**: pick minimum-latency config
//! 4. **Cache**: persist to `~/.t0_autotune/` for zero-overhead re-use
//!
//! # Usage
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use t0_gpu::t0::auto_gemm::GemmTuner;
//! let mut tuner = GemmTuner::new();
//! // First call: benchmarks ~8 candidates (~2-5s), selects best
//! let cfg = tuner.tune(&rt, 4096, 4096, 4096).unwrap();
//! // Subsequent calls: instant cache hit
//! let cfg2 = tuner.tune(&rt, 4096, 4096, 4096).unwrap();
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use super::gemm_gen::GemmConfig;
use super::cost_model::{self, DataFormat};

/// Results of a GEMM autotuning run.
#[derive(Clone, Debug)]
pub struct GemmTuneResult {
    /// Best configuration found
    pub best: GemmConfig,
    /// TFLOPS measured for the best configuration
    pub best_tflops: f64,
    /// All (config_name, measured_tflops) sorted by performance
    pub all: Vec<(String, f64)>,
    /// Problem dimensions
    pub key: (u32, u32, u32),
    /// Whether result came from disk cache
    pub from_cache: bool,
}

/// GEMM autotuner — benchmarks multiple GemmConfigs and selects the fastest.
///
/// Caches results both in-memory and on disk (`~/.t0_autotune/`).
pub struct GemmTuner {
    /// In-memory cache: (M, N, K) → result
    cache: HashMap<(u32, u32, u32), GemmTuneResult>,
    /// Disk cache directory
    cache_dir: PathBuf,
    /// Maximum candidates to benchmark (cost_model pre-filters the rest)
    pub max_candidates: usize,
    /// Number of benchmark iterations (takes median)
    pub n_iters: usize,
}

impl GemmTuner {
    /// Create a new tuner with default settings.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            cache_dir: default_cache_dir(),
            max_candidates: 8,
            n_iters: 5,
        }
    }

    /// Create with a custom cache directory.
    pub fn with_cache_dir(dir: PathBuf) -> Self {
        Self {
            cache_dir: dir,
            ..Self::new()
        }
    }

    /// Tune GEMM for given (M, N, K) dimensions.
    ///
    /// Returns the best GemmConfig based on actual GPU measurements.
    /// First call benchmarks `max_candidates` configs (~2-5s).
    /// Subsequent calls return from cache (~0ns).
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    pub fn tune(
        &mut self,
        rt: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
        m: u32, n: u32, k: u32,
    ) -> Result<GemmConfig, String> {
        let key = (m, n, k);

        // 1. In-memory cache
        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.best.clone());
        }

        // 2. Disk cache
        if let Some(cached) = self.load_cache(m, n, k) {
            eprintln!("[autotune] Cache hit: {}×{}×{} → {} ({:.1} TF)",
                m, n, k, cached.best.name(), cached.best_tflops);
            let cfg = cached.best.clone();
            self.cache.insert(key, cached);
            return Ok(cfg);
        }

        // 3. Generate candidates via cost_model
        let cost_results = cost_model::auto_schedule_gemm(m, n, k, DataFormat::BF16);
        let candidates: Vec<GemmConfig> = cost_results.iter()
            .take(self.max_candidates)
            .map(|c| c.config.to_gemm_config())
            .collect();

        if candidates.is_empty() {
            eprintln!("[autotune] No feasible candidates, falling back to legacy");
            return Ok(super::gemm_gen::auto_select_legacy(m, k, n));
        }

        eprintln!("[autotune] Benchmarking {} candidates for {}×{}×{}...",
            candidates.len(), m, n, k);

        // 4. Benchmark each candidate (gemm_gen path)
        let mut results: Vec<(GemmConfig, f64)> = Vec::new();
        for cfg in &candidates {
            match self.benchmark_one(rt, cfg, m, n, k) {
                Ok(tflops) => {
                    eprintln!("[autotune]   {} → {:.1} TFLOPS", cfg.name(), tflops);
                    results.push((cfg.clone(), tflops));
                }
                Err(e) => {
                    eprintln!("[autotune]   {} → FAILED: {}", cfg.name(), e);
                }
            }
        }

        // 4b. tile_ir candidates DISABLED in auto_gemm.
        // Root cause: tile_ir kernels hang when dispatched after gemm_gen kernels
        // in the same queue session (confirmed 2026-03-30, two GPU hard hangs).
        // tile_ir config selection is handled by tile_auto_select() instead,
        // which uses static heuristics based on problem dimensions.
        // TODO: investigate queue state incompatibility between gemm_gen and tile_ir paths.

        if results.is_empty() {
            return Err("autotune: all candidates failed".into());
        }

        // 5. Sort by TFLOPS descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best = results[0].clone();
        eprintln!("[autotune] ✓ Best: {} ({:.1} TFLOPS)", best.0.name(), best.1);

        let tune_result = GemmTuneResult {
            best: best.0.clone(),
            best_tflops: best.1,
            all: results.iter().map(|(c, t)| (c.name(), *t)).collect(),
            key,
            from_cache: false,
        };

        // 6. Persist
        self.save_cache(m, n, k, &tune_result);
        self.cache.insert(key, tune_result);

        Ok(best.0)
    }

    /// Benchmark a single GemmConfig on the GPU.
    ///
    /// Compiles the kernel, allocates random bf16 data, dispatches
    /// `n_iters` times, and returns median TFLOPS.
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn benchmark_one(
        &self,
        rt: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
        cfg: &GemmConfig,
        m: u32, n: u32, k: u32,
    ) -> Result<f64, String> {
        use crate::gpu_backend::{GpuKernel, KernelLoadConfig};

        // 1. Generate and compile kernel
        let kernel_ir = super::gemm_gen::generate(cfg);
        let elf = kernel_ir.compile(super::ir::Target::GFX1100)?;
        let lds_size = kernel_ir.lds_size();

        let gpu_kernel = GpuKernel::load(
            &rt.device, &elf,
            &KernelLoadConfig {
                workgroup_size: [cfg.wg_size, 1, 1],
                lds_size,
            },
        )?;

        // 2. Allocate buffers (bf16 for A/B, f32 for C)
        let a_bytes = (m as usize) * (k as usize) * 2; // bf16
        let b_bytes = (n as usize) * (k as usize) * 2; // bf16
        let c_bytes = (m as usize) * (n as usize) * 4; // f32
        let sk = cfg.split_k.unwrap_or(1);
        let c_total = if sk > 1 { c_bytes * sk as usize } else { c_bytes };

        let a_buf = rt.alloc(a_bytes.max(4096))?;
        let b_buf = rt.alloc(b_bytes.max(4096))?;
        let c_buf = rt.alloc(c_total.max(4096))?;

        // Zero buffers (content doesn't matter for timing)
        a_buf.zero();
        b_buf.zero();
        c_buf.zero();

        // 3. Build kernargs
        let ka = super::gemm_gen::build_kernargs(
            a_buf.gpu_addr(), b_buf.gpu_addr(), c_buf.gpu_addr(),
            k, n, m, cfg,
        );

        // 4. Compute grid
        let (grid_x, grid_y) = super::gemm_gen::compute_grid_auto(cfg, m, n);

        // 5. Warmup
        rt.dispatch(&gpu_kernel, [grid_x, grid_y, 1], &ka)?;

        // 6. Benchmark
        let mut times_ns: Vec<u64> = Vec::new();
        for _ in 0..self.n_iters {
            let start = std::time::Instant::now();
            rt.dispatch(&gpu_kernel, [grid_x, grid_y, 1], &ka)?;
            times_ns.push(start.elapsed().as_nanos() as u64);
        }

        // Recycle buffers
        rt.recycle(a_buf);
        rt.recycle(b_buf);
        rt.recycle(c_buf);

        // 7. Median time → TFLOPS
        times_ns.sort();
        let median_ns = times_ns[times_ns.len() / 2];
        let total_flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = total_flops / (median_ns as f64 * 1e-9) / 1e12;

        Ok(tflops)
    }

    /// Benchmark a single TileGemm (tile_ir path) on the GPU.
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn benchmark_tile_ir(
        &self,
        rt: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
        spec: &super::tile_ir::TileGemm,
        m: u32, n: u32, k: u32,
    ) -> Result<f64, String> {
        use crate::gpu_backend::{GpuKernel, KernelLoadConfig};

        // Safety: reject configs with LDS > 64KB (CWSR hang on GFX1100)
        let lds_total = spec.lds_total();
        if lds_total > 65536 {
            return Err(format!("LDS {}B > 64KB limit", lds_total));
        }

        // 1. Generate and compile kernel via tile_ir
        // Use catch_unwind because tile_ir has asserts (e.g., tile_k power-of-2)
        let spec_clone = spec.clone();
        let compile_result = std::panic::catch_unwind(move || {
            let kernel_ir = super::tile_ir::lower_gemm(&spec_clone);
            let elf = kernel_ir.compile(super::ir::Target::GFX1100);
            let lds_size = kernel_ir.lds_size();
            (elf, lds_size)
        });
        let (elf_result, lds_size) = match compile_result {
            Ok((elf, lds)) => (elf, lds),
            Err(_) => return Err("tile_ir compilation panic (likely invalid config)".into()),
        };
        let elf = elf_result?;

        let gpu_kernel = GpuKernel::load(
            &rt.device, &elf,
            &KernelLoadConfig {
                workgroup_size: [spec.wg_size(), 1, 1],
                lds_size,
            },
        )?;

        // 2. Allocate buffers
        let a_bytes = (m as usize) * (k as usize) * 2;
        let b_bytes = (n as usize) * (k as usize) * 2;
        let c_bytes = (m as usize) * (n as usize) * 4;

        let a_buf = rt.alloc(a_bytes.max(4096))?;
        let b_buf = rt.alloc(b_bytes.max(4096))?;
        let c_buf = rt.alloc(c_bytes.max(4096))?;

        a_buf.zero();
        b_buf.zero();
        c_buf.zero();

        // 3. Build kernargs (tile_ir format)
        let ka = super::tile_ir::build_kernargs_m(
            a_buf.gpu_addr(), b_buf.gpu_addr(), c_buf.gpu_addr(),
            k, n, m, spec,
        );
        let grid = super::tile_ir::compute_grid(spec, m, n);

        // 4. Warmup
        rt.dispatch(&gpu_kernel, grid, &ka)?;

        // 5. Benchmark
        let mut times_ns: Vec<u64> = Vec::new();
        for _ in 0..self.n_iters {
            let start = std::time::Instant::now();
            rt.dispatch(&gpu_kernel, grid, &ka)?;
            times_ns.push(start.elapsed().as_nanos() as u64);
        }

        rt.recycle(a_buf);
        rt.recycle(b_buf);
        rt.recycle(c_buf);

        // 6. Median time → TFLOPS
        times_ns.sort();
        let median_ns = times_ns[times_ns.len() / 2];
        let total_flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = total_flops / (median_ns as f64 * 1e-9) / 1e12;

        Ok(tflops)
    }

    // ── Cache persistence ──

    fn cache_path(&self, m: u32, n: u32, k: u32) -> PathBuf {
        self.cache_dir.join(format!("gemm_{}x{}x{}.json", m, n, k))
    }

    fn load_cache(&self, m: u32, n: u32, k: u32) -> Option<GemmTuneResult> {
        let path = self.cache_path(m, n, k);
        let content = std::fs::read_to_string(&path).ok()?;

        // Parse minimal JSON: {"best":"name","tflops":79.2,"tile_m":128,...}
        let tile_m = parse_u32(&content, "tile_m")?;
        let tile_n = parse_u32(&content, "tile_n")?;
        let tile_k = parse_u32(&content, "tile_k")?;
        let wg_size = parse_u32(&content, "wg_size")?;
        let tflops = parse_f64(&content, "tflops")?;
        let split_k_val = parse_u32(&content, "split_k").unwrap_or(1);
        let wgp = content.contains("\"wgp\":true");
        let swap = content.contains("\"swap\":true");

        let cfg = GemmConfig {
            tile_m, tile_n, tile_k, wg_size,
            use_lds: true, double_buffer: true,
            split_k: if split_k_val > 1 { Some(split_k_val) } else { None },
            lds_pad: parse_u32(&content, "lds_pad").unwrap_or(0),
            n_col_passes: parse_u32(&content, "n_col_passes").unwrap_or(1),
            swap_grid: swap,
            wgp_mode: wgp,
            transpose: super::gemm_gen::GemmTranspose::NT,
            epilogue: super::gemm_gen::EpilogueOp::StoreF32,
        };

        Some(GemmTuneResult {
            best: cfg,
            best_tflops: tflops,
            all: Vec::new(),
            key: (m, n, k),
            from_cache: true,
        })
    }

    fn save_cache(&self, m: u32, n: u32, k: u32, result: &GemmTuneResult) {
        let _ = std::fs::create_dir_all(&self.cache_dir);
        let path = self.cache_path(m, n, k);
        let cfg = &result.best;
        let sk = cfg.split_k.unwrap_or(1);

        let all_json: Vec<String> = result.all.iter()
            .map(|(name, tf)| format!("[\"{}\",{:.2}]", name, tf))
            .collect();

        let json = format!(
            concat!(
                "{{\"tile_m\":{},\"tile_n\":{},\"tile_k\":{},\"wg_size\":{},",
                "\"split_k\":{},\"lds_pad\":{},\"n_col_passes\":{},",
                "\"wgp\":{},\"swap\":{},\"tflops\":{:.2},",
                "\"all\":[{}]}}"
            ),
            cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.wg_size,
            sk, cfg.lds_pad, cfg.n_col_passes,
            cfg.wgp_mode, cfg.swap_grid,
            result.best_tflops,
            all_json.join(","),
        );

        if let Err(e) = std::fs::write(&path, &json) {
            eprintln!("[autotune] Cache write failed: {}", e);
        }
    }

    /// Invalidate cache for a specific problem size.
    pub fn invalidate(&mut self, m: u32, n: u32, k: u32) {
        self.cache.remove(&(m, n, k));
        let _ = std::fs::remove_file(self.cache_path(m, n, k));
    }

    /// Clear all cached results.
    pub fn clear(&mut self) {
        self.cache.clear();
        let _ = std::fs::remove_dir_all(&self.cache_dir);
    }

    /// Print a report of all cached results.
    pub fn report(&self) {
        eprintln!("╔═══════════════════════════════════════════════════╗");
        eprintln!("║  GEMM Autotune Cache ({} entries)               ║", self.cache.len());
        eprintln!("╠═══════════════════════════════════════════════════╣");
        for ((m, n, k), result) in &self.cache {
            eprintln!("║  {}×{}×{} → {} ({:.1} TF) {}",
                m, n, k, result.best.name(), result.best_tflops,
                if result.from_cache { "[cached]" } else { "[measured]" });
        }
        eprintln!("╚═══════════════════════════════════════════════════╝");
    }
}

// ── Global tuner (lazy singleton) ──

#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
use std::sync::Mutex;

#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
static GLOBAL_TUNER: std::sync::OnceLock<Mutex<GemmTuner>> = std::sync::OnceLock::new();

/// Get or create the global GEMM tuner.
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub fn global_tuner() -> &'static Mutex<GemmTuner> {
    GLOBAL_TUNER.get_or_init(|| Mutex::new(GemmTuner::new()))
}

/// One-shot: tune and dispatch GEMM with the best configuration.
///
/// Equivalent to Triton's `@triton.autotune` + `@triton.jit`:
/// - First call for a given (M,N,K): benchmarks candidates, selects best (~2-5s)  
/// - Subsequent calls: cache hit, zero overhead
///
/// # Example
/// ```rust,no_run
/// # use std::sync::Arc;
/// auto_gemm(&rt, &a_buf, &b_buf, &c_buf, 4096, 4096, 4096).unwrap();
/// ```
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub fn auto_gemm(
    rt: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
    a_buf: &crate::gpu_backend::GpuBuffer,
    b_buf: &crate::gpu_backend::GpuBuffer,
    c_buf: &crate::gpu_backend::GpuBuffer,
    m: u32, n: u32, k: u32,
) -> Result<f64, String> {
    // 1. Tune (or cache hit)
    let cfg = {
        let mut tuner = global_tuner().lock().map_err(|e| e.to_string())?;
        tuner.tune(rt, m, n, k)?
    };

    // 2. Compile kernel (GpuRuntime kernel cache handles dedup)
    let kernel = rt.ensure_kernel_t0(
        &cfg.name(),
        || super::gemm_gen::generate(&cfg),
        [cfg.wg_size, 1, 1],
        super::gemm_gen::generate(&cfg).lds_size(),
    )?;

    // 3. Build kernargs + dispatch
    let ka = super::gemm_gen::build_kernargs(
        a_buf.gpu_addr(), b_buf.gpu_addr(), c_buf.gpu_addr(),
        k, n, m, &cfg,
    );
    let (grid_x, grid_y) = super::gemm_gen::compute_grid_auto(&cfg, m, n);
    rt.dispatch(&kernel, [grid_x, grid_y, 1], &ka)?;

    // 4. Return achieved TFLOPS (from cache)
    let tuner = global_tuner().lock().map_err(|e| e.to_string())?;
    let tflops = tuner.cache.get(&(m, n, k))
        .map(|r| r.best_tflops)
        .unwrap_or(0.0);
    Ok(tflops)
}

// ── Helpers ──

fn default_cache_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".t0_autotune")
    } else {
        PathBuf::from("/tmp/.t0_autotune")
    }
}

fn parse_u32(json: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    if end == 0 { return None; }
    rest[..end].parse().ok()
}

fn parse_f64(json: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(rest.len());
    if end == 0 { return None; }
    rest[..end].parse().ok()
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_helpers() {
        let json = r#"{"tile_m":128,"tile_n":64,"tile_k":32,"wg_size":128,"tflops":79.21,"split_k":1}"#;
        assert_eq!(parse_u32(json, "tile_m"), Some(128));
        assert_eq!(parse_u32(json, "tile_n"), Some(64));
        assert_eq!(parse_u32(json, "tile_k"), Some(32));
        assert_eq!(parse_u32(json, "wg_size"), Some(128));
        assert_eq!(parse_u32(json, "split_k"), Some(1));
        let tf = parse_f64(json, "tflops").unwrap();
        assert!((tf - 79.21).abs() < 0.01);
    }

    #[test]
    fn test_cache_dir_default() {
        let dir = default_cache_dir();
        assert!(dir.to_str().unwrap().contains("t0_autotune"));
    }

    #[test]
    fn test_tuner_new() {
        let tuner = GemmTuner::new();
        assert_eq!(tuner.max_candidates, 8);
        assert_eq!(tuner.n_iters, 5);
        assert!(tuner.cache.is_empty());
    }

    /// GPU E2E: tune 4096³ GEMM and verify TFLOPS > 50
    #[test]
    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    fn test_tune_4096() {
        let rt = crate::ignis::gpu_context::GpuRuntime::new()
            .expect("GpuRuntime::new");

        let mut tuner = GemmTuner::with_cache_dir(
            PathBuf::from("/tmp/t0_autotune_test")
        );
        tuner.max_candidates = 4; // fewer for test speed
        tuner.n_iters = 3;

        let cfg = tuner.tune(&rt, 4096, 4096, 4096)
            .expect("tune failed");

        eprintln!("Best config: {}", cfg.name());
        let result = tuner.cache.get(&(4096, 4096, 4096)).unwrap();
        eprintln!("TFLOPS: {:.1}", result.best_tflops);

        assert!(result.best_tflops > 50.0,
            "Expected > 50 TFLOPS, got {:.1}", result.best_tflops);

        // Print full ranking
        for (name, tf) in &result.all {
            eprintln!("  {} → {:.1} TF", name, tf);
        }

        // Verify cache hit
        let cfg2 = tuner.tune(&rt, 4096, 4096, 4096).unwrap();
        assert_eq!(cfg.name(), cfg2.name(), "cache should return same config");

        // Cleanup
        let _ = std::fs::remove_dir_all("/tmp/t0_autotune_test");
    }

    /// Test cache roundtrip (save + load)
    #[test]
    fn test_cache_roundtrip() {
        let dir = PathBuf::from("/tmp/t0_autotune_cache_test");
        let _ = std::fs::remove_dir_all(&dir);

        let mut tuner = GemmTuner::with_cache_dir(dir.clone());
        let cfg = GemmConfig::tile_128x64_k32();
        let result = GemmTuneResult {
            best: cfg.clone(),
            best_tflops: 79.2,
            all: vec![
                (cfg.name(), 79.2),
                ("t0_gemm_64x64_k16_ldsdb".into(), 65.3),
            ],
            key: (4096, 4096, 4096),
            from_cache: false,
        };

        tuner.save_cache(4096, 4096, 4096, &result);

        // Load back
        let loaded = tuner.load_cache(4096, 4096, 4096)
            .expect("cache load failed");
        assert_eq!(loaded.best.tile_m, 128);
        assert_eq!(loaded.best.tile_n, 64);
        assert_eq!(loaded.best.tile_k, 32);
        assert!((loaded.best_tflops - 79.2).abs() < 0.1);
        assert!(loaded.from_cache);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
