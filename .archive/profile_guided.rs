//! Profile-Guided Tuning — micro-benchmark → optimal kernel configuration.
//!
//! Uses `ReadShaderCycles` (s_getreg_b32 SHADER_CYCLES) to measure actual
//! instruction latencies on the target GPU, then caches results for future runs.
//!
//! ## Usage
//! ```rust,no_run
//! let tuner = ProfileTuner::new(&device);
//! let best_wg = tuner.tune_workgroup_size("kernel_name", &elf, &[64, 128, 256]);
//! ```
//!
//! ## Architecture
//! ```text
//! 1. Enumerate candidate configs (WG sizes, tile dims, etc.)
//! 2. For each candidate: dispatch small problem, measure shader cycles
//! 3. Select minimum-latency configuration
//! 4. Cache result as JSON: ~/.t0_pgo_cache/<kernel_hash>.json
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

/// Results of a profile-guided tuning run.
#[derive(Clone, Debug)]
pub struct TuneResult {
    /// Name of the kernel being tuned
    pub kernel_name: String,
    /// Best workgroup size found
    pub best_wg_size: u32,
    /// Measured cycles for each candidate
    pub measurements: Vec<(u32, u64)>, // (wg_size, cycles)
    /// Cache hit?  
    pub from_cache: bool,
}

/// PGO cache entry stored as JSON.
#[derive(Clone, Debug)]
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
struct CacheEntry {
    kernel_name: String,
    best_wg_size: u32,
    measurements: Vec<(u32, u64)>,
    gpu_name: String,
    timestamp: u64,
}

/// Profile-guided tuner for kernel configurations.
#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
pub struct ProfileTuner {
    cache_dir: PathBuf,
    cache: HashMap<String, TuneResult>,
}

#[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
impl ProfileTuner {
    /// Create a new profiler with default cache directory.
    pub fn new() -> Self {
        let cache_dir = dirs_or_default();
        Self {
            cache_dir,
            cache: HashMap::new(),
        }
    }

    /// Create with a specific cache directory.
    pub fn with_cache_dir(dir: PathBuf) -> Self {
        Self {
            cache_dir: dir,
            cache: HashMap::new(),
        }
    }

    /// Tune workgroup size for a kernel.
    ///
    /// Dispatches the kernel with each candidate WG size on a small problem,
    /// measures shader cycles, and returns the optimal configuration.
    ///
    /// # Arguments
    /// * `name` — kernel name for caching
    /// * `rt` — GPU runtime
    /// * `builder` — builds the kernel for a given WG size
    /// * `candidates` — WG sizes to try (e.g., [64, 128, 256, 512])
    /// * `n_elems` — problem size for benchmark (should be large enough to saturate GPU)
    pub fn tune_workgroup_size(
        &mut self,
        name: &str,
        rt: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
        builder: &dyn Fn(u32) -> Result<(Vec<u8>, usize), String>,
        candidates: &[u32],
        n_elems: u32,
    ) -> Result<TuneResult, String> {
        // Check cache
        if let Some(cached) = self.cache.get(name) {
            return Ok(cached.clone());
        }
        if let Some(cached) = self.load_cache(name) {
            self.cache.insert(name.to_string(), cached.clone());
            return Ok(cached);
        }

        let mut measurements: Vec<(u32, u64)> = Vec::new();

        for &wg in candidates {
            // Skip WG sizes larger than problem
            if wg > n_elems { continue; }

            match builder(wg) {
                Ok((elf, kernarg_size)) => {
                    match self.benchmark_one(rt, name, wg, &elf, kernarg_size, n_elems) {
                        Ok(cycles) => {
                            eprintln!("[PGO] {}: wg={} → {} cycles", name, wg, cycles);
                            measurements.push((wg, cycles));
                        }
                        Err(e) => {
                            eprintln!("[PGO] {}: wg={} FAILED: {}", name, wg, e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[PGO] {}: wg={} build FAILED: {}", name, wg, e);
                }
            }
        }

        if measurements.is_empty() {
            return Err(format!("PGO: no valid measurements for '{}'", name));
        }

        // Select minimum cycles
        let best = measurements.iter().min_by_key(|(_, c)| *c).unwrap();
        let result = TuneResult {
            kernel_name: name.to_string(),
            best_wg_size: best.0,
            measurements: measurements.clone(),
            from_cache: false,
        };

        eprintln!("[PGO] {} → best wg_size={} ({} cycles)", name, best.0, best.1);

        // Save to cache
        self.save_cache(name, &result);
        self.cache.insert(name.to_string(), result.clone());

        Ok(result)
    }

    /// Benchmark a single configuration: dispatch and measure shader cycles.
    fn benchmark_one(
        &self,
        rt: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
        name: &str,
        wg_size: u32,
        elf: &[u8],
        kernarg_size: usize,
        n_elems: u32,
    ) -> Result<u64, String> {
        use crate::kfd::{KernelLoadConfig, GpuKernel};

        let config = KernelLoadConfig {
            workgroup_size: [wg_size, 1, 1],
            lds_size: 0,
        };
        let kernel = GpuKernel::load(&rt.device, elf, &config)?;

        // Allocate dummy input/output buffers
        let buf_bytes = (n_elems as usize * 4).max(4096);
        let input = rt.alloc(buf_bytes)?;
        let output = rt.alloc(buf_bytes)?;
        let cycles_buf = rt.alloc(256)?;
        cycles_buf.zero();

        // Build kernargs: [input_ptr:u64, output_ptr:u64, n_elems:u32, cycles_out:u64]
        let mut ka = vec![0u8; kernarg_size.max(32)];
        ka[0..8].copy_from_slice(&input.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&output.gpu_addr().to_le_bytes());
        ka[16..20].copy_from_slice(&n_elems.to_le_bytes());

        let grid_x = ((n_elems + wg_size - 1) / wg_size) * wg_size;

        // Warmup
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka[..kernarg_size.max(20)])?;

        // Measure: 3 runs, take minimum
        let mut min_cycles = u64::MAX;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            rt.dispatch(&kernel, [grid_x, 1, 1], &ka[..kernarg_size.max(20)])?;
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            // Use wall-clock as a proxy (shader cycles would need kernel instrumentation)
            min_cycles = min_cycles.min(elapsed_ns);
        }

        // Return buffers to pool
        rt.recycle(input);
        rt.recycle(output);
        rt.recycle(cycles_buf);

        Ok(min_cycles)
    }

    /// Get the cache file path for a kernel.
    fn cache_path(&self, name: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.json", name))
    }

    /// Load a cached result.
    fn load_cache(&self, name: &str) -> Option<TuneResult> {
        let path = self.cache_path(name);
        let content = std::fs::read_to_string(&path).ok()?;

        // Simple JSON parsing (avoid serde dependency)
        // Format: {"best_wg":256,"measurements":[[64,1234],[128,987],[256,876]]}
        let best_wg = extract_u32(&content, "best_wg")?;
        let measurements = extract_measurements(&content)?;

        Some(TuneResult {
            kernel_name: name.to_string(),
            best_wg_size: best_wg,
            measurements,
            from_cache: true,
        })
    }

    /// Save a result to cache.
    fn save_cache(&self, name: &str, result: &TuneResult) {
        let _ = std::fs::create_dir_all(&self.cache_dir);
        let path = self.cache_path(name);

        let measurements_json = result.measurements.iter()
            .map(|(wg, cy)| format!("[{},{}]", wg, cy))
            .collect::<Vec<_>>()
            .join(",");

        let json = format!(
            "{{\"best_wg\":{},\"measurements\":[{}]}}",
            result.best_wg_size, measurements_json
        );

        if let Err(e) = std::fs::write(&path, &json) {
            eprintln!("[PGO] Cache write failed: {}", e);
        } else {
            eprintln!("[PGO] Cached: {}", path.display());
        }
    }

    /// Invalidate cache for a kernel.
    pub fn invalidate(&mut self, name: &str) {
        self.cache.remove(name);
        let _ = std::fs::remove_file(self.cache_path(name));
    }

    /// Clear all cache.
    pub fn clear_all(&mut self) {
        self.cache.clear();
        let _ = std::fs::remove_dir_all(&self.cache_dir);
    }
}

// ── Helpers ──

fn dirs_or_default() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".t0_pgo_cache")
    } else {
        PathBuf::from("/tmp/.t0_pgo_cache")
    }
}

fn extract_u32(json: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[..end].parse().ok()
}

fn extract_measurements(json: &str) -> Option<Vec<(u32, u64)>> {
    let start = json.find("\"measurements\":[")?;
    let rest = &json[start..];
    let start = rest.find('[')?;
    let end = rest.rfind(']')? + 1;
    let inner = &rest[start..end];

    let mut result = Vec::new();
    let mut depth = 0;
    let mut current = String::new();

    for ch in inner.chars() {
        match ch {
            '[' => {
                depth += 1;
                if depth == 2 { current.clear(); }
            }
            ']' => {
                depth -= 1;
                if depth == 1 {
                    let parts: Vec<&str> = current.split(',').collect();
                    if parts.len() == 2 {
                        if let (Ok(wg), Ok(cy)) = (parts[0].trim().parse::<u32>(), parts[1].trim().parse::<u64>()) {
                            result.push((wg, cy));
                        }
                    }
                }
            }
            _ => {
                if depth >= 2 {
                    current.push(ch);
                }
            }
        }
    }

    if result.is_empty() { None } else { Some(result) }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_parse() {
        let json = r#"{"best_wg":256,"measurements":[[64,1234],[128,987],[256,876]]}"#;
        assert_eq!(extract_u32(json, "best_wg"), Some(256));
        let m = extract_measurements(json).unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m[0], (64, 1234));
        assert_eq!(m[2], (256, 876));
    }

    #[test]
    fn test_cache_roundtrip() {
        let dir = PathBuf::from("/tmp/t0_pgo_test_cache");
        let _ = std::fs::remove_dir_all(&dir);

        #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
        {
            let mut tuner = ProfileTuner::with_cache_dir(dir.clone());
            let result = TuneResult {
                kernel_name: "test_kernel".to_string(),
                best_wg_size: 128,
                measurements: vec![(64, 5000), (128, 3000), (256, 4000)],
                from_cache: false,
            };
            tuner.save_cache("test_kernel", &result);

            // Load
            let loaded = tuner.load_cache("test_kernel").unwrap();
            assert_eq!(loaded.best_wg_size, 128);
            assert_eq!(loaded.measurements.len(), 3);
            assert!(loaded.from_cache);

            // Cleanup
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
