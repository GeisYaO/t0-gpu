//! # test_gemm_correctness — GPU vs CPU GEMM Correctness & Precision Test
//!
//! Tests all T0 GEMM configurations against a CPU bf16 reference implementation.
//! Uses random bf16 data to detect systematic errors (not just all-ones).
//!
//! Run: cargo run --example test_gemm_correctness --features rocm --release
//!
//! Reports: max absolute error, mean absolute error, relative error, PASS/FAIL

use t0_gpu::t0::Target;
use t0_gpu::t0::gemm_gen::{GemmConfig, generate, compute_grid, compute_grid_split_k};

// ── bf16 helpers ──

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Round to nearest even
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits + round) >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

// ── CPU reference GEMM ──
// Y[M,N] = X[M,K] × WT[N,K]  (both X, WT row-major bf16, Y f32)
fn cpu_gemm_bf16(x: &[u16], wt: &[u16], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a = bf16_to_f32(x[i * k + kk]);
                let b = bf16_to_f32(wt[j * k + kk]);
                acc += a * b;
            }
            y[i * n + j] = acc;
        }
    }
    y
}

// ── Random bf16 data generation ──
fn rand_bf16(n: usize, seed: u64) -> Vec<u16> {
    // Simple xorshift64 PRNG
    let mut state = seed ^ 0xDEADBEEFCAFEBABE;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Generate float in [-1, 1] range, then convert to bf16
        let f = ((state >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        out.push(f32_to_bf16(f));
    }
    out
}

fn main() -> Result<(), String> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  T0 GEMM Correctness & Precision Test                      ║");
    eprintln!("║  GPU (bf16 WMMA) vs CPU reference (f32 accumulate)         ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // Configs to test
    let configs: Vec<GemmConfig> = vec![
        GemmConfig::tile_64x64_k16(),
        GemmConfig::tile_64x64_k32(),
        GemmConfig::tile_128x64_k32(),
        GemmConfig { split_k: Some(2), ..GemmConfig::tile_128x64_k32() },
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k16() },
        GemmConfig { split_k: Some(8), ..GemmConfig::tile_64x64_k16() },
        GemmConfig::tile_32x128_k16(),
        GemmConfig { split_k: Some(4), ..GemmConfig::tile_32x128_k16() },
    ];

    // Test sizes: (M, K, N)
    let sizes: Vec<(u32, u32, u32)> = vec![
        (64, 64, 64),        // tiny
        (128, 128, 128),     // small
        (256, 256, 256),     // medium
        (64, 256, 64),       // rectangular
        (128, 1024, 256),    // tall K
        (256, 512, 512),     // asymmetric
    ];

    // Compile all kernels
    eprintln!("\n── Compiling {} kernel variants ──", configs.len());
    let mut compiled = Vec::new();
    for cfg in &configs {
        eprint!("  [{}] ... ", cfg.name());
        let kernel_ir = generate(cfg);
        let elf = kernel_ir.compile(Target::GFX1100)?;
        eprintln!("✓ {} bytes", elf.len());
        compiled.push(elf);
    }

    #[cfg(any(feature = "rocm", feature = "wsl_dxg"))]
    {
        use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig, DispatchPool};

        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 32)?;

        // Load all kernels
        let mut gpu_kernels = Vec::new();
        for (ci, cfg) in configs.iter().enumerate() {
            let gk = GpuKernel::load(&device, &compiled[ci], &KernelLoadConfig {
                workgroup_size: [cfg.wg_size, 1, 1],
                lds_size: cfg.lds_total(),
            })?;
            gpu_kernels.push(gk);
        }

        let mut total_tests = 0u32;
        let mut total_pass = 0u32;
        let mut total_fail = 0u32;

        eprintln!("\n══ Running Correctness Tests ══\n");
        eprintln!("{:>6} {:>6} {:>6} | {:>26} | {:>10} {:>10} {:>10} | {}",
            "M", "K", "N", "Config", "MaxErr", "AvgErr", "RelErr", "Result");
        eprintln!("{}", "-".repeat(105));

        for &(m, k, n) in &sizes {
            // Generate random bf16 data
            let x_data = rand_bf16((m * k) as usize, (m as u64) * 1000 + (k as u64));
            let wt_data = rand_bf16((n * k) as usize, (n as u64) * 2000 + (k as u64));

            // CPU reference
            let y_ref = cpu_gemm_bf16(&x_data, &wt_data, m as usize, k as usize, n as usize);

            // Upload X and WT
            let x_bytes = (m * k * 2) as usize;
            let w_bytes = (n * k * 2) as usize;
            let y_bytes = (m * n * 4) as usize;

            let x_buf = device.alloc_vram(x_bytes)?;
            let w_buf = device.alloc_vram(w_bytes)?;
            x_buf.write(unsafe {
                std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_bytes)
            });
            w_buf.write(unsafe {
                std::slice::from_raw_parts(wt_data.as_ptr() as *const u8, w_bytes)
            });

            for (ci, cfg) in configs.iter().enumerate() {
                let sk = cfg.split_k.unwrap_or(1);
                // Check compatibility
                let effective_n = cfg.tile_n * cfg.n_col_passes;
                if m % cfg.tile_m != 0 || n % effective_n != 0 || k % (cfg.tile_k * sk) != 0 {
                    continue;
                }

                total_tests += 1;

                let (grid_x, grid_y) = if sk > 1 {
                    compute_grid_split_k(cfg, m, n, sk)
                } else {
                    compute_grid(cfg, m, n)
                };

                // Allocate output + optional split-K workspace
                let y_stride = if sk > 1 { (m * n * 4) } else { 0u32 };
                let y_alloc = if sk > 1 {
                    device.alloc_vram(y_bytes * sk as usize)?
                } else {
                    device.alloc_vram(y_bytes)?
                };

                // Zero-fill output
                let zeros = vec![0u8; if sk > 1 { y_bytes * sk as usize } else { y_bytes }];
                y_alloc.write(&zeros);

                // Build kernargs
                let mut ka = [0u8; 40];
                ka[0..8].copy_from_slice(&x_buf.gpu_addr().to_le_bytes());
                ka[8..16].copy_from_slice(&w_buf.gpu_addr().to_le_bytes());
                ka[16..24].copy_from_slice(&y_alloc.gpu_addr().to_le_bytes());
                ka[24..28].copy_from_slice(&k.to_le_bytes());
                ka[28..32].copy_from_slice(&n.to_le_bytes());
                ka[32..36].copy_from_slice(&0u32.to_le_bytes());
                ka[36..40].copy_from_slice(&y_stride.to_le_bytes());

                let ka_buf = pool.write_kernargs(0, &ka);
                queue.submit(&gpu_kernels[ci], [grid_x, grid_y, 1], ka_buf);
                queue.wait_idle()?;

                // For split-K: reduce partial results
                let mut y_gpu = vec![0f32; (m * n) as usize];
                if sk > 1 {
                    // Sum split-K partials on CPU
                    let total_elems = (m * n) as usize;
                    let mut partials = vec![0f32; total_elems * sk as usize];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            y_alloc.host_ptr as *const f32,
                            partials.as_mut_ptr(),
                            total_elems * sk as usize,
                        );
                    }
                    for s in 0..sk as usize {
                        for i in 0..total_elems {
                            y_gpu[i] += partials[s * total_elems + i];
                        }
                    }
                } else {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            y_alloc.host_ptr as *const f32,
                            y_gpu.as_mut_ptr(),
                            (m * n) as usize,
                        );
                    }
                }

                // Compare
                let mut max_err = 0.0f32;
                let mut sum_err = 0.0f64;
                let mut sum_ref = 0.0f64;
                let mut max_err_idx = 0usize;
                let total = (m * n) as usize;

                for i in 0..total {
                    let err = (y_gpu[i] - y_ref[i]).abs();
                    if err > max_err {
                        max_err = err;
                        max_err_idx = i;
                    }
                    sum_err += err as f64;
                    sum_ref += (y_ref[i].abs()) as f64;
                }

                let avg_err = sum_err / total as f64;
                let rel_err = if sum_ref > 0.0 { sum_err / sum_ref } else { 0.0 };

                // bf16 has ~7-bit mantissa, so relative error should be < K * 2^-7
                // For K=1024, that's ~8.0. Absolute tolerance scales with K.
                let tol = (k as f32) * 0.01; // 1% of K (generous for bf16)
                let pass = max_err < tol;

                let short_name = cfg.name().replace("t0_gemm_", "");
                let marker = if pass { "✅ PASS" } else { "❌ FAIL" };

                if pass {
                    total_pass += 1;
                } else {
                    total_fail += 1;
                }

                eprintln!("{:>6} {:>6} {:>6} | {:>26} | {:>10.4e} {:>10.4e} {:>10.4e} | {}",
                    m, k, n, short_name, max_err, avg_err, rel_err, marker);

                if !pass {
                    let row = max_err_idx / n as usize;
                    let col = max_err_idx % n as usize;
                    eprintln!("  └─ worst at [{},{}]: gpu={:.6} ref={:.6} diff={:.6}",
                        row, col, y_gpu[max_err_idx], y_ref[max_err_idx],
                        y_gpu[max_err_idx] - y_ref[max_err_idx]);
                }

                drop(y_alloc);
            }

            drop(x_buf);
            drop(w_buf);
        }

        eprintln!("\n══ Summary ══");
        eprintln!("  Total: {} tests, {} PASS, {} FAIL", total_tests, total_pass, total_fail);
        if total_fail > 0 {
            eprintln!("  ⚠️  {} tests FAILED!", total_fail);
            std::process::exit(1);
        } else {
            eprintln!("  ✅ All tests passed!");
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        eprintln!("\nKFD runtime not available. Compile with --features rocm");
    }

    Ok(())
}
